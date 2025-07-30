"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import __init__
from openpoints.dataset.build import DATASETS as BUILD_DATASETS
from openpoints.dataset.OpenTrench3D.OpenTrench3D import OpenTrench3D
BUILD_DATASETS.module_dict['OpenTrench3D'] = OpenTrench3D
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port, load_checkpoint_inv
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings
import openpoints.dataset.vis3d as vis3d    # ── 新增：导入模块，准备打补丁
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from openpoints.dataset.build import build_dataset_from_cfg
warnings.simplefilter(action='ignore', category=FutureWarning)

class ImbalancedDatasetSampler(Sampler):
    """
    针对子云级别的不均衡：让那些含有更多少数类点的样本被抽到的概率更高。
    假设 dataset[i] 返回 dict，其中 dict['label'] 是 shape (Ni,) 的标签数组。
    """
    def __init__(self, dataset):
        sample_weights = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            # 优先读取 data['y']，它是 (Ni,1) 或 (Ni,) 的 Tensor
            if 'y' in item:
                labels = item['y']
            # 兼容老代码：如果有 'label'
            elif 'label' in item:
                labels = item['label']
            else:
                raise KeyError(
                    f"Sampler 找不到标签字段，请检查 dataset[{idx}].keys() = {list(item.keys())}"
                )
            # 把 Tensor 转成 numpy 一维数组
            if isinstance(labels, torch.Tensor):
                labels = labels.squeeze(-1).cpu().numpy()
            else:
                labels = np.array(labels).squeeze()   # 按你的 dataset API 可能要改成 ['labels']
            counts = np.bincount(labels, minlength=dataset.num_classes)
            # 少数类是标签 0，对应 counts[0]
            minority_count = counts[0]
            # 如果一个子云里完全没有 minority（Other），就不给它采样机会
            if minority_count == 0:
                sample_weights.append(0.0)
            else:
                # 子云里其他类点越多，权重越高
                sample_weights.append(float(minority_count))
        self.weights = torch.DoubleTensor(sample_weights)

    def __iter__(self):
        # 每个 epoch 有放回地抽 len(dataset) 次
        return iter(torch.multinomial(
            self.weights, len(self.weights), replacement=True
        ).tolist())

    def __len__(self):
        return len(self.weights)

def focal_loss(inputs, targets, alpha, gamma=2.0, ignore_index=None):
    """
    inputs: (B, C, N) logits  targets: (B, N) long  
    alpha: Tensor(C,) class balance weights  
    """
    B, C, N = inputs.shape
    preds = inputs.permute(0,2,1).reshape(-1, C)
    gts   = targets.view(-1)
    valid = (gts >= 0) & (gts < C)
    preds = preds[valid]
    gts   = gts[valid]
    # 若无有效样本，直接返回 0 loss
    if preds.numel() == 0:
        return torch.tensor(0.0, device=inputs.device)
    # 计算 focal loss
    logpt = -F.cross_entropy(preds, gts, weight=None, reduction='none')
    pt    = torch.exp(logpt)
    at    = alpha[gts]
    loss  = -at * ((1 - pt) ** gamma) * logpt
    return loss.mean()


# ★—— Monkey-patch 全局 write_obj ——★
def _patched_write_obj(points, colors, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as fout:
        for i in range(points.shape[0]):
            x, y, z = points[i]
            # 如果 colors 是 (N,3)，取第 i 行；否则当 (3,) 用
            ci = colors[i] if (hasattr(colors, "ndim") and colors.ndim == 2) else colors
            fout.write('v %f %f %f %f %f %f\n' % (
                float(x), float(y), float(z),
                float(ci[0]), float(ci[1]), float(ci[2])
            ))
# 覆盖掉原模块里的 write_obj
vis3d.write_obj = _patched_write_obj

def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def generate_data_list(cfg):
    if 's3dis' in cfg.dataset.common.NAME.lower():
        raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data_list = glob.glob(os.path.join(cfg.dataset.common.data_root, cfg.dataset.test.split, "*.pth"))
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        if cfg.dataset.test.split == 'val':
            split_no = 1
        else:
            split_no = 2
        data_list = get_semantickitti_file_list(os.path.join(cfg.dataset.common.data_root, 'sequences'),
                                                str(cfg.dataset.test.test_id + 11))[split_no]
        # —— 新增对 OpenTrench3D (.ply) 的支持 —— 
    elif 'opentrench3d' in cfg.dataset.common.NAME.lower():
        # 取命令行或 YAML 里指定的 dataset.test.root
        base_dir = cfg.dataset.test.root
        # 如果指定了 area，就用 <root>/<area>/<split>
        area = getattr(cfg.dataset.test, 'area', None)
        if area:
            ply_dir = os.path.join(base_dir, area, cfg.dataset.test.split)
        else:
            ply_dir = os.path.join(base_dir, cfg.dataset.test.split)
        # 如果目录不存在，再退回到 root
        if not os.path.isdir(ply_dir):
            ply_dir = base_dir
        # 扫描所有 .ply 文件
        data_list = glob.glob(os.path.join(ply_dir, '*.ply'))
    else:
        # 修复 args.data_name 未定义的问题，并输出实际 NAME
        raise Exception(f"Dataset '{cfg.dataset.common.NAME}' not supported yet")
    return data_list


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
           label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        coord = load_pc_kitti(data_path[0])
        if cfg.dataset.test.split != 'test':
            label = load_label_kitti(data_path[1], remap_lut_read)
        # —— 新增 OpenTrench3D .ply 文件读取 —— 
    elif 'opentrench3d' in cfg.dataset.common.NAME.lower():
        from plyfile import PlyData
        # ① 读坐标/颜色/原始 class
        ply   = PlyData.read(data_path)
        v     = ply['vertex']
        coord = np.stack([v['x'],v['y'],v['z']],axis=-1).astype(np.float32)
        feat  = np.stack([v['red'],v['green'],v['blue']],axis=-1).astype(np.float32) / 255.0
        labels = np.array(v['class'], dtype=np.int64)

        # ② 过滤掉 ignore_label
        mask = labels != cfg.ignore_label
        coord, feat, labels = coord[mask], feat[mask], labels[mask]

        # ③ 二分类重映射：0=Other, 1=Trench
        trench_id = cfg.dataset.common.trench_class_id
        labels = np.where(labels == trench_id, 1, 0).astype(np.int64)

        label = labels
    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
                # 如果不做体素化，就把所有点都当作一个子云
        idx_points.append(np.arange(coord.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


def main(gpu, cfg):
    # —— 强制开启可视化输出 (.obj) —— 
    cfg.visualize = cfg.get('visualize', False)
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(    
                                            batch_size        = cfg.get('val_batch_size', cfg.batch_size),
                                            dataset_cfg       = cfg.dataset,
                                            dataloader_cfg    = cfg.dataloader.val,
                                            datatransforms_cfg= cfg.datatransforms,
                                            split             = 'val',
                                            distributed       = False,
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    
    # 确保读取到的新 num_classes=2
    if hasattr(val_loader.dataset, 'num_classes'):
        val_loader.dataset.num_classes = cfg.num_classes
    # 直接使用 cfg.num_classes，不再依赖 dataset 原始值
    num_classes = cfg.num_classes
    logging.info(f"number of classes of the dataset (forced): {num_classes}")

    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1, epoch=epoch)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return val_miou
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                data_list = generate_data_list(cfg)
                    # —— 如果命令行指定了 file_name，就只推理这一张 —— 
                file_name = getattr(cfg.dataset.test, 'file_name', None)
                if file_name:
                    # 只保留后缀匹配的那一条
                    matches = [p for p in data_list if p.endswith(file_name)]
                    if not matches:
                        raise ValueError(f"找不到指定文件: {file_name}")
                    data_list = matches
                logging.info(f"length of test dataset: {len(data_list)}")
                test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

                if test_miou is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
                            f'\niou per cls is: {test_ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
                return test_miou

            elif 'encoder' in cfg.mode:
                if 'inv' in cfg.mode:
                    logging.info(f'Finetuning from {cfg.pretrained_path}')
                    load_checkpoint_inv(model.encoder, cfg.pretrained_path)
                else:
                    logging.info(f'Finetuning from {cfg.pretrained_path}')
                    load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))

            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
    else:
        logging.info('Training from scratch')

    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    # 先构造 dataset 本身
        # 先构造一个临时的 train loader，用它来拿到 train_dataset
    _tmp_loader = build_dataloader_from_cfg(
        batch_size        = cfg.batch_size,
        dataset_cfg       = cfg.dataset,
        dataloader_cfg    = cfg.dataloader.train,
        datatransforms_cfg= cfg.datatransforms,
        split             = 'train',
        distributed       = cfg.distributed,
    )
    train_dataset = _tmp_loader.dataset

    if cfg.get('use_imbalanced_sampler', False):
        sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,               # sampler 已经决定了“每次一个样本”
            sampler=sampler,
            num_workers=cfg.dataloader.train.num_workers,
            collate_fn=_tmp_loader.collate_fn,  # ← 改成用 tmp_loader 的 collate_fn
            pin_memory=cfg.dataloader.train.pin_memory,
            drop_last=cfg.dataloader.train.drop_last,
        )
    else:
        train_loader = build_dataloader_from_cfg(
            batch_size        = cfg.batch_size,
            dataset_cfg       = cfg.dataset,
            dataloader_cfg    = cfg.dataloader.train,
            datatransforms_cfg= cfg.datatransforms,
            split             = 'train',
            distributed       = cfg.distributed,
        )

    # —— DEBUG: 打印采样器 & 损失配置 ——  
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    if hasattr(train_loader.dataset, 'num_classes'):
        train_loader.dataset.num_classes = cfg.num_classes
        batch = next(iter(train_loader))
        labels = batch['y'].squeeze(-1).view(-1).cpu()
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"[DEBUG batch] unique labels: {unique.tolist()}, counts: {counts.tolist()}")
    logging.info(f"[DEBUG] use_imbalanced_sampler: {cfg.get('use_imbalanced_sampler', False)}")
    if cfg.get('use_imbalanced_sampler', False):
        # sampler 权重分布
        sampler_obj = train_loader.sampler
        logging.info(f"[DEBUG] sampler weights: mean={sampler_obj.weights.mean():.4f}, std={sampler_obj.weights.std():.4f}")

    # 打印 CrossEntropy 权重或 Focal alpha
    if cfg.get('cls_weighed_loss', False):
        w = cfg.criterion_args.weight
        # 如果已经是 Tensor，就转成 list；否则直接打印原 list
        if isinstance(w, torch.Tensor):
            w = w.cpu().tolist()
        logging.info(f"[DEBUG] CE class weights: {w}")
    if cfg.get('use_focal_loss', False):
        alpha = torch.tensor(cfg.criterion_args.weight if cfg.get('cls_weighed_loss') else [1]*cfg.num_classes)
        logging.info(f"[DEBUG] focal alpha: {alpha.tolist()}, gamma: {cfg.get('focal_gamma', None)}")

    # 首三批标签分布
    for i, batch in enumerate(train_loader):
        lbl = batch['y'].squeeze(-1).view(-1).cpu()
        uniq, cnt = torch.unique(lbl, return_counts=True)
        logging.info(f"[DEBUG batch {i}] labels: {uniq.tolist()}, counts: {cnt.tolist()}")
        if i >= 2: break

    # —— STEP2: 构建加权交叉熵或 Focal Loss —— 
# 先按原逻辑构建加权/非加权的 CrossEntropy
    if cfg.get('cls_weighed_loss', False):
        # 直接从 YAML 里拿 [w0, w1] 并转成 Tensor
        cfg.criterion_args.weight = torch.tensor(
            cfg.criterion_args.weight,
            dtype=torch.float32
        ).cuda()
    else:
        cfg.criterion_args.weight = None
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # —— DEBUG: 打印交叉熵的 class 权重（已经是 Tensor 了）——
    if isinstance(cfg.criterion_args.weight, torch.Tensor):
        w = cfg.criterion_args.weight.cpu().tolist()
    else:
        w = cfg.criterion_args.weight
    logging.info(f"[DEBUG] CE class weights: {w}")

    # 如果在命令行或 YAML 打开了 focal loss，就覆盖上面那个 criterion
    if cfg.get('use_focal_loss', False):
        # 准备 alpha
        if cfg.get('cls_weighed_loss', False) and cfg.criterion_args.weight is not None:
            alpha = torch.tensor(cfg.criterion_args.weight, dtype=torch.float32).cuda()
        else:
            alpha = torch.ones(cfg.num_classes, dtype=torch.float32).cuda()
        gamma = cfg.get('focal_gamma', 2.0)
        # 覆盖成 Focal Loss，不再传递已被过滤的 ignore_index
        criterion = lambda logits, target: focal_loss(
            logits, target,
            alpha=alpha,
            gamma=gamma,
            
        )
        logging.info(f"[DEBUG] focal alpha: {alpha.cpu().tolist()}, gamma: {gamma}")

    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    total_iter = 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_miou, train_macc, train_oa, _, _, total_iter = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, epoch=epoch, total_iter=total_iter)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False
    # do not save file to wandb to save wandb space
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')

    if cfg.world_size < 2:  # do not support multi gpu testing
        # test
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'.csv')
        if 'sphere' in cfg.dataset.common.NAME.lower():
            # TODO: 
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg, epoch=epoch)
        else:
            data_list = generate_data_list(cfg)
            test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                f'\niou per cls is: {test_ious}')
        if writer is not None:
            writer.add_scalar('test_miou', test_miou, epoch)
            writer.add_scalar('test_macc', test_macc, epoch)
            writer.add_scalar('test_oa', test_oa, epoch)
        write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
        logging.info(f'save results in {cfg.csv_path}')
        
        if cfg.use_voting:
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
            set_random_seed(cfg.seed)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
                                                                         epoch=epoch)
            if writer is not None:
                writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)

            ious_table = [f'{item:.2f}' for item in val_ious]
            data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}', f'{val_miou:.2f}'] + ious_table + [
                str(best_epoch), cfg.run_dir]
            with open(cfg.csv_path, 'w', encoding='UT8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        logging.warning('Testing using multiple GPUs is not allowed for now. Running testing after this training is required.')
    if writer is not None:
        writer.close()
    # dist.destroy_process_group() # comment this line due to https://github.com/guochengqian/PointNeXt/issues/95
    wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['epoch'] = epoch
        total_iter += 1 
        data['iter'] = total_iter 
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(data)
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)
            # mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            # print(f"Memory after backward is {mem}")
            
        # update confusion matrix
        pred = logits.argmax(dim=1).view(-1)    # (B*N,)
        tgt  = target.view(-1)                  # (B*N,)
        # 只保留 0 <= label < num_classes
        valid = (tgt >= 0) & (tgt < cfg.num_classes)
        pred = pred[valid]
        tgt  = tgt[valid]
        cm.update(pred, tgt)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs, total_iter


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['epoch'] = epoch
        data['iter'] = total_iter 
        logits = model(data)
        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            # 展平并仅统计有效标签
            pred = logits.argmax(dim=1).view(-1)
            tgt  = target.view(-1)
            valid = (tgt >= 0) & (tgt < cfg.num_classes)
            cm.update(pred[valid], tgt[valid])
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        coord = data['pos'].cpu().numpy()[0]
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        label = target[0].cpu().numpy()
        if cfg.ignore_index is not None:
            if (label == cfg.ignore_index).sum() > 0:
                pred[label == cfg.ignore_index] = cfg.num_classes
                label[label == cfg.ignore_index] = cfg.num_classes
        vis_multi_points([coord, coord], labels=[label, pred])
        """
        # tp, union, count = cm.tp, cm.union, cm.count
        # if cfg.distributed:
        #     dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        # miou, macc, oa, ious, accs = get_mious(tp, union, count)
        # with np.printoptions(precision=2, suppress=True):
        #     logging.info(f'{idx}-th cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
        #                 f'\niou per cls is: {ious}')

    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


@torch.no_grad()
def validate_sphere(model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1):
    """
    validation for sphere sampled input points with mask.
    in this case, between different batches, there are overlapped points.
    thus, one point can be evaluated multiple times.
    In this validate_mask, we will avg the logits.
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    if cfg.visualize:
        # 动态导入原函数（保留引用以防万一）
        write_obj = vis3d.write_obj
        # 覆盖 write_obj，修复数组格式化问题
        def write_obj(points, colors, fname):
            # points: (N,3)  colors: (N,3) or (3,)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'w') as fout:
                for i in range(points.shape[0]):
                    x, y, z = points[i]
                    ci = colors[i] if colors.ndim == 2 else colors
                    # 强制转成 Python float，确保 %f 能用
                    fout.write('v %f %f %f %f %f %f\n' % (
                        float(x), float(y), float(z),
                        float(ci[0]), float(ci[1]), float(ci[2])
                    ))
        # vis 目录
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    all_logits, idx_points = [], []
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['epoch'] = epoch
        data['iter'] = total_iter 
        logits = model(data)
        all_logits.append(logits)
        idx_points.append(data['input_inds'])
    all_logits = torch.cat(all_logits, dim=0).transpose(1, 2).reshape(-1, cfg.num_classes)
    idx_points = torch.cat(idx_points, dim=0).flatten()

    if cfg.distributed:
        dist.all_reduce(all_logits), dist.all_reduce(idx_points)

    # average overlapped predictions to subsampled points
    all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')

    # now, project the original points to the subsampled points
    # these two targets would be very similar but not the same
    # val_points_targets = all_targets[val_points_projections]
    # torch.allclose(val_points_labels, val_points_targets)
    all_logits = all_logits.argmax(dim=1)
    val_points_labels = torch.from_numpy(val_loader.dataset.clouds_points_labels[0]).squeeze(-1).to(all_logits.device)
    val_points_projections = torch.from_numpy(val_loader.dataset.projections[0]).to(all_logits.device).long()
    val_points_preds = all_logits[val_points_projections]

    del all_logits, idx_points
    torch.cuda.empty_cache()

    cm.update(val_points_preds, val_points_labels)
    miou, macc, oa, ious, accs = cm.all_metrics()

    if cfg.get('visualize', False):
        dataset_name = cfg.dataset.common.NAME.lower()
        coord = val_loader.dataset.clouds_points[0]
        colors = val_loader.dataset.clouds_points_colors[0].astype(np.float32)
        gt = val_points_labels.cpu().numpy().squeeze()
        pred = val_points_preds.cpu().numpy().squeeze()
        gt = cfg.cmap[gt, :]
        pred = cfg.cmap[pred, :]
        # output pred labels
        # save per room
        rooms = val_loader.dataset.clouds_rooms[0]

        for idx in tqdm(range(len(rooms)-1), desc='save visualization'):
            start_idx, end_idx = rooms[idx], rooms[idx+1]
            write_obj(coord[start_idx:end_idx], colors[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'input-{dataset_name}-{idx}.obj'))
            # output ground truth labels
            write_obj(coord[start_idx:end_idx], gt[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'gt-{dataset_name}-{idx}.obj'))
            # output pred labels
            write_obj(coord[start_idx:end_idx], pred[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{dataset_name}-{idx}.obj'))
    return miou, macc, oa, ious, accs

 # ── Monkey‐patch 全局 vis3d.write_obj ─────────────────────────────
def _patched_write_obj(points, colors, fname):
     os.makedirs(os.path.dirname(fname), exist_ok=True)
     with open(fname, 'w') as fout:
         for i in range(points.shape[0]):
             x, y, z = points[i]
             ci = colors[i] if colors.ndim == 2 else colors
             fout.write('v %f %f %f %f %f %f\n' % (
                 float(x), float(y), float(z),
                 float(ci[0]), float(ci[1]), float(ci[2])
             ))
vis3d.write_obj = _patched_write_obj  # ← 用补丁覆盖原函数der.

@torch.no_grad()
def test(model, data_list, cfg, num_votes=1):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        # 如果没有提供 cmap，就自动生成一个随机颜色表
        if cfg.cmap is None:
            # 每个类别随机一种 RGB 颜色，范围 [0,1]
            cfg.cmap = np.random.rand(cfg.num_classes, 3).astype(np.float32)
        else:
            # 否则对已有 cmap 做归一化
            cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'
    for cloud_idx, data_path in enumerate(data_list):
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)))
        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]]")
            if not (nearest_neighbor and idx_subcloud>0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part =  feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    if 'semantickitti' in cfg.dataset.common.NAME.lower():
                        data['heights'] = torch.from_numpy((coord_part[:, gravity_dim:gravity_dim + 1] - coord_part[:, gravity_dim:gravity_dim + 1].min()).astype(np.float32)).unsqueeze(0)
                    else:
                        data['heights'] = torch.from_numpy(coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                logits = model(data)
                """visualization in debug mode. !!! visulization is not correct, should remove ignored idx.
                from openpoints.dataset.vis3d import vis_points, vis_multi_points
                vis_multi_points([coord, coord_part], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
                """

            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        if not nearest_neighbor:
            # average merge overlapped multi voxels logits to original point set
            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
        else:
            # interpolate logits by nearest neighbor
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)
        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        vis_multi_points([coord, coord], labels=[label.cpu().numpy(), all_logits.argmax(dim=1).squeeze().cpu().numpy()])
        """
        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'

            write_obj(coord, feat,
                      os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt,
                        os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            # output pred labels
            write_obj(coord, pred,
                      os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{file_name}.obj'))

        if cfg.get('save_pred', False):
            if 'semantickitti' in cfg.dataset.common.NAME.lower():
                pred = pred + 1
                pred = pred.cpu().numpy().squeeze()
                pred = pred.astype(np.uint32)
                upper_half = pred >> 16  # get upper half for instances
                lower_half = pred & 0xFFFF  # get lower half for semantics (lower_half.shape) (100k+, )
                lower_half = remap_lut_write[lower_half]  # do the remapping of semantics
                pred = (upper_half << 16) + lower_half  # reconstruct full label
                pred = pred.astype(np.uint32)
                frame_id = data_path[0].split('/')[-1][:-4]
                store_path = os.path.join(cfg.save_path, frame_id + '.label')
                pred.tofile(store_path)
            elif 'scannet' in cfg.dataset.common.NAME.lower():
                pred = pred.cpu().numpy().squeeze()
                label_int_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
                pred=np.vectorize(label_int_mapping.get)(pred)
                save_file_name=data_path.split('/')[-1].split('_')
                save_file_name=save_file_name[0]+'_'+save_file_name[1]+'.txt'
                save_file_name=os.path.join(cfg.save_path,save_file_name)
                np.savetxt(save_file_name, pred, fmt="%d")
            elif 'opentrench3d' in cfg.dataset.common.NAME.lower():
                # —— 新增 OpenTrench3D 写入逻辑 —— 
                # pred: torch.Tensor, shape (N,)
                pred_np = pred.cpu().numpy().squeeze().astype(np.int32)
                # 从路径中取文件名（去掉扩展名）
                frame_id = os.path.splitext(os.path.basename(data_path))[0]
                out_path = os.path.join(cfg.save_path, frame_id + '.txt')
                # 保存为纯文本：每行一个类别索引
                np.savetxt(out_path, pred_np, fmt='%d')
            else:
                # —— 其他数据集的通用写法 —— 
                pred_np = pred.cpu().numpy().squeeze().astype(np.int32)
                frame_id = os.path.splitext(os.path.basename(data_path))[0]
                out_path = os.path.join(cfg.save_path, frame_id + '.txt')
                np.savetxt(out_path, pred_np, fmt='%d')

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'[{cloud_idx}]/[{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                    f'\niou per cls is: {ious}')
            all_cm.value += cm.value

    if 'scannet' in cfg.dataset.common.NAME.lower():
        logging.info(f" Please select and zip all the files (DON'T INCLUDE THE FOLDER) in {cfg.save_path} and submit it to"
                     f" Scannet Benchmark https://kaldir.vc.in.tum.de/scannet_benchmark/. ")

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, all_cm
    else:
        return None, None, None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
