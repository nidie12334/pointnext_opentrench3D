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
import os, numpy as np, torch
import json
warnings.simplefilter(action='ignore', category=FutureWarning)

class FastImbalancedSampler(Sampler):
    """
    轻量版不均衡采样：  
    只打开 <root>/<area>/<split> 下的 PLY 文件，跳过 header 后只扫描 class 列，
    忽略 ignore_label，trench_id 之外的都当成“负类”计数，生成权重。
    """
    def __init__(self, root, area, split, ignore_label, trench_id, cache_path=None):
        # 1) 找到要采样的文件列表
        ply_dir = os.path.join(root, area, split)
        paths = sorted(glob.glob(os.path.join(ply_dir, '*.ply')))

        # 2) 如果有缓存，就直接加载 JSON
        if cache_path and os.path.exists(cache_path):
            self.weights = torch.DoubleTensor(json.load(open(cache_path)))
            return

        # 3) 否则逐文件扫描 class 列
        weights = []
        for ply_path in paths:
            cnt_neg = 0
            with open(ply_path, 'r') as f:
                # 跳过 header
                for line in f:
                    if line.strip() == 'end_header':
                        break
                # 统计负类：cls != trench_id 且 != ignore_label
                for l in f:
                    cls = int(l.strip().split()[-1])
                    if cls == ignore_label:
                        continue
                    if cls != trench_id:
                        cnt_neg += 1
            weights.append(float(cnt_neg) if cnt_neg > 0 else 0.0)

        # 4) 缓存权重，下次直接加载
        if cache_path:
            json.dump(weights, open(cache_path, 'w'))

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        # 以 weights 为概率，返回 len(weights) 次有放回抽样结果
        return iter(torch.multinomial(self.weights, len(self.weights), replacement=True).tolist())

    def __len__(self):
        return len(self.weights)

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
    # 这个函数实现了Focal Loss计算，用于点云分割任务中的损失函数，特别适合处理类别不均衡问题，通过alpha（类权重）和gamma（焦点参数）调整。
    # 作用：基于输入的logits和targets计算Focal Loss，支持类平衡权重alpha。Focal Loss通过降低易分类样本的权重，焦点关注难分类样本，提高模型对少数类的敏感度。
    B, C, N = inputs.shape
    # 展平inputs到 (-1, C)（所有点的logits），targets到 (-1,)。
    preds = inputs.permute(0,2,1).reshape(-1, C)
    gts   = targets.view(-1)
    # 过滤有效标签：valid = (gts >= 0) & (gts < C)，排除无效或越界标签（C是num_classes）。
    valid = (gts >= 0) & (gts < C)
    preds = preds[valid]
    gts   = gts[valid]
    # 若无有效样本，直接返回 0 loss
    if preds.numel() == 0:
        return torch.tensor(0.0, device=inputs.device)
    # 计算交叉熵的负logpt（使用F.cross_entropy，reduction='none'得到逐点损失）
    logpt = -F.cross_entropy(preds, gts, weight=None, reduction='none')
    # pt = exp(logpt)（softmax概率的等效，表示分类置信度）。
    pt    = torch.exp(logpt)
    # at = alpha[gts]（为每个有效标签取对应类权重）。
    at    = alpha[gts]
    # 损失 = -at * (1 - pt)^gamma * logpt，求所有有效点的均值
    loss  = -at * ((1 - pt) ** gamma) * logpt
    return loss.mean()


def compute_effective_class_weights(dataset, num_classes, beta=0.9999):
     """
     Compute class-balanced weights for each class based on the effective number of samples
     defined in Cui et al., CVPR 2019. For a class with n samples, the effective number
     is (1 - beta**n) / (1 - beta), and the weight is proportional to (1 - beta)/(1 - beta**n).
     The weights are normalised so that their sum equals num_classes.

     Args:
         dataset: training dataset, each item should provide labels in item['y'] or item['label']
         num_classes: total number of classes
         beta: hyperparameter beta in [0,1). Larger beta (e.g. 0.9999) gives smoother weighting.

     Returns:
         torch.Tensor of shape (num_classes,) containing normalised class weights.
     """
     import numpy as _np
     counts = _np.zeros(num_classes, dtype=_np.float64)
     for idx in range(len(dataset)):
         item = dataset[idx]
         if 'y' in item:
             labels = item['y']
         elif 'label' in item:
             labels = item['label']
         else:
             continue
         if hasattr(labels, 'cpu'):
             labels_np = labels.squeeze(-1).cpu().numpy()
         else:
             labels_np = _np.array(labels).squeeze()
         for cls in range(num_classes):
             counts[cls] += (labels_np == cls).sum()
     counts[counts == 0] = 1.0
     weights = (1.0 - beta) / (1.0 - _np.power(beta, counts))
     weights_sum = weights.sum()
     if weights_sum > 0:
         weights = weights / weights_sum * num_classes
     weights_tensor = torch.tensor(weights, dtype=torch.float32)
     return weights_tensor

def compute_effective_class_weights_by_voxel(dataset, num_classes, beta=0.9999, voxel_size=None):
    """
    统计所有子云中，每个类别出现过的不同体素数，计算 CB 权重。
    """
    # 这个函数计算基于体素的类平衡权重（Class-Balanced weights），用于损失函数如Focal Loss的alpha参数。
    # 作用：针对点云数据的不均衡，统计每个类出现的独特体素数（而非点数），避免密度影响；然后用Cui et al. (CVPR 2019)的有效样本数公式计算权重，让少数类权重更高。
    # 构建思路：点云中点可能密集，用体素化“稀疏化”计数；用平均每文件n_i缩小量级，防止数值溢出；beta接近1让权重平滑（大类n_i大，beta^{n_i}≈0，权重≈1；小类n_i小，权重高）
    import numpy as _np
    from openpoints.dataset.data_util import voxelize

    # 累加每类出现的体素单元数量
    voxel_counts = _np.zeros(num_classes, dtype=int)
    for idx in range(len(dataset)):
        item = dataset[idx]
        coords = item['pos'].cpu().numpy()           # (N,3)
        labels = item['y'].squeeze(-1).cpu().numpy()  # (N,)
        # voxelize 返回: (sorted_idx, voxel_idx, count)
        _, voxel_idx, _ = voxelize(coords, voxel_size, mode=1)
        for cls in range(num_classes):
            if (labels == cls).any():
                voxel_counts[cls] += _np.unique(voxel_idx[labels == cls]).size
    # === 改用“平均每文件”级别的 n_i 来缩小量级 ===
    # 计算平均体素数：总体素数 / 文件数
    counts_mean = voxel_counts.astype(_np.float64) / len(dataset)
    # 防止太小或为0
    counts_mean[counts_mean < 1.0] = 1.0
    logging.info(f"[DEBUG] per-file average voxel_counts: {counts_mean.tolist()}")
    # 计算 CB 权重并归一化
    raw_w = (1.0 - beta) / (1.0 - _np.power(beta, counts_mean))
    weights = raw_w / raw_w.sum() * num_classes
    weights_tensor = torch.tensor(weights, dtype=torch.float32).cuda()
    logging.info(f"[DEBUG] computed CB alpha by voxel(avg): {weights_tensor.cpu().tolist()}")
    return weights_tensor

class TverskyLoss(nn.Module):
    """
    Tversky Loss for point cloud segmentation.
    """
    # 这个类实现了Tversky Loss，用于点云分割任务的损失函数，继承torch.nn.Module。
    # 作用：Tversky Loss是Dice Loss的泛化，通过α和β调节假阳性(FP)和假阴性(FN)的权重，让损失更关注不均衡类（e.g., α小强调召回，β小强调精度）。适用于点云中少数类分割，提高IoU-like指标。
    # 构建思路：基于Tversky指数（类似IoU），损失=1-指数；用eps防分母0；支持reduction='mean'/'sum'。

    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6, reduction='mean'):
        # 初始化参数：alpha控制FP权重（默认0.3，低值强调减少FP，提高精度），beta控制FN（默认0.7，高值强调减少FN，提高召回）。
        # eps=1e-6防数值不稳（高中：小数加eps避免除0）。
        # reduction决定损失聚合：'mean'求平均（默认，稳定训练），'sum'求和。
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (B, C, N)  网络输出的 logits
        targets: (B, N)    对应的点级标签
        """
        # 1) 将 logits 转为各类概率，形状 (B, C, N)
        probs = F.softmax(inputs, dim=1)
        # 2) 构建 one-hot 编码掩码，形状 (B, C, N)
        with torch.no_grad():
            one_hot = torch.zeros_like(probs)                 # 全零张量
            one_hot.scatter_(1, targets.unsqueeze(1), 1)      # 在 class 维度填 1
        # 3) 计算 TP、FP、FN（对 batch 和点两维求和）
        dims = (0, 2)  # 汇总维度：批次和点数
        TP = torch.sum(probs * one_hot, dims)              # 真实正类被正确预测的概率和        
        FP = torch.sum(probs * (1 - one_hot), dims)        # 错误预测为正类的概率和 
        FN = torch.sum((1 - probs) * one_hot, dims)        # 漏检的真实正类概率和
        # 4) 计算 Tversky 指数： (TP + eps) / (TP + α·FP + β·FN + eps)
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        # 5) 损失 = 1 - Tversky 指数
        loss = 1.0 - tversky
        # 6) 依据 reduction 聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
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
    """
    流程：
      1. 根据 cfg.dataset.common.NAME 判断数据集类型；
      2. 进入对应分支读取原始文件，提取坐标(coord)、特征(feat)和标签(label)；
      3. 对特征做归一化或预处理，对标签做必要过滤与重映射；
      4. 将坐标平移至原点（最小坐标为 0）；

    函数作用：
      将不同格式的数据文件统一加载为：
        - coord: (N,3) 的 float32 数组
        - feat:  (N,F) 的 float32 数组（F 依数据集而异）
        - label: (N,)  的 int64 数组或 None

    内部原理：
      通过字符串匹配选择数据集分支，分别调用 NumPy/torch/load、PlyData 解析等库函数；
      归一化将原始颜色或法向量映射到 [0,1]，掩码与重映射处理不平衡类别。
      """
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
        ply   = PlyData.read(data_path)
        v     = ply['vertex']
        # ① 读坐标/颜色/原始 class
        coord = np.stack([v['x'],v['y'],v['z']],axis=-1).astype(np.float32)
        # 同时读取 法向量 (nx,ny,nz) 和 曲率 (curvature)
        normals = np.stack([v['nx'], v['ny'], v['nz']], axis=-1).astype(np.float32)
        curv    = np.expand_dims(np.array(v['curvature'], dtype=np.float32), 1)  # (N,1)
        feat    = np.concatenate([normals, curv], axis=1)                        # (N,4)
        print(">>> ENTERED OPENTRENCH3D BRANCH", flush=True)
        print(f">>> RAW feat.shape = {feat.shape}", flush=True)

        # ② 过滤掉 ignore_label
        labels = np.array(v['class'], dtype=np.int64)
        # ② 过滤掉 ignore_label
        ignore_lbl = getattr(cfg, 'ignore_label', cfg.dataset.get('ignore_label', None))
        if ignore_lbl is not None:
            mask = labels != ignore_lbl
            coord, feat, labels = coord[mask], feat[mask], labels[mask]


        # ③ 二分类重映射：0=Other, 1=Trench
        trench_id = cfg.dataset.common.trench_class_id
        labels = np.where(labels == trench_id, 1, 0).astype(np.int64)

        label = labels
    # 最后：将坐标平移，确保最小值为 0
    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)
    """  
    流程：
  1. 判断 cfg.dataset.common.get('voxel_size') 是否非 None；
  2. 若非 None，调用 voxelize() 得到：
       - idx_sort: 按体素编号排序后的点索引
       - voxel_idx: 每个点对应的体素编号
       - count: 每个体素包含的点数
     然后根据 cfg.test_mode 分两种子云采样方式：
       a) 'nearest_neighbor'：每个体素随机选一个点，打乱顺序，并记录逆向索引
       b) 其他（multi_voxel）：对每个体素循环采样一个子云，打乱后直接追加
  3. 若 voxel_size 为 None，则将所有点作为单一子云
  4. 最终将 idx_points、voxel_idx、reverse_idx_part、reverse_idx_sort 返回
函数作用：
  将完整点云按体素网格划分成多个子云（子点集）的索引列表，以支持分批或多次投票推理。
内部原理：
  - voxelize 利用空间量化将点映射到体素格并输出排序、编号与计数信息；
  - 前缀和 + 取模 或 直接迭代，快速计算每个体素的采样索引；
  - 随机打乱并记录逆向索引，方便后续将模型输出点结果还原到原始顺序。

    """
    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
         # 1) 体素化：排序索引、体素编号、每体素点数
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        # 2) 最近邻模式：每体素仅保留一个随机点
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            # 2.1) 计算每个体素的随机采样偏移
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            # 2.2) 根据偏移选择点索引，得到子云
            idx_part = idx_sort[idx_select]
            # 2.3) 获取子云点总数并打乱顺序
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            # 2.4) 构建逆向索引，用于恢复原序
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            # 2.5) 整体体素排序的逆向索引
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        # 3) 多体素模式：对每个可能 i%count 子云采样
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    # 4) 如果没有体素化，就把所有点都当作一个子云
    else:
        idx_points.append(np.arange(coord.shape[0]))
    # 返回子云索引列表及重建所需的逆向索引
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
    logging.info(f"[DEBUG CFG] use_focal_loss={cfg.get('use_focal_loss')}, "
                 f"use_cb_focal_loss={cfg.get('use_cb_focal_loss')}, "
                 f"cb_beta={cfg.get('cb_beta')}, "
                 f"focal_gamma={cfg.get('focal_gamma')}")
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
                                            split             = 'val_pre',
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
        split             = 'train_pre',
        distributed       = cfg.distributed,
    )
    train_dataset = _tmp_loader.dataset
    logging.info(f"[DEBUG] train_dataset type: {type(train_dataset)}, length: {len(train_dataset)}")
    logging.info(f"[DEBUG CFG dataset] {cfg.dataset}")

    if cfg.get('use_imbalanced_sampler', False):
        sampler = FastImbalancedSampler(
            root=cfg.dataset.common.root,
            area=cfg.dataset.common.area,
            split='train_pre',                              # or cfg.dataset.split
            ignore_label=cfg.dataset.ignore_label,    # e.g. 4
            trench_id=cfg.dataset.common.trench_class_id,    # e.g. 2
            cache_path=os.path.join(cfg.run_dir, 'imbalance_weights.json')
        )
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
            split             = cfg.dataset.split,
            distributed       = cfg.distributed,
        )

    # —— DEBUG: 打印采样器 & 损失配置 ——  
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    if hasattr(train_loader.dataset, 'num_classes'):
        train_loader.dataset.num_classes = cfg.num_classes
        batch = next(iter(train_loader))
        # —— DEBUG: 检查批次中的特征键及通道数 ——  
        logging.info(f">>> DEBUG keys in batch: {batch.keys()}")  
        if 'feat' in batch:
            logging.info(f">>> DEBUG feat shape: {batch['feat'].shape}")  
        if 'x' in batch:
            logging.info(f">>> DEBUG x shape: {batch['x'].shape}")  
        # 以下为原有标签分布打印  
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
        logging.info("[DEBUG] >>> entering use_focal_loss block")
        alpha = torch.tensor(cfg.criterion_args.weight if cfg.get('cls_weighed_loss') else [1]*cfg.num_classes)
        logging.info(f"[DEBUG] focal alpha: {alpha.tolist()}, gamma: {cfg.get('focal_gamma', None)}")

    # 首三批标签分布
    for i, batch in enumerate(train_loader):
        lbl = batch['y'].squeeze(-1).view(-1).cpu()
        uniq, cnt = torch.unique(lbl, return_counts=True)
        logging.info(f"[DEBUG batch {i}] labels: {uniq.tolist()}, counts: {cnt.tolist()}")
        if i >= 2: break

    # —— STEP2: 构建加权交叉熵或 Focal Loss —— 
 # —— STEP2: 构建损失（优先 Tversky，其次 Focal，其次 CrossEntropy）—— 
    if cfg.get('use_tversky_loss', False):
        # 使用本地实现的 TverskyLoss，无需外部依赖
        alpha_t = cfg.get('tversky_alpha', 0.3)
        beta_t  = cfg.get('tversky_beta', 0.7)
        criterion = TverskyLoss(alpha=alpha_t, beta=beta_t, reduction='mean').cuda()
        logging.info(f"[DEBUG] 使用 Tversky Loss, alpha={alpha_t}, beta={beta_t}")
    elif cfg.get('use_focal_loss', False):
        logging.info("[DEBUG] >>> entering use_focal_loss block")
        if cfg.get('use_cb_focal_loss', False):
            logging.info("[DEBUG] >>> entering CB-Focal-Loss branch")
            beta_cb = cfg.get('cb_beta', 0.9999)
            alpha = compute_effective_class_weights_by_voxel(
                train_dataset,
                cfg.num_classes,
                beta=beta_cb,
                voxel_size=cfg.dataset.common.voxel_size
            )
            logging.info(f"[DEBUG]   → computed CB alpha by voxel: {alpha.cpu().tolist()}")
            logging.info(f"[DEBUG] using CB-Focal-Loss, beta: {beta_cb}, alpha (CB weights): {alpha.cpu().tolist()}")
        else:
            if cfg.get('cls_weighed_loss', False) and cfg.criterion_args.weight is not None:
                alpha = torch.tensor(cfg.criterion_args.weight, dtype=torch.float32).cuda()
            else:
                alpha = torch.ones(cfg.num_classes, dtype=torch.float32).cuda()
            logging.info(f"[DEBUG] using standard Focal-Loss alpha: {alpha.cpu().tolist()}")
        gamma = cfg.get('focal_gamma', 2.0)
        criterion = lambda logits, target: focal_loss(
            logits, target,
            alpha=alpha,
            gamma=gamma,
        )
        logging.info(f"[DEBUG] focal gamma: {gamma}")
    else:
        if cfg.get('cls_weighed_loss', False):
            cfg.criterion_args.weight = torch.tensor(
                cfg.criterion_args.weight,
                dtype=torch.float32
            ).cuda()
        else:
            cfg.criterion_args.weight = None
        criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
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
                # —— 只在第一条 batch 打印一次调试信息 ——  
        if idx == 0:
            print(f">>> DEBUG keys in batch: {list(data.keys())}")
            print(f">>> DEBUG x shape: {data['x'].shape}")

        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        # 重点：这样分步做的核心目的是 解耦 和 配置驱动！       
        # 从 data['x'] 取出原始 4 通道特征
        raw_feat   = data['x']                # (B,4,N)
        normals    = raw_feat[:, :3, :]       # (B,3,N)
        curvature  = raw_feat[:, 3:4, :]      # (B,1,N)
        data['normals']   = normals
        data['curvature'] = curvature
        
        
        feat_arg = cfg.feature_keys if isinstance(cfg.feature_keys, str) \
                   else ','.join(cfg.feature_keys)
        data['x'] = get_features_by_keys(data, feat_arg)   # 它直接输出 (B,4,N)，正好给 Conv1d


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
                
               # —— 拆分原始 data['x']（4 通道特征） ——  
        raw_feat   = data['x']                  # (B,4,N)
        normals    = raw_feat[:, :3, :]         # (B,3,N)
        curvature  = raw_feat[:, 3:4, :]        # (B,1,N)
        data['normals']   = normals
        data['curvature'] = curvature
        # —— 再拼接 7 通道输入 ——  
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
        # 1) 先把所有张量推到 GPU
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        # 2) 拆分 feat → normals(3通道) & curvature(1通道)
        normals   = data['feat'][:, :3, :]
        curvature = data['feat'][:, 3:4, :]
        data['normals']   = normals
        data['curvature'] = curvature

        # 3) 再按 FEATURE_KEYS 拼成 x（3+3+1=7通道）  
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
                               # —— 拆分原始 feat_part（4 通道特征） ——  
                # 此时 data['x'] 已包含 feat_part  
                raw_feat_part   = data['x'].squeeze(0)          # (N,4)
                normals_part    = raw_feat_part[:, :3]          # (N,3)
                curvature_part  = raw_feat_part[:, 3:]          # (N,1)
                data['normals']   = torch.from_numpy(normals_part).unsqueeze(0).cuda(non_blocking=True)
                data['curvature'] = torch.from_numpy(curvature_part).unsqueeze(0).cuda(non_blocking=True)
                # —— 再拼接 7 通道输入 ——  
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
