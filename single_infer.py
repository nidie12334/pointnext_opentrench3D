#!/usr/bin/env python3
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.models import build_model_from_cfg
from openpoints.dataset.build import build_dataloader_from_cfg

def single_point_inference(cfg_path, ckpt_path, ply_path, out_dir):
    """
    单帧 .ply 推理脚本——复用主流程的 train/val 数据加载与预处理
    1) 从 ply_path 推断 dataset.root 和 split
    2) 补齐 cfg.dataset.common 及 cfg.dataloader.<split>
    3) build_dataloader_from_cfg(split, cfg) 自动 Pad/Collate
    4) 模型前向，输出 per-point logits
    5) 保存彩色 PLY & XY 投影图
    """
    os.makedirs(out_dir, exist_ok=True)

    # —— 1. 加载配置 & 模型 —— 
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.mode = 'test'
    cfg.pretrained_path = ckpt_path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, pretrained_path=ckpt_path)
    model.eval()

    # —— 2. 从 ply_path 推断 dataset.root 和 split —— 
    #    e.g. ply_path = ".../OpenTrench3D/water/val/Area_1_Site_16.ply"
    ply_dir = os.path.dirname(ply_path)        # ".../water/val"
    root_dir = os.path.dirname(ply_dir)        # ".../water"
    split_name = os.path.basename(ply_dir)     # "val"

    # 覆盖 dataset.root & dataset.split
    cfg.dataset.root = root_dir
    cfg.dataset.split = split_name

    # —— 3. 补齐 dataset.common —— 
    # Main pipeline 依赖 cfg.dataset.common; 如果原 YAML 里没它，就手动补一个
    if not hasattr(cfg.dataset, 'common'):
        from openpoints.utils import EasyConfig as _EC
        cfg.dataset.common = _EC({
            'NAME': cfg.dataset.type,
            'type': cfg.dataset.type,
            'root': cfg.dataset.root,
            'area': getattr(cfg.dataset, 'area', None)
        })

    # —— 4. 补齐 dataloader.<split> —— 
    # 验证/测试阶段需要对应的 dataloader 配置；如果没就补一个
    if not hasattr(cfg.dataloader, split_name):
        from openpoints.utils import EasyConfig as _EC
        setattr(cfg.dataloader, split_name, _EC({
            'batch_size': 1,
            'num_workers': 1,
            'shuffle': False,
            'pin_memory': False,
            'drop_last': False,
            'collate_fn': 'pad_collate_fn'
        }))

    # —— 5. 构建 DataLoader —— 
    test_loader = build_dataloader_from_cfg(split_name, cfg)

    # —— 6. 取出唯一一帧 batch 并上 GPU —— 
    batch = next(iter(test_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # —— 7. 前向推理 —— 
    with torch.no_grad():
        out = model(batch)
        logits = out['logits'] if isinstance(out, dict) else out
        pred = logits.argmax(dim=-1)[0].cpu().numpy()  # [N]

    # —— 8. 从 batch 拿回原始 coords —— 
    coords = batch['orig_pos'][0].cpu().numpy()     # [N,3]

    # —— 9. 保存彩色 PLY —— 
    num_cls = cfg.model.num_classes
    cmap = plt.get_cmap('tab20')
    colors = cmap(pred / num_cls)[:, :3]

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(coords)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    ply_out = os.path.join(out_dir, os.path.basename(ply_path).replace('.ply','_pred.ply'))
    o3d.io.write_point_cloud(ply_out, pcd_vis)
    print("已保存彩色 PLY：", ply_out)

    # —— 10. XY 平面投影 —— 
    x, y = coords[:,0], coords[:,1]
    plt.figure(figsize=(6,6))
    sc = plt.scatter(x, y, c=pred, cmap='tab20', s=1)
    plt.axis('equal'); plt.axis('off')
    plt.title('XY 平面投影')
    plt.colorbar(sc, ticks=range(num_cls)).set_label('Class')
    png_out = os.path.join(out_dir, os.path.basename(ply_path).replace('.ply','_xy_proj.png'))
    plt.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.show()
    print("已保存投影图：", png_out)


if __name__ == "__main__":
    # —— 改成你本地的绝对路径 —— 
    CFG_PATH  = '/home/tech/pointnext/cfgs/segmentation/water_pretrain.yaml'
    CKPT_PATH = '/home/tech/pointnext/segmentation-train-water_pretrain-ngpus1-20250719-100122-5qEP5dPiPuywV7ZfBEQebT_ckpt_best.pth'
    PLY_PATH  = '/home/tech/pointnext/datasets/OpenTrench3D/water/val/Area_1_Site_16.ply'
    OUT_DIR   = '/home/tech/pointnext/inference_single'

    single_point_inference(CFG_PATH, CKPT_PATH, PLY_PATH, OUT_DIR)
