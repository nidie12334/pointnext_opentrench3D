import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.models import build_model_from_cfg


def single_pointcloud_inference(
    cfg_path: str,
    ckpt_path: str,
    ply_path: str,
    out_dir: str
):
    """
    对单个 .ply 点云文件进行分割推理：
    1) 输入 dict {'pos': [1,3,N], 'batch': [N]} 
    2) 模型输出 per-point logits → [N, num_classes]
    3) 保存彩色 PLY & XY 散点投影
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载配置 & 模型
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.mode = 'test'
    cfg.pretrained_path = ckpt_path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, pretrained_path=ckpt_path)
    model.eval()

    # 2. 读取 .ply 点云并构造 [1,3,N] 的 pos
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points)    # [N,3]

    # 转为 tensor [1, N, 3]
    coords_tensor = torch.from_numpy(coords).float().to(device).unsqueeze(0)  # [1,N,3]
    # 置换为 [1,3,N]
    pos = coords_tensor.permute(0, 2, 1)  # [1,3,N]

    # 3. 构造 batch 索引 [N]
    batch_idx = torch.zeros(pos.size(2), dtype=torch.long, device=device)

    # 4. 前向推理
    with torch.no_grad():
        out = model({'pos': pos, 'batch': batch_idx})
        logits = out['logits'] if isinstance(out, dict) else out
        # 如果是 [1, N, C] → squeeze 到 [N, C]
        if logits.dim() == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        pred = logits.argmax(dim=-1).cpu().numpy()  # [N]

    # 5. 保存彩色 PLY
    num_cls = cfg.model.num_classes
    cmap = plt.get_cmap('tab20')
    colors = cmap(pred / num_cls)[:, :3]  # [N,3]

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(coords)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    ply_out = os.path.join(
        out_dir,
        os.path.basename(ply_path).replace('.ply', '_pred.ply')
    )
    o3d.io.write_point_cloud(ply_out, pcd_vis)
    print(f"已保存彩色 PLY：{ply_out}")

    # 6. XY 投影 & 可视化
    x, y = coords[:, 0], coords[:, 1]
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x, y, c=pred, cmap='tab20', s=1)
    plt.axis('equal')
    plt.axis('off')
    plt.title('XY Plane Projection')
    cbar = plt.colorbar(sc, ticks=range(num_cls))
    cbar.set_label('Class')

    png_out = os.path.join(
        out_dir,
        os.path.basename(ply_path).replace('.ply', '_xy_proj.png')
    )
    plt.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"已保存投影图：{png_out}")


if __name__ == "__main__":
    # 请替换为你本地的绝对路径
    CFG_PATH  = '/home/tech/pointnext/cfgs/segmentation/water_pretrain.yaml'
    CKPT_PATH = '/home/tech/pointnext/segmentation-train-water_pretrain-ngpus1-20250719-100122-5qEP5dPiPuywV7ZfBEQebT_ckpt_best.pth'
    PLY_PATH  = '/home/tech/pointnext/Area_1_Site_16.ply'
    OUT_DIR   = '/home/tech/pointnext/inference_single'

    single_pointcloud_inference(
        cfg_path=CFG_PATH,
        ckpt_path=CKPT_PATH,
        ply_path=PLY_PATH,
        out_dir=OUT_DIR
    )
