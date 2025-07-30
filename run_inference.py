import os
import torch

from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.models import build_model_from_cfg
from openpoints.dataset.build import build_dataloader_from_cfg
from examples.segmentation.main import test

def inference_via_dataloader(cfg_path, ckpt_path, out_dir):
    # 加载配置
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.mode = 'test'
    cfg.pretrained_path = ckpt_path
    cfg.save_pred = True
    cfg.save_path = out_dir

    # 构建模型并加载权重
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, pretrained_path=ckpt_path)
    model.eval()

    # 构建测试集 DataLoader
    test_loader = build_dataloader_from_cfg('test', cfg)

    # 推理并保存结果
    test(model, test_loader, cfg)


if __name__ == "__main__":
    # —— 请替换成你本地的绝对路径 —— 
    CFG_PATH  = '/home/tech/pointnext/cfgs/segmentation/water_pretrain.yaml'
    CKPT_PATH = '/home/tech/pointnext/segmentation-train-water_pretrain-ngpus1-20250719-100122-5qEP5dPiPuywV7ZfBEQebT_ckpt_best.pth'
    OUT_DIR   = '/home/tech/pointnext/inference_single'

    os.makedirs(OUT_DIR, exist_ok=True)
    inference_via_dataloader(CFG_PATH, CKPT_PATH, OUT_DIR)

