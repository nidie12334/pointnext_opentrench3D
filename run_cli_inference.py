# run_cli_inference.py
import os
import subprocess

# —— 请改成你的绝对路径 —— 
CFG = '/home/tech/pointnext/cfgs/segmentation/water_pretrain.yaml'
CKPT= '/home/tech/pointnext/segmentation-train-water_pretrain-ngpus1-20250719-100122-5qEP5dPiPuywV7ZfBEQebT_ckpt_best.pth'
OUT = '/home/tech/pointnext/inference_single'

os.makedirs(OUT, exist_ok=True)

cmd = [
    'python', 'examples/segmentation/main.py',
    '--cfg', CFG,
    'mode=test',
    f'pretrained_path={CKPT}',
    'save_pred=True',
    f'save_path={OUT}',
]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
