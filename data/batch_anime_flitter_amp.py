import os
import shutil
from pathlib import Path

import torch
from torch.cuda import amp
from PIL import Image
from timm import create_model
from torchvision import transforms
from tqdm import tqdm

# —————— 配置区域 ——————
DATASET_DIR = Path("data_test")
CHECKPOINT_PATH = Path("model/mobilenetv3_v1-3_dist.ckpt")
OUTPUT_FILE = Path("anime_folders.txt")
FILTERED_DIR = Path("dataset_anime_filtered")

NUM_CLASSES = 5
LABELS = ['3d', 'bangumi', 'comic', 'illustration', 'not_painting']
ANIME_LABELS = {'bangumi', 'comic', 'illustration'}

# 调小一点更快触发批量推理；如果显存足够，可调大
BATCH_SIZE = 868

# —————— 准备工作 ——————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILTERED_DIR.mkdir(exist_ok=True)
processed = set()
if OUTPUT_FILE.exists():
    processed = {line.strip() for line in OUTPUT_FILE.open()}

# Load model
model = create_model('mobilenetv3_large_100', pretrained=False, num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt)
filtered = {
    k.replace('model.', ''): v
    for k, v in state_dict.items()
    if not ('.total_ops' in k or '.total_params' in k)
}
model.load_state_dict(filtered, strict=False)
model.to(device).eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# 打开输出文件
out_f = OUTPUT_FILE.open('a')

# 缓存 batch
batch_imgs = []
batch_folder_names = []

def flush_batch():
    """对当前 batch_imgs 做推理，并移动对应文件夹。"""
    global batch_imgs, batch_folder_names
    if not batch_imgs:
        return
    tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad(), amp.autocast():
        preds = model(tensor).argmax(dim=1).cpu().tolist()
    for folder_name, p in zip(batch_folder_names, preds):
        if folder_name not in processed and LABELS[p] in ANIME_LABELS:
            processed.add(folder_name)
            # 写入 txt
            out_f.write(folder_name + "\n"); out_f.flush()
            # 移动目录
            src = DATASET_DIR / folder_name
            dst = FILTERED_DIR / folder_name
            if src.exists():
                print(f"[MOVE] {src} -> {dst}")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
            else:
                print(f"[WARN] Expected to move {src}, but it does not exist.")
    # 清空 batch
    batch_imgs = []
    batch_folder_names = []

# 主循环
for folder in tqdm(sorted(DATASET_DIR.iterdir()), desc="Folders"):
    if not folder.is_dir() or folder.name in processed:
        continue
    imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    for img_path in imgs:
        try:
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_folder_names.append(folder.name)
        except Exception:
            continue

        # 达到阈值就 flush
        if len(batch_imgs) >= BATCH_SIZE:
            flush_batch()

# 处理残余
flush_batch()
out_f.close()
