import os
import time
from pathlib import Path

import torch
from PIL import Image
from timm import create_model
from torchvision import transforms
from tqdm import tqdm

# Configuration
DATASET_DIR = Path("dataset_img")
CHECKPOINT_PATH = Path("model/mobilenetv3_v1-3_dist.ckpt")
OUTPUT_FILE = Path("anime_folders.txt")
NUM_CLASSES = 5
LABELS = ['3d', 'bangumi', 'comic', 'illustration', 'not_painting']
ANIME_LABELS = {'bangumi', 'comic', 'illustration'}  # labels considered as anime
# ANIME_LABELS = {'3d', 'not_painting'}  # labels considered as anime
MAX_RETRIES = 3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = create_model('mobilenetv3_large_100', pretrained=False, num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt)
# Filter out FLOPs/params stats and Lightning prefixes
filtered_state = {
    k.replace('model.', ''): v
    for k, v in state_dict.items()
    if not ('.total_ops' in k or '.total_params' in k)
}
model.load_state_dict(filtered_state, strict=False)
model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Read already processed folders to allow resume
processed = set()
if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open('r') as f:
        processed = {line.strip() for line in f if line.strip()}

# Prepare output file in append mode
with OUTPUT_FILE.open('a') as out_f:
    for folder in tqdm(sorted(DATASET_DIR.iterdir()), desc="Folders"):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        if folder_name in processed:
            continue  # skip already done

        # Retry mechanism per folder
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Load all images in folder
                img_paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
                if not img_paths:
                    break  # no images to check

                # Batch preprocess
                batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in img_paths])
                batch = batch.to(device)

                # Inference
                with torch.no_grad():
                    logits = model(batch)
                    preds = logits.argmax(dim=1).cpu().tolist()

                # Check if any prediction is in ANIME_LABELS
                if any(LABELS[p] in ANIME_LABELS for p in preds):
                    out_f.write(folder_name + "\n")
                    out_f.flush()

                break  # success, move to next folder

            except Exception as e:
                print(f"[Attempt {attempt}/{MAX_RETRIES}] Error processing '{folder_name}': {e}")
                time.sleep(1)  # brief pause before retry
