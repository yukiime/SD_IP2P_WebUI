import os
import random
import shutil

def move_random_folders(src_dir, dst_dir, count=500):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取所有子文件夹列表
    all_folders = [
        name for name in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, name))
    ]

    # 检查子文件夹数量是否足够
    total = len(all_folders)
    if total < count:
        raise ValueError(f"源目录只有 {total} 个子文件夹，无法抽取 {count} 个")

    # 随机抽取
    selected = random.sample(all_folders, count)

    # 移动操作
    for folder_name in selected:
        src_path = os.path.join(src_dir, folder_name)
        dst_path = os.path.join(dst_dir, folder_name)
        shutil.move(src_path, dst_path)
        print(f"Moved: {folder_name}")

if __name__ == "__main__":
    src_dir = "dataset_anime_filtered"       # 源目录
    dst_dir = "data_transfer"     # 目标目录
    move_random_folders(src_dir, dst_dir, count=500)
