import json
import random
from pathlib import Path
from collections import defaultdict

def build_seeds_json(dataset_root: str, output_file: str = "seeds.json"):
    """
    生成目录顺序随机化的 seeds.json
    参数：
        dataset_root: 数据集根目录路径
        output_file: 输出的 JSON 文件名
    """
    dataset_path = Path(dataset_root)
    
    # 存储目录与种子的映射关系
    seed_dict = defaultdict(list)
    
    # 遍历所有子目录
    for dir_path in dataset_path.iterdir():
        if not dir_path.is_dir():
            continue
        
        dir_name = dir_path.name
        seed_set = set()
        
        # 收集有效的种子对
        for img_file in dir_path.glob("*_0.jpg"):
            seed = img_file.stem.split("_")[0]
            if (dir_path / f"{seed}_1.jpg").exists():
                seed_set.add(seed)
        
        if seed_set:
            # 目录内种子保持排序以保证可重复性
            seed_dict[dir_name] = sorted(seed_set)
    
    # 将目录列表随机排序
    dir_list = list(seed_dict.items())
    random.shuffle(dir_list)  # 核心修改：随机打乱目录顺序
    
    # 转换为最终格式
    sorted_seeds = [[dir_name, seeds] for dir_name, seeds in dir_list]
    
    # 保存为 JSON 文件
    with open(dataset_path / output_file, "w") as f:
        json.dump(sorted_seeds, f, indent=2)
    
    print(f"生成完成：共 {len(sorted_seeds)} 个目录，目录顺序已随机化")

if __name__ == "__main__":
    # 使用示例
    build_seeds_json(
        dataset_root="./dataset_anime_filtered",
        output_file="seeds.json"
    )