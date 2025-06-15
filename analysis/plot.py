import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_three_models(metrics_file1, label1,
                      metrics_file2, label2,
                      metrics_file3, label3,
                      output_dir, output_name="comparison_plot.pdf"):
    """
    读取三个 .jsonl 文件中的 sim_direction 和 sim_image，
    并绘制三条折线图，最后保存到 output_dir/output_name。
    """
    def load_data(path):
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
        x = [d['sim_direction'] for d in data]
        y = [d['sim_image']     for d in data]
        return x, y

    # 读数据
    x1, y1 = load_data(metrics_file1)
    x2, y2 = load_data(metrics_file2)
    x3, y3 = load_data(metrics_file3)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, marker='o', linestyle='-', linewidth=2, label=label1)
    plt.plot(x2, y2, marker='s', linestyle='--', linewidth=2, label=label2)
    plt.plot(x3, y3, marker='^', linestyle='-.', linewidth=2, label=label3)

    plt.xlabel("CLIP Text–Image Direction Similarity")
    plt.ylabel("CLIP Image Similarity")
    plt.title("Performance comparison")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # 保存输出
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":

    metrics_file1 = "jsonl/n=100_p=test_s=100_r=512_e=0_1.jsonl"
    metrics_file2 = "jsonl/n=100_p=test_s=100_r=512_e=0_2.jsonl"
    metrics_file3 = "jsonl/n=100_p=test_s=100_r=512_e=0_3.jsonl"

    
    plot_three_models(
        metrics_file1, "AnimeStyle1.5",
        metrics_file2, "Common",
        metrics_file3, "AnimeStyleV1.3",
        output_dir="plot/",
        output_name="three_model_comparison.pdf"
    )
