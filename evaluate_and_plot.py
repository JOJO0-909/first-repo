import scanpy as sc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

# 全局绘图设置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def calculate_clisi(X, labels, k=30):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    _, indices = nn.kneighbors(X)
    lisi_scores = []
    for idx in indices:
        neigh_labels = labels[idx]
        _, counts = np.unique(neigh_labels, return_counts=True)
        probs = counts / k
        simpson_index = np.sum(probs ** 2)
        lisi_scores.append(1.0 / simpson_index)
    return np.mean(lisi_scores)


def calculate_npr(adata, X_low, k=15):
    nn_high = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(adata.X)
    indices_high = nn_high.kneighbors(adata.X, return_distance=False)
    nn_low = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_low)
    indices_low = nn_low.kneighbors(X_low, return_distance=False)
    overlaps = [len(np.intersect1d(indices_high[i], indices_low[i])) / k for i in range(adata.n_obs)]
    return np.mean(overlaps)


def evaluate_and_plot(dataset_name="95k"):
    print(f"\n\n================ 正在评估 {dataset_name} 数据集 ================")
    tasks = [
        (f"lsi_result_{dataset_name}.h5ad", "X_LSI", "Standard LSI"),
        (f"peakvi_result_{dataset_name}.h5ad", "X_PeakVI", "PeakVI"),
        (f"phate_result_{dataset_name}.h5ad", "X_PHATE", "PHATE")
    ]

    all_metrics = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"scATAC-seq Dimensionality Reduction Visualization ({dataset_name} Peaks)", fontsize=16)

    for idx, (file, key, name) in enumerate(tasks):
        if not os.path.exists(file):
            print(f"找不到 {file}，跳过 {name}")
            continue

        print(f"\n正在分析: {name}...")
        adata = sc.read_h5ad(file)
        X_low = adata.obsm[key]
        y_true = adata.obs['groundtruth'].values

        # 1. 执行 Leiden 聚类 (分辨率 0.365)
        sc.pp.neighbors(adata, use_rep=key, n_neighbors=15)
        sc.tl.leiden(adata, resolution=0.365, key_added='leiden_pred')
        y_pred = adata.obs['leiden_pred'].values

        # 2. 计算各项指标
        metrics = {
            "Method": name,
            "Dataset": dataset_name,
            "ARI": round(adjusted_rand_score(y_true, y_pred), 4),
            "NMI": round(normalized_mutual_info_score(y_true, y_pred), 4),
            "ASW": round(silhouette_score(X_low, y_true), 4),
            "cLISI": round(calculate_clisi(X_low, y_true), 4),
            "NPR": round(calculate_npr(adata, X_low), 4),
            "Time (s)": round(adata.uns.get('run_time', 0), 2),
            "Peak Memory (MB)": round(adata.uns.get('peak_memory_mb', 0), 2)
        }
        all_metrics.append(metrics)

        # 3. 生成 UMAP 可视化
        sc.tl.umap(adata)
        # 上排画真实标签，下排画预测标签
        sc.pl.umap(adata, color='groundtruth', ax=axes[0, idx], show=False, title=f"{name} (True Labels)")
        sc.pl.umap(adata, color='leiden_pred', ax=axes[1, idx], show=False, title=f"{name} (Leiden r=0.365)")

    plt.tight_layout()
    plt.savefig(f"UMAP_Visualization_{dataset_name}.png", dpi=300)
    print(f"-> 图像已保存为 UMAP_Visualization_{dataset_name}.png")

    return all_metrics


if __name__ == "__main__":
    results_36k = evaluate_and_plot("36k")
    results_95k = evaluate_and_plot("95k")

    # 合并汇总表格
    final_df = pd.DataFrame(results_36k + results_95k)
    print("\n" + "=" * 90)
    print("🏆 2026 MCM/ICM 终极全性能基准测试报告 (Leiden clustering)")
    print("=" * 90)
    print(final_df.to_string(index=False))
    print("=" * 90)

    final_df.to_csv("benchmark_results_comprehensive.csv", index=False)