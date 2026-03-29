import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# --- 1. 读取数据 ---
lsi_path = r"E:\大创\lsi_embedding_final.csv"
lsi_df = pd.read_csv(lsi_path, index_col=0)

mat_path = r"E:\大创\pbmc_110xAtac.mat"
mat_data = loadmat(mat_path)
# 提取真实标签（确保路径和变量名正确）
slabel = mat_data['groundtruth'].flatten()
slabel = np.array([str(s[0]) if isinstance(s, np.ndarray) else str(s) for s in slabel])

# --- 2. 构建 AnnData 对象 ---
# 使用 LSI 降维后的矩阵作为主要表示
adata = ad.AnnData(X=np.zeros((lsi_df.shape[0], 1)))
adata.obsm['X_lsi'] = lsi_df.values
adata.obs['true_label'] = slabel 

# --- 3. 构建邻面图 (KNN Graph) ---
# Louvain 同样基于邻面图，这里保持 20 个邻居
sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=20)

# --- 4. 执行 Louvain 聚类 ---
# 注意：这里将 sc.tl.leiden 替换为 sc.tl.louvain
res = 0.5
sc.tl.louvain(adata, resolution=res, key_added='louvain_pred')

# --- 5. 性能评估 (ARI 和 NMI) ---
ari_score = adjusted_rand_score(adata.obs['true_label'], adata.obs['louvain_pred'])
nmi_score = normalized_mutual_info_score(adata.obs['true_label'], adata.obs['louvain_pred'])

print("-" * 30)
print(f"Louvain 聚类分辨率 (Resolution): {res}")
print(f"识别出的簇数量: {len(adata.obs['louvain_pred'].unique())}")
print(f"真实标签类别数: {len(np.unique(slabel))}")
print(f"ARI 得分 (调整兰德指数): {ari_score:.4f}")
print(f"NMI 得分 (标准化互信息): {nmi_score:.4f}")
print("-" * 30)

# --- 6. 可视化 ---
sc.tl.umap(adata)
sc.pl.umap(adata, color=['louvain_pred', 'true_label'],
           title=['Louvain Clustering', 'Ground Truth'],
           show=True)