import scipy.io as sio
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score )
from sklearn.neighbors import NearestNeighbors  
import gc

#  1. 加载数据
mat_data = sio.loadmat(r"E:\大创\pbmc_110xAtac.mat")
sparse_A = sp.csr_matrix(mat_data['X'][0, 1])  # 第二组数据（2592*95460peak），改0为1处理第一组
sparse_A_binarized = sparse_A.astype(bool).astype(int)
groundtruth = mat_data['groundtruth'].ravel()  # 真实标签

# 2.邻域保持度计算函数（核心）
def calculate_neighborhood_preservation(original_sparse, reduced_dense, k=10):
    """
    计算降维前后的邻域保持度（NPR）：评估细胞近邻关系的保留效果
    :param original_sparse: 原始高维稀疏矩阵（cell×peak）
    :param reduced_dense: 降维后低维稠密矩阵（cell×n_components）
    :param k: 近邻数量（适配单细胞数据，默认10）
    :return: 平均邻域保持度（0~1，越高表示邻域结构保留越好）
    """
    # 原始稀疏矩阵转为稠密（2592个细胞×95460个peak，内存可控）
    original_dense = original_sparse.toarray() if sp.issparse(original_sparse) else original_sparse
    
    # 1. 原始高维空间找k近邻（排除自身）
    nn_original = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(original_dense)
    _, indices_original = nn_original.kneighbors(original_dense)
    original_neighs = [set(neigh[1:]) for neigh in indices_original]  # 去掉自身索引
    
    # 2. 降维后低维空间找k近邻（排除自身）
    nn_reduced = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(reduced_dense)
    _, indices_reduced = nn_reduced.kneighbors(reduced_dense)
    reduced_neighs = [set(neigh[1:]) for neigh in indices_reduced]
    
    # 3. 计算每个细胞的Jaccard相似度（交集/并集），求平均
    total_npr = 0.0
    for o_neigh, r_neigh in zip(original_neighs, reduced_neighs):
        intersection = len(o_neigh & r_neigh)
        union = len(o_neigh | r_neigh)
        if union > 0:  # 避免除以0（理论上不会发生）
            total_npr += intersection / union
    return total_npr / len(original_dense)

#2. 迭代LSI函数（新增双指标计算） 
def iterative_lsi_with_all_metrics(adata, n_components=30, resolution=0.7, top_peak_ratio=0.25):
    ari_list = []          # 原有：ARI
    nmi_list = []          # 原有：NMI
    silhouette_list = []   # 新增：轮廓系数
    npr_list = []          # 新增：邻域保持度
    peak_num_list = [adata.shape[1]]  # 记录peak数
    
    # 保存原始高维数据（仅需一次存储，用于邻域保持度计算）
    original_high_dim = adata.X.copy()
    
    for round in range(2):
        print(f"\n===== 第{round+1}次迭代 =====")
        
        # 步骤1：TF-IDF加权（不变）
        tf = normalize(adata.X, norm='l1', axis=1)
        peak_open_counts = np.array((adata.X > 0).sum(axis=0)).ravel()
        idf = np.log(adata.shape[0] / (peak_open_counts + 1e-10))
        idf_diag = sp.diags(idf, shape=(adata.shape[1], adata.shape[1]), format='csr')
        tf_idf = tf @ idf_diag
        del tf, idf
        gc.collect()
        
        # 步骤2：随机SVD降维（不变）
        lsi = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='randomized',
            n_iter=3
        )
        lsi_result = lsi.fit_transform(tf_idf)
        adata.obsm[f'X_lsi_{round+1}'] = lsi_result
        del tf_idf
        gc.collect()
        
        # 步骤3：Leiden聚类（不变）
        sc.pp.neighbors(adata, use_rep=f'X_lsi_{round+1}')
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=f'leiden_{round+1}',
            flavor="igraph",
            n_iterations=2,
            directed=False
        )
        cluster_labels = adata.obs[f'leiden_{round+1}'].cat.codes.values
        
        # 步骤4：计算所有指标（原有+新增）
        # 4.1 原有指标：ARI、NMI
        ari = adjusted_rand_score(groundtruth, cluster_labels)
        nmi = normalized_mutual_info_score(groundtruth, cluster_labels, average_method='arithmetic')
        ari_list.append(ari)
        nmi_list.append(nmi)
        
        # 4.2 新增指标：轮廓系数（基于降维后数据+聚类标签）
        # 轮廓系数衡量聚类紧致度与分离度，用cosine距离适配高维数据
        silhouette = silhouette_score(lsi_result, cluster_labels, metric='cosine')
        silhouette_list.append(silhouette)
        
        # 4.3 新增指标：邻域保持度（对比原始高维数据与当前降维数据）
        # 注意：第2次迭代时adata.X已筛选peak，需用原始高维数据对比
        npr = calculate_neighborhood_preservation(original_high_dim, lsi_result, k=10)
        npr_list.append(npr)
        
        # 步骤5：输出当前迭代所有指标
        print(f"第{round+1}次迭代：")
        print(f"  ARI={ari:.3f} | NMI={nmi:.3f} | 轮廓系数={silhouette:.3f} | 邻域保持度={npr:.3f}")
        print(f"  当前peak数={adata.shape[1]}")
        
        # 步骤6：筛选高变异peak（不变）
        if round < 1:
            groups = cluster_labels
            n_clusters = len(np.unique(groups))
            
            group_sums = []
            group_sq_sums = []
            for g in range(n_clusters):
                group_mask = groups == g
                group_data = adata.X[group_mask]
                group_sums.append(group_data.sum(axis=0).A.ravel())
                group_sq_sums.append((group_data.power(2)).sum(axis=0).A.ravel())
            
            group_sums = np.array(group_sums)
            group_sq_sums = np.array(group_sq_sums)
            group_counts = np.array([np.sum(groups == g) for g in range(n_clusters)])
            
            total_mean = adata.X.mean(axis=0).A.ravel()
            SSA = np.sum(group_counts.reshape(-1, 1) * (group_sums / group_counts.reshape(-1, 1) - total_mean)**2, axis=0)
            SSE = np.sum(group_sq_sums - (group_sums**2) / group_counts.reshape(-1, 1), axis=0)
            f_stats = (SSA / (n_clusters - 1)) / (SSE / (adata.shape[0] - n_clusters) + 1e-10)
            
            n_top_peaks = int(adata.shape[1] * top_peak_ratio)
            top_peak_idx = np.argsort(f_stats)[-n_top_peaks:]
            adata = adata[:, top_peak_idx].copy()
            peak_num_list.append(adata.shape[1])
    
    adata.obsm['X_lsi'] = adata.obsm['X_lsi_2']
    # 返回值新增轮廓系数和邻域保持度列表
    return adata, ari_list, nmi_list, silhouette_list, npr_list, peak_num_list

#  3. 执行分析并输出所有指标结果
adata = sc.AnnData(
    X=sparse_A_binarized,
    obs=pd.DataFrame(index=[f'cell_{i}' for i in range(sparse_A_binarized.shape[0])]),
    var=pd.DataFrame(index=[f'peak_{i}' for i in range(sparse_A_binarized.shape[1])])
)

# 运行并获取所有指标（修改返回值接收）
adata, ari_list, nmi_list, silhouette_list, npr_list, peak_num_list = iterative_lsi_with_all_metrics(adata)

#  4. 输出最终汇总结果 
print(f"\n===== 最终评估结果汇总 =====")
print(f"第1次迭代（原始peak）：")
print(f"  ARI={ari_list[0]:.3f} | NMI={nmi_list[0]:.3f} | 轮廓系数={silhouette_list[0]:.3f} | 邻域保持度={npr_list[0]:.3f}")
print(f"第2次迭代（筛选后peak）：")
print(f"  ARI={ari_list[1]:.3f} | NMI={nmi_list[1]:.3f} | 轮廓系数={silhouette_list[1]:.3f} | 邻域保持度={npr_list[1]:.3f}")
print(f"\n指标提升幅度：")
print(f"  ARI提升：{ari_list[1] - ari_list[0]:.3f} | NMI提升：{nmi_list[1] - nmi_list[0]:.3f}")
print(f"  轮廓系数提升：{silhouette_list[1] - silhouette_list[0]:.3f} | 邻域保持度提升：{npr_list[1] - npr_list[0]:.3f}")

#  5. 保存所有结果到CSV（含新增指标） 
result_df = pd.DataFrame({
    'ARI_第一次迭代': [ari_list[0]],
    'ARI_第二次迭代': [ari_list[1]],
    'NMI_第一次迭代': [nmi_list[0]],
    'NMI_第二次迭代': [nmi_list[1]],
    '轮廓系数_第一次迭代': [silhouette_list[0]],
    '轮廓系数_第二次迭代': [silhouette_list[1]],
    '邻域保持度_第一次迭代': [npr_list[0]],
    '邻域保持度_第二次迭代': [npr_list[1]],
    'Peak数_第一次迭代': [peak_num_list[0]],
    'Peak数_第二次迭代': [peak_num_list[1]]
})
result_df.to_csv("lsi_all_metrics_results.csv", index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 lsi_all_metrics_results.csv")

