import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from scipy import stats

def load_chromvar_data(mat_file_path):
    """
    加载ATAC-seq数据并准备ChromVAR分析
    """
    print("正在加载ATAC-seq数据...")
    mat_data = scipy.io.loadmat(mat_file_path)
    X = mat_data['X']
    
    # 提取稀疏矩阵
    if X.shape == (1, 2):
        sparse_matrix_1 = X[0, 0]  # 第一个矩阵
        sparse_matrix_2 = X[0, 1]  # 第二个矩阵
    else:
        sparse_matrix_1 = X
        sparse_matrix_2 = None
    
    print(f"矩阵1: {sparse_matrix_1.shape}")
    if sparse_matrix_2 is not None:
        print(f"矩阵2: {sparse_matrix_2.shape}")
    
    return sparse_matrix_1, sparse_matrix_2

def compute_chromvar_deviations(sparse_matrix, motif_peak_matrix):
    """计算ChromVAR偏差分数"""
    n_cells, n_peaks = sparse_matrix.shape
    
    # 计算motif可及性
    dense_matrix = sparse_matrix.toarray()
    motif_accessibility = np.dot(dense_matrix, motif_peak_matrix)
    
    # 计算期望可及性
    total_accessibility = np.sum(dense_matrix, axis=1)
    total_motif_coverage = np.sum(motif_peak_matrix, axis=0)
    expected_accessibility = np.outer(total_accessibility, total_motif_coverage) / n_peaks
    
    # 计算偏差分数
    deviations = (motif_accessibility - expected_accessibility) / np.sqrt(expected_accessibility + 1e-8)
    deviations_zscore = stats.zscore(deviations, axis=0)
    
    return deviations_zscore, motif_accessibility
def reduce_dimension_chromvar(sparse_matrix, target_dim=30):
    """ChromVAR降维主函数"""
    n_peaks = sparse_matrix.shape[1]
    motif_peak_matrix, motif_names = load_motif_annotations(n_peaks)
    
    deviations, motif_accessibility = compute_chromvar_deviations(
        sparse_matrix, motif_peak_matrix
    )
    
    # 使用SVD进行最终降维
    svd = TruncatedSVD(n_components=target_dim, random_state=42)
    chromvar_embeddings = svd.fit_transform(deviations)
    
    return {
        'embeddings': chromvar_embeddings,
        'deviations': deviations,
        'explained_variance': svd.explained_variance_ratio_.sum()
    }
def load_motif_annotations(motif_file=None, n_peaks=36600, random_seed=None):
    """
    加载motif注释信息
    参数n_peaks指定peak的数量，random_seed确保可重复性
    """
    if motif_file is None:
        # 创建示例motif-peak关联矩阵
        print(f"使用示例motif注释，peak数量: {n_peaks}")
        n_motifs = 500   # 常见的转录因子motif数量
        
        # 设置随机种子确保不同矩阵的motif注释独立
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 随机生成motif-peak关联
        motif_peak_matrix = np.random.binomial(1, 0.02, (n_peaks, n_motifs))
        motif_names = [f"TF_{i}" for i in range(n_motifs)]
        
        return motif_peak_matrix, motif_names
    else:
        # 从文件加载motif注释
        print(f"从文件加载motif注释: {motif_file}")
        pass
def compute_chromvar_deviations(sparse_matrix, motif_peak_matrix, motif_names):
    """
    计算ChromVAR偏差分数 - 完全兼容稀疏和稠密矩阵
    """
    print("计算ChromVAR偏差分数...")
    
    n_cells, n_peaks = sparse_matrix.shape
    n_motifs = motif_peak_matrix.shape[1]
    
    print(f"数据维度: {n_cells} cells, {n_peaks} peaks, {n_motifs} motifs")
    print(f"输入矩阵类型: {type(sparse_matrix)}")
    
    # 1. 计算motif可及性 - 统一处理方法
    print("步骤1: 计算motif可及性...")
    
    # 方法1：统一转换为稠密矩阵处理（简单但可能内存消耗大）
    if sp.issparse(sparse_matrix):
        dense_matrix = sparse_matrix.toarray()
    else:
        dense_matrix = sparse_matrix
    
    motif_accessibility = np.dot(dense_matrix, motif_peak_matrix)
    
    print(f"motif可及性矩阵形状: {motif_accessibility.shape}")
    
    # 2. 计算期望可及性
    print("步骤2: 计算期望可及性...")
    total_accessibility = np.sum(dense_matrix, axis=1)
    total_motif_coverage = np.sum(motif_peak_matrix, axis=0)
    
    expected_accessibility = np.outer(total_accessibility, total_motif_coverage) / n_peaks
    
    # 3. 计算偏差分数
    print("步骤3: 计算偏差分数...")
    expected_safe = expected_accessibility + 1e-8
    deviations = (motif_accessibility - expected_accessibility) / np.sqrt(expected_safe)
    
    # 4. Z-score标准化
    deviations_zscore = stats.zscore(deviations, axis=0)
    
    print(f"最终偏差分数矩阵形状: {deviations_zscore.shape}")
    
    return deviations_zscore, motif_accessibility
def reduce_dimension_chromvar(sparse_matrix, target_dim=50, motif_file=None, matrix_id=1):
    """
    ChromVAR降维主函数
    matrix_id: 矩阵标识符，用于生成独立的随机种子
    """
    print(f"开始ChromVAR降维: {sparse_matrix.shape} -> {target_dim}维")
    
    # 1. 加载motif注释，根据矩阵维度调整，使用独立随机种子
    n_peaks = sparse_matrix.shape[1]  # 获取实际的peak数量
    random_seed = 42 + matrix_id  # 为每个矩阵生成不同的随机种子
    motif_peak_matrix, motif_names = load_motif_annotations(
        motif_file, n_peaks=n_peaks, random_seed=random_seed
    )
    
    # 2. 计算ChromVAR偏差分数
    deviations, motif_accessibility = compute_chromvar_deviations(
        sparse_matrix, motif_peak_matrix, motif_names
    )
    
    # 3. 对偏差分数进行降维
    print("对ChromVAR偏差分数进行降维...")
    if deviations.shape[1] > target_dim:
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        chromvar_embeddings = svd.fit_transform(deviations)
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"累计解释方差: {explained_variance:.4f}")
    else:
        chromvar_embeddings = deviations
        explained_variance = 1.0
    
    results = {
        'embeddings': chromvar_embeddings,
        'deviations': deviations,
        'motif_accessibility': motif_accessibility,
        'motif_names': motif_names,
        'explained_variance': explained_variance
    }
    
return results
def visualize_chromvar_results(chromvar_results, title="ChromVAR降维结果"):
    """
    可视化ChromVAR结果
    """
    embeddings = chromvar_results['embeddings']
    deviations = chromvar_results['deviations']
    motif_names = chromvar_results['motif_names']
    
    print("生成ChromVAR可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=30)
    axes[0, 0].set_title('ChromVAR嵌入 - t-SNE')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    
    # 2. 转录因子活性热图（前20个）
    top_tf_indices = np.argsort(np.std(deviations, axis=0))[-20:]
    top_tf_deviations = deviations[:, top_tf_indices]
    top_tf_names = [motif_names[i] for i in top_tf_indices]
    
    im = axes[0, 1].imshow(top_tf_deviations.T, aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_title('Top 20 转录因子活性')
    axes[0, 1].set_yticks(range(len(top_tf_names)))
    axes[0, 1].set_yticklabels(top_tf_names)
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. 偏差分数分布
    axes[1, 0].hist(deviations.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('偏差分数分布')
    axes[1, 0].set_xlabel('偏差分数')
    axes[1, 0].set_ylabel('频率')
    
    # 4. 最可变转录因子
    tf_variability = np.std(deviations, axis=0)
    top_variable_indices = np.argsort(tf_variability)[-10:]
    top_variable_tf = [motif_names[i] for i in top_variable_indices]
    top_variable_scores = tf_variability[top_variable_indices]
    
    axes[1, 1].barh(range(len(top_variable_tf)), top_variable_scores)
    axes[1, 1].set_yticks(range(len(top_variable_tf)))
    axes[1, 1].set_yticklabels(top_variable_tf)
    axes[1, 1].set_title('最可变转录因子')
    axes[1, 1].set_xlabel('标准差')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'tsne_2d': embeddings_2d,
        'top_variable_tf': list(zip(top_variable_tf, top_variable_scores))
}
def evaluate_chromvar_results(chromvar_results, sparse_matrix, true_labels=None):
    """
    评估ChromVAR结果 - 增强版，包含聚类指标
    
    参数:
    - chromvar_results: ChromVAR分析结果
    - sparse_matrix: 原始稀疏矩阵
    - true_labels: 真实标签（可选）
    """
    print("\n=== ChromVAR结果评估 ===")
    
    embeddings = chromvar_results['embeddings']
    deviations = chromvar_results['deviations']
    
    evaluation_results = {}
    
    # 1. 计算motif活性的生物学一致性
    tf_variability = np.std(deviations, axis=0)
    variability_score = np.mean(tf_variability)
    evaluation_results['variability_score'] = variability_score
    print(f"转录因子活性变异度: {variability_score:.4f}")
    
    # 2. 嵌入质量评估
    from sklearn.neighbors import NearestNeighbors
    n_neighbors = min(10, embeddings.shape[0]-1)
    if n_neighbors > 0:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        neighborhood_preservation = np.mean(distances)
        evaluation_results['neighborhood_preservation'] = neighborhood_preservation
        print(f"邻域保持度: {neighborhood_preservation:.4f}")
    else:
        evaluation_results['neighborhood_preservation'] = 0
        print("样本太少，跳过邻域保持度计算")
    
    # 3. 生物学意义评估
    evaluation_results['embedding_quality'] = chromvar_results['explained_variance']
    print(f"嵌入质量(解释方差): {chromvar_results['explained_variance']:.4f}")
    
    # 4. 计算聚类指标
    n_clusters = min(8, embeddings.shape[0] // 20)  # 自适应聚类数量
    if n_clusters > 1:
        clustering_metrics = calculate_clustering_metrics(
            embeddings, true_labels, n_clusters=n_clusters
        )
        evaluation_results.update(clustering_metrics)
    else:
        print("样本太少，跳过聚类评估")
        evaluation_results.update({
            'silhouette_score': 0,
            'ari': None if true_labels is None else 0,
            'nmi': None if true_labels is None else 0
        })
    
return evaluation_results
def main_chromvar_analysis(mat_file_path, target_dim=50):
    """
    ChromVAR分析主流程 - 增强版，包含完整评估
    """
    print("=" * 60)
    print("开始ChromVAR分析")
    print("=" * 60)
    
    # 1. 加载数据
    mat_data = scipy.io.loadmat(mat_file_path)
    sparse_matrix_1, sparse_matrix_2 = load_chromvar_data(mat_file_path)
    
    # 2. 尝试加载真实标签
    true_labels = load_ground_truth_labels(mat_data)
    if true_labels is not None:
        print(f"找到真实标签，样本数量: {len(true_labels)}")
        # 确保标签数量与矩阵行数匹配
        if len(true_labels) != sparse_matrix_1.shape[0]:
            print("⚠️ 真实标签数量与矩阵行数不匹配，将不使用真实标签")
            true_labels = None
    else:
        print("未找到真实标签，将只计算无监督指标")
    
    all_results = {}
    
    # 3. 对第一个矩阵进行ChromVAR分析
    print("\n" + "="*40)
    print(">>> 分析矩阵1 <<<")
    print("="*40)
    try:
        chromvar_results_1 = reduce_dimension_chromvar(
            sparse_matrix_1, target_dim, matrix_id=1
        )
        viz_results_1 = visualize_chromvar_results(chromvar_results_1, "矩阵1 - ChromVAR分析")
        
        # 正确传递参数：现在函数接受3个参数
        eval_results_1 = evaluate_chromvar_results(
            chromvar_results_1, sparse_matrix_1, true_labels
        )
        
        all_results['matrix1'] = {
            'results': chromvar_results_1,
            'visualization': viz_results_1,
            'evaluation': eval_results_1
        }
        
        # 保存结果
        np.save('chromvar_embeddings_matrix1.npy', chromvar_results_1['embeddings'])
        np.save('chromvar_deviations_matrix1.npy', chromvar_results_1['deviations'])
        
        print("✅ 矩阵1分析成功")
        
    except Exception as e:
        print(f"❌ 矩阵1分析失败: {e}")
        import traceback
        traceback.print_exc()
        all_results['matrix1'] = None
    
    # 4. 对第二个矩阵进行分析（如果存在）
    if sparse_matrix_2 is not None:
        print("\n" + "="*40)
        print(">>> 分析矩阵2 <<<")
        print("="*40)
        try:
            chromvar_results_2 = reduce_dimension_chromvar(
                sparse_matrix_2, target_dim, matrix_id=2
            )
            viz_results_2 = visualize_chromvar_results(chromvar_results_2, "矩阵2 - ChromVAR分析")
            
            # 正确传递参数：现在函数接受3个参数
            eval_results_2 = evaluate_chromvar_results(
                chromvar_results_2, sparse_matrix_2, true_labels
            )
            
            all_results['matrix2'] = {
                'results': chromvar_results_2,
                'visualization': viz_results_2,
                'evaluation': eval_results_2
            }
            
            # 保存结果
            np.save('chromvar_embeddings_matrix2.npy', chromvar_results_2['embeddings'])
            np.save('chromvar_deviations_matrix2.npy', chromvar_results_2['deviations'])
            
            print("✅ 矩阵2分析成功")
            
        except Exception as e:
            print(f"❌ 矩阵2分析失败: {e}")
            import traceback
            traceback.print_exc()
            all_results['matrix2'] = None
    else:
        all_results['matrix2'] = None
    
    # 5. 生成综合评估报告
    print("\n" + "="*60)
    print("综合评估报告")
    print("="*60)
    generate_comprehensive_report(all_results, true_labels is not None)
    
    # 6. 保存转录因子名称和评估结果
    save_final_results(all_results)
    
    print("✅ ChromVAR分析完成!")
    
return all_results
if __name__ == "__main__":
    results = main_chromvar_analysis('pbmc_10xAtac.mat', target_dim=30)
