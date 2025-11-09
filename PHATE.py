import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.io import loadmat
from scipy.sparse import issparse, csr_matrix
import phate
import leidenalg
import igraph as ig
import warnings
warnings.filterwarnings('ignore')
import time
from itertools import product

class ATACseq无预处理优化:
    def __init__(self):
        """初始化ATAC-seq无预处理参数优化器"""
        pass
    
    def 加载数据(self, 文件路径='pbmc_10xAtac.mat'):
        """直接加载原始ATAC-seq数据，不进行预处理"""
        print("=== 直接加载ATAC-seq原始数据 ===")
        try:
            mat数据 = loadmat(文件路径)
            
            # 直接提取数据，不进行任何预处理
            if 'X' in mat数据:
                x变量 = mat数据['X']
                print(f"找到X变量: 形状 {x变量.shape}")
                
                # 直接使用原始数据
                self.原始数据_00 = x变量[0, 0]
                self.原始数据_01 = x变量[0, 1]
                
                print(f"批次00原始数据: {self.原始数据_00.shape}")
                print(f"批次01原始数据: {self.原始数据_01.shape}")
                print("✓ 跳过所有预处理步骤，使用原始数据")
            
            # 加载ground truth - 修复变量名错误
            self.ground_truth = None
            if 'groundtruth' in mat数据:
                self.ground_truth = mat数据['groundtruth'].flatten()  # 修复这里
                print(f"Ground truth: {self.ground_truth.shape}")
                
                # 检查匹配情况
                if self.ground_truth.shape[0] == self.原始数据_00.shape[0]:
                    self.ground_truth_批次 = 0
                    print("✓ Ground truth匹配批次00")
                elif self.ground_truth.shape[0] == self.原始数据_01.shape[0]:
                    self.ground_truth_批次 = 1
                    print("✓ Ground truth匹配批次01")
                else:
                    self.ground_truth_批次 = None
                    print("⚠ Ground truth与批次不匹配")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def 定义参数空间(self):
        """定义参数搜索空间"""
        self.参数组合 = {
            '维度列表': [2, 5, 10, 15, 20, 25, 30, 40, 50],
            'knn列表': [3, 5, 8, 10, 15],
            'decay列表': [10, 20, 40, 60],
            't列表': ['auto', 5, 10, 15]
        }
        
        print("=== 参数搜索空间 ===")
        print(f"维度: {self.参数组合['维度列表']}")
        print(f"KNN: {self.参数组合['knn列表']}")
        print(f"Decay: {self.参数组合['decay列表']}")
        print(f"t: {self.参数组合['t列表']}")
        
        return self.参数组合
    
    def 计算邻域保持度(self, 原始数据, 降维数据, k=15):
        """
        计算邻域保持度 - 评估降维前后邻域结构的一致性
        返回邻域保持度得分 (0-1之间，越高表示邻域保持越好)
        """
        try:
            # 确保使用稠密数组进行计算
            if issparse(原始数据):
                原始数据 = 原始数据.toarray()
            if issparse(降维数据):
                降维数据 = 降维数据.toarray()
                
            # 限制k值不超过样本数
            k = min(k, 原始数据.shape[0] - 1)
            
            # 计算原始空间和降维空间的k近邻
            nbrs_原始 = NearestNeighbors(n_neighbors=k+1).fit(原始数据)
            nbrs_降维 = NearestNeighbors(n_neighbors=k+1).fit(降维数据)
            
            distances_原始, indices_原始 = nbrs_原始.kneighbors(原始数据)
            distances_降维, indices_降维 = nbrs_降维.kneighbors(降维数据)
            
            总保持度 = 0
            有效样本数 = 0
            
            for i in range(len(indices_原始)):
                # 跳过自身，取前k个邻居
                原始邻居集 = set(indices_原始[i][1:k+1])
                降维邻居集 = set(indices_降维[i][1:k+1])
                
                # 计算Jaccard相似度
                交集大小 = len(原始邻居集 & 降维邻居集)
                并集大小 = len(原始邻居集 | 降维邻居集)
                
                if 并集大小 > 0:
                    jaccard相似度 = 交集大小 / 并集大小
                    总保持度 += jaccard相似度
                    有效样本数 += 1
            
            平均邻域保持度 = 总保持度 / 有效样本数 if 有效样本数 > 0 else 0
            return 平均邻域保持度
            
        except Exception as e:
            print(f"  邻域保持度计算失败: {e}")
            return 0.0
    
    def 执行PHATE降维(self, n_components, knn, decay, t):
        """使用原始数据执行PHATE降维"""
        try:
            phate操作器 = phate.PHATE(
                n_components=n_components,
                knn=knn,
                decay=decay,
                t=t,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            # 直接对原始数据降维
            print(f"  执行PHATE降维...")
            phate_00 = phate操作器.fit_transform(self.原始数据_00)
            phate_01 = phate操作器.fit_transform(self.原始数据_01)
            
            # 合并结果
            phate合并 = np.vstack([phate_00, phate_01])
            
            return {
                'phate_00': phate_00,
                'phate_01': phate_01,
                'phate合并': phate合并,
                '批次标签': np.array([0] * len(phate_00) + [1] * len(phate_01))
            }
            
        except Exception as e:
            print(f"  PHATE降维失败: {e}")
            return None
    
    def leiden聚类(self, 数据):
        """Leiden聚类"""
        try:
            n_neighbors = min(30, 数据.shape[0] // 10)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(数据)
            distances, indices = nbrs.kneighbors(数据)
            
            sources, targets, weights = [], [], []
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):
                    sources.append(i)
                    targets.append(indices[i, j])
                    weights.append(1.0 / (distances[i, j] + 1e-8))
            
            graph = ig.Graph(directed=False)
            graph.add_vertices(len(数据))
            graph.add_edges(list(zip(sources, targets)))
            graph.es['weight'] = weights
            
            # 尝试不同参数名
            try:
                partition = leidenalg.find_partition(
                    graph, 
                    leidenalg.ModularityVertexPartition,
                    resolution_parameter=1.0,
                    weights=weights,
                    seed=42
                )
            except TypeError:
                try:
                    partition = leidenalg.find_partition(
                        graph, 
                        leidenalg.ModularityVertexPartition,
                        resolution=1.0,
                        weights=weights,
                        seed=42
                    )
                except TypeError:
                    partition = leidenalg.find_partition(
                        graph, 
                        leidenalg.ModularityVertexPartition,
                        weights=weights,
                        seed=42
                    )
            
            聚类标签 = np.array(partition.membership)
            print(f"  Leiden聚类完成: {len(np.unique(聚类标签))}个聚类")
            return 聚类标签
            
        except Exception as e:
            print(f"  Leiden失败，使用KMeans: {e}")
            from sklearn.cluster import KMeans
            n_clusters = min(10, 数据.shape[0] // 20)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            聚类标签 = kmeans.fit_predict(数据)
            print(f"  KMeans聚类完成: {len(np.unique(聚类标签))}个聚类")
            return 聚类标签
    
    def 计算指标(self, 预测标签, 真实标签):
        """计算ARI和NMI"""
        if 真实标签 is None or len(预测标签) != len(真实标签):
            return None, None
        
        try:
            ari = adjusted_rand_score(真实标签, 预测标签)
            nmi = normalized_mutual_info_score(真实标签, 预测标签)
            return ari, nmi
        except Exception as e:
            print(f"  指标计算失败: {e}")
            return None, None
    
    def 评估参数组合(self, n_components, knn, decay, t):
        """评估单个参数组合"""
        start_time = time.time()
        
        try:
            # 执行PHATE降维
            降维结果 = self.执行PHATE降维(n_components, knn, decay, t)
            if 降维结果 is None:
                return None
            
            # 计算邻域保持度
            print(f"  计算邻域保持度...")
            邻域保持度_00 = self.计算邻域保持度(self.原始数据_00, 降维结果['phate_00'])
            邻域保持度_01 = self.计算邻域保持度(self.原始数据_01, 降维结果['phate_01'])
            平均邻域保持度 = (邻域保持度_00 + 邻域保持度_01) / 2
            
            # 执行聚类
            聚类标签 = self.leiden聚类(降维结果['phate合并'])
            
            # 计算指标
            ari, nmi = None, None
            轮廓系数 = None
            
            if self.ground_truth is not None and self.ground_truth_批次 is not None:
                if self.ground_truth_批次 == 0:
                    批次标签 = 聚类标签[:len(降维结果['phate_00'])]
                    ari, nmi = self.计算指标(批次标签, self.ground_truth)
                    if len(np.unique(批次标签)) > 1:
                        轮廓系数 = silhouette_score(降维结果['phate_00'], 批次标签)
                elif self.ground_truth_批次 == 1:
                    批次标签 = 聚类标签[len(降维结果['phate_00']):]
                    ari, nmi = self.计算指标(批次标签, self.ground_truth)
                    if len(np.unique(批次标签)) > 1:
                        轮廓系数 = silhouette_score(降维结果['phate_01'], 批次标签)
            
            # 如果没有外部标签，计算轮廓系数
            if 轮廓系数 is None and len(np.unique(聚类标签)) > 1:
                轮廓系数 = silhouette_score(降维结果['phate合并'], 聚类标签)
            
            耗时 = time.time() - start_time
            
            return {
                'n_components': n_components,
                'knn': knn,
                'decay': decay,
                't': t,
                'ARI': ari,
                'NMI': nmi,
                '轮廓系数': 轮廓系数,
                '邻域保持度': 平均邻域保持度,
                '邻域保持度_00': 邻域保持度_00,
                '邻域保持度_01': 邻域保持度_01,
                '聚类数量': len(np.unique(聚类标签)),
                '耗时': 耗时,
                '降维结果': 降维结果,
                '聚类标签': 聚类标签
            }
            
        except Exception as e:
            print(f"  参数评估失败: {e}")
            return None
    
    def 自动参数优化(self, 最大组合数=20):
        """自动搜索最佳参数和维度"""
        print("\n=== 开始无预处理参数优化 ===")
        
        # 获取参数空间
        参数组合 = self.定义参数空间()
        
        # 生成所有组合
        all_combinations = list(product(
            参数组合['维度列表'],
            参数组合['knn列表'], 
            参数组合['decay列表'],
            参数组合['t列表']
        ))
        
        # 随机选择组合测试
        if len(all_combinations) > 最大组合数:
            np.random.seed(42)
            selected_combinations = np.random.choice(
                len(all_combinations), 最大组合数, replace=False
            )
            test_combinations = [all_combinations[i] for i in selected_combinations]
            print(f"随机选择 {最大组合数} 个参数组合进行测试")
        else:
            test_combinations = all_combinations
            print(f"测试所有 {len(test_combinations)} 个参数组合")
        
        结果列表 = []
        
        for i, (n_components, knn, decay, t) in enumerate(test_combinations):
            print(f"\n[{i+1}/{len(test_combinations)}] 测试: 维度{n_components}, knn{knn}, decay{decay}, t{t}")
            
            结果 = self.评估参数组合(n_components, knn, decay, t)
            if 结果 is not None:
                结果列表.append(结果)
                if 结果['ARI'] is not None:
                    print(f"  ✓ 完成 - ARI: {结果['ARI']:.4f}, 邻域保持: {结果['邻域保持度']:.4f}, 聚类: {结果['聚类数量']}个")
                else:
                    print(f"  ✓ 完成 - 轮廓系数: {结果['轮廓系数']:.4f}, 邻域保持: {结果['邻域保持度']:.4f}, 聚类: {结果['聚类数量']}个")
        
        # 找到最佳结果
        if 结果列表:
            # 优先使用ARI
            有效结果 = [r for r in 结果列表 if r['ARI'] is not None]
            if 有效结果:
                最佳结果 = max(有效结果, key=lambda x: x['ARI'])
                print(f"\n🎯 基于ARI找到最佳结果: ARI = {最佳结果['ARI']:.4f}")
            else:
                # 使用轮廓系数
                有效结果 = [r for r in 结果列表 if r['轮廓系数'] is not None]
                if 有效结果:
                    最佳结果 = max(有效结果, key=lambda x: x['轮廓系数'])
                    print(f"\n🎯 基于轮廓系数找到最佳结果: 轮廓系数 = {最佳结果['轮廓系数']:.4f}")
                else:
                    最佳结果 = 结果列表[0]
                    print(f"\n⚠ 使用第一个可用结果")
            
            self.最佳结果 = 最佳结果
            self.所有结果 = 结果列表
            
            return 最佳结果
        else:
            print("❌ 所有参数组合都失败了")
            return None
    
    def 生成优化报告(self):
        """生成优化报告"""
        if not hasattr(self, '最佳结果'):
            print("未找到优化结果")
            return
        
        最佳 = self.最佳结果
        
        print("\n" + "="*80)
        print("          ATAC-seq无预处理参数优化报告")
        print("="*80)
        
        print(f"🎯 最佳参数组合:")
        print(f"  维度: {最佳['n_components']}")
        print(f"  KNN: {最佳['knn']}")
        print(f"  Decay: {最佳['decay']}")
        print(f"  t: {最佳['t']}")
        
        print(f"\n📊 降维效果:")
        print(f"  邻域保持度: {最佳['邻域保持度']:.4f}")
        print(f"  批次00邻域保持度: {最佳['邻域保持度_00']:.4f}")
        print(f"  批次01邻域保持度: {最佳['邻域保持度_01']:.4f}")
        
        print(f"\n📈 聚类效果:")
        print(f"  聚类数量: {最佳['聚类数量']}")
        if 最佳['ARI'] is not None:
            print(f"  ARI: {最佳['ARI']:.4f}")
        if 最佳['NMI'] is not None:
            print(f"  NMI: {最佳['NMI']:.4f}")
        if 最佳['轮廓系数'] is not None:
            print(f"  轮廓系数: {最佳['轮廓系数']:.4f}")
        print(f"  耗时: {最佳['耗时']:.1f}秒")
        
        # 显示前5个最佳结果
        print(f"\n🏆 前5个最佳参数组合:")
        有效结果 = [r for r in self.所有结果 if r['ARI'] is not None]
        if not 有效结果:
            有效结果 = [r for r in self.所有结果 if r['轮廓系数'] is not None]
        
        if 有效结果:
            if 'ARI' in 有效结果[0] and 有效结果[0]['ARI'] is not None:
                排序结果 = sorted(有效结果, key=lambda x: x['ARI'], reverse=True)
                排序依据 = "ARI"
            else:
                排序结果 = sorted(有效结果, key=lambda x: x['轮廓系数'], reverse=True)
                排序依据 = "轮廓系数"
            
            print(f"排序依据: {排序依据}")
            print(f"{'排名':<4} {'维度':<6} {'KNN':<4} {'Decay':<6} {'t':<8} {'ARI':<8} {'NMI':<8} {'轮廓系数':<10} {'邻域保持':<10}")
            print("-" * 85)
            for i, 结果 in enumerate(排序结果[:5]):
                ari_str = f"{结果['ARI']:.4f}" if 结果['ARI'] is not None else "N/A"
                nmi_str = f"{结果['NMI']:.4f}" if 结果['NMI'] is not None else "N/A"
                轮廓_str = f"{结果['轮廓系数']:.4f}" if 结果['轮廓系数'] is not None else "N/A"
                邻域_str = f"{结果['邻域保持度']:.4f}"
                print(f"{i+1:<4} {结果['n_components']:<6} {结果['knn']:<4} {结果['decay']:<6} {str(结果['t']):<8} {ari_str:<8} {nmi_str:<8} {轮廓_str:<10} {邻域_str:<10}")
        
        print("="*80)
    
    def 保存最佳结果(self):
        """保存最佳结果"""
        if not hasattr(self, '最佳结果'):
            return
        
        最佳 = self.最佳结果
        
        # 保存降维结果
        pd.DataFrame(最佳['降维结果']['phate_00']).to_csv('best_raw_phate_X00.csv', index=False)
        pd.DataFrame(最佳['降维结果']['phate_01']).to_csv('best_raw_phate_X01.csv', index=False)
        pd.DataFrame(最佳['降维结果']['phate合并']).to_csv('best_raw_phate_combined.csv', index=False)
        
        # 保存聚类结果
        pd.DataFrame({'cluster': 最佳['聚类标签']}).to_csv('best_raw_leiden_clusters.csv', index=False)
        
        # 保存参数信息
        参数信息 = pd.DataFrame([{
            'n_components': 最佳['n_components'],
            'knn': 最佳['knn'],
            'decay': 最佳['decay'],
            't': 最佳['t'],
            'ARI': 最佳['ARI'],
            'NMI': 最佳['NMI'],
            '轮廓系数': 最佳['轮廓系数'],
            '邻域保持度': 最佳['邻域保持度'],
            '邻域保持度_00': 最佳['邻域保持度_00'],
            '邻域保持度_01': 最佳['邻域保持度_01'],
            '聚类数量': 最佳['聚类数量'],
            '耗时_秒': 最佳['耗时']
        }])
        参数信息.to_csv('best_raw_parameters.csv', index=False)
        
        print("✓ 最佳结果已保存:")
        print("  - best_raw_phate_X00.csv (批次00降维结果)")
        print("  - best_raw_phate_X01.csv (批次01降维结果)")
        print("  - best_raw_phate_combined.csv (合并降维结果)")
        print("  - best_raw_leiden_clusters.csv (聚类结果)")
        print("  - best_raw_parameters.csv (参数信息)")
    
    def 执行完整优化(self):
        """执行完整优化流程"""
        print("开始ATAC-seq无预处理参数优化")
        print("="*50)
        
        # 1. 加载数据（无预处理）
        if not self.加载数据():
            return
        
        # 2. 自动参数优化
        最佳结果 = self.自动参数优化(最大组合数=20)
        
        if 最佳结果 is None:
            print("参数优化失败")
            return
        
        # 3. 生成报告
        self.生成优化报告()
        
        # 4. 保存结果
        self.保存最佳结果()
        
        print(f"\n🎉 无预处理参数优化完成！")
        print(f"最佳维度: {最佳结果['n_components']}")
        if 最佳结果['ARI'] is not None:
            print(f"最佳ARI: {最佳结果['ARI']:.4f}")
        print(f"最佳邻域保持度: {最佳结果['邻域保持度']:.4f}")

# 执行优化
if __name__ == "__main__":
    优化器 = ATACseq无预处理优化()
    优化器.执行完整优化()