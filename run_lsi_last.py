import scipy.io as sio
import scipy.sparse as sp
import scanpy as sc
import numpy as np
import time
import tracemalloc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def run_standard_lsi(adata, n_components=30):
    print("-> 开始 Standard LSI 降维...")
    tracemalloc.start()
    t_start = time.time()

    # 1. TF-IDF 变换
    tf = normalize(adata.X, norm='l1', axis=1)
    peak_open_counts = np.array((adata.X > 0).sum(axis=0)).ravel()
    idf = np.log(adata.shape[0] / (peak_open_counts + 1e-10))
    idf_diag = sp.diags(idf, shape=(adata.shape[1], adata.shape[1]), format='csr')
    tf_idf = tf @ idf_diag

    # 2. SVD 降维
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_result = svd.fit_transform(tf_idf)
    adata.obsm['X_LSI'] = lsi_result

    t_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    adata.uns['run_time'] = t_end - t_start
    adata.uns['peak_memory_mb'] = peak / 10 ** 6
    print(f"完成！耗时: {adata.uns['run_time']:.2f}s, 峰值内存: {adata.uns['peak_memory_mb']:.2f}MB")
    return adata


if __name__ == "__main__":
    mat_data = sio.loadmat(r"E:\大创\pbmc_10xAtac.mat")
    groundtruth = mat_data['groundtruth'].flatten()

    for idx, (name, dim_name) in enumerate([(0, '36k'), (1, '95k')]):
        print(f"\n========== 正在处理 {dim_name} 数据集 ==========")
        sparse_X = sp.csr_matrix(mat_data['X'][0, idx])
        sparse_X_bin = sparse_X.astype(bool).astype(int)  # 二值化处理 x_ij \in {0,1}

        adata = sc.AnnData(X=sparse_X_bin)
        adata.obs['groundtruth'] = [str(i) for i in groundtruth]

        adata = run_standard_lsi(adata)
        adata.write(f"lsi_result_{dim_name}.h5ad")