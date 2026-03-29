import scipy.io as sio
import scipy.sparse as sp
import scanpy as sc
import time
import tracemalloc
import phate


def run_phate_dimred(adata, n_components=30):
    print("-> 开始 PHATE 流形降维...")
    tracemalloc.start()
    t_start = time.time()

    # 配置 PHATE
    phate_op = phate.PHATE(n_components=n_components, n_jobs=-1, random_state=42)
    phate_results = phate_op.fit_transform(adata.X)

    adata.obsm['X_PHATE'] = phate_results

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
        sparse_X_bin = sparse_X.astype(bool).astype(int)

        adata = sc.AnnData(X=sparse_X_bin)
        adata.obs['groundtruth'] = [str(i) for i in groundtruth]

        adata = run_phate_dimred(adata)
        adata.write(f"phate_result_{dim_name}.h5ad")