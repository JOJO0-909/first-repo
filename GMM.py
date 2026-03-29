"""
GMM Clustering - Remove First 2 Dimensions (Technical Noise)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GMM Clustering - Remove Technical Noise (First 2 Dimensions)")
print("="*70)

# ==================== 1. Load Data ====================
lsi_path = r"C:\Users\jinzhi\Desktop\聚类分析\lsi_embedding_final.csv"
label_path = r"C:\Users\jinzhi\Desktop\聚类分析\细胞真实标签_groundtruth.csv"

print("\n[1/7] Loading data...")
lsi_df = pd.read_csv(lsi_path, index_col=0)
X_full = lsi_df.values
n_samples, n_features = X_full.shape
print(f"   ✓ Full data: {n_samples} samples, {n_features} features")

label_df = pd.read_csv(label_path)
true_labels = label_df['True_Label'].values
n_true_classes = len(np.unique(true_labels))
print(f"   ✓ True classes: {n_true_classes}")

# ==================== 2. Remove First 2 Dimensions ====================
print("\n[2/7] Removing technical noise (first 2 dimensions)...")
print(f"   Original dimensions: {n_features}")

# Remove first 2 columns (dimension 0 and 1)
X_clean = X_full[:, 2:]  # Keep only dimensions 2-29
n_clean_features = X_clean.shape[1]

print(f"   Removed dimensions: 0, 1 (LSI_0, LSI_1)")
print(f"   Clean data: {n_samples} samples, {n_clean_features} features")
print(f"   Features kept: LSI_2 to LSI_{n_clean_features-1}")

# ==================== 3. Compare with/without noise ====================
print("\n[3/7] Comparing with and without noise removal...")

# Standardize both versions
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

scaler_clean = StandardScaler()
X_clean_scaled = scaler_clean.fit_transform(X_clean)

# Quick GMM to see difference (using true k)
k_test = n_true_classes
print(f"   Quick test with k={k_test}...")

# With full dimensions
gmm_full = GaussianMixture(n_components=k_test, random_state=42, n_init=3, max_iter=100)
labels_full = gmm_full.fit_predict(X_full_scaled)
ari_full = adjusted_rand_score(true_labels, labels_full)
nmi_full = normalized_mutual_info_score(true_labels, labels_full)

# With cleaned dimensions (no first 2)
gmm_clean = GaussianMixture(n_components=k_test, random_state=42, n_init=3, max_iter=100)
labels_clean = gmm_clean.fit_predict(X_clean_scaled)
ari_clean = adjusted_rand_score(true_labels, labels_clean)
nmi_clean = normalized_mutual_info_score(true_labels, labels_clean)

print("\n   Comparison (k={}):".format(k_test))
print("-"*50)
print(f"   Full dimensions (30 dims):  ARI={ari_full:.4f}, NMI={nmi_full:.4f}")
print(f"   Clean dimensions (28 dims): ARI={ari_clean:.4f}, NMI={nmi_clean:.4f}")
print("-"*50)
print(f"   Improvement: ARI +{ari_clean-ari_full:.4f}, NMI +{nmi_clean-nmi_full:.4f}")

# ==================== 4. Test different k values on clean data ====================
print("\n[4/7] Testing different k values on clean data...")

max_k = min(30, int(np.sqrt(n_samples)) + 5)
k_range = range(2, max_k + 1)
ari_scores = []
nmi_scores = []

print(f"   Testing k from {min(k_range)} to {max(k_range)}...")
print("-"*70)

for idx, k in enumerate(k_range):
    print(f"   [{idx+1}/{len(k_range)}] Testing k = {k:2d}...", end=' ', flush=True)
    
    try:
        gmm = GaussianMixture(
            n_components=k, 
            random_state=42, 
            n_init=3,
            max_iter=200,
            covariance_type='diag',
            tol=1e-3
        )
        
        cluster_labels = gmm.fit_predict(X_clean_scaled)
        
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        
        print(f"ARI={ari:.4f}, NMI={nmi:.4f}")
        
    except Exception as e:
        print(f"Failed: {str(e)[:50]}")
        ari_scores.append(np.nan)
        nmi_scores.append(np.nan)

# ==================== 5. Find best k ====================
print("\n[5/7] Finding best k...")

valid_idx = [i for i, ari in enumerate(ari_scores) if not np.isnan(ari)]
valid_k = [k_range[i] for i in valid_idx]
valid_ari = [ari_scores[i] for i in valid_idx]
valid_nmi = [nmi_scores[i] for i in valid_idx]

if valid_k:
    best_k_ari = valid_k[np.argmax(valid_ari)]
    best_k_nmi = valid_k[np.argmax(valid_nmi)]
    best_ari = max(valid_ari)
    best_nmi = max(valid_nmi)
    
    print("\n" + "="*70)
    print("Results Summary (After Removing First 2 Dimensions)")
    print("="*70)
    print(f"True classes: {n_true_classes}")
    print(f"Best k by ARI:  k={best_k_ari:2d} (ARI={best_ari:.4f})")
    print(f"Best k by NMI:  k={best_k_nmi:2d} (NMI={best_nmi:.4f})")
    print("="*70)
    
    # Show top 10
    print("\nTop 10 k by ARI:")
    print("-"*70)
    results = list(zip(valid_k, valid_ari, valid_nmi))
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    print(f"{'k':<6} {'ARI':<10} {'NMI':<10}")
    print("-"*70)
    for k, ari, nmi in results_sorted[:10]:
        print(f"{k:<6} {ari:<10.4f} {nmi:<10.4f}")
    print("-"*70)

# ==================== 6. Visualization ====================
print("\n[6/7] Generating visualizations...")

# Plot ARI and NMI
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ARI plot
axes[0].plot(valid_k, valid_ari, 'bo-', linewidth=2, markersize=6)
axes[0].axvline(x=n_true_classes, color='red', linestyle='--', linewidth=2, 
                label=f'True k={n_true_classes}')
axes[0].axvline(x=best_k_ari, color='green', linestyle='--', linewidth=2, 
                label=f'Best ARI k={best_k_ari}')
axes[0].fill_between(valid_k, valid_ari, alpha=0.3)
axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
axes[0].set_ylabel('ARI Score', fontsize=12)
axes[0].set_title(f'ARI after removing first 2 dimensions (Best k={best_k_ari})', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# NMI plot
axes[1].plot(valid_k, valid_nmi, 'ro-', linewidth=2, markersize=6)
axes[1].axvline(x=n_true_classes, color='red', linestyle='--', linewidth=2, 
                label=f'True k={n_true_classes}')
axes[1].axvline(x=best_k_nmi, color='green', linestyle='--', linewidth=2, 
                label=f'Best NMI k={best_k_nmi}')
axes[1].fill_between(valid_k, valid_nmi, alpha=0.3)
axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
axes[1].set_ylabel('NMI Score', fontsize=12)
axes[1].set_title(f'NMI after removing first 2 dimensions (Best k={best_k_nmi})', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ==================== 7. UMAP Visualization ====================
print(f"\n[7/7] Generating UMAP for best k={best_k_ari}...")

# Final GMM on clean data
gmm_final = GaussianMixture(
    n_components=best_k_ari, 
    random_state=42, 
    n_init=5,
    max_iter=300,
    covariance_type='diag'
)
final_labels = gmm_final.fit_predict(X_clean_scaled)

# UMAP on clean data
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = reducer.fit_transform(X_clean_scaled)

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# GMM result on clean data
scatter1 = axes[0].scatter(X_umap[:, 0], X_umap[:, 1], 
                           c=final_labels, cmap='tab10', s=8, alpha=0.6)
axes[0].set_title(f'GMM Clustering (Clean Data, k={best_k_ari}, ARI={best_ari:.3f})', fontsize=12)
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
plt.colorbar(scatter1, ax=axes[0], label='GMM Cluster')

# True labels
scatter2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], 
                           c=true_labels, cmap='tab10', s=8, alpha=0.6)
axes[1].set_title(f'Ground Truth (True k={n_true_classes})', fontsize=12)
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
plt.colorbar(scatter2, ax=axes[1], label='True Label')

plt.tight_layout()
plt.show()

# ==================== 8. Save Results ====================
print("\n[8/8] Saving results...")

# Save clean data (without first 2 dimensions)
clean_df = pd.DataFrame(X_clean, columns=[f'LSI_{i+2}' for i in range(n_clean_features)])
clean_df.index = lsi_df.index
clean_df.to_csv(r"C:\Users\jinzhi\Desktop\聚类分析\lsi_embedding_clean.csv")
print(f"   ✓ Clean data saved (28 dimensions)")

# Save comparison results
comparison_df = pd.DataFrame({
    'k': valid_k,
    'ARI': valid_ari,
    'NMI': valid_nmi
})
comparison_path = r"C:\Users\jinzhi\Desktop\聚类分析\gmm_clean_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"   ✓ Comparison saved: {comparison_path}")

# Save final result
result_df = lsi_df.copy()
result_df['gmm_cluster'] = final_labels
result_df['true_label'] = true_labels
output_path = r"C:\Users\jinzhi\Desktop\聚类分析\lsi_embedding_gmm_clean.csv"
result_df.to_csv(output_path)
print(f"   ✓ Final result saved: {output_path}")

# Save report
report_path = r"C:\Users\jinzhi\Desktop\聚类分析\gmm_clean_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("GMM Clustering Report (After Removing First 2 Dimensions)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Original dimensions: {n_features}\n")
    f.write(f"Removed dimensions: LSI_0, LSI_1 (technical noise)\n")
    f.write(f"Clean dimensions: {n_clean_features} (LSI_2 to LSI_{n_clean_features-1})\n\n")
    f.write(f"Data: {n_samples} samples\n")
    f.write(f"True classes: {n_true_classes}\n")
    f.write(f"Best k (ARI): {best_k_ari} (ARI={best_ari:.4f})\n")
    f.write(f"Best k (NMI): {best_k_nmi} (NMI={best_nmi:.4f})\n\n")
    f.write("Comparison with full data (k={}):\n".format(n_true_classes))
    f.write(f"  Full data: ARI={ari_full:.4f}, NMI={nmi_full:.4f}\n")
    f.write(f"  Clean data: ARI={ari_clean:.4f}, NMI={nmi_clean:.4f}\n")
    f.write(f"  Improvement: ARI +{ari_clean-ari_full:.4f}, NMI +{nmi_clean-nmi_full:.4f}\n\n")
    f.write("Top 10 k by ARI:\n")
    for k, ari, nmi in results_sorted[:10]:
        f.write(f"  k={k:2d}: ARI={ari:.4f}, NMI={nmi:.4f}\n")

print(f"   ✓ Report saved: {report_path}")

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)
print(f"Best k (clean data): {best_k_ari} (ARI={best_ari:.4f}, NMI={best_nmi:.4f})")
print(f"True k: {n_true_classes}")
print(f"Improvement vs full data: ARI +{ari_clean-ari_full:.4f}")
print("="*70)