import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv('../data/smoking_driking_dataset_Ver01.csv')

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Remove near-zero variance features
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('variance', VarianceThreshold(threshold=0.01))
])

# Apply preprocessing
X = pipeline.fit_transform(df)
print(f"Processed data shape: {X.shape}")

# ---------- Clustering ----------

# 1. K-means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# 2. GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)

# 3. Hierarchical clustering on PCA-reduced data
# Safe PCA reduction
max_samples = 5000
if X.shape[0] > max_samples:
    sample_idx = np.random.choice(X.shape[0], size=max_samples, replace=False)
    X_hier = X[sample_idx]
else:
    X_hier = X

pca = PCA(n_components=20, random_state=42)
X_hier_reduced = pca.fit_transform(X_hier)

hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(X_hier_reduced)

# ---------- Evaluation Function ----------
def evaluate_clustering_safe(X_eval, labels, algorithm_name, sample_size=5000):
    n_samples = X_eval.shape[0]
    if n_samples > sample_size:
        idx = np.random.choice(n_samples, size=sample_size, replace=False)
        X_eval_sample = X_eval[idx]
        labels_sample = labels[idx]
    else:
        X_eval_sample = X_eval
        labels_sample = labels

    sil = silhouette_score(X_eval_sample, labels_sample)
    ch = calinski_harabasz_score(X_eval_sample, labels_sample)
    db = davies_bouldin_score(X_eval_sample, labels_sample)
    print(f"{algorithm_name}: Silhouette={sil:.4f}, Calinski-Harabasz={ch:.2f}, Davies-Bouldin={db:.4f}")


# Evaluate
print("\nClustering evaluation:")
evaluate_clustering_safe(X, kmeans_labels, "K-means")
evaluate_clustering_safe(X, gmm_labels, "GMM")
evaluate_clustering_safe(X_hier_reduced, hier_labels, "Hierarchical (PCA, sampled)")
