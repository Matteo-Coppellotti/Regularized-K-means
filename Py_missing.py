import numpy as np
import pandas as pd
import skfda
from skfda.representation.grid import FDataGrid
from skfda.ml.clustering import KMeans
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKLearnKMeans
from skimage.feature import hog
import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.pca import TangentPCA
from sklearn.preprocessing import StandardScaler
from persim import PersistenceImager
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import os
import glob
import re
import shutil

curve_types = ["gauss"] #"wave", , "warped_bump", "spline_like"

data_dir = "data"
recon_data_dir = "data_reconstructed"
dir = "output"

with open("./data/gen_log.txt") as f:
    text = f.read()

comb_match = re.search(r'^combination\s*[:=]\s*(\d+)', text, re.MULTILINE)
combination = comb_match.group(1) if comb_match else None

scalar_keys = ("n_obs_per_clust", "n_clust", "N", "seed")
scalar_pat  = rf'^({"|".join(scalar_keys)})\s*[:=]\s*(\d+)\s*$'
scalars = {k: int(v) for k, v in re.findall(scalar_pat, text, re.MULTILINE)}

n_obs_per_clust = scalars["n_obs_per_clust"]
n_clust         = scalars["n_clust"]
N               = scalars["N"]
seed            = scalars["seed"]

print("combination     =", combination)        
print("n_obs_per_clust =", n_obs_per_clust)    
print("n_clust         =", n_clust)            
print("N               =", N)                  
print("seed            =", seed)  

output_dir = "output_competitors/reconstructed"

methods = {}
for dim in ["1d", "2d", "3d"]:
    collected = set()
    for type in ["scatter", "area"]:
        for curve in curve_types:
            temp_path = os.path.join(recon_data_dir, dim, type, curve)
            if not os.path.isdir(temp_path):
                continue
            for method in os.listdir(temp_path):
                #check that method is a directory and it isn't empty, if emplty delete the directory
                method_path = os.path.join(temp_path, method)
                if not os.path.isdir(method_path) or not os.listdir(method_path):
                    shutil.rmtree(method_path, ignore_errors=True)
                    continue
                collected.add(method)
    methods[dim] = sorted(collected)

competitors = {
    "1d": ["scikit", "scikit_pca", "scikit_fpca"],
    "2d": ["scikit", "scikit_pca", "scikit_hog", "geomstats"],
    "3d": ["scikit", "scikit_pca", "geomstats"]
}

# 1D
for dims in ["1d", "2d", "3d"]: #
    for method in methods[dims]:
        for competitor in competitors[dims]:
            for type in ["scatter", "area"]:
                for curve in curve_types:
                    out_path = os.path.join(output_dir, dims, method, type, competitor, curve)
                    if os.path.exists(out_path):
                        shutil.rmtree(out_path)
                    os.makedirs(out_path, exist_ok=True)

                    if dims == "1d":
                        if competitor == "scikit":
                            for i in range(N):
                                Y = pd.read_csv(os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv"), header=None).values
                                Y = np.nan_to_num(Y)
                                nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
                                grid_points = nodes.flatten()
                                fd = FDataGrid(data_matrix=Y, grid_points=grid_points)

                                init_indices = [0, n_obs_per_clust, 2*n_obs_per_clust]
                                init_centroids = fd[init_indices]

                                # Cluster
                                model = KMeans(n_clusters=3, init=init_centroids)
                                model.fit(fd)

                                # Save labels
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)
                        
                        if competitor == "scikit_pca":
                            # Sort mesh nodes for consistent reshaping
                            nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
                            sorted_indices = np.lexsort((nodes[:, 0],))  # sort by x-coordinate

                            for i in range(N):
                                # Load function evaluations (rows = functions, cols = node values)
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values
                                Y = np.nan_to_num(Y)

                                # Ensure shape is always (n_obs, n_nodes)
                                if Y.ndim == 1:
                                    Y = Y.reshape(1, -1)

                                # Sort node values consistently
                                Y_sorted = Y[:, sorted_indices]

                                # Optional: standardize
                                # Y_sorted = StandardScaler().fit_transform(Y_sorted)

                                # PCA: retain 95% variance
                                pca = PCA(n_components=0.95)
                                Y_pca = pca.fit_transform(Y_sorted)

                                # Init centroids by observation index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_pca[init_indices]

                                # KMeans clustering
                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_pca)

                                # Save memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                        if competitor == "scikit_fpca":
                            for i in range(N):
                                # Load data
                                Y = pd.read_csv(os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv"), header=None).values
                                Y = np.nan_to_num(Y)
                                nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
                                grid_points = nodes.flatten()
                                fd = FDataGrid(data_matrix=Y, grid_points=grid_points)

                                # Apply FPCA
                                fpca = FPCA(n_components=min(20, len(fd)))  # or some upper bound
                                fpca.fit(fd)

                                # Compute cumulative explained variance
                                cumvar = np.cumsum(fpca.explained_variance_ratio_)
                                n_keep = np.searchsorted(cumvar, 0.95) + 1  # Number of PCs to keep

                                # Re-fit with optimal number of components
                                fpca = FPCA(n_components=n_keep)
                                X_fpca = fpca.fit_transform(fd)

                                # Use sklearn KMeans
                                init_indices = [0, n_obs_per_clust, 2*n_obs_per_clust]
                                init_centroids = X_fpca[init_indices]

                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(X_fpca)

                                # Save results
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                    if dims == "2d":
                        if competitor == "scikit":
                            # Load and sort nodes (3D coordinates)
                            nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values  # (n_nodes, 3)
                            sorted_indices = np.lexsort((nodes[:, 1], nodes[:, 0]))  # sort by x, y, z
                            nodes_sorted = nodes[sorted_indices]

                            x_unique = np.unique(nodes_sorted[:, 0])
                            y_unique = np.unique(nodes_sorted[:, 1])
                            grid_points = [x_unique, y_unique]
                            n_x, n_y,= len(x_unique), len(y_unique)

                            for i in range(N):
                                # Load function evaluations (n_samples x n_nodes)
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Sort node values to match x-y-z grid layout
                                Y_sorted = Y[:, sorted_indices]  # shape: (n_samples, n_nodes)

                                # Reshape to (n_samples, n_x, n_y, n_z)
                                data_matrix = Y_sorted.reshape((Y.shape[0], n_x, n_y))

                                # Build FDataGrid object
                                fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

                                # Initialize centroids by index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = fd[init_indices]

                                # KMeans clustering
                                model = KMeans(n_clusters=n_clust, init=init_centroids)
                                model.fit(fd)

                                # Save cluster memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                        if competitor == "scikit_pca":
                            for i in range(N):
                                # Load function evaluations (each row = one function)
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Apply PCA to reduce dimensionality (95% variance retained)
                                pca = PCA(n_components=0.95)
                                Y_pca = pca.fit_transform(Y)  # shape: (n_samples, n_pcs)

                                # Define init centroids (by index)
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_pca[init_indices]

                                # KMeans clustering
                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_pca)

                                # Save memberships (1 row per run)
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                        if competitor == "scikit_hog":
                            # Determine grid shape (2D)
                            nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
                            h = np.unique(nodes[:, 0]).size
                            w = np.unique(nodes[:, 1]).size

                            for i in range(N):
                                # Load function evaluations
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                hog_features = []
                                for func in Y:
                                    img = func.reshape((h, w))  # reshape flat function to 2D image
                                    img = (img - img.min()) / (img.max() - img.min())  # optional normalization

                                    features = hog(
                                        img,
                                        orientations=9,
                                        pixels_per_cell=(5, 5),
                                        cells_per_block=(2, 2),
                                        block_norm='L2-Hys',
                                        visualize=False,
                                        feature_vector=True
                                    )
                                    hog_features.append(features)

                                Y_hog = np.array(hog_features)

                                # Optional: reduce HOG features with PCA
                                # pca = PCA(n_components=0.95)
                                # Y_hog = pca.fit_transform(Y_hog)

                                # Initialize centroids by index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_hog[init_indices]

                                # KMeans clustering
                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_hog)

                                # Save cluster memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False) 

                        if competitor == "geomstats":
                            for i in range(N):
                                # Load function evaluations
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Standardize
                                Y = StandardScaler().fit_transform(Y)

                                # Define Euclidean manifold
                                manifold = Euclidean(dim=Y.shape[1])

                                # Use arithmetic mean as base point
                                base_point = np.mean(Y, axis=0)

                                # Tangent PCA with correct constructor
                                tpca = TangentPCA(space=manifold, n_components=10)
                                tpca.fit(Y, base_point=base_point)
                                Y_tpca = tpca.transform(Y)

                                # KMeans clustering
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_tpca[init_indices]

                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_tpca)

                                # Save memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False) 

                    if dims == "3d":
                        if competitor == "scikit":
                            # Load and sort nodes (3D coordinates)
                            nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values  # (n_nodes, 3)
                            sorted_indices = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))  # sort by x, y, z
                            nodes_sorted = nodes[sorted_indices]

                            x_unique = np.unique(nodes_sorted[:, 0])
                            y_unique = np.unique(nodes_sorted[:, 1])
                            z_unique = np.unique(nodes_sorted[:, 2])
                            grid_points = [x_unique, y_unique, z_unique]
                            n_x, n_y, n_z = len(x_unique), len(y_unique), len(z_unique)

                            for i in range(N):
                                # Load function evaluations (n_samples x n_nodes)
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Sort node values to match x-y-z grid layout
                                Y_sorted = Y[:, sorted_indices]  # shape: (n_samples, n_nodes)

                                # Reshape to (n_samples, n_x, n_y, n_z)
                                data_matrix = Y_sorted.reshape((Y.shape[0], n_x, n_y, n_z))

                                # Build FDataGrid object
                                fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

                                # Initialize centroids by index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = fd[init_indices]

                                # KMeans clustering
                                model = KMeans(n_clusters=n_clust, init=init_centroids)
                                model.fit(fd)

                                # Save cluster memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                        if competitor == "scikit_pca":
                            # Sort 3D mesh nodes for consistent reshaping
                            nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
                            sorted_indices = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))
                            nodes_sorted = nodes[sorted_indices]

                            for i in range(N):
                                # Load function evaluations (each row = one function)
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Sort nodes consistently with mesh grid
                                Y_sorted = Y[:, sorted_indices]  # shape: (n_samples, n_nodes)

                                # Optional standardization
                                # Y_sorted = StandardScaler().fit_transform(Y_sorted)

                                # Apply PCA to reduce dimensionality (retain 95% variance)
                                pca = PCA(n_components=0.95)
                                Y_pca = pca.fit_transform(Y_sorted)

                                # Init centroids by index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_pca[init_indices]

                                # KMeans clustering
                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_pca)

                                # Save memberships
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                        if competitor == "geomstats":
                            for i in range(N):
                                # Load flattened functions
                                data_path = os.path.join(recon_data_dir, dims, type, curve, method, f"{method}_{i}.csv")
                                Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)
                                Y = np.nan_to_num(Y)

                                # Optional: Standardize the data
                                Y = StandardScaler().fit_transform(Y)

                                # Define Euclidean manifold for flattened functions
                                manifold = Euclidean(dim=Y.shape[1])

                                # Use Euclidean mean as base point
                                base_point = np.mean(Y, axis=0)

                                # Apply Tangent PCA
                                tpca = TangentPCA(space=manifold, n_components=10)
                                tpca.fit(Y, base_point=base_point)
                                Y_tpca = tpca.transform(Y)

                                # Init centroids by index
                                init_indices = [0, n_obs_per_clust, 2 * n_obs_per_clust]
                                init_centroids = Y_tpca[init_indices]

                                # KMeans clustering in tangent PCA space
                                model = SKLearnKMeans(n_clusters=n_clust, init=init_centroids, n_init=1, random_state=seed)
                                model.fit(Y_tpca)

                                # Save labels
                                memb_path = os.path.join(out_path, "memberships.csv")
                                row_df = pd.DataFrame([model.labels_])
                                row_df.to_csv(memb_path, mode='a', index=False, header=False)

                    print(f"{dims}/{method}/{type}/{competitor}/{curve} done")

    
