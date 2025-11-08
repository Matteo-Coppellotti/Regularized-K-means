# --- FILE: run_scikit_fda.py ---
# Python script to run scikit-fda clustering

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

output_dir = "output_competitors"
os.makedirs(output_dir, exist_ok=True)
for dims in ["1d", "2d", "3d"]:
    os.makedirs(os.path.join(output_dir, dims), exist_ok=True)


# 1D
for dims in ["1d"]:
    for curve in curve_types:
        # Create new output directory
        out_path = os.path.join(output_dir, dims, "smooth_scikit", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        for i in range(N):
            Y = pd.read_csv(os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv"), header=None).values
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

# 2D KMeans using FDataGrid
for dims in ["2d"]:
    for curve in curve_types:
        # Create output directory
        out_path = os.path.join(output_dir, dims, "smooth_scikit", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

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
            data_path = os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv")
            Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)

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

# 2D HOG + KMeans
for dims in ["2d"]:
    for curve in curve_types:
        out_path = os.path.join(output_dir, dims, "smooth_scikit_hog", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        # Determine grid shape (2D)
        nodes = pd.read_csv(os.path.join(data_dir, dims, "nodes.csv"), header=None).values
        h = np.unique(nodes[:, 0]).size
        w = np.unique(nodes[:, 1]).size

        for i in range(N):
            # Load function evaluations
            data_path = os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv")
            Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)

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

for dims in ["2d"]:
    for curve in curve_types:
        out_path = os.path.join(output_dir, dims, "smooth_geomstats", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        for i in range(N):
            # Load function evaluations
            data_path = os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv")
            Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)

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

# 3D KMeans using FDataGrid
for dims in ["3d"]:
    for curve in curve_types:
        # Create output directory
        out_path = os.path.join(output_dir, dims, "smooth_scikit", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

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
            data_path = os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv")
            Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)

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

for dims in ["3d"]:
    for curve in curve_types:
        out_path = os.path.join(output_dir, dims, "smooth_geomstats", curve)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        for i in range(N):
            # Load flattened functions
            data_path = os.path.join(data_dir, dims, "smooth", curve, f"{curve}_{i}.csv")
            Y = pd.read_csv(data_path, header=None).values  # shape: (n_samples, n_nodes)

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
