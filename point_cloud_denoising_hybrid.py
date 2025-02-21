import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Callable
import time
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

class StructuredDenoisingComparator:
    def __init__(self, synthetic_dir: str = "data/synthetic", real_dir: str = "data/visibility_dataset"):
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.methods = {}
        self.results = {}
        
        # Synthetic data parameters
        self.models = ['bunny', 'cube', 'dragon', 'sphere', 'vase']
        self.densities = ['D01', 'D02']
        self.noise_levels = ['L01', 'L02', 'L03', 'L04']
        
        # Real data parameters
        self.real_point_clouds = sorted(list((self.real_dir / "point_clouds").glob("*.xyz")))
        
        
    def add_method(self, name: str, method: Callable):
        """Add a denoising method to the comparator"""
        self.methods[name] = method
        
    def load_point_cloud(self, filepath: str) -> o3d.geometry.PointCloud:
        """Load point cloud from file"""
        return o3d.io.read_point_cloud(str(filepath))
    
    def evaluate(self, original_pcd: o3d.geometry.PointCloud, 
                denoised_pcd: o3d.geometry.PointCloud) -> dict:
        """Memory-efficient evaluation using point-by-point processing"""
        # Get points as numpy arrays
        original_points = np.asarray(original_pcd.points)
        denoised_points = np.asarray(denoised_pcd.points)
        
        # Create KD-trees for efficient neighbor search
        original_tree = o3d.geometry.KDTreeFlann(original_pcd)
        denoised_tree = o3d.geometry.KDTreeFlann(denoised_pcd)
        
        # Estimate normals
        original_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        denoised_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        original_normals = np.asarray(original_pcd.normals)
        denoised_normals = np.asarray(denoised_pcd.normals)
        
        # Initialize metrics
        mse_sum = 0.0
        max_error = 0.0
        normal_consistency_sum = 0.0
        chamfer_sum_1 = 0.0
        chamfer_sum_2 = 0.0
        
        # Process original to denoised distances
        print("Computing metrics (original to denoised)...")
        for i in range(len(original_points)):
            [k, idx, dist] = denoised_tree.search_knn_vector_3d(original_points[i], 1)
            min_dist = np.sqrt(dist[0])
            mse_sum += min_dist ** 2
            max_error = max(max_error, min_dist)
            
            # Normal consistency
            normal_sim = np.abs(np.dot(original_normals[i], denoised_normals[idx[0]]))
            normal_consistency_sum += normal_sim
            
            chamfer_sum_1 += min_dist
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(original_points)} original points")
        
        # Process denoised to original distances
        print("Computing metrics (denoised to original)...")
        for i in range(len(denoised_points)):
            [k, idx, dist] = original_tree.search_knn_vector_3d(denoised_points[i], 1)
            min_dist = np.sqrt(dist[0])
            chamfer_sum_2 += min_dist
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(denoised_points)} denoised points")
        
        # Calculate final metrics
        mse = mse_sum / len(original_points)
        mean_error = np.sqrt(mse)
        normal_consistency = normal_consistency_sum / len(original_points)
        chamfer_distance = (chamfer_sum_1 / len(original_points) + chamfer_sum_2 / len(denoised_points)) / 2
        density_ratio = len(denoised_points) / len(original_points)
        
        return {
            'mse': mse,
            'chamfer_distance': chamfer_distance,
            'max_error': max_error,
            'mean_error': mean_error,
            'normal_consistency': normal_consistency,
            'density_ratio': density_ratio
        }
        
    def _compute_chamfer_distance(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Compute Chamfer distance between two point clouds"""
        def nearest_neighbor(src, dst):
            distances = np.linalg.norm(src[:, np.newaxis] - dst, axis=2)
            return np.mean(np.min(distances, axis=1))
        
        dist1 = nearest_neighbor(points1, points2)
        dist2 = nearest_neighbor(points2, points1)
        return (dist1 + dist2) / 2
    
    def _process_methods(self, input_pcd, reference_pcd, results_data, output_dir,
                        prefix, dataset_type, noise_level, save_format='ply'):
        """Process point cloud with all methods"""
        for method_name, method in self.methods.items():
            print(f"Applying {method_name}...")
            
            try:
                start_time = time.time()
                denoised_pcd = method(input_pcd)
                processing_time = time.time() - start_time
                
                # Save denoised point cloud
                self.save_point_cloud(denoised_pcd, output_dir, prefix, 
                                    noise_level, method_name, save_format)
                
                # Evaluate results
                metrics = self.evaluate(reference_pcd, denoised_pcd)
                metrics['processing_time'] = processing_time
                
                # Store results
                result_entry = {
                    'dataset_type': dataset_type,
                    'model': prefix,
                    'noise_level': noise_level,
                    'method': method_name,
                    **metrics
                }
                results_data.append(result_entry)
                
            except Exception as e:
                print(f"Error in {method_name}: {str(e)}")
    
    def save_point_cloud(self, pcd, output_dir: Path, prefix: str, 
                        noise_level: str = None, method: str = None,
                        save_format: str = 'ply'):
        """Save point cloud to specified directory"""
        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if noise_level and method:
            filename = f"{prefix}_{noise_level}_{method}"
        elif noise_level:
            filename = f"{prefix}_{noise_level}_noisy"
        else:
            filename = f"{prefix}_original"
            
        if save_format == 'xyz' and hasattr(pcd, 'uvs'):
            # Save as XYZ with additional attributes
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else np.zeros((len(points), 3))
            uvs = pcd.uvs if hasattr(pcd, 'uvs') else np.zeros((len(points), 2))
            labels = pcd.labels if hasattr(pcd, 'labels') else np.zeros((len(points), 1))
            
            data = np.hstack((points, colors, uvs, labels))
            np.savetxt(output_dir / f"{filename}.xyz", data, delimiter=' ')
        else:
            # Save as PLY
            o3d.io.write_point_cloud(str(output_dir / f"{filename}.ply"), pcd)
        
    def process_dataset(self, include_synthetic=True, include_real=True) -> pd.DataFrame:
        """Process both synthetic and real datasets"""
        results_data = []
        output_dir = Path("results/point_clouds")
        
        # Process synthetic dataset
        if include_synthetic:
            for density in self.densities:
                density_dir = self.synthetic_dir / density
                
                for model in self.models:
                    print(f"\nProcessing synthetic model {model} with density {density}")
                    
                    # Load and save original model
                    original_path = density_dir / f"{model}.ply"
                    original_pcd = self.load_point_cloud(original_path)
                    self.save_point_cloud(original_pcd, output_dir / "synthetic" / density / model, model)
                    
                    # Process each noise level
                    for noise_level in self.noise_levels:
                        noisy_path = density_dir / f"{model}_{density}_{noise_level}.ply"
                        noisy_pcd = self.load_point_cloud(noisy_path)
                        
                        # Save noisy point cloud
                        self.save_point_cloud(noisy_pcd, output_dir / "synthetic" / density / model, 
                                           model, noise_level)
                        
                        print(f"Processing noise level {noise_level}")
                        self._process_methods(noisy_pcd, original_pcd, results_data, 
                                           output_dir / "synthetic" / density / model,
                                           model, density, noise_level)
        
        # Process real dataset
        if include_real:
            for idx, pc_path in enumerate(self.real_point_clouds):
                print(f"\nProcessing real point cloud {pc_path.name}")
                
                # Load point cloud
                original_pcd = self.load_point_cloud(pc_path)
                prefix = f"real_{idx+1}"
                
                # Save original
                self.save_point_cloud(original_pcd, output_dir / "real", prefix, 
                                    save_format='xyz')
                
                # Process with different methods
                self._process_methods(original_pcd, original_pcd, results_data,
                                   output_dir / "real", prefix, "real", "original",
                                   save_format='xyz')
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results_data)
        return self.results_df
    
    # def visualize_results(self, metric: str = 'mse', plot_type: str = 'boxplot'):
    #     """Visualize results for specific metric"""
    #     plt.figure(figsize=(15, 8))
        
    #     if plot_type == 'boxplot':
    #         sns.boxplot(data=self.results_df, x='method', y=metric, hue='noise_level')
    #     elif plot_type == 'heatmap':
    #         pivot_table = self.results_df.pivot_table(
    #             values=metric,
    #             index=['model', 'density'],
    #             columns=['method', 'noise_level'],
    #             aggfunc='mean'
    #         )
    #         sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
        
    #     plt.title(f'{metric} Comparison Across Methods and Noise Levels')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
    

def bilateral_filter(pcd, batch_size=15000):
    """Memory-efficient bilateral filtering without large array operations"""
    points = np.asarray(pcd.points)
    
    # Estimate normals (create new point cloud to avoid modifying input)
    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(points)
    input_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(input_pcd.normals)
    
    # Build KD-tree
    pcd_tree = o3d.geometry.KDTreeFlann(input_pcd)
    
    # Estimate average spacing from subset of points
    sample_indices = np.random.choice(len(points), min(1000, len(points)), replace=False)
    distances = []
    for i in sample_indices:
        [_, _, dist] = pcd_tree.search_knn_vector_3d(points[i], 2)
        distances.append(np.sqrt(dist[1]))
    
    avg_spacing = np.mean(distances)
    search_radius = avg_spacing * 3
    sigma_s = avg_spacing * 2
    sigma_n = 0.5
    
    new_points = []
    total_points = len(points)
    
    for i in range(total_points):
        # Find neighbors
        [k, idx, dist] = pcd_tree.search_radius_vector_3d(points[i], search_radius)
        
        if k < 3:
            new_points.append(points[i])
            continue
        
        # Process each point individually
        current_point = points[i]
        current_normal = normals[i]
        
        # Get neighbors
        neighbor_points = points[idx]
        neighbor_normals = normals[idx]
        
        # Calculate weights one by one
        total_weight = 0.0
        weighted_sum = np.zeros(3)
        
        for j in range(k):
            # Spatial weight
            spatial_dist = np.sum((neighbor_points[j] - current_point) ** 2)
            spatial_weight = np.exp(-spatial_dist / (2 * sigma_s ** 2))
            
            # Normal weight
            normal_sim = np.abs(np.dot(neighbor_normals[j], current_normal))
            normal_weight = np.exp(-(1 - normal_sim) / (2 * sigma_n ** 2))
            
            # Combined weight
            weight = spatial_weight * normal_weight
            total_weight += weight
            weighted_sum += weight * neighbor_points[j]
        
        # Calculate new position
        if total_weight > 0:
            new_pos = weighted_sum / total_weight
        else:
            new_pos = current_point
            
        new_points.append(new_pos)
        
        # Progress tracking
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total_points} points")
    
    # Create output point cloud
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(np.array(new_points))
    return result_pcd

def guided_filter(pcd, radius=0.01, epsilon=0.1, iterations=2):
    """Guided filter denoising"""
    # Create new point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    
    for iteration in range(iterations):
        # Build KD-tree
        pcd_tree = o3d.geometry.KDTreeFlann(filtered_pcd)
        points = np.asarray(filtered_pcd.points)
        points_copy = np.copy(points)
        num_points = len(points)
        
        for i in range(num_points):
            # Find neighbors within radius
            [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
            
            if k < 3:
                continue
            
            # Get neighbors and compute statistics
            neighbors = points[idx]
            mean = np.mean(neighbors, axis=0)
            
            # Compute covariance matrix
            centered = neighbors - mean
            cov = (centered.T @ centered) / (k - 1)
            
            try:
                # Compute guided filter matrices
                e = np.linalg.inv(cov + epsilon * np.eye(3))
                A = cov @ e
                b = mean - A @ mean
                
                # Update point position
                points_copy[i] = A @ points[i] + b
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                points_copy[i] = mean
            
            # Progress tracking
            if (i + 1) % 1000 == 0:
                print(f"Iteration {iteration + 1}/{iterations}, "
                      f"Processed {i + 1}/{num_points} points")
        
        # Update points for next iteration
        filtered_pcd.points = o3d.utility.Vector3dVector(points_copy)
    
    return filtered_pcd

def compute_weighted_covariance_matrix(points: np.ndarray, center_point: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted covariance matrix for local plane estimation.
    
    Args:
        points: (N, 3) array of point coordinates
        center_point: (3,) array of the center point
        sigma: gaussian weight parameter
    
    Returns:
        covariance_matrix: (3, 3) weighted covariance matrix
        centroid: (4,) weighted centroid
    """
    # Calculate weights using Gaussian kernel
    diff = points - center_point
    squared_dist = np.sum(diff * diff, axis=1)
    weights = np.exp(-squared_dist / (2 * sigma * sigma))
    weights = weights / np.sum(weights)
    
    # Compute weighted centroid
    weighted_points = points * weights[:, np.newaxis]
    centroid = np.mean(weighted_points, axis=0)
    centroid_4d = np.append(centroid, 1)  # Homogeneous coordinates
    
    # Compute weighted covariance matrix
    centered_points = points - centroid
    covariance_matrix = np.zeros((3, 3))
    
    for i in range(len(points)):
        point = centered_points[i:i+1].T
        covariance_matrix += weights[i] * (point @ point.T)
        
    return covariance_matrix, centroid_4d

def calculate_local_planes(pcd: o3d.geometry.PointCloud, k: int = 30, sigma: float = 0.1) -> np.ndarray:
    """
    Calculate local planes for each point in the point cloud.
    
    Args:
        pcd: Open3D point cloud
        k: number of nearest neighbors
        sigma: gaussian weight parameter
    
    Returns:
        plane_params: (N, 4) array containing plane parameters (normal_x, normal_y, normal_z, d)
    """
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    n_points = len(points)
    plane_params = np.zeros((n_points, 4))
    
    print("Calculating local planes...")
    for idx in range(n_points):
        # Find k nearest neighbors
        [k_found, indices, _] = pcd_tree.search_knn_vector_3d(points[idx], k)
        if k_found < 3:
            continue
            
        neighbor_points = points[indices]
        
        # Compute weighted covariance matrix
        cov_matrix, centroid = compute_weighted_covariance_matrix(
            neighbor_points, points[idx], sigma
        )
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Normal is eigenvector corresponding to smallest eigenvalue
        normal = eigenvectors[:, 0]
        d = np.dot(normal, centroid[:3])
        
        plane_params[idx] = np.array([normal[0], normal[1], normal[2], d])
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{n_points} points")
            
    return plane_params

def get_projected_point(point: np.ndarray, plane_params: np.ndarray) -> np.ndarray:
    """
    Project a point onto a plane defined by its parameters.
    
    Args:
        point: (3,) array of point coordinates
        plane_params: (4,) array of plane parameters (normal_x, normal_y, normal_z, d)
    
    Returns:
        projected_point: (3,) array of projected point coordinates
    """
    normal = plane_params[:3]
    d = plane_params[3]
    
    # Project point onto plane
    proj_dist = np.dot(normal, point) - d
    projected_point = point - proj_dist * normal
    
    return projected_point

def weighted_multi_projection(pcd: o3d.geometry.PointCloud, k: int = 30, sigma: float = 0.1) -> o3d.geometry.PointCloud:
    """
    Denoise point cloud using Weighted Multi-projection method.
    
    Args:
        pcd: Open3D point cloud
        k: number of nearest neighbors
        sigma: gaussian weight parameter
    
    Returns:
        denoised_pcd: Denoised point cloud
    """
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    n_points = len(points)
    
    # Calculate local planes for all points
    plane_params = calculate_local_planes(pcd, k, sigma)
    
    # Initialize array for denoised points
    denoised_points = np.zeros_like(points)
    
    print("\nApplying weighted multi-projection...")
    for idx in range(n_points):
        # Find k nearest neighbors
        [k_found, indices, distances] = pcd_tree.search_knn_vector_3d(points[idx], k)
        if k_found < 3:
            denoised_points[idx] = points[idx]
            continue
            
        # Calculate spatial weights
        weights = np.exp(-np.array(distances) / (2 * sigma * sigma))
        weights = weights / np.sum(weights)
        
        # Project point onto each neighboring plane and combine
        projected_point = np.zeros(3)
        for i, neighbor_idx in enumerate(indices):
            proj = get_projected_point(points[idx], plane_params[neighbor_idx])
            projected_point += weights[i] * proj
            
        denoised_points[idx] = projected_point
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{n_points} points")
    
    # Create new point cloud with denoised points
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(denoised_points)
    
    return denoised_pcd





class Edge:
    def __init__(self, src: int, dst: int, weight: float = 1.0):
        self.src = src
        self.dst = dst
        self.weight = weight

class Graph:
    def __init__(self, num_vertices: int):
        self.num_vertices = num_vertices
        self.adj_matrix = np.zeros((num_vertices, num_vertices))
        self.edges = []
        
    def add_edge(self, edge: Edge):
        self.adj_matrix[edge.src, edge.dst] = edge.weight
        self.adj_matrix[edge.dst, edge.src] = edge.weight  # Undirected graph
        self.edges.append(edge)
        
    def get_sparse_matrix(self):
        return csr_matrix(self.adj_matrix)

def derive_minimum_spanning_tree(points: np.ndarray, k: int = 5) -> Tuple[Graph, List[Edge]]:
    """
    Compute minimum spanning tree from point cloud.
    
    Args:
        points: (N, 3) array of point coordinates
        k: Number of nearest neighbors
        
    Returns:
        graph: Graph object
        mst_edges: List of edges in the minimum spanning tree
    """
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create graph
    n_points = len(points)
    graph = Graph(n_points)
    
    # Add edges
    for i in range(n_points):
        for j in range(1, k):  # Skip first neighbor (self)
            if distances[i, j] > 0:
                edge = Edge(i, indices[i, j], distances[i, j])
                graph.add_edge(edge)
    
    # Compute minimum spanning tree
    sparse_matrix = graph.get_sparse_matrix()
    mst_matrix = minimum_spanning_tree(sparse_matrix)
    mst_matrix = mst_matrix.toarray()
    
    # Extract MST edges
    mst_edges = []
    rows, cols = np.nonzero(mst_matrix)
    for i, j in zip(rows, cols):
        if i < j:  # Only add each edge once
            mst_edges.append(Edge(i, j, mst_matrix[i, j]))
    
    return graph, mst_edges

def bipartite_separation(points: np.ndarray, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate points into two sets using bipartite graph approximation.
    
    Args:
        points: (N, 3) array of point coordinates
        k: Number of nearest neighbors
        
    Returns:
        red_indices: Indices of points in first set
        blue_indices: Indices of points in second set
    """
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create graph
    n_points = len(points)
    graph = Graph(n_points)
    
    # Add edges
    for i in range(n_points):
        for j in range(k):
            edge = Edge(i, indices[i, j], distances[i, j])
            graph.add_edge(edge)
    
    # Compute shortest paths from vertex 0
    sparse_matrix = graph.get_sparse_matrix()
    distances_matrix = dijkstra(sparse_matrix, indices=0, directed=False)
    
    # Sort vertices by distance
    sorted_indices = np.argsort(distances_matrix)
    
    # Separate into red and blue sets
    red_indices = sorted_indices[::2]  # Even indices
    blue_indices = sorted_indices[1::2]  # Odd indices
    
    return red_indices, blue_indices

def total_variation_step(points: np.ndarray, red_indices: np.ndarray, 
                        blue_indices: np.ndarray, lambda_param: float = 0.1) -> np.ndarray:
    """
    Apply total variation regularization step.
    
    Args:
        points: (N, 3) array of point coordinates
        red_indices: Indices of red set points
        blue_indices: Indices of blue set points
        lambda_param: Regularization parameter
        
    Returns:
        new_points: Updated point coordinates
    """
    new_points = points.copy()
    
    # Find nearest neighbors between sets
    red_points = points[red_indices]
    blue_points = points[blue_indices]
    
    nbrs_red = NearestNeighbors(n_neighbors=1).fit(red_points)
    nbrs_blue = NearestNeighbors(n_neighbors=1).fit(blue_points)
    
    # Update red points based on blue neighbors
    _, indices = nbrs_blue.kneighbors(red_points)
    blue_neighbors = blue_points[indices.flatten()]
    new_points[red_indices] = (1 - lambda_param) * red_points + lambda_param * blue_neighbors
    
    # Update blue points based on red neighbors
    _, indices = nbrs_red.kneighbors(blue_points)
    red_neighbors = red_points[indices.flatten()]
    new_points[blue_indices] = (1 - lambda_param) * blue_points + lambda_param * red_neighbors
    
    return new_points

def bipartite_graph_denoising(pcd: o3d.geometry.PointCloud, 
                            num_iterations: int = 10,
                            k: int = 30,
                            lambda_param: float = 0.1) -> o3d.geometry.PointCloud:
    """
    Denoise point cloud using bipartite graph approximation and total variation.
    
    Args:
        pcd: Input point cloud
        num_iterations: Number of denoising iterations
        k: Number of nearest neighbors for graph construction
        lambda_param: Total variation regularization parameter
        
    Returns:
        denoised_pcd: Denoised point cloud
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)
    
    # Compute MST and initial graph structure
    graph, mst_edges = derive_minimum_spanning_tree(points)
    
    # Iterative denoising
    current_points = points.copy()
    for iter in range(num_iterations):
        # Separate points into bipartite sets
        red_indices, blue_indices = bipartite_separation(current_points, k)
        
        # Apply total variation regularization
        current_points = total_variation_step(
            current_points, red_indices, blue_indices, lambda_param
        )
        
        print(f"Completed iteration {iter + 1}/{num_iterations}")
    
    # Create output point cloud
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(current_points)
    
    return denoised_pcd

def optimized_bipartite_separation(points: np.ndarray, k: int = 30, cached_neighbors=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized version of bipartite separation using cached neighbors.
    """
    n_points = len(points)
    
    if cached_neighbors is None:
        # Only compute nearest neighbors if not provided
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=40)
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
    else:
        distances, indices = cached_neighbors
    
    # Use sparse matrix for efficient graph representation
    rows = np.repeat(np.arange(n_points), k)
    cols = indices.ravel()
    data = distances.ravel()
    
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    # Compute shortest paths from vertex 0 using sparse matrix
    distances_matrix = dijkstra(sparse_matrix, indices=0, directed=False)
    
    # Efficient array operations for separation
    sorted_indices = np.argsort(distances_matrix)
    red_indices = sorted_indices[::2]
    blue_indices = sorted_indices[1::2]
    
    return red_indices, blue_indices

def optimized_total_variation_step(points: np.ndarray, red_indices: np.ndarray, 
                                 blue_indices: np.ndarray, lambda_param: float = 0.1,
                                 cached_tree_red=None, cached_tree_blue=None) -> np.ndarray:
    """
    Optimized total variation step using cached KD-trees.
    """
    new_points = points.copy()
    red_points = points[red_indices]
    blue_points = points[blue_indices]
    
    if cached_tree_red is None or cached_tree_blue is None:
        # Build trees only if not provided
        nbrs_red = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=40).fit(red_points)
        nbrs_blue = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=40).fit(blue_points)
    else:
        nbrs_red, nbrs_blue = cached_tree_red, cached_tree_blue
    
    # Vectorized operations for point updates
    _, blue_indices_local = nbrs_blue.kneighbors(red_points)
    blue_neighbors = blue_points[blue_indices_local.flatten()]
    new_points[red_indices] = (1 - lambda_param) * red_points + lambda_param * blue_neighbors
    
    _, red_indices_local = nbrs_red.kneighbors(blue_points)
    red_neighbors = red_points[red_indices_local.flatten()]
    new_points[blue_indices] = (1 - lambda_param) * blue_points + lambda_param * red_neighbors
    
    return new_points

def optimized_bipartite_graph_denoising(pcd: o3d.geometry.PointCloud, 
                                      num_iterations: int = 10,
                                      k: int = 30,
                                      lambda_param: float = 0.1) -> o3d.geometry.PointCloud:
    """
    Optimized version of bipartite graph denoising with performance improvements.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    current_points = points.copy()
    
    # Pre-compute nearest neighbors for initial graph construction
    print("Pre-computing nearest neighbors...")
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=40)
    nbrs.fit(points)
    cached_neighbors = nbrs.kneighbors(points)
    print(f"Nearest neighbor computation took {time.time() - start_time:.2f} seconds")
    
    # Initialize progress tracking
    total_operations = num_iterations
    start_time = time.time()
    last_update_time = start_time
    
    for iter in range(num_iterations):
        iter_start = time.time()
        
        # Use cached neighbors for bipartite separation
        red_indices, blue_indices = optimized_bipartite_separation(
            current_points, k, cached_neighbors if iter == 0 else None
        )
        
        # Pre-compute KD-trees for red and blue sets
        red_points = current_points[red_indices]
        blue_points = current_points[blue_indices]
        nbrs_red = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=40).fit(red_points)
        nbrs_blue = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=40).fit(blue_points)
        
        # Apply optimized total variation step
        current_points = optimized_total_variation_step(
            current_points, red_indices, blue_indices, lambda_param,
            nbrs_red, nbrs_blue
        )
        
        # Progress and timing updates
        current_time = time.time()
        iter_time = current_time - iter_start
        elapsed_time = current_time - start_time
        
        if current_time - last_update_time >= 1.0:  # Update every second
            progress = (iter + 1) / total_operations * 100
            print(f"Progress: {progress:.1f}% | Iteration {iter + 1}/{num_iterations} "
                  f"| Iteration time: {iter_time:.2f}s | Total time: {elapsed_time:.2f}s")
            last_update_time = current_time
            
        # Early stopping condition based on convergence
        if iter > 0:
            change = np.mean(np.abs(current_points - points))
            if change < 1e-6:
                print(f"Converged after {iter + 1} iterations")
                break
    
    # Create output point cloud
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(current_points)
    
    return denoised_pcd

def main():
    # Initialize comparator
    comparator = StructuredDenoisingComparator()
    
    # Add methods to comparator
    comparator.add_method('bipartite_graph_denoising', optimized_bipartite_graph_denoising)
    comparator.add_method('Bilateral', bilateral_filter)
    comparator.add_method('Guided Filter', guided_filter)
    comparator.add_method('Weighted multi projection', guided_filter)
    
    # Process dataset and generate results
    results_df = comparator.process_dataset(include_synthetic=True, include_real=True)
    
    # Save results and visualizations
    # comparator.save_results('denoising_results')

if __name__ == "__main__":
    main()