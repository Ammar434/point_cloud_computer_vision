import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import open3d as o3d
from typing import Dict, List, Tuple
import json

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import tempfile
import shutil

class VideoGenerator:
    def __init__(self, output_dir: str = "denoising_results/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def add_gaussian_noise(self, pcd: o3d.geometry.PointCloud, mean: float = 0, std: float = 0.05) -> o3d.geometry.PointCloud:
        """Add Gaussian noise to point cloud"""
        points = np.asarray(pcd.points)
        noise = np.random.normal(mean, std, size=points.shape)
        points_noisy = points + noise
        
        # Create new point cloud with noisy points
        pcd_noisy = o3d.geometry.PointCloud()
        pcd_noisy.points = o3d.utility.Vector3dVector(points_noisy)
        
        # Copy colors if they exist
        if pcd.has_colors():
            pcd_noisy.colors = pcd.colors
            
        return pcd_noisy
        
    def __del__(self):
        """Cleanup temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def setup_camera(self, pcd: o3d.geometry.PointCloud):
        """Setup initial camera parameters"""
        center = pcd.get_center()
        bbox = pcd.get_axis_aligned_bounding_box()
        scale = np.linalg.norm(bbox.get_extent())
        
        return {
            'center': center,
            'scale': scale,
            'up': [0, 1, 0],
            'zoom': 0.7,
            'distance_factor': 1.5
        }
    
    def capture_rotating_frames(self, pcd: o3d.geometry.PointCloud, 
                              n_frames: int = 120,
                              window_size=(1920, 1080)) -> List[str]:
        """Capture frames while rotating the point cloud"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=window_size[0], height=window_size[1], visible=False)
        
        # Add default colors to point cloud if it doesn't have any
        if not pcd.has_colors():
            points = np.asarray(pcd.points)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = 0.2  # R value
            colors[:, 1] = 0.2  # G value
            colors[:, 2] = 0.8  # B value (predominantly blue)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
        vis.add_geometry(pcd)
        
        # Setup render options
        opt = vis.get_render_option()
        opt.point_size = 3.0  # Increased point size
        opt.background_color = np.array([1, 1, 1])  # White background
        
        # Setup camera
        cam_params = self.setup_camera(pcd)
        ctr = vis.get_view_control()
        ctr.set_up(cam_params['up'])
        ctr.set_zoom(cam_params['zoom'])
        
        frame_paths = []
        for i in range(n_frames):
            # Calculate rotation angle
            theta = i * (360.0 / n_frames)
            
            # Update camera position
            radius = cam_params['scale'] * cam_params['distance_factor']
            x = radius * np.cos(np.radians(theta))
            z = radius * np.sin(np.radians(theta))
            front = [-x, 0, -z]
            
            ctr.set_front(front)
            ctr.set_lookat(cam_params['center'])
            
            # Render and capture frame
            vis.poll_events()
            vis.update_renderer()
            
            frame_path = self.temp_dir / f"frame_{i:04d}.png"
            vis.capture_screen_image(str(frame_path), do_render=True)
            frame_paths.append(str(frame_path))
        
        vis.destroy_window()
        return frame_paths
    
    def create_video(self, frame_paths: List[str], output_path: str, fps: int = 30):
        """Create video from frames"""
        # Read first frame to get dimensions
        frame = cv2.imread(frame_paths[0])
        height, width = frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        video.release()
    
    def generate_360_video(self, pcd: o3d.geometry.PointCloud, 
                          output_path: str,
                          n_frames: int = 120,
                          fps: int = 30,
                          add_noise: bool = True):
        """Generate complete 360° rotation video"""
        # Add Gaussian noise if requested
        if add_noise and "original" in output_path:
            pcd = self.add_gaussian_noise(pcd, mean=0, std=0.05)
            
        # Capture rotating frames
        frame_paths = self.capture_rotating_frames(pcd, n_frames)
        
        # Create video
        self.create_video(frame_paths, output_path, fps)
        
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import open3d as o3d
from typing import Dict, List, Tuple
import cv2
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import open3d as o3d
from typing import Dict, List, Tuple
import cv2
import tempfile
import shutil

class DenoisingAnalyzer:
    def __init__(self, results_dir: str = "results/point_clouds"):
        """Initialize the analyzer with paths to results"""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        self.vis_dir = self.output_dir / "visualizations"
        self.video_dir = self.output_dir / "videos"
        self.stats_dir = self.output_dir / "statistics"
        self.latex_dir = self.output_dir / "latex"
        
        for dir_path in [self.vis_dir, self.video_dir, self.stats_dir, self.latex_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Parameters
        self.models = ['bunny', 'cube', 'dragon', 'sphere', 'vase']
        self.densities = ['D01', 'D02']
        self.noise_levels = ['L01', 'L02', 'L03', 'L04']
        self.methods = ['Bilateral', 'Guided Filter', 'bipartite_graph_denoising', 'Weighted multi projection']
        
        self.results_df = None
    def generate_videos(self):
        """Generate 360° videos for all models"""
        # Initialize video generator
        video_generator = VideoGenerator(str(self.video_dir))
        
        # Process synthetic dataset
        synthetic_dir = self.results_dir / "synthetic"
        if synthetic_dir.exists():
            for density in self.densities:
                density_dir = synthetic_dir / density
                
                for model in self.models:
                    model_dir = density_dir / model
                    if not model_dir.exists():
                        continue
                    
                    print(f"\nProcessing videos for {model} with density {density}")
                    output_dir = self.video_dir / density / model
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate video for original model
                    original_path = model_dir / f"{model}_original.ply"
                    if original_path.exists():
                        original_pcd = o3d.io.read_point_cloud(str(original_path))
                        video_generator.generate_360_video(
                            original_pcd,
                            str(output_dir / f"{model}_original_360.mp4"))
                        print(f"Generated video for original {model}")
                    
                    # Process each noise level
                    for noise_level in self.noise_levels:
                        # Generate video for noisy model
                        noisy_path = model_dir / f"{model}_{noise_level}_noisy.ply"
                        if noisy_path.exists():
                            noisy_pcd = o3d.io.read_point_cloud(str(noisy_path))
                            video_generator.generate_360_video(
                                noisy_pcd,
                                str(output_dir / f"{model}_{noise_level}_noisy_360.mp4"))
                            print(f"Generated video for {noise_level} noisy {model}")
                        
                        # Generate videos for denoised results
                        for method in self.methods:
                            method_path = model_dir / f"{model}_{noise_level}_{method}.ply"
                            if method_path.exists():
                                denoised_pcd = o3d.io.read_point_cloud(str(method_path))
                                video_generator.generate_360_video(
                                    denoised_pcd,
                                    str(output_dir / f"{model}_{noise_level}_{method}_360.mp4"))
                                print(f"Generated video for {noise_level} {method} {model}")
        
        # Process real dataset
        real_dir = self.results_dir / "real"
        if real_dir.exists():
            output_dir = self.video_dir / "real"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each real point cloud
            for real_idx in range(1, 4):
                model_name = f"real_{real_idx}"
                print(f"\nProcessing videos for {model_name}")
                
                # Generate video for original point cloud
                original_path = real_dir / f"{model_name}_original.ply"
                if original_path.exists():
                    original_pcd = o3d.io.read_point_cloud(str(original_path))
                    video_generator.generate_360_video(
                        original_pcd,
                        str(output_dir / f"{model_name}_original_360.mp4"))
                    print(f"Generated video for original {model_name}")
                
                # Generate videos for each denoising method
                for method in self.methods:
                    method_path = real_dir / f"{model_name}_original_{method}.ply"
                    if method_path.exists():
                        denoised_pcd = o3d.io.read_point_cloud(str(method_path))
                        video_generator.generate_360_video(
                            denoised_pcd,
                            str(output_dir / f"{model_name}_{method}_360.mp4"))
                        print(f"Generated video for {method} {model_name}")

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("Starting analysis...")
        
        # # Compute metrics and generate statistics
        # print("Computing metrics and generating statistics...")
        # self.analyze_dataset()
        
        # # Generate method comparisons
        # print("Generating method comparisons...")
        # self.generate_method_comparison()
        
        # # Generate LaTeX tables
        # print("Generating LaTeX tables...")
        # self.generate_latex_tables()
        
        # Generate 360° videos
        print("Generating 360° videos...")
        self.generate_videos()
        
        # print("\nAnalysis complete. Results saved in:", self.output_dir)
        # print("- CSV files with detailed statistics in:", self.stats_dir)
        # print("- LaTeX tables for the paper in:", self.latex_dir)
        # print("- 360° videos in:", self.video_dir)
    
    def get_method_file(self, base_dir: Path, model: str, noise_level: str, method: str) -> Path:
        """Get the path to a method's result file"""
        if noise_level:
            return base_dir / f"{model}_{noise_level}_{method}.ply"
        return base_dir / f"{model}_original_{method}.ply"
    
    def analyze_synthetic_data(self) -> List[Dict]:
        """Analyze synthetic dataset results"""
        results = []
        
        for density in self.densities:
            density_dir = self.results_dir / "synthetic" / density
            
            for model in self.models:
                model_dir = density_dir / model
                if not model_dir.exists():
                    continue
                
                # Load original (ground truth) model
                original_path = model_dir / f"{model}_original.ply"
                if not original_path.exists():
                    continue
                original_pcd = o3d.io.read_point_cloud(str(original_path))
                
                for noise_level in self.noise_levels:
                    # Load noisy model for reference
                    noisy_path = model_dir / f"{model}_{noise_level}_noisy.ply"
                    if not noisy_path.exists():
                        continue
                    noisy_pcd = o3d.io.read_point_cloud(str(noisy_path))
                    
                    # Compute noise baseline metrics
                    noise_metrics = self.compute_metrics(original_pcd, noisy_pcd)
                    results.append({
                        'dataset_type': 'synthetic',
                        'density': density,
                        'model': model,
                        'noise_level': noise_level,
                        'method': 'Noisy',
                        **noise_metrics
                    })
                    
                    # Analyze each denoising method
                    for method in self.methods:
                        method_path = self.get_method_file(model_dir, model, noise_level, method)
                        if not method_path.exists():
                            continue
                        
                        denoised_pcd = o3d.io.read_point_cloud(str(method_path))
                        metrics = self.compute_metrics(original_pcd, denoised_pcd)
                        
                        results.append({
                            'dataset_type': 'synthetic',
                            'density': density,
                            'model': model,
                            'noise_level': noise_level,
                            'method': method,
                            **metrics
                        })
        
        return results
    
    def analyze_real_data(self) -> List[Dict]:
        """Analyze real dataset results"""
        results = []
        real_dir = self.results_dir / "real"
        
        for real_idx in range(1, 4):  # real_1, real_2, real_3
            model_name = f"real_{real_idx}"
            
            # Load original point cloud
            original_path = real_dir / f"{model_name}_original.ply"
            if not original_path.exists():
                continue
                
            original_pcd = o3d.io.read_point_cloud(str(original_path))
            
            # Analyze each denoising method
            for method in self.methods:
                method_path = real_dir / f"{model_name}_original_{method}.ply"
                if not method_path.exists():
                    continue
                
                denoised_pcd = o3d.io.read_point_cloud(str(method_path))
                metrics = self.compute_metrics(original_pcd, denoised_pcd)
                
                results.append({
                    'dataset_type': 'real',
                    'model': model_name,
                    'method': method,
                    **metrics
                })
        
        return results
    
    def analyze_dataset(self):
        """Analyze complete dataset"""
        all_results = []
        
        # Process synthetic data
        synthetic_results = self.analyze_synthetic_data()
        all_results.extend(synthetic_results)
        
        # Process real data
        real_results = self.analyze_real_data()
        all_results.extend(real_results)
        
        # Create DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        # Save complete results
        self.results_df.to_csv(self.stats_dir / 'complete_results.csv', index=False)
        
        return self.results_df
    
    def compute_metrics(self, original_pcd: o3d.geometry.PointCloud, 
                       denoised_pcd: o3d.geometry.PointCloud) -> Dict:
        """Compute comparison metrics between point clouds"""
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
        print("Computing metrics...")
        for i in range(len(original_points)):
            [k, idx, dist] = denoised_tree.search_knn_vector_3d(original_points[i], 1)
            min_dist = np.sqrt(dist[0])
            mse_sum += min_dist ** 2
            max_error = max(max_error, min_dist)
            
            # Normal consistency
            normal_sim = np.abs(np.dot(original_normals[i], denoised_normals[idx[0]]))
            normal_consistency_sum += normal_sim
            chamfer_sum_1 += min_dist
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(original_points)} original points")
        
        # Process denoised to original distances
        for i in range(len(denoised_points)):
            [k, idx, dist] = original_tree.search_knn_vector_3d(denoised_points[i], 1)
            min_dist = np.sqrt(dist[0])
            chamfer_sum_2 += min_dist
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(denoised_points)} denoised points")
        
        # Calculate final metrics
        mse = mse_sum / len(original_points)
        mean_error = np.sqrt(mse)
        normal_consistency = normal_consistency_sum / len(original_points)
        chamfer_distance = (chamfer_sum_1 / len(original_points) + 
                          chamfer_sum_2 / len(denoised_points)) / 2
        density_ratio = len(denoised_points) / len(original_points)
        
        return {
            'mse': mse,
            'chamfer_distance': chamfer_distance,
            'max_error': max_error,
            'mean_error': mean_error,
            'normal_consistency': normal_consistency,
            'density_ratio': density_ratio
        }
    
    def generate_method_comparison(self):
        """Generate method comparison tables and plots"""
        if self.results_df is None:
            self.analyze_dataset()
        
        # Synthetic data analysis by noise level
        synthetic_noise = self.results_df[
            self.results_df['dataset_type'] == 'synthetic'
        ].pivot_table(
            values=['mse', 'chamfer_distance', 'normal_consistency'],
            index=['method', 'noise_level'],
            aggfunc=['mean', 'std']
        ).round(4)
        
        # Synthetic data analysis by model type
        synthetic_model = self.results_df[
            self.results_df['dataset_type'] == 'synthetic'
        ].pivot_table(
            values=['mse', 'chamfer_distance', 'normal_consistency'],
            index=['method', 'model'],
            aggfunc=['mean', 'std']
        ).round(4)
        
        # Real data analysis
        real_comparison = self.results_df[
            self.results_df['dataset_type'] == 'real'
        ].pivot_table(
            values=['mse', 'chamfer_distance', 'normal_consistency'],
            index=['method', 'model'],
            aggfunc=['mean', 'std']
        ).round(4)
        
        # Save tables
        synthetic_noise.to_csv(self.stats_dir / 'synthetic_noise_comparison.csv')
        synthetic_model.to_csv(self.stats_dir / 'synthetic_model_comparison.csv')
        real_comparison.to_csv(self.stats_dir / 'real_comparison.csv')
        
        # Generate plots
        self.plot_comparisons()
        
        return synthetic_noise, synthetic_model, real_comparison
    
    def plot_comparisons(self):
        """Generate comparison plots"""
        metrics = ['mse', 'chamfer_distance', 'normal_consistency']
        
        # Plot synthetic data comparisons
        synthetic_data = self.results_df[self.results_df['dataset_type'] == 'synthetic']
        
        for metric in metrics:
            # Noise level comparison
            plt.figure(figsize=(15, 6))
            sns.boxplot(data=synthetic_data, x='method', y=metric, hue='noise_level')
            plt.title(f'{metric.upper()} Comparison by Noise Level')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'{metric}_noise_comparison.png')
            plt.close()
            
            # Model comparison
            plt.figure(figsize=(15, 6))
            sns.boxplot(data=synthetic_data, x='method', y=metric, hue='model')
            plt.title(f'{metric.upper()} Comparison by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'{metric}_model_comparison.png')
            plt.close()
        
        # Plot real data comparisons
        real_data = self.results_df[self.results_df['dataset_type'] == 'real']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=real_data, x='method', y=metric)
            plt.title(f'{metric.upper()} Comparison (Real Data)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'{metric}_real_comparison.png')
            plt.close()
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for the paper"""
        if self.results_df is None:
            self.analyze_dataset()
        
        tables = {}
        
        # Overall performance table
        tables['overall_performance'] = self.results_df.pivot_table(
            values=['mse', 'chamfer_distance', 'normal_consistency'],
            index=['dataset_type', 'method'],
            aggfunc=['mean', 'std']
        ).round(4).to_latex()
        
        # Noise level analysis
        tables['noise_analysis'] = self.results_df[
            self.results_df['dataset_type'] == 'synthetic'
        ].pivot_table(
            values=['mse', 'chamfer_distance'],
            index=['method', 'noise_level'],
            aggfunc=['mean', 'std']
        ).round(4).to_latex()
        
        # Save tables
        for name, table in tables.items():
            with open(self.latex_dir / f'{name}.tex', 'w') as f:
                f.write(table)
        
        return tables
            
def main():
    analyzer = DenoisingAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main()