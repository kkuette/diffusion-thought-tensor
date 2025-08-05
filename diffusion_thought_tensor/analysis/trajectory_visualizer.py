"""
Phase 3: Thought Trajectory Visualization

Advanced visualization tools for understanding thought evolution patterns,
including 3D trajectories, temporal dynamics, and emergent structures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class ThoughtTrajectoryVisualizer:
    """
    Creates sophisticated visualizations of thought evolution trajectories
    during the diffusion denoising process.
    """
    
    def __init__(self, color_scheme: str = 'viridis'):
        self.color_scheme = color_scheme
        self.trajectory_data = []
        
    def visualize_3d_trajectory(self, 
                              thought_history: List[torch.Tensor],
                              timesteps: List[int],
                              title: str = "3D Thought Evolution Trajectory",
                              save_path: Optional[str] = None) -> None:
        """
        Create 3D visualization of thought evolution using PCA projection.
        
        Args:
            thought_history: List of thought tensors over time
            timesteps: Corresponding timesteps
            title: Plot title
            save_path: Optional path to save the plot
        """
        if len(thought_history) < 3:
            print("Need at least 3 snapshots for 3D trajectory")
            return
        
        # Flatten thought tensors for PCA
        flattened_thoughts = []
        for thought in thought_history:
            flat = thought.flatten().detach().cpu().numpy()
            flattened_thoughts.append(flat)
        
        flattened_thoughts = np.array(flattened_thoughts)
        
        # Apply PCA to reduce to 3D
        pca = PCA(n_components=3)
        trajectory_3d = pca.fit_transform(flattened_thoughts)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color trajectory by timestep (reverse order since we go from high to low)
        colors = plt.cm.get_cmap(self.color_scheme)(np.linspace(0, 1, len(timesteps)))
        
        # Plot trajectory line
        ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 
                'k-', alpha=0.6, linewidth=2, label='Trajectory')
        
        # Plot points colored by timestep
        scatter = ax.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 
                           c=timesteps, cmap=self.color_scheme, s=60, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2], 
                  c='red', s=200, marker='o', label='Start (Noise)', alpha=0.9)
        ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], 
                  c='green', s=200, marker='s', label='End (Clean)', alpha=0.9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Diffusion Timestep', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax.set_title(title)
        ax.legend()
        
        # Store PCA info
        total_variance = pca.explained_variance_ratio_.sum()
        print(f"3D projection captures {total_variance:.1%} of total variance")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory saved to {save_path}")
        else:
            plt.show()
    
    def visualize_thought_flow_field(self,
                                   thought_history: List[torch.Tensor],
                                   timesteps: List[int],
                                   grid_size: int = 20,
                                   save_path: Optional[str] = None) -> None:
        """
        Create flow field visualization showing thought evolution vectors.
        """
        if len(thought_history) < 2:
            print("Need at least 2 snapshots for flow field")
            return
        
        # Flatten and reduce dimensionality
        flattened_thoughts = []
        for thought in thought_history:
            flat = thought.flatten().detach().cpu().numpy()
            flattened_thoughts.append(flat)
        
        flattened_thoughts = np.array(flattened_thoughts)
        
        # Use PCA to project to 2D
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(flattened_thoughts)
        
        # Compute velocity vectors
        velocities = np.diff(trajectory_2d, axis=0)
        
        # Create flow field visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Trajectory with velocity vectors
        ax1.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'k-', alpha=0.6, linewidth=2)
        scatter1 = ax1.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                              c=timesteps, cmap=self.color_scheme, s=60, alpha=0.8)
        
        # Add velocity vectors
        for i in range(len(velocities)):
            if i % max(1, len(velocities)//10) == 0:  # Sample vectors for clarity
                ax1.arrow(trajectory_2d[i, 0], trajectory_2d[i, 1],
                         velocities[i, 0]*0.1, velocities[i, 1]*0.1,
                         head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('Thought Trajectory with Flow Vectors')
        ax1.grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Timestep')
        
        # Plot 2: Speed profile over time
        speeds = np.linalg.norm(velocities, axis=1)
        ax2.plot(timesteps[:-1], speeds, 'b-', linewidth=2, marker='o')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Evolution Speed')
        ax2.set_title('Thought Evolution Speed Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Flow field visualization saved to {save_path}")
        else:
            plt.show()
    
    def create_interactive_trajectory(self,
                                    thought_history: List[torch.Tensor],
                                    timesteps: List[int],
                                    coherence_scores: List[float],
                                    diversity_scores: List[float],
                                    save_path: Optional[str] = None) -> None:
        """
        Create interactive 3D trajectory visualization with Plotly.
        """
        if len(thought_history) < 3:
            print("Need at least 3 snapshots for interactive trajectory")
            return
        
        # Prepare data
        flattened_thoughts = []
        for thought in thought_history:
            flat = thought.flatten().detach().cpu().numpy()
            flattened_thoughts.append(flat)
        
        flattened_thoughts = np.array(flattened_thoughts)
        
        # Apply PCA
        pca = PCA(n_components=3)
        trajectory_3d = pca.fit_transform(flattened_thoughts)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory_3d[:, 0],
            y=trajectory_3d[:, 1],
            z=trajectory_3d[:, 2],
            mode='lines+markers',
            line=dict(color='rgba(0,0,0,0.6)', width=4),
            marker=dict(
                size=8,
                color=timesteps,
                colorscale=self.color_scheme,
                showscale=True,
                colorbar=dict(title="Timestep")
            ),
            text=[f'Step: {t}<br>Coherence: {c:.3f}<br>Diversity: {d:.3f}' 
                  for t, c, d in zip(timesteps, coherence_scores, diversity_scores)],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
            name='Trajectory'
        ))
        
        # Mark start and end
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[0, 0]],
            y=[trajectory_3d[0, 1]],
            z=[trajectory_3d[0, 2]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='circle'),
            name='Start (Noise)',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[-1, 0]],
            y=[trajectory_3d[-1, 1]],
            z=[trajectory_3d[-1, 2]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='square'),
            name='End (Clean)',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Thought Evolution Trajectory',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive trajectory saved to {save_path}")
        else:
            fig.show()
    
    def visualize_thought_manifold(self,
                                 thought_history: List[torch.Tensor],
                                 timesteps: List[int],
                                 method: str = 'tsne',
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize thought evolution on a 2D manifold using t-SNE or UMAP.
        
        Args:
            thought_history: List of thought tensors
            timesteps: Corresponding timesteps
            method: 'tsne' or 'pca' for dimensionality reduction
            save_path: Optional save path
        """
        # Flatten thoughts
        flattened_thoughts = []
        for thought in thought_history:
            flat = thought.flatten().detach().cpu().numpy()
            flattened_thoughts.append(flat)
        
        flattened_thoughts = np.array(flattened_thoughts)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            if len(thought_history) < 4:
                print("Need at least 4 points for t-SNE, using PCA instead")
                method = 'pca'
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(thought_history)-1))
                embedding = reducer.fit_transform(flattened_thoughts)
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(flattened_thoughts)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Manifold with trajectory
        scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                                  c=timesteps, cmap=self.color_scheme, s=80, alpha=0.8)
        axes[0].plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.5, linewidth=2)
        
        # Mark start and end
        axes[0].scatter(embedding[0, 0], embedding[0, 1], c='red', s=200, marker='o', 
                       label='Start', alpha=0.9, edgecolor='black', linewidth=2)
        axes[0].scatter(embedding[-1, 0], embedding[-1, 1], c='green', s=200, marker='s', 
                       label='End', alpha=0.9, edgecolor='black', linewidth=2)
        
        axes[0].set_xlabel(f'{method.upper()} Component 1')
        axes[0].set_ylabel(f'{method.upper()} Component 2')
        axes[0].set_title(f'Thought Evolution on {method.upper()} Manifold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Timestep')
        
        # Plot 2: Distance from start over time
        start_point = embedding[0]
        distances = [np.linalg.norm(point - start_point) for point in embedding]
        
        axes[1].plot(timesteps, distances, 'b-', linewidth=2, marker='o')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Distance from Initial State')
        axes[1].set_title('Thought Divergence from Initial State')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Manifold visualization saved to {save_path}")
        else:
            plt.show()
    
    def create_thought_evolution_dashboard(self,
                                         evolution_data: Dict,
                                         save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard showing all evolution metrics.
        
        Args:
            evolution_data: Dictionary containing timesteps, thoughts, and metrics
            save_path: Optional path to save HTML dashboard
        """
        timesteps = evolution_data['timesteps']
        coherence = evolution_data['coherence_scores']
        diversity = evolution_data['diversity_scores']
        alignment = evolution_data['alignment_scores']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Coherence Evolution', 'Diversity Evolution', 
                          'Alignment Evolution', 'All Metrics Combined'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=timesteps, y=coherence, mode='lines+markers', 
                      name='Coherence', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timesteps, y=diversity, mode='lines+markers', 
                      name='Diversity', line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=timesteps, y=alignment, mode='lines+markers', 
                      name='Alignment', line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        # Combined plot
        fig.add_trace(
            go.Scatter(x=timesteps, y=coherence, mode='lines', 
                      name='Coherence', line=dict(color='blue', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=timesteps, y=diversity, mode='lines', 
                      name='Diversity', line=dict(color='green', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=timesteps, y=alignment, mode='lines', 
                      name='Alignment', line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Thought Evolution Dashboard - Phase 3 Analysis",
            height=800,
            showlegend=True
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Diffusion Timestep", row=1, col=1)
        fig.update_xaxes(title_text="Diffusion Timestep", row=1, col=2)
        fig.update_xaxes(title_text="Diffusion Timestep", row=2, col=1)
        fig.update_xaxes(title_text="Diffusion Timestep", row=2, col=2)
        
        # Update y-axes
        fig.update_yaxes(title_text="Coherence Score", row=1, col=1)
        fig.update_yaxes(title_text="Diversity Score", row=1, col=2)
        fig.update_yaxes(title_text="Alignment Score", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")
        else:
            fig.show()

def create_thought_heatmap_sequence(thought_history: List[torch.Tensor], 
                                  timesteps: List[int],
                                  save_path: Optional[str] = None) -> None:
    """Create sequence of heatmaps showing thought evolution"""
    n_snapshots = len(thought_history)
    cols = min(5, n_snapshots)
    rows = (n_snapshots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_snapshots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (thought, timestep) in enumerate(zip(thought_history, timesteps)):
        row = i // cols
        col = i % cols
        
        # Convert tensor to 2D for heatmap
        if len(thought.shape) > 2:
            if thought.shape[0] == 1:  # Remove batch dimension
                tensor_2d = thought.squeeze(0)
            else:
                tensor_2d = thought[0]
            
            if len(tensor_2d.shape) > 2:
                tensor_2d = tensor_2d.flatten(1)  # Flatten last dimensions
        else:
            tensor_2d = thought
        
        tensor_np = tensor_2d.detach().cpu().numpy()
        
        im = axes[row, col].imshow(tensor_np, cmap='viridis', aspect='auto')
        axes[row, col].set_title(f'Step {timestep}')
        axes[row, col].set_xlabel('Dimension')
        axes[row, col].set_ylabel('Feature')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[row, col])
    
    # Hide unused subplots
    for i in range(n_snapshots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Thought Tensor Evolution Heatmaps', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap sequence saved to {save_path}")
    else:
        plt.show()