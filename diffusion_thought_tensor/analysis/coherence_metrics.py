"""
Phase 3: Advanced Thought Coherence and Pattern Detection

Sophisticated metrics for measuring thought coherence, detecting emergent patterns,
and analyzing the structure of thought evolution during diffusion.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict
import json

@dataclass
class CoherenceAnalysis:
    """Results of coherence analysis"""
    global_coherence: float
    local_coherence: float
    temporal_stability: float
    structural_integrity: float
    pattern_strength: float
    emergent_properties: Dict[str, Any]

@dataclass  
class PatternDetection:
    """Results of pattern detection analysis"""
    recurring_motifs: List[Dict]
    attractor_states: List[torch.Tensor]
    phase_transitions: List[Dict]
    complexity_measures: Dict[str, float]
    network_properties: Dict[str, float]

class AdvancedCoherenceAnalyzer:
    """
    Advanced analyzer for thought coherence and emergent patterns.
    Implements multiple sophisticated metrics beyond basic correlation.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.analysis_cache = {}
        
    def analyze_coherence_multiscale(self, 
                                   thought_tensor: torch.Tensor,
                                   scales: List[int] = [1, 2, 4, 8]) -> CoherenceAnalysis:
        """
        Analyze coherence at multiple spatial and temporal scales.
        
        Args:
            thought_tensor: Input thought tensor [batch, ...spatial_dims]
            scales: Different scales to analyze
            
        Returns:
            CoherenceAnalysis object with comprehensive metrics
        """
        # Global coherence (entire tensor)
        global_coherence = self._compute_global_coherence(thought_tensor)
        
        # Local coherence (spatial neighborhoods) 
        local_coherence = self._compute_local_coherence(thought_tensor, scales)
        
        # Temporal stability (if multiple timesteps available)
        temporal_stability = self._compute_temporal_stability(thought_tensor)
        
        # Structural integrity (preserve important patterns)
        structural_integrity = self._compute_structural_integrity(thought_tensor)
        
        # Pattern strength (how well-defined are patterns)
        pattern_strength = self._compute_pattern_strength(thought_tensor)
        
        # Emergent properties
        emergent_props = self._detect_emergent_properties(thought_tensor)
        
        return CoherenceAnalysis(
            global_coherence=global_coherence,
            local_coherence=local_coherence,
            temporal_stability=temporal_stability,
            structural_integrity=structural_integrity,
            pattern_strength=pattern_strength,
            emergent_properties=emergent_props
        )
    
    def _compute_global_coherence(self, thought_tensor: torch.Tensor) -> float:
        """Compute global coherence using spectral analysis"""
        # Flatten tensor while preserving structure
        flat_tensor = thought_tensor.flatten(1)  # [batch, features]
        
        if flat_tensor.size(1) < 2:
            return 0.0
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(flat_tensor.T)
        
        # Remove NaN and diagonal elements
        mask = ~torch.isnan(corr_matrix) & ~torch.eye(corr_matrix.size(0), dtype=bool, device=corr_matrix.device)
        
        if mask.sum() == 0:
            return 0.0
        
        # Global coherence is mean absolute correlation
        global_coherence = corr_matrix[mask].abs().mean().item()
        
        return global_coherence
    
    def _compute_local_coherence(self, thought_tensor: torch.Tensor, scales: List[int]) -> float:
        """Compute local coherence using multi-scale spatial analysis"""
        if len(thought_tensor.shape) < 3:
            return 0.0  # Need spatial dimensions
        
        coherence_scores = []
        batch_size = thought_tensor.size(0)
        
        for scale in scales:
            if thought_tensor.size(-1) < scale or thought_tensor.size(-2) < scale:
                continue
                
            # Create overlapping patches
            # Handle different tensor shapes
            if len(thought_tensor.shape) == 4:  # [batch, depth, height, width]
                tensor_2d = thought_tensor.view(batch_size, -1, thought_tensor.size(-2), thought_tensor.size(-1))
                tensor_2d = tensor_2d.mean(dim=1, keepdim=True)  # Average across depth
            elif len(thought_tensor.shape) == 3:  # [batch, height, width]  
                tensor_2d = thought_tensor.unsqueeze(1)  # Add channel dimension
            else:
                continue  # Skip invalid shapes
                
            patches = F.unfold(
                tensor_2d,
                kernel_size=scale,
                stride=max(1, scale//2)
            )  # [batch, scale*scale, num_patches]
            
            if patches.size(-1) < 2:
                continue
            
            # Compute coherence within each patch
            patch_coherences = []
            for i in range(patches.size(-1)):
                patch = patches[:, :, i]  # [batch, scale*scale]
                
                if patch.size(1) > 1:
                    patch_corr = torch.corrcoef(patch.T)
                    mask = ~torch.isnan(patch_corr) & ~torch.eye(patch_corr.size(0), dtype=bool, device=patch_corr.device)
                    
                    if mask.sum() > 0:
                        coherence = patch_corr[mask].abs().mean().item()
                        patch_coherences.append(coherence)
            
            if patch_coherences:
                coherence_scores.append(np.mean(patch_coherences))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _compute_temporal_stability(self, thought_tensor: torch.Tensor) -> float:
        """Compute temporal stability (requires sequence of thoughts)"""
        # For single tensor, estimate stability from spatial smoothness
        if len(thought_tensor.shape) < 3:
            return 0.5  # Default for insufficient dimensions
        
        # Compute spatial gradients as proxy for stability
        grad_x = torch.diff(thought_tensor, dim=-1)
        grad_y = torch.diff(thought_tensor, dim=-2)
        
        # Make gradients compatible for combination
        # Crop to minimum size
        min_h = min(grad_x.size(-2), grad_y.size(-2))
        min_w = min(grad_x.size(-1), grad_y.size(-1))
        
        grad_x_cropped = grad_x[..., :min_h, :min_w]
        grad_y_cropped = grad_y[..., :min_h, :min_w]
        
        # Stability inversely related to gradient magnitude
        grad_magnitude = torch.sqrt(grad_x_cropped**2 + grad_y_cropped**2)
        stability = 1.0 / (1.0 + grad_magnitude.mean().item())
        
        return stability
    
    def _compute_structural_integrity(self, thought_tensor: torch.Tensor) -> float:
        """Measure how well structure is preserved"""
        # Use SVD to analyze structural properties
        flat_tensor = thought_tensor.flatten(1)
        
        if flat_tensor.size(1) < 2:
            return 0.0
        
        try:
            # Compute SVD
            U, S, V = torch.svd(flat_tensor)
            
            # Structural integrity based on singular value distribution
            # Higher integrity = more concentrated energy in top components
            total_energy = S.sum()
            if total_energy > 0:
                # Compute entropy of normalized singular values
                normalized_s = S / total_energy
                # Convert to numpy for entropy calculation
                s_np = normalized_s.cpu().numpy()
                s_np = s_np[s_np > 1e-10]  # Remove very small values
                
                if len(s_np) > 1:
                    structural_entropy = entropy(s_np)
                    # Convert entropy to integrity (lower entropy = higher integrity)
                    max_entropy = np.log(len(s_np))
                    integrity = 1.0 - (structural_entropy / max_entropy) if max_entropy > 0 else 0.0
                else:
                    integrity = 1.0
            else:
                integrity = 0.0
                
        except Exception:
            integrity = 0.0
        
        return integrity
    
    def _compute_pattern_strength(self, thought_tensor: torch.Tensor) -> float:
        """Measure strength of patterns using frequency analysis"""
        if len(thought_tensor.shape) < 3:
            return 0.0
        
        # Convert to numpy for signal processing
        tensor_np = thought_tensor.detach().cpu().numpy()
        
        pattern_strengths = []
        
        # Analyze each spatial dimension
        for dim in [-2, -1]:  # Last two dimensions (height, width)
            # Compute FFT along each dimension
            fft_result = np.fft.fft(tensor_np, axis=dim)
            power_spectrum = np.abs(fft_result)**2
            
            # Pattern strength based on power concentration
            total_power = power_spectrum.sum(axis=dim, keepdims=True)
            normalized_power = power_spectrum / (total_power + 1e-10)
            
            # Higher concentration = stronger patterns
            pattern_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-10), axis=dim)
            max_entropy = np.log(power_spectrum.shape[dim])
            
            if max_entropy > 0:
                pattern_strength = 1.0 - (pattern_entropy.mean() / max_entropy)
                pattern_strengths.append(pattern_strength)
        
        return np.mean(pattern_strengths) if pattern_strengths else 0.0
    
    def _detect_emergent_properties(self, thought_tensor: torch.Tensor) -> Dict[str, Any]:
        """Detect emergent properties in thought structure"""
        properties = {}
        
        # Complexity measures
        properties['effective_dimensionality'] = self._compute_effective_dimensionality(thought_tensor)
        properties['fractal_dimension'] = self._estimate_fractal_dimension(thought_tensor)
        properties['information_content'] = self._compute_information_content(thought_tensor)
        
        # Symmetry detection
        properties['symmetries'] = self._detect_symmetries(thought_tensor)
        
        # Critical points
        properties['critical_points'] = self._find_critical_points(thought_tensor)
        
        return properties
    
    def _compute_effective_dimensionality(self, thought_tensor: torch.Tensor) -> float:
        """Compute effective dimensionality using participation ratio"""
        flat_tensor = thought_tensor.flatten(1)
        
        if flat_tensor.size(1) < 2:
            return 1.0
        
        try:
            # Compute covariance matrix
            centered = flat_tensor - flat_tensor.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(centered.T, centered) / (centered.size(0) - 1)
            
            # Compute eigenvalues
            eigenvals = torch.linalg.eigvals(cov_matrix).real
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
            
            # Participation ratio
            if len(eigenvals) > 0:
                participation_ratio = (eigenvals.sum()**2) / (eigenvals**2).sum()
                effective_dim = participation_ratio.item()
            else:
                effective_dim = 1.0
                
        except Exception:
            effective_dim = 1.0
        
        return effective_dim
    
    def _estimate_fractal_dimension(self, thought_tensor: torch.Tensor) -> float:
        """Estimate fractal dimension using box-counting method (simplified)"""
        if len(thought_tensor.shape) < 3:
            return 2.0  # Default 2D
        
        # Simplify to 2D by taking mean across batch
        tensor_2d = thought_tensor.mean(dim=0).detach().cpu().numpy()
        
        # Threshold to binary
        threshold = np.median(tensor_2d)
        binary_image = (tensor_2d > threshold).astype(int)
        
        # Box counting (simplified version)
        sizes = []
        counts = []
        
        min_size = 2
        max_size = min(binary_image.shape) // 4
        
        for box_size in range(min_size, max_size + 1, 2):
            count = 0
            for i in range(0, binary_image.shape[0] - box_size + 1, box_size):
                for j in range(0, binary_image.shape[1] - box_size + 1, box_size):
                    box = binary_image[i:i+box_size, j:j+box_size]
                    if np.any(box):
                        count += 1
            
            if count > 0:
                sizes.append(box_size)
                counts.append(count)
        
        if len(sizes) < 2:
            return 2.0
        
        # Fit line to log-log plot
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        try:
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -slope
        except Exception:
            fractal_dim = 2.0
        
        return fractal_dim
    
    def _compute_information_content(self, thought_tensor: torch.Tensor) -> float:
        """Compute information content using entropy"""
        # Quantize tensor values
        tensor_flat = thought_tensor.flatten().detach().cpu().numpy()
        
        # Create histogram
        hist, _ = np.histogram(tensor_flat, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute entropy
        information_content = entropy(hist)
        
        return information_content
    
    def _detect_symmetries(self, thought_tensor: torch.Tensor) -> Dict[str, float]:
        """Detect various symmetries in thought tensor"""
        symmetries = {}
        
        if len(thought_tensor.shape) < 3:
            return symmetries
        
        tensor_2d = thought_tensor.mean(dim=0).detach().cpu().numpy()
        
        # Horizontal symmetry
        flipped_horizontal = np.fliplr(tensor_2d)
        horizontal_symmetry = np.corrcoef(tensor_2d.flatten(), flipped_horizontal.flatten())[0, 1]
        symmetries['horizontal'] = horizontal_symmetry if not np.isnan(horizontal_symmetry) else 0.0
        
        # Vertical symmetry
        flipped_vertical = np.flipud(tensor_2d)
        vertical_symmetry = np.corrcoef(tensor_2d.flatten(), flipped_vertical.flatten())[0, 1]
        symmetries['vertical'] = vertical_symmetry if not np.isnan(vertical_symmetry) else 0.0
        
        # Rotational symmetry (90 degrees)
        rotated_90 = np.rot90(tensor_2d)
        if rotated_90.shape == tensor_2d.shape:
            rotational_symmetry = np.corrcoef(tensor_2d.flatten(), rotated_90.flatten())[0, 1]
            symmetries['rotational_90'] = rotational_symmetry if not np.isnan(rotational_symmetry) else 0.0
        
        return symmetries
    
    def _find_critical_points(self, thought_tensor: torch.Tensor) -> Dict[str, int]:
        """Find critical points (maxima, minima, saddle points)"""
        critical_points = {'maxima': 0, 'minima': 0, 'saddle_points': 0}
        
        if len(thought_tensor.shape) < 3:
            return critical_points
        
        tensor_2d = thought_tensor.mean(dim=0).detach().cpu().numpy()
        
        # Compute gradients
        gradients = np.gradient(tensor_2d)
        if len(gradients) == 2:
            grad_y, grad_x = gradients
        else:
            # Handle case where gradient returns different number of arrays
            grad_y = gradients[0] if len(gradients) > 0 else np.zeros_like(tensor_2d)
            grad_x = gradients[1] if len(gradients) > 1 else np.zeros_like(tensor_2d)
        
        # Find points where gradient is near zero
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        threshold = np.percentile(gradient_magnitude, 5)  # Bottom 5% of gradients
        
        critical_mask = gradient_magnitude < threshold
        critical_locations = np.where(critical_mask)
        
        # Classify critical points using Hessian
        for i, j in zip(critical_locations[0], critical_locations[1]):
            if i > 0 and i < tensor_2d.shape[0]-1 and j > 0 and j < tensor_2d.shape[1]-1:
                # Compute Hessian elements
                fxx = tensor_2d[i-1, j] - 2*tensor_2d[i, j] + tensor_2d[i+1, j]
                fyy = tensor_2d[i, j-1] - 2*tensor_2d[i, j] + tensor_2d[i, j+1]
                fxy = (tensor_2d[i-1, j-1] - tensor_2d[i-1, j+1] - 
                       tensor_2d[i+1, j-1] + tensor_2d[i+1, j+1]) / 4
                
                # Determinant and trace of Hessian
                det = fxx * fyy - fxy**2
                trace = fxx + fyy
                
                if det > 0:
                    if trace > 0:
                        critical_points['minima'] += 1
                    else:
                        critical_points['maxima'] += 1
                elif det < 0:
                    critical_points['saddle_points'] += 1
        
        return critical_points

class EmergentPatternDetector:
    """
    Detects emergent patterns in thought evolution sequences.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def detect_patterns_in_sequence(self, 
                                  thought_sequence: List[torch.Tensor],
                                  timesteps: List[int]) -> PatternDetection:
        """
        Detect patterns across a sequence of thought states.
        
        Args:
            thought_sequence: List of thought tensors over time
            timesteps: Corresponding timesteps
            
        Returns:
            PatternDetection object with found patterns
        """
        # Find recurring motifs
        recurring_motifs = self._find_recurring_motifs(thought_sequence)
        
        # Detect attractor states
        attractor_states = self._detect_attractor_states(thought_sequence)
        
        # Find phase transitions
        phase_transitions = self._detect_phase_transitions(thought_sequence, timesteps)
        
        # Compute complexity measures
        complexity_measures = self._compute_complexity_evolution(thought_sequence)
        
        # Analyze network properties
        network_properties = self._analyze_thought_network(thought_sequence)
        
        return PatternDetection(
            recurring_motifs=recurring_motifs,
            attractor_states=attractor_states,
            phase_transitions=phase_transitions,
            complexity_measures=complexity_measures,
            network_properties=network_properties
        )
    
    def _find_recurring_motifs(self, thought_sequence: List[torch.Tensor]) -> List[Dict]:
        """Find recurring patterns in thought evolution"""
        if len(thought_sequence) < 3:
            return []
        
        motifs = []
        
        # Convert to embeddings for pattern matching
        embeddings = []
        for thought in thought_sequence:
            # Flatten and reduce dimensionality
            flat = thought.flatten().detach().cpu().numpy()
            embeddings.append(flat)
        
        embeddings = np.array(embeddings)
        
        # Find similar states using clustering
        if embeddings.shape[0] >= 2:
            try:
                n_clusters = min(5, len(embeddings) // 2)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    
                    # Find clusters with multiple members (recurring patterns)
                    for cluster_id in range(n_clusters):
                        cluster_members = np.where(labels == cluster_id)[0]
                        if len(cluster_members) >= 2:
                            motifs.append({
                                'cluster_id': int(cluster_id),
                                'occurrences': cluster_members.tolist(),
                                'strength': float(silhouette_score(embeddings, labels)) if len(set(labels)) > 1 else 0.0
                            })
            except Exception:
                pass  # Clustering failed
        
        return motifs
    
    def _detect_attractor_states(self, thought_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """Detect attractor states in thought dynamics"""
        if len(thought_sequence) < 5:
            return []
        
        attractors = []
        
        # Look for states where evolution slows down (potential attractors)
        velocities = []
        for i in range(1, len(thought_sequence)):
            prev_thought = thought_sequence[i-1].flatten()
            curr_thought = thought_sequence[i].flatten()
            velocity = torch.norm(curr_thought - prev_thought).item()
            velocities.append(velocity)
        
        # Find local minima in velocity (slow evolution points)
        velocities = np.array(velocities)
        
        # Simple peak detection for minima
        for i in range(1, len(velocities) - 1):
            if velocities[i] < velocities[i-1] and velocities[i] < velocities[i+1]:
                # This is a local minimum - potential attractor
                if velocities[i] < np.percentile(velocities, 25):  # Bottom quartile
                    attractors.append(thought_sequence[i+1])  # +1 because velocities is offset
        
        return attractors
    
    def _detect_phase_transitions(self, thought_sequence: List[torch.Tensor], timesteps: List[int]) -> List[Dict]:
        """Detect sudden changes indicating phase transitions"""
        if len(thought_sequence) < 3:
            return []
        
        transitions = []
        
        # Compute thought "energy" or magnitude over time
        energies = []
        for thought in thought_sequence:
            energy = torch.norm(thought).item()
            energies.append(energy)
        
        # Detect sudden changes in energy
        energy_diff = np.diff(energies)
        threshold = np.std(energy_diff) * 2  # 2 standard deviations
        
        for i, diff in enumerate(energy_diff):
            if abs(diff) > threshold:
                transitions.append({
                    'timestep': timesteps[i+1] if i+1 < len(timesteps) else timesteps[-1],
                    'magnitude': float(abs(diff)),
                    'direction': 'increase' if diff > 0 else 'decrease',
                    'energy_before': float(energies[i]),
                    'energy_after': float(energies[i+1])
                })
        
        return transitions
    
    def _compute_complexity_evolution(self, thought_sequence: List[torch.Tensor]) -> Dict[str, float]:
        """Compute how complexity evolves during thought process"""
        complexity_measures = {}
        
        if len(thought_sequence) < 2:
            return complexity_measures
        
        # Track various complexity measures over time
        entropies = []
        effective_dims = []
        
        coherence_analyzer = AdvancedCoherenceAnalyzer(self.device)
        
        for thought in thought_sequence:
            # Information entropy
            tensor_flat = thought.flatten().detach().cpu().numpy()
            hist, _ = np.histogram(tensor_flat, bins=50, density=True)
            hist = hist[hist > 0]
            if len(hist) > 1:
                ent = entropy(hist)
                entropies.append(ent)
            
            # Effective dimensionality
            eff_dim = coherence_analyzer._compute_effective_dimensionality(thought)
            effective_dims.append(eff_dim)
        
        # Compute trends
        if entropies:
            complexity_measures['entropy_trend'] = np.polyfit(range(len(entropies)), entropies, 1)[0]
            complexity_measures['entropy_final'] = entropies[-1]
        
        if effective_dims:
            complexity_measures['dimensionality_trend'] = np.polyfit(range(len(effective_dims)), effective_dims, 1)[0]
            complexity_measures['dimensionality_final'] = effective_dims[-1]
        
        return complexity_measures
    
    def _analyze_thought_network(self, thought_sequence: List[torch.Tensor]) -> Dict[str, float]:
        """Analyze thoughts as nodes in a network"""
        if len(thought_sequence) < 3:
            return {}
        
        # Create similarity network
        similarities = []
        n = len(thought_sequence)
        
        for i in range(n):
            for j in range(i+1, n):
                thought_i = thought_sequence[i].flatten()
                thought_j = thought_sequence[j].flatten()
                
                # Cosine similarity
                similarity = F.cosine_similarity(thought_i.unsqueeze(0), thought_j.unsqueeze(0)).item()
                similarities.append((i, j, abs(similarity)))
        
        # Create network graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges above threshold
        threshold = np.percentile([sim[2] for sim in similarities], 75)  # Top 25% similarities
        for i, j, sim in similarities:
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
        
        # Compute network properties
        network_props = {}
        
        if G.number_of_edges() > 0:
            network_props['clustering_coefficient'] = nx.average_clustering(G)
            network_props['density'] = nx.density(G)
            
            # Path lengths (if connected)
            if nx.is_connected(G):
                network_props['average_path_length'] = nx.average_shortest_path_length(G)
            else:
                # Compute for largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    network_props['average_path_length'] = nx.average_shortest_path_length(subgraph)
            
            # Centrality measures
            centralities = nx.degree_centrality(G)
            network_props['max_centrality'] = max(centralities.values())
            network_props['centrality_variance'] = np.var(list(centralities.values()))
        
        return network_props

def export_coherence_analysis(analysis: CoherenceAnalysis, 
                            pattern_detection: PatternDetection,
                            filepath: str) -> None:
    """Export complete coherence and pattern analysis to JSON"""
    
    # Convert tensors to serializable format
    def tensor_to_dict(tensor):
        return {
            'shape': list(tensor.shape),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item()
        }
    
    export_data = {
        'coherence_analysis': {
            'global_coherence': analysis.global_coherence,
            'local_coherence': analysis.local_coherence,
            'temporal_stability': analysis.temporal_stability,
            'structural_integrity': analysis.structural_integrity,
            'pattern_strength': analysis.pattern_strength,
            'emergent_properties': analysis.emergent_properties
        },
        'pattern_detection': {
            'recurring_motifs': pattern_detection.recurring_motifs,
            'attractor_states': [tensor_to_dict(tensor) for tensor in pattern_detection.attractor_states],
            'phase_transitions': pattern_detection.phase_transitions,
            'complexity_measures': pattern_detection.complexity_measures,
            'network_properties': pattern_detection.network_properties
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Coherence and pattern analysis exported to {filepath}")