import torch
import numpy as np
from scipy import ndimage
import random

class PointLabelSampler:
    """
    Simulates sparse point annotations from dense segmentation masks.
    Supports multiple sampling strategies for experimental exploration.
    """
    
    def __init__(self, num_points=100, strategy='random', ignore_index=-1):
        """
        Args:
            num_points (int): Number of points to sample per image
            strategy (str): Sampling strategy - 'random', 'uniform', 'boundary', 'cluster'
            ignore_index (int): Value for unlabeled pixels
        """
        self.num_points = num_points
        self.strategy = strategy
        self.ignore_index = ignore_index
        
    def random_sampling(self, mask):
        """Completely random sampling across the image"""
        h, w = mask.shape
        sparse_mask = torch.full_like(mask, self.ignore_index)
        
        
        indices = torch.randperm(h * w)[:self.num_points]
        y_coords = indices // w
        x_coords = indices % w
        
        sparse_mask[y_coords, x_coords] = mask[y_coords, x_coords]
        return sparse_mask
    
    def uniform_grid_sampling(self, mask):
        """Uniform grid sampling - better spatial coverage"""
        h, w = mask.shape
        sparse_mask = torch.full_like(mask, self.ignore_index)
        
        
        grid_size = int(np.sqrt(h * w / self.num_points))
        grid_size = max(1, grid_size)
        
        
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                if len(sparse_mask[sparse_mask != self.ignore_index]) >= self.num_points:
                    break
                
                y = min(h-1, i + random.randint(0, min(grid_size-1, h-i-1)))
                x = min(w-1, j + random.randint(0, min(grid_size-1, w-j-1)))
                sparse_mask[y, x] = mask[y, x]
        
        return sparse_mask
    
    def class_balanced_sampling(self, mask):
        """Sample points ensuring each class is represented"""
        h, w = mask.shape
        sparse_mask = torch.full_like(mask, self.ignore_index)
        
        unique_classes = torch.unique(mask)
        points_per_class = self.num_points // len(unique_classes)
        
        for class_id in unique_classes:
            
            class_mask = (mask == class_id)
            class_indices = torch.nonzero(class_mask, as_tuple=False)
            
            if len(class_indices) == 0:
                continue
            
            
            num_samples = min(points_per_class, len(class_indices))
            sampled_indices = class_indices[torch.randperm(len(class_indices))[:num_samples]]
            
            for idx in sampled_indices:
                sparse_mask[idx[0], idx[1]] = mask[idx[0], idx[1]]
        
        return sparse_mask
    
    def boundary_aware_sampling(self, mask):
        """Sample more points near class boundaries (harder regions)"""
        h, w = mask.shape
        sparse_mask = torch.full_like(mask, self.ignore_index)
        
        
        mask_np = mask.cpu().numpy()
        gradient = ndimage.sobel(mask_np.astype(float))
        gradient_magnitude = np.abs(gradient)
        
        
        prob_map = gradient_magnitude / (gradient_magnitude.sum() + 1e-7)
        prob_map = prob_map.flatten()
        
        
        indices = np.random.choice(h * w, size=self.num_points, p=prob_map, replace=False)
        y_coords = indices // w
        x_coords = indices % w
        
        for y, x in zip(y_coords, x_coords):
            sparse_mask[y, x] = mask[y, x]
        
        return sparse_mask
    
    def cluster_sampling(self, mask, num_clusters=10):
        """Sample points in spatial clusters (simulates annotator behavior)"""
        h, w = mask.shape
        sparse_mask = torch.full_like(mask, self.ignore_index)
        
        points_per_cluster = self.num_points // num_clusters
        
        
        cluster_centers = [
            (random.randint(0, h-1), random.randint(0, w-1))
            for _ in range(num_clusters)
        ]
        
        for center_y, center_x in cluster_centers:
            
            radius = min(h, w) // 8
            
            for _ in range(points_per_cluster):
                
                y = int(np.clip(np.random.normal(center_y, radius/3), 0, h-1))
                x = int(np.clip(np.random.normal(center_x, radius/3), 0, w-1))
                
                sparse_mask[y, x] = mask[y, x]
        
        return sparse_mask
    
    def __call__(self, mask):
        """
        Args:
            mask: (H, W) tensor with dense labels
        Returns:
            sparse_mask: (H, W) tensor with sparse point labels and ignore_index elsewhere
        """
        if self.strategy == 'random':
            return self.random_sampling(mask)
        elif self.strategy == 'uniform':
            return self.uniform_grid_sampling(mask)
        elif self.strategy == 'balanced':
            return self.class_balanced_sampling(mask)
        elif self.strategy == 'boundary':
            return self.boundary_aware_sampling(mask)
        elif self.strategy == 'cluster':
            return self.cluster_sampling(mask)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    mask = torch.zeros((256, 256), dtype=torch.long)
    mask[50:100, 50:150] = 1
    mask[150:200, 100:200] = 2
    mask[80:120, 180:240] = 3
    
    
    strategies = ['random', 'uniform', 'balanced', 'boundary', 'cluster']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    
    axes[0].imshow(mask.numpy(), cmap='tab10')
    axes[0].set_title('Original Dense Mask')
    axes[0].axis('off')
    
    
    for idx, strategy in enumerate(strategies):
        sampler = PointLabelSampler(num_points=200, strategy=strategy)
        sparse_mask = sampler(mask)
        
        
        vis_mask = sparse_mask.numpy().copy()
        vis_mask[vis_mask == -1] = np.nan  
        
        axes[idx + 1].imshow(mask.numpy(), cmap='tab10', alpha=0.3)
        axes[idx + 1].imshow(vis_mask, cmap='tab10', alpha=1.0)
        axes[idx + 1].set_title(f'{strategy.capitalize()} Sampling')
        axes[idx + 1].axis('off')
        
        
        num_labeled = (sparse_mask != -1).sum().item()
        print(f"{strategy}: {num_labeled} labeled pixels ({num_labeled/(256*256)*100:.2f}%)")
    
    plt.tight_layout()
    plt.savefig('sampling_strategies.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'sampling_strategies.png'")
