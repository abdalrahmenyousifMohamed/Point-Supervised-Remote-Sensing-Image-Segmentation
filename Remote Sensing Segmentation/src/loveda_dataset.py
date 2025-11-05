"""
LoveDA Dataset Implementation for Point-Supervised Segmentation
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LoveDADataset(Dataset):
    """
    LoveDA Remote Sensing Dataset
    
    Classes:
    0: Background (ignored in some versions)
    1: Building
    2: Road
    3: Water
    4: Barren
    5: Forest
    6: Agricultural
    """
    
    def __init__(self, 
                 root_dir, 
                 split='train',  
                 scene='both',   
                 point_sampler=None,
                 transform=None,
                 image_size=512):
        """
        Args:
            root_dir: Path to LoveDA directory
            split: 'train' or 'val'
            scene: 'urban', 'rural', or 'both'
            point_sampler: PointLabelSampler instance (None for full supervision)
            transform: Custom augmentation (if None, uses default)
            image_size: Target image size (will crop from 1024x1024)
        """
        self.root_dir = Path(root_dir)
        self.split = split.capitalize()  
        self.scene = scene
        self.point_sampler = point_sampler
        self.image_size = image_size
        
        
        self.class_names = [
            'Background', 'Building', 'Road', 'Water', 
            'Barren', 'Forest', 'Agricultural'
        ]
        
        self.class_colors = [
            [255, 255, 255],  
            [255, 0, 0],      
            [255, 255, 0],    
            [0, 0, 255],      
            [159, 129, 183],  
            [0, 255, 0],      
            [255, 195, 128],  
        ]
        
        
        self.image_paths = []
        self.mask_paths = []
        
        scenes = ['Urban', 'Rural'] if scene == 'both' else [scene.capitalize()]
        
        for scene_type in scenes:
            img_dir = self.root_dir / self.split / scene_type / 'images_png'
            mask_dir = self.root_dir / self.split / scene_type / 'masks_png'
            
            if not img_dir.exists():
                print(f"Warning: {img_dir} does not exist!")
                continue
            
            
            img_files = sorted(list(img_dir.glob('*.png')))
            
            for img_path in img_files:
                
                mask_path = mask_dir / img_path.name
                
                if mask_path.exists():
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        
        print(f"Loaded {len(self.image_paths)} images from LoveDA {self.split} set ({scene})")
        
        
        if transform is None:
            self.transform = self.get_default_transform()
        else:
            self.transform = transform
    
    def get_default_transform(self):
        """Default augmentation pipeline"""
        if self.split == 'Train':
            return A.Compose([
                A.RandomCrop(height=self.image_size, width=self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.CenterCrop(height=self.image_size, width=self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        
        
        
        if mask.max() > 6:
            mask = mask - 1  
            mask = np.clip(mask, 0, 6)
        
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        
        if self.point_sampler is not None:
            mask = self.point_sampler(mask)
        
        return image, mask
    
    def visualize_sample(self, idx, save_path=None):
        """Visualize a sample with colored mask overlay"""
        import matplotlib.pyplot as plt
        
        
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(self.class_colors):
            colored_mask[mask == class_id] = color
        
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(image)
        axes[2].imshow(colored_mask, alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def get_class_distribution(self):
        """Calculate class distribution in the dataset"""
        class_counts = np.zeros(7)
        
        print("Calculating class distribution...")
        for mask_path in self.mask_paths:
            mask = np.array(Image.open(mask_path))
            for class_id in range(7):
                class_counts[class_id] += (mask == class_id).sum()
        
        total_pixels = class_counts.sum()
        class_percentages = (class_counts / total_pixels) * 100
        
        print("\nClass Distribution:")
        print("-" * 50)
        for i, (name, count, pct) in enumerate(zip(self.class_names, class_counts, class_percentages)):
            print(f"{i}: {name:15s} - {count:12.0f} pixels ({pct:5.2f}%)")
        print("-" * 50)
        
        return class_counts, class_percentages



if __name__ == "__main__":
    from point_sampler import PointLabelSampler
    
    
    dataset = LoveDADataset(
        root_dir='./LoveDA',
        split='train',
        scene='both',
        point_sampler=None,  
        image_size=512
    )
    
    print(f"\nDataset size: {len(dataset)} images")
    
    
    dataset.get_class_distribution()
    
    
    print("\nVisualizing samples...")
    for i in range(min(3, len(dataset))):
        dataset.visualize_sample(i, save_path=f'loveda_sample_{i}.png')
    
    
    print("\nTesting with point sampling...")
    point_sampler = PointLabelSampler(num_points=200, strategy='uniform')
    dataset_sparse = LoveDADataset(
        root_dir='./LoveDA',
        split='train',
        scene='urban',
        point_sampler=point_sampler,
        image_size=512
    )
    
    
    image, mask = dataset_sparse[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of labeled pixels: {(mask != -1).sum()}")
    print(f"Percentage labeled: {(mask != -1).sum() / mask.numel() * 100:.2f}%")
