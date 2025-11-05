"""
Complete Training Pipeline for Point-Supervised Segmentation
Includes: Data loading, Model, Training loop, Evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from pathlib import Path
import json
import matplotlib.pyplot as plt




class RemoteSensingDataset(Dataset):
    """
    Custom dataset for remote sensing segmentation.
    Adapt this to your specific dataset structure.
    """
    def __init__(self, image_dir, mask_dir, point_sampler=None, transform=None):
        """
        Args:
            image_dir: Path to directory containing images
            mask_dir: Path to directory containing masks
            point_sampler: PointLabelSampler instance (None for full supervision)
            transform: Data augmentation transforms
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.point_sampler = point_sampler
        self.transform = transform
        
        
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + 
                                  list(self.image_dir.glob('*.jpg')) +
                                  list(self.image_dir.glob('*.tif')))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path))
        
        
        mask_path = self.mask_dir / img_path.name
        mask = np.array(Image.open(mask_path))
        
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        
        if self.point_sampler is not None:
            mask = self.point_sampler(mask)
        
        return image, mask


class SegmentationTrainer:
    """
    Training and evaluation framework for point-supervised segmentation
    """
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device='cuda',
                 num_classes=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        ious = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                
                predictions = torch.argmax(outputs, dim=1)
                iou = self.calculate_iou(predictions, masks)
                ious.append(iou)
        
        avg_loss = val_loss / len(self.val_loader)
        avg_miou = np.mean(ious)
        
        self.val_losses.append(avg_loss)
        self.val_mious.append(avg_miou)
        
        return avg_loss, avg_miou
    
    def calculate_iou(self, predictions, targets):
        """Calculate mean IoU on labeled pixels only"""
        
        valid_mask = (targets != -1)
        
        if not valid_mask.any():
            return 0.0
        
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        ious = []
        for c in range(self.num_classes):
            pred_c = (predictions == c)
            target_c = (targets == c)
            
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            
            if union > 0:
                iou = (intersection / union).item()
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_miou = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            
            train_loss = self.train_epoch()
            
            
            val_loss, val_miou = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
            
            
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_miou': val_miou,
                }, save_dir / 'best_model.pth')
                print(f"✓ Saved best model (mIoU: {val_miou:.4f})")
        
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        
        return history
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        
        ax2.plot(self.val_mious, label='Val mIoU', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.set_title('Validation mIoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")


def main():
    """
    Main training script
    """
    
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    NUM_POINTS = 200  
    SAMPLING_STRATEGY = 'uniform'  
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    point_sampler = PointLabelSampler(
        num_points=NUM_POINTS,
        strategy=SAMPLING_STRATEGY
    )
    
    
    
    train_dataset = RemoteSensingDataset(
        image_dir='data/train/images',
        mask_dir='data/train/masks',
        point_sampler=point_sampler
    )
    
    val_dataset = RemoteSensingDataset(
        image_dir='data/val/images',
        mask_dir='data/val/masks',
        point_sampler=None  
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    
    model = smp.UNet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=NUM_CLASSES
    )
    
    
    criterion = PartialCrossEntropyLoss(ignore_index=-1)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=NUM_CLASSES
    )
    
    
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    
    trainer.plot_training_curves()
    
    print("\nTraining completed!")
    print(f"Best validation mIoU: {max(history['val_mious']):.4f}")


if __name__ == "__main__":
    main()
