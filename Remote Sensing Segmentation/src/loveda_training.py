
"""
Complete Training Script for LoveDA Dataset with Point Supervision
Run this after setting up the dataset and required modules
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from loveda_dataset import LoveDADataset
from loss import PartialCrossEntropyLoss
from sampler import PointLabelSampler


class LoveDATrainer:
    """Training pipeline specifically for LoveDA dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.model = self._create_model()
        
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_per_class_iou': [],
            'learning_rates': []
        }
        
        self.best_miou = 0.0
    
    def _create_model(self):
        """Create segmentation model"""
        model = smp.Unet(
            encoder_name=self.config['encoder'],
            encoder_weights='imagenet',
            in_channels=3,
            classes=self.config['num_classes']
        )
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {self.config['encoder']}-UNet")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders"""
        from loveda_dataset import LoveDADataset
        from sampler import PointLabelSampler
        
        if self.config['num_points'] > 0:
            point_sampler = PointLabelSampler(
                num_points=self.config['num_points'],
                strategy=self.config['sampling_strategy']
            )
        else:
            point_sampler = None
        
        train_dataset = LoveDADataset(
            root_dir=self.config['data_root'],
            split='train',
            scene=self.config['scene'],
            point_sampler=point_sampler,
            image_size=self.config['image_size']
        )
        
        val_dataset = LoveDADataset(
            root_dir=self.config['data_root'],
            split='val',
            scene=self.config['scene'],
            point_sampler=None,
            image_size=self.config['image_size']
        )
        use_pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=use_pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=use_pin_memory
        )
        
        print(f"\nDataset: LoveDA ({self.config['scene']})")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Point annotations: {self.config['num_points']} ({self.config['sampling_strategy']})")
        
        return train_loader, val_loader
    
    def _create_criterion(self):
        """Create loss function"""
        from loss import PartialCrossEntropyLoss
        
        if self.config.get('use_class_weights', False):
            weights = None
        else:
            weights = None
        
        criterion = PartialCrossEntropyLoss(
            ignore_index=-1,
            weight=weights
        )
        
        return criterion
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        
        return optimizer
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        
        intersection = torch.zeros(self.config['num_classes'])
        union = torch.zeros(self.config['num_classes'])
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                
                for class_id in range(self.config['num_classes']):
                    pred_class = (predictions == class_id)
                    mask_class = (masks == class_id)
                    
                    intersection[class_id] += (pred_class & mask_class).sum().cpu()
                    union[class_id] += (pred_class | mask_class).sum().cpu()
        
        avg_loss = val_loss / len(self.val_loader)
        
        iou_per_class = []
        for class_id in range(self.config['num_classes']):
            if union[class_id] > 0:
                iou = (intersection[class_id] / union[class_id]).item()
            else:
                iou = 0.0
            iou_per_class.append(iou * 100)
        
        if self.config.get('ignore_background', False):
            mean_iou = np.mean(iou_per_class[1:])
        else:
            mean_iou = np.mean(iou_per_class)
        
        return avg_loss, mean_iou, iou_per_class
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            
            val_loss, val_miou, per_class_iou = self.validate()
            
            self.scheduler.step(val_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_miou)
            self.history['val_per_class_iou'].append(per_class_iou)
            self.history['learning_rates'].append(current_lr)
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val mIoU: {val_miou:.2f}% | LR: {current_lr:.6f}")
            print("Per-class IoU:", ", ".join([f"{iou:.1f}" for iou in per_class_iou]))
            
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Saved best model (mIoU: {val_miou:.2f}%)")
            
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        self.save_history()
        self.plot_training_curves()
        
        print("\n" + "="*60)
        print(f"Training Completed! Best mIoU: {self.best_miou:.2f}%")
        print("="*60)
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Saved training history to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['val_miou'], label='Val mIoU', 
                       color='green', linewidth=2, marker='o')
        axes[0, 1].axhline(y=self.best_miou, color='r', linestyle='--', 
                          label=f'Best: {self.best_miou:.2f}%')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU (%)')
        axes[0, 1].set_title('Validation mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['learning_rates'], linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(self.history['val_per_class_iou']) > 0:
            per_class_array = np.array(self.history['val_per_class_iou'])
            class_names = ['BG', 'Build', 'Road', 'Water', 'Barren', 'Forest', 'Agri']
            
            for i, name in enumerate(class_names):
                axes[1, 1].plot(per_class_array[:, i], label=name, linewidth=1.5)
            
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('IoU (%)')
            axes[1, 1].set_title('Per-Class IoU Evolution')
            axes[1, 1].legend(ncol=2, fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        plt.close()


def main():
    """Main training script"""
    
    config = {
        'data_root': './LoveDA',
        'scene': 'both',
        'image_size': 512,
        'num_classes': 7,
        'num_points': 200,
        'sampling_strategy': 'uniform',
        'encoder': 'resnet34',
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'num_workers': 4,
        'save_freq': 10,
        'checkpoint_dir': 'checkpoints/loveda_experiment',
        'ignore_background': False,
        'use_class_weights': False,
    }
    
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")
    
    trainer = LoveDATrainer(config)
    
    history = trainer.train()
    
    print("\n✓ Training completed successfully!")
    print(f"Best model saved in: {config['checkpoint_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
