import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for point-supervised segmentation.
    Only computes loss on labeled pixels, ignoring unlabeled regions.
    
    Args:
        ignore_index (int): Label value to ignore (default: -1 for unlabeled pixels)
        reduction (str): Specifies the reduction to apply ('mean', 'sum', 'none')
        weight (Tensor, optional): Manual rescaling weight for each class
    """
    def __init__(self, ignore_index=-1, reduction='mean', weight=None):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - model output logits
            targets: (B, H, W) - ground truth labels with ignore_index for unlabeled
            
        Returns:
            loss: Scalar tensor
        """
        
        valid_mask = (targets != self.ignore_index)
        
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        
        
        ce_loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'  
        )
        
        
        masked_loss = ce_loss * valid_mask.float()
        
        
        if self.reduction == 'mean':
            
            loss = masked_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            loss = masked_loss.sum()
        else:  
            loss = masked_loss
            
        return loss


class PartialCrossEntropyDiceLoss(nn.Module):
    """
    Combined Partial Cross-Entropy + Dice Loss for better performance.
    This is useful for exploring different loss combinations in experiments.
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=-1):
        super(PartialCrossEntropyDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.partial_ce = PartialCrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        
    def dice_loss(self, predictions, targets):
        """Compute Dice loss on labeled pixels only"""
        valid_mask = (targets != self.ignore_index)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        probs = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        
        dice_scores = []
        for c in range(num_classes):
            
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()
            
            
            pred_c = pred_c * valid_mask.float()
            target_c = target_c * valid_mask.float()
            
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.append(dice)
        
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss
    
    def forward(self, predictions, targets):
        ce_loss = self.partial_ce(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice
        return total_loss



if __name__ == "__main__":
    
    batch_size = 4
    num_classes = 5
    height, width = 256, 256
    
    
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    
    targets = torch.full((batch_size, height, width), -1, dtype=torch.long)
    
    
    for b in range(batch_size):
        num_points = 100
        y_coords = torch.randint(0, height, (num_points,))
        x_coords = torch.randint(0, width, (num_points,))
        labels = torch.randint(0, num_classes, (num_points,))
        
        targets[b, y_coords, x_coords] = labels
    
    
    criterion = PartialCrossEntropyLoss()
    loss = criterion(predictions, targets)
    
    print(f"Partial Cross-Entropy Loss: {loss.item():.4f}")
    print(f"Percentage of labeled pixels: {((targets != -1).sum() / targets.numel() * 100):.2f}%")
