

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


from src.loss import PartialCrossEntropyLoss
from src.sampler import PointLabelSampler
from src.complete_pipeline import RemoteSensingDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5


sampler = PointLabelSampler(num_points=200, strategy='uniform')


train_dataset = RemoteSensingDataset(
    image_dir='data/train/images',
    mask_dir='data/train/masks',
    point_sampler=sampler
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=NUM_CLASSES
).to(device)


criterion = PartialCrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


model.train()
for epoch in range(10):  
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training completed!")