from torch.utils.data import DataLoader
from src.dataset import FloodNetDataset
from src.transforms import train_transforms, val_transforms
from src.model import UNet

train_dataset = FloodNetDataset(split="train", transform=train_transforms)
val_dataset = FloodNetDataset(split="val",transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

#testing purpose
for epoch in range(3):
    model.train()
    for batch_idx, [images, masks] in enumerate(islice(train_loader,3),1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.5f}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for  batch_idx, [images, masks] in enumerate(islice(val_loader, 2),1):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()
    print(f"Epoch: {epoch} | val_loss: {val_loss/2:.5f}")   
