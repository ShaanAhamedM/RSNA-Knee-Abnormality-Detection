import torch
import pandas as pd
import time
from src.dataset import RSNAKneeZipDataset, get_dataloader
from src.models import RSNAKneeModel
from src.losses import Poly1Loss, AsymmetricFocalLoss, HomoscedasticTaskUncertaintyLoss
from src.metrics import calculate_macro_roc_auc

print("=== Unit Test: Pipeline Verification ===")

df_folds = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv")
sample_df = df_folds[df_folds['ACL'].notnull()].head(4) # 4 labeled studies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device locked: {device}")

# 1. Test Dataset
t0 = time.time()
dataset = RSNAKneeZipDataset(
    df=sample_df,
    target_size=256,
    slices_per_study=4,
    in_chans=1,
    is_train=False
)
sample = dataset[0]
print(f"Sample 0 fetched in {time.time() - t0:.2f}s")
print(f"Image bag shape: {sample['image'].shape}") # Expected: (4, 1, 256, 256)
print(f"Labels shape: {sample['labels'].shape}") # Expected: (12,)
print(f"Labels values: {sample['labels'].numpy()}")

# 2. Test DataLoader with 2 workers
loader = get_dataloader(sample_df, target_size=256, slices_per_study=4, batch_size=2, num_workers=0)
batch = next(iter(loader))
print(f"Batch image shape: {batch['image'].shape}") # Expected: (2, 4, 1, 256, 256)
print(f"Batch labels shape: {batch['labels'].shape}")

# 3. Test Model Forward Pass
model = RSNAKneeModel(
    backbone_name='efficientnet_b0',
    num_classes=12,
    in_chans=1,
    pretrained=False,
    use_mil=True
).to(device)

images = batch['image'].to(device)
labels = batch['labels'].to(device)

logits = model(images)
print(f"Model output logits shape: {logits.shape}") # Expected: (2, 12)

# 4. Test Losses
poly_loss_fn = Poly1Loss(epsilon=1.0)
asl_loss_fn = AsymmetricFocalLoss()
uncertainty_loss_fn = HomoscedasticTaskUncertaintyLoss(num_tasks=12).to(device)

loss_poly = poly_loss_fn(logits, labels)
loss_asl = asl_loss_fn(logits, labels)
loss_unc = uncertainty_loss_fn(logits, labels)

print(f"PolyLoss: {loss_poly.item():.4f}")
print(f"ASL Loss: {loss_asl.item():.4f}")
print(f"Uncertainty Loss: {loss_unc.item():.4f}")

# 5. Test Backward Pass with AMP
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

optimizer.zero_grad()
with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
    out = model(images)
    loss = poly_loss_fn(out, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print("Backward pass + AMP optimizer step completed successfully!")
print("VRAM Allocated:", torch.cuda.memory_allocated() / (1024**2), "MB")
print("VRAM Reserved:", torch.cuda.memory_reserved() / (1024**2), "MB")
print("=== All Unit Tests Passed! ===")
