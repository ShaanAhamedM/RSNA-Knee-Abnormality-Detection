import os
import torch
import timm

os.makedirs('checkpoints', exist_ok=True)
out_path = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\densenet121_backbone_in1.pt"

print("Instantiating DenseNet121 with in_chans=1...")
model = timm.create_model('densenet121.ra_in1k', pretrained=True, in_chans=1, num_classes=0)
torch.save(model.state_dict(), out_path)
print(f"Saved DenseNet121 backbone to {out_path} ({os.path.getsize(out_path) / (1024*1024):.1f} MB)!")
