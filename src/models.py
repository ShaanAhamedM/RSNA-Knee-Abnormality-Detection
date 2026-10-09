"""
Architectural Engineering:
2.5D Volumetric Backbones with Gated Attention Multiple Instance Learning (Attention-MIL)
Optimized for 6 GB VRAM with Gradient Checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GatedAttentionMIL(nn.Module):
    """
    Ilse et al. (ICML 2018) Gated Attention-MIL Pooling Head:
    a_k = exp(w^T (tanh(V h_k) * sigm(U h_k))) / sum_j exp(w^T (tanh(V h_j) * sigm(U h_j)))
    h_bag = sum_k a_k * h_k
    """
    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.linear_v = nn.Linear(in_features, hidden_dim)
        self.linear_u = nn.Linear(in_features, hidden_dim)
        self.linear_w = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Tensor of shape (Batch_size, Bag_size, in_features)
        Returns:
            bag_repr: Tensor of shape (Batch_size, in_features)
            attention_weights: Tensor of shape (Batch_size, Bag_size, 1)
        """
        # tanh(V h) and sigmoid(U h)
        v_h = torch.tanh(self.linear_v(h))
        u_h = torch.sigmoid(self.linear_u(h))
        
        # Element-wise gating
        gated = v_h * u_h
        
        # Unnormalized attention logits
        a_logits = self.linear_w(gated) # (B, Bag_size, 1)
        
        # Softmax across slices/bag dimension
        alpha = F.softmax(a_logits, dim=1)
        
        # Weighted sum: (B, Bag_size, 1) * (B, Bag_size, in_features) -> sum along dim 1
        bag_repr = torch.sum(alpha * h, dim=1) # (B, in_features)
        return bag_repr, alpha

class RSNAKneeModel(nn.Module):
    """
    Unified 2D / 2.5D Attention-MIL Neural Architecture for RSNA Knee Abnormality Detection.
    Accepts arbitrary timm backbones, with gradient checkpointing and gated attention pooling.
    """
    def __init__(
        self,
        backbone_name: str = 'convnext_small',
        num_classes: int = 12,
        in_chans: int = 1,
        pretrained: bool = True,
        use_mil: bool = True,
        hidden_dim: int = 256,
        dropout_p: float = 0.3,
        grad_checkpointing: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.use_mil = use_mil
        
        # Build timm backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0, # pooled feature vector
            drop_rate=dropout_p
        )
        
        # Enable gradient checkpointing if requested (critical for 6GB VRAM)
        if grad_checkpointing and hasattr(self.backbone, 'set_grad_checkpointing'):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass
                
        self.num_features = self.backbone.num_features
        
        # Attention-MIL Head
        if self.use_mil:
            self.mil_pool = GatedAttentionMIL(self.num_features, hidden_dim=hidden_dim)
            
        # Final classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If use_mil is True:
            Input shape: (B, Bag_size, C, H, W)
            Output shape: (B, num_classes)
        If use_mil is False:
            Input shape: (B, C, H, W)
            Output shape: (B, num_classes)
        """
        if self.use_mil:
            if x.dim() == 4:
                # Add bag dimension of 1 if 4D tensor passed
                x = x.unsqueeze(1)
            b, bag_size, c, h, w = x.shape
            
            # Collapse batch and bag for backbone feature extraction
            x_flat = x.view(b * bag_size, c, h, w)
            features_flat = self.backbone(x_flat) # (b * bag_size, num_features)
            
            # Reshape back to bag
            features = features_flat.view(b, bag_size, self.num_features)
            
            # Gated Attention Pooling
            bag_features, self.last_attention = self.mil_pool(features) # (b, num_features)
            logits = self.classifier(bag_features) # (b, num_classes)
            return logits
        else:
            features = self.backbone(x)
            logits = self.classifier(features)
            return logits
