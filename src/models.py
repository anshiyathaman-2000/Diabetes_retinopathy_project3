"""
Deep Learning Model Architectures for Diabetic Retinopathy Severity Classification.
Implements:
1. Model A: ResNet50 Baseline CNN.
2. Model B: Proposed CNN-Transformer-BiLSTM Hybrid Architecture.
3. Modular Ablation variants:
   - CNN-only (ResNet50)
   - CNN-Transformer (without BiLSTM)
   - CNN-BiLSTM (without Transformer)
   - Full Hybrid (CNN-Transformer-BiLSTM)
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

def build_2d_sincos_position_embedding(h: int, w: int, embed_dim: int) -> torch.Tensor:
    """Generates 2D sinusoidal position embeddings of shape [1, h*w, embed_dim]."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    dim_each = embed_dim // 2

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij"
    )

    omega = torch.arange(dim_each // 2, dtype=torch.float32) / (dim_each / 2.0)
    omega = 1.0 / (10000 ** omega)

    out_y = grid_y.flatten()[:, None] * omega[None, :]
    out_x = grid_x.flatten()[:, None] * omega[None, :]

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

    pos_emb = torch.cat([pos_y, pos_x], dim=1)
    return pos_emb.unsqueeze(0)

class BaselineResNet50(nn.Module):
    """
    Baseline Transfer Learning Model using Pretrained ResNet-50.
    Replaces the 1000-class ImageNet classification head with a regularized
    DR severity prediction head (5 grades or 2 binary classes).
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        # Replace original classification head with Dropout + Linear
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings before classification."""
        return self.backbone(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block (layer4) for Grad-CAM."""
        return self.backbone.layer4[-1]

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4
    ) -> list:
        return [
            {
                "params": [p for p in self.backbone.parameters() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay
            },
            {
                "params": [p for p in self.classifier.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay
            }
        ]

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


class CNNTransformerBiLSTM(nn.Module):
    """
    Proposed Hybrid Deep Learning Architecture for Diabetic Retinopathy.
    
    Stages:
    1. CNN Backbone (ResNet-50): Extracts local retinal vascular & lesion feature maps.
       Input: [B, 3, H, W] -> Output: [B, 2048, H/32, W/32] (e.g. 8x8 for 256x256 input).
    2. 1x1 Projection + Positional Encoding:
       Projects 2048 channels -> embed_dim (e.g. 128 or 256). Flattens spatial grid into
       a sequence of tokens [B, 64, embed_dim] with 2D spatial sinusoidal or learnable positions.
    3. Transformer Encoder:
       Self-attention modeling global co-occurrences of microvascular lesions across quadrants.
       Uses Pre-LN (norm_first=True) for stable optimization.
    4. Bidirectional LSTM:
       Captures ordered spatial-sequential dependencies across retinal tokens.
    5. Contextual Pooling & Classification Head:
       Concatenates mean and max temporal/spatial pooling (or mean pooling) -> LayerNorm -> Dropout -> Linear.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        embed_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.2,
        bilstm_hidden_dim: int = 128,
        bilstm_layers: int = 1,
        bilstm_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        use_transformer: bool = True,
        use_bilstm: bool = True,
        norm_first: bool = True,
        pos_embedding_type: str = "2d_sincos",
        pool_type: str = "mean_max"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_transformer = use_transformer
        self.use_bilstm = use_bilstm
        self.pool_type = pool_type
        self.embed_dim = embed_dim
        self.pos_embedding_type = pos_embedding_type

        # Step 1: CNN Backbone (ResNet-50 up to layer4)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Step 2: Feature Projection (2048 -> embed_dim)
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.proj_norm = nn.LayerNorm(embed_dim)

        # For 256x256 input, spatial map is 8x8 = 64 tokens
        self.num_tokens = 64
        if pos_embedding_type == "2d_sincos" and (embed_dim % 4 == 0):
            initial_pos = build_2d_sincos_position_embedding(8, 8, embed_dim)
            self.pos_embedding = nn.Parameter(initial_pos)
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(p=transformer_dropout)

        # Step 3: Transformer Encoder
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.transformer_norm = nn.LayerNorm(embed_dim)
        else:
            self.transformer = None
            self.transformer_norm = None

        # Step 4: Bidirectional LSTM
        if self.use_bilstm:
            lstm_in_dim = embed_dim
            self.bilstm = nn.LSTM(
                input_size=lstm_in_dim,
                hidden_size=bilstm_hidden_dim,
                num_layers=bilstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=bilstm_dropout if bilstm_layers > 1 else 0.0
            )
            raw_seq_dim = bilstm_hidden_dim * 2
        else:
            self.bilstm = None
            raw_seq_dim = embed_dim

        # Step 5: Classification Head
        if self.pool_type == "mean_max":
            head_in_dim = raw_seq_dim * 2
        else: # 'mean' or 'max'
            head_in_dim = raw_seq_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(head_in_dim, num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass up to feature representations."""
        # CNN Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        cnn_features = self.layer4(x) # [B, 2048, H', W']

        B, C, H, W = cnn_features.shape
        proj_features = self.proj(cnn_features) # [B, embed_dim, H, W]

        # Flatten into sequence of tokens: [B, H*W, embed_dim]
        tokens = proj_features.flatten(2).permute(0, 2, 1)
        tokens = self.proj_norm(tokens)

        # Interpolate positional embeddings if image size changes
        seq_len = tokens.size(1)
        if seq_len == self.pos_embedding.size(1):
            pos_emb = self.pos_embedding
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=seq_len,
                mode='linear'
            ).permute(0, 2, 1)

        tokens = tokens + pos_emb
        tokens = self.pos_dropout(tokens)

        # Transformer processing
        if self.use_transformer and self.transformer is not None:
            tokens = self.transformer(tokens)
            tokens = self.transformer_norm(tokens)

        # BiLSTM processing
        if self.use_bilstm and self.bilstm is not None:
            lstm_out, _ = self.bilstm(tokens) # [B, seq_len, 2 * hidden_dim]
            seq_out = lstm_out
        else:
            seq_out = tokens

        return seq_out, cnn_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.forward_features(x)

        if self.pool_type == "mean":
            pooled = torch.mean(seq_out, dim=1)
        elif self.pool_type == "max":
            pooled, _ = torch.max(seq_out, dim=1)
        else: # 'mean_max' default
            mean_pool = torch.mean(seq_out, dim=1)
            max_pool, _ = torch.max(seq_out, dim=1)
            pooled = torch.cat([mean_pool, max_pool], dim=1)

        logits = self.classifier(pooled)
        return logits

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of the CNN backbone for Grad-CAM."""
        return self.layer4[-1]

    def get_backbone_params(self):
        for m in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                yield p

    def get_head_params(self):
        head_modules = [self.proj, self.proj_norm, self.classifier]
        if self.use_transformer and self.transformer is not None:
            head_modules.extend([self.transformer, self.transformer_norm])
        if self.use_bilstm and self.bilstm is not None:
            head_modules.append(self.bilstm)
        for m in head_modules:
            for p in m.parameters():
                yield p
        yield self.pos_embedding

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4
    ) -> list:
        return [
            {
                "params": [p for p in self.get_backbone_params() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay
            },
            {
                "params": [p for p in self.get_head_params() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay
            }
        ]

    def freeze_backbone(self):
        for p in self.get_backbone_params():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.get_backbone_params():
            p.requires_grad = True

    def unfreeze_layer4_only(self):
        for p in self.get_backbone_params():
            p.requires_grad = False
        for p in self.layer4.parameters():
            p.requires_grad = True


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Model factory instantiating either Baseline ResNet50,
    the Proposed CNN-Transformer-BiLSTM, or Ablation variants.
    """
    model_name = config["model"]["name"].lower()
    num_classes = config["task"]["num_classes"]
    pretrained = config["model"].get("pretrained", True)
    dropout = config["model"].get("dropout", 0.3)
    hybrid_cfg = config["model"].get("hybrid", {})

    embed_dim = hybrid_cfg.get("cnn_embed_dim", 256)
    transformer_layers = hybrid_cfg.get("transformer_layers", 2)
    transformer_heads = hybrid_cfg.get("transformer_heads", 8)
    transformer_ff_dim = hybrid_cfg.get("transformer_feedforward", 512)
    transformer_dropout = hybrid_cfg.get("transformer_dropout", 0.2)
    bilstm_hidden_dim = hybrid_cfg.get("bilstm_hidden_dim", 128)
    bilstm_layers = hybrid_cfg.get("bilstm_layers", 1)
    bilstm_dropout = hybrid_cfg.get("bilstm_dropout", 0.1)
    norm_first = hybrid_cfg.get("norm_first", True)
    pos_embedding_type = hybrid_cfg.get("pos_embedding_type", "2d_sincos")
    pool_type = hybrid_cfg.get("pool_type", "mean_max")

    if model_name in ["resnet50", "baseline"]:
        model = BaselineResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name in ["cnn_transformer_bilstm", "hybrid", "proposed"]:
        model = CNNTransformerBiLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            embed_dim=embed_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
            bilstm_hidden_dim=bilstm_hidden_dim,
            bilstm_layers=bilstm_layers,
            bilstm_dropout=bilstm_dropout,
            classifier_dropout=dropout,
            use_transformer=True,
            use_bilstm=True,
            norm_first=norm_first,
            pos_embedding_type=pos_embedding_type,
            pool_type=pool_type
        )
    elif model_name == "cnn_transformer": # Ablation: No BiLSTM
        model = CNNTransformerBiLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            embed_dim=embed_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
            classifier_dropout=dropout,
            use_transformer=True,
            use_bilstm=False,
            norm_first=norm_first,
            pos_embedding_type=pos_embedding_type,
            pool_type=pool_type
        )
    elif model_name == "cnn_bilstm": # Ablation: No Transformer
        model = CNNTransformerBiLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            embed_dim=embed_dim,
            bilstm_hidden_dim=bilstm_hidden_dim,
            bilstm_layers=bilstm_layers,
            bilstm_dropout=bilstm_dropout,
            classifier_dropout=dropout,
            use_transformer=False,
            use_bilstm=True,
            norm_first=norm_first,
            pos_embedding_type=pos_embedding_type,
            pool_type=pool_type
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'resnet50', 'hybrid', 'cnn_transformer', or 'cnn_bilstm'.")

    return model
