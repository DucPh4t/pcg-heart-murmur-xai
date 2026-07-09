import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

try:
    import config as cfg
except Exception:
    cfg = None

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class FrequencyGuidedAttention(nn.Module):
    """
    Frequency-guided attention.

    The module pools over time and channels, then learns a 1D attention map over
    frequency bands. This keeps temporal structure intact while reweighting
    frequency regions that are more informative for murmur detection.
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        # Conv1d operates along the frequency axis.
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Freq(H), Time(W))
        
        # Average energy per frequency band.
        avg_out = x.mean(dim=3, keepdim=False).mean(dim=1, keepdim=True) # (B, 1, H)
        
        # Maximum activation per frequency band.
        max_out = x.amax(dim=(1, 3), keepdim=False).unsqueeze(1) # (B, 1, H)
        
        out = torch.cat([avg_out, max_out], dim=1) # (B, 2, H)
        
        # Learn one attention weight per frequency band.
        out = self.conv(out) # (B, 1, H)
        out = self.sigmoid(out) # (B, 1, H)
        
        out = out.unsqueeze(3) # (B, 1, H, 1)
        
        return x * out

class FGA_Module(nn.Module):
    def __init__(self, in_channels, reduction=16, freq_kernel_size=5):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.freq_attention = FrequencyGuidedAttention(kernel_size=freq_kernel_size)

    def forward(self, x):
        # Channel attention reweights feature channels before frequency attention.
        x = self.channel_attention(x)
        x = self.freq_attention(x)
        return x

class TemporalAttentionPool(nn.Module):
    """
    Temporal attention pooling over the final feature map.

    Frequency is averaged first, then a small Conv1D network learns how much
    each time step should contribute to the recording-level representation.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # Average over frequency, preserving channel and time axes.
        x_time = x.mean(dim=2) # (B, C, W)
        
        attn_weights = self.attention(x_time) # (B, 1, W)
        
        out = (x_time * attn_weights).sum(dim=2) # (B, C)
        
        return out

class ResNet18_FGA(nn.Module):
    """ResNet18 backbone with FGA and temporal attention for 3-class murmur detection."""
    
    def __init__(self, num_classes=3, pretrained=True, input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        
        # Convert ImageNet RGB conv1 weights to spectrogram channels. If
        # location channels are enabled, channel 0 keeps the audio pretrained
        # initialization while location channels start at zero so the model
        # begins from the audio-only baseline and learns location conditioning.
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            audio_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
            self.conv1.weight.data.zero_()
            self.conv1.weight.data[:, :1] = audio_weight
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.fga1 = FGA_Module(64, reduction=4)
        self.fga2 = FGA_Module(128, reduction=8)
        self.fga3 = FGA_Module(256, reduction=16)
        self.fga4 = FGA_Module(512, reduction=16)
        
        self.temporal_pool = TemporalAttentionPool(512)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.attention_maps = {}
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.fga1(x)
        
        x = self.layer2(x)
        x = self.fga2(x)
        
        x = self.layer3(x)
        x = self.fga3(x)
        self.attention_maps['layer3'] = x.detach()
        
        x = self.layer4(x)
        x = self.fga4(x)
        self.attention_maps['layer4'] = x.detach()
        
        x = self.temporal_pool(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


class PatientMILModel(nn.Module):
    """Patient-level multiple-instance model over variable recording bags."""

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        input_channels=1,
        use_location_embedding=True,
        num_locations=5,
        location_embed_dim=16,
    ):
        super().__init__()
        self.encoder = ResNet18_FGA(
            num_classes=num_classes,
            pretrained=pretrained,
            input_channels=input_channels,
        )
        self.use_location_embedding = use_location_embedding
        feature_dim = 512
        if use_location_embedding:
            self.location_embedding = nn.Embedding(
                num_locations + 1,
                location_embed_dim,
                padding_idx=num_locations,
            )
            instance_dim = feature_dim + location_embed_dim
        else:
            self.location_embedding = None
            instance_dim = feature_dim

        self.attention = nn.Sequential(
            nn.Linear(instance_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(instance_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, bags, mask, locations=None, return_attention=False):
        bsz, n_rec, channels, height, width = bags.shape
        flat = bags.view(bsz * n_rec, channels, height, width)
        features = self.encoder.forward_features(flat).view(bsz, n_rec, -1)

        if self.use_location_embedding:
            if locations is None:
                raise ValueError("locations are required when location embedding is enabled")
            loc_emb = self.location_embedding(locations)
            features = torch.cat([features, loc_emb], dim=-1)

        attn_logits = self.attention(features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1)
        patient_feature = torch.sum(features * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(patient_feature)

        if return_attention:
            return logits, attn_weights
        return logits

def get_model(num_classes=3, pretrained=True, input_channels=None):
    if input_channels is None:
        input_channels = getattr(cfg, "INPUT_CHANNELS", 1) if cfg is not None else 1
    return ResNet18_FGA(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
    )


def get_patient_mil_model(num_classes=3, pretrained=True, input_channels=None):
    if input_channels is None:
        input_channels = getattr(cfg, "INPUT_CHANNELS", 1) if cfg is not None else 1
    return PatientMILModel(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
        use_location_embedding=getattr(cfg, "MIL_USE_LOCATION_EMBEDDING", True),
        num_locations=getattr(cfg, "NUM_LOCATION_CHANNELS", 5),
        location_embed_dim=getattr(cfg, "MIL_LOCATION_EMBED_DIM", 16),
    )
