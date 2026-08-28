
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float or list): Weighting factor for each class. 
                               If float, used for positive class.
                               If list, [alpha_neg, alpha_pos].
        gamma (float): Focusing parameter. Higher = more focus on hard examples.
        reduction (str): 'mean', 'sum', or 'none'
    
    References:
        - Lin et al., "Focal Loss for Dense Object Detection" (2017)
        - Commonly used for imbalanced medical datasets
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # Single alpha: apply to positive class, 1-alpha to negative
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            # List of alphas per class
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, take the first element
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for the target class
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients (Importance weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.data.squeeze().cpu().numpy()
class GradCAMPlusPlus:
    """
    Grad-CAM++: Generalized Graduate-based Visual Explanations for 
    Deep Convolutional Networks.
    Provides better localization and highlights multiple instances of the same class.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for the target class
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients # (1, C, H, W)
        activations = self.activations # (1, C, H, W)
        
        # Grad-CAM++ specific weight calculation
        # See paper: Grad-CAM++: Generalized Gradient-based Visual Explanations...
        
        # Calculate alpha weights (pixel-wise weighting)
        # Using a simplified but effective version of the alpha formula
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        
        # Summing over spatial dims (H, W)
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-7)
        
        # Weighted importance for each pixel
        weights = torch.sum(aij * torch.clamp(gradients, min=0), dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and Norm
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.data.squeeze().cpu().numpy()

class ScoreCAM:
    """
    Score-CAM: Improved visual explanations via score-based weighting.
    Eliminates gradient noise and provides much cleaner, sharper heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            # 1. Get activations: (1, C, H, W)
            _ = self.model(x)
            activations = self.activations
            b, c, h, w = activations.shape
            
            # 2. Project activations back to input space using Masking
            # Upsample activations to input size
            upsampler = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
            masks = upsampler(activations.permute(1, 0, 2, 3)) # (C, 1, H_in, W_in)
            
            # Normalize masks 0-1
            masks = (masks - masks.min()) / (masks.max() - masks.min() + 1e-7)
            
            # Mask the input and get scores
            masked_input = x * masks # (C, 1, H_in, W_in)
            
            # Memory efficient batching for scores
            batch_size = 32
            scores = []
            for i in range(0, c, batch_size):
                chunk = masked_input[i:i+batch_size]
                out = self.model(chunk)
                scores.append(torch.softmax(out, dim=1)[:, class_idx].cpu())
            
            weights = torch.cat(scores).view(c, 1, 1, 1).to(x.device)
            
            # 3. Final CAM is weighted sum of activations
            cam = torch.sum(weights * activations.permute(1, 0, 2, 3), dim=0, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            
            return cam.squeeze().cpu().numpy()

def visualize_cam(mask, img_tensor, save_path="gradcam.png", alpha=0.4, title="XAI Visualization"):
    """
    Professional visualization with high contrast and proper overlay.
    """
    # 1. Advanced Normalization (Focus on top-95% to avoid dull heatmaps)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-7)
    
    # 2. Use 'Inferno' or 'Jet' for better clinical visibility
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # 3. Process Spectrogram
    img = img_tensor.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    img_rgb = np.stack([img]*3, axis=-1) 
    
    # 4. Resize and Overlay
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # Use a sharper blend formula
    cam = heatmap * alpha + img_rgb * (1.1 - alpha)
    cam = np.clip(cam, 0, 1)
    
    # 5. Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(np.flipud(cam)) 
    plt.title(title, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
