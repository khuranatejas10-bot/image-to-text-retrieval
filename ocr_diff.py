import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class LinearAttention(nn.Module):
    """
    Linear Attention Module as proposed in OCR-Diff paper (Section III-A & Ref [10]).
    Reduces memory and computational complexity from O(N^2) to O(N) by applying
    Softmax along feature dimensions of Query and Key before context aggregation.
    """
    def __init__(self, in_channels):
        super(LinearAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        # Project to Q, K, V
        proj_query = self.query_conv(x).view(batch_size, C, N)  # B x C x N
        proj_key = self.key_conv(x).view(batch_size, C, N)      # B x C x N
        proj_value = self.value_conv(x).view(batch_size, C, N)  # B x C x N

        # Apply softmax across spatial dimension N (linear attention formulation)
        q_softmax = F.softmax(proj_query, dim=2)  # B x C x N
        k_softmax = F.softmax(proj_key, dim=1)    # B x C x N (softmax across channels)

        # Context vector via matrix multiplication
        # k_softmax: B x C x N -> transposed to B x N x C
        context = torch.bmm(k_softmax, proj_value.transpose(1, 2)) # B x C x C
        out = torch.bmm(context, q_softmax)                         # B x C x N

        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out


class ResBlock(nn.Module):
    """
    Residual Block with optional Time Embedding and Feature Conditioning.
    Uses GroupNorm and SiLU activation for stable gradient flow.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None

        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        if self.time_mlp is not None and time_emb is not None:
            time_effect = self.time_mlp(F.silu(time_emb))
            h = h + time_effect.unsqueeze(-1).unsqueeze(-1)

        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)


class FeatureExtractor(nn.Module):
    """
    Feature Extractor network (OCR-Diff Paper Fig. 2).
    Constructed by stacking 5 residual blocks with a skip connection to extract fine text details
    from upsampled low-resolution text images x_up.
    Input: x_up (B x D x H x W)
    Output: x_f (B x D x H x W)
    """
    def __init__(self, in_channels=3, mid_channels=64):
        super(FeatureExtractor, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResBlock(mid_channels, mid_channels, time_emb_dim=None) for _ in range(5)
        ])
        self.conv_out = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x_up):
        h = self.conv_in(x_up)
        h_init = h
        for block in self.blocks:
            h = block(h)
        h = h + h_init  # Main skip connection
        x_f = self.conv_out(h)
        return x_f


class CustomizedConditionalUNet(nn.Module):
    """
    Customized Conditional U-Net for OCR-Diff (Fig. 1 & Fig. 2).
    Features:
    - 10 Residual Blocks with skip connections
    - Conditioning on noisy text image X_t, extracted text feature x_f, and embedded time step tau_t
    - Linear Attention module in the bottleneck middle layer to capture spatial text dependencies
    """
    def __init__(self, in_channels=6, out_channels=3, K=64, time_emb_dim=128):
        super(CustomizedConditionalUNet, self).__init__()
        self.K = K
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * K, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Concatenation of noisy image X_t (3 channels) + extracted feature x_f (3 channels) = 6 channels
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Encoder Residual Blocks (Downsampling)
        self.down1 = ResBlock(64, 128, time_emb_dim=time_emb_dim)
        self.down2 = ResBlock(128, 256, time_emb_dim=time_emb_dim)
        self.downsample = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Bottleneck with Linear Attention
        self.mid1 = ResBlock(512, 512, time_emb_dim=time_emb_dim)
        self.attention = LinearAttention(512)
        self.mid2 = ResBlock(512, 512, time_emb_dim=time_emb_dim)

        # Decoder Residual Blocks (Upsampling)
        self.upsample = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(256 + 256, 128, time_emb_dim=time_emb_dim)
        self.up2 = ResBlock(128 + 128, 64, time_emb_dim=time_emb_dim)

        self.outc = nn.Conv2d(64 + 64, out_channels, kernel_size=3, padding=1)

    def compute_time_embedding(self, t, device):
        """
        Computes sinusoidal positional time embedding vector tau_t in R^{2K} (Eq. 2):
        tau_t = [ sin(t / 10000^{k/(K-1)}), cos(t / 10000^{k/(K-1)}) ]_{k=1}^K
        """
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)

        if t.dim() == 1:
            t = t.unsqueeze(1) # B x 1

        k = torch.arange(self.K, dtype=torch.float32, device=device) # K
        freqs = t / (10000.0 ** (k / max(1, self.K - 1)))            # B x K

        sin_emb = torch.sin(freqs)
        cos_emb = torch.cos(freqs)
        tau_t = torch.cat([sin_emb, cos_emb], dim=-1)                # B x 2K
        return tau_t

    def forward(self, X_t, x_f, t):
        """
        Input:
        - X_t: Noisy text image (B x 3 x H x W)
        - x_f: Text image feature from FeatureExtractor (B x 3 x H x W)
        - t: Time step integer/tensor
        Output:
        - Predicted noise E_theta (B x 3 x H x W)
        """
        device = X_t.device
        tau_t = self.compute_time_embedding(t, device)
        time_emb = self.time_mlp(tau_t)

        # Input concatenation (X_t, x_f)
        x_in = torch.cat([X_t, x_f], dim=1)  # B x 6 x H x W

        h_in = self.inc(x_in)                # B x 64 x H x W
        h_d1 = self.down1(h_in, time_emb)     # B x 128 x H x W
        h_d2 = self.down2(h_d1, time_emb)     # B x 256 x H x W

        h_down = self.downsample(h_d2)       # B x 512 x (H/2) x (W/2)

        # Bottleneck with Linear Attention
        h_mid = self.mid1(h_down, time_emb)
        h_mid = self.attention(h_mid)
        h_mid = self.mid2(h_mid, time_emb)

        h_up = self.upsample(h_mid)          # B x 256 x H x W

        # Skip connections
        h_u1 = self.up1(torch.cat([h_up, h_d2], dim=1), time_emb)   # B x 128 x H x W
        h_u2 = self.up2(torch.cat([h_u1, h_d1], dim=1), time_emb)   # B x 64 x H x W

        out = self.outc(torch.cat([h_u2, h_in], dim=1))              # B x 3 x H x W
        return out


class OCRDiffPipeline:
    """
    OCR-Diff Two-Stage Training & Reconstruction Manager (IEEE IoTJ 2024).
    Implements:
    - Cosine Beta Schedule (Nichol & Dhariwal)
    - Forward Diffusion corruption (Stage 1 Pretraining)
    - Pretraining loss function (MSE)
    - Reverse Diffusion sampling (Stage 2 Fine-tuning / Inference)
    - Residual reconstruction: X_hat = X_hat_0 + x_up
    - Combined Fine-tuning loss (MSE + Weighted Cross-Entropy)
    """
    def __init__(self, T=1000, K=64, device=None):
        self.T = T
        self.K = K
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network modules
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.unet = CustomizedConditionalUNet(K=K).to(self.device)

        # Compute Cosine Beta Schedule parameters (Eq. Section IV paper)
        s = 8e-3
        t_vec = torch.arange(0, T + 1, dtype=torch.float64)
        f_t = torch.cos(((t_vec / T + s) / (1 + s)) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]

        betas = []
        for i in range(1, T + 1):
            b = 1.0 - (alphas_cumprod[i] / alphas_cumprod[i - 1])
            betas.append(min(max(b.item(), 1e-5), 0.999))

        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculation for posterior variance sigma_t^2 (Eq. 4)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def forward_diffusion(self, X, t, noise=None):
        """
        Stage 1 Forward Diffusion Corruption (Eq. 1):
        X_t = sqrt(alpha_bar_t) * X + sqrt(1 - alpha_bar_t) * E
        """
        if noise is None:
            noise = torch.randn_like(X)

        alpha_bar_t = self.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
        X_t = torch.sqrt(alpha_bar_t) * X + torch.sqrt(1.0 - alpha_bar_t) * noise
        return X_t, noise

    def compute_pretrain_loss(self, X, x_up, t):
        """
        Stage 1 Pretraining Loss (Eq. 3):
        Minimizes MSE between actual noise E and predicted noise E_theta(X_t, x_f, tau_t)
        """
        x_f = self.feature_extractor(x_up)
        X_t, noise = self.forward_diffusion(X, t)
        predicted_noise = self.unet(X_t, x_f, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def reverse_diffusion_sample(self, x_up, num_steps=20):
        """
        Stage 2 Reverse Diffusion Sampling with Residual Learning (Eq. 4 & Fig. 1b):
        X_hat = X_hat_0 + x_up
        Accelerated sampling support using step stride for real-time inference.
        """
        self.feature_extractor.eval()
        self.unet.eval()

        batch_size = x_up.size(0)
        x_f = self.feature_extractor(x_up)

        # Start from Gaussian Noise X_hat_T ~ N(0, I)
        X_hat_t = torch.randn_like(x_up)

        step_stride = max(1, self.T // num_steps)
        time_steps = list(range(1, self.T + 1, step_stride))[::-1]

        for t_val in time_steps:
            t_tensor = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            pred_noise = self.unet(X_hat_t, x_f, t_tensor)

            alpha_t = self.alphas[t_val - 1]
            alpha_bar_t = self.alphas_cumprod[t_val - 1]
            beta_t = self.betas[t_val - 1]

            # Reverse step formula (Eq. 4)
            coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (X_hat_t - coef * pred_noise)

            if t_val > 1:
                sigma_t = torch.sqrt(self.posterior_variance[t_val - 1])
                noise = torch.randn_like(X_hat_t)
                X_hat_t = mean + sigma_t * noise
            else:
                X_hat_t = mean

        # Residual learning reconstruction (Eq. Section III-B): X_hat = X_hat_0 + x_up
        X_hat_0 = X_hat_t
        X_reconstructed = X_hat_0 + x_up
        return torch.clamp(X_reconstructed, 0.0, 1.0)

    def compute_finetune_loss(self, X, x_up, recognizer, target_labels, lambda_weight=5e-4):
        """
        Stage 2 Fine-tuning Loss (Eq. 5):
        Combines Image Reconstruction MSE and Recognizer Negative Cross Entropy:
        Loss = || X - X_hat ||^2 - lambda * sum(w_i * y_i * log(y_hat_i))
        """
        x_f = self.feature_extractor(x_up)
        # 1-step estimated reverse prediction for gradient backprop
        t = torch.ones((X.size(0),), device=self.device, dtype=torch.long)
        pred_noise = self.unet(X, x_f, t)

        alpha_1 = self.alphas[0]
        alpha_bar_1 = self.alphas_cumprod[0]
        coef = (1.0 - alpha_1) / torch.sqrt(1.0 - alpha_bar_1)
        X_hat_0 = (1.0 / torch.sqrt(alpha_1)) * (X - coef * pred_noise)
        X_hat = X_hat_0 + x_up

        mse_loss = F.mse_loss(X_hat, X)

        rec_loss = torch.tensor(0.0, device=self.device)
        if recognizer is not None and target_labels is not None:
            # Recognizer parameters frozen
            with torch.no_grad():
                preds = recognizer(X_hat)
            rec_loss = F.cross_entropy(preds, target_labels)

        total_loss = mse_loss + lambda_weight * rec_loss
        return total_loss, mse_loss, rec_loss

    def enhance_image_np(self, image_np, num_steps=20):
        """
        High-level inference API for NumPy OpenCV images (H x W x C, uint8, BGR/RGB).
        Returns enhanced OpenCV image (H x W x C, uint8).
        """
        is_gray = len(image_np.shape) == 2
        if is_gray:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        # Cap target diffusion dimensions per OCR-Diff paper (128 x 32 canonical text patch, max 256 x 64)
        aspect = orig_w / float(max(1, orig_h))
        target_h = 32
        target_w = int(round(target_h * aspect))
        target_w = max(64, min(256, (target_w // 8) * 8))

        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Convert to tensor [0, 1] B x C x H x W
        tensor_in = torch.tensor(resized, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            enhanced_tensor = self.reverse_diffusion_sample(tensor_in, num_steps=num_steps)

        # Convert back to NumPy uint8
        out_np = (enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Resize back to original dimensions
        out_final = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        if is_gray:
            return cv2.cvtColor(out_final, cv2.COLOR_RGB2GRAY)
        else:
            return cv2.cvtColor(out_final, cv2.COLOR_RGB2BGR)

# Singleton global instance helper
_ocr_diff_instance = None

def get_ocr_diff_pipeline():
    global _ocr_diff_instance
    if _ocr_diff_instance is None:
        _ocr_diff_instance = OCRDiffPipeline(T=100)
    return _ocr_diff_instance
