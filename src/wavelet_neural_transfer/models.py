import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pywt
import matplotlib.pyplot as plt

class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor for perceptual loss."""
    
    def __init__(self, device='cpu'):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        self.vgg19_features = vgg19.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2", 
            '6': "relu2_1",
            '8': "relu2_2",
            '11': "relu3_1", 
            '13': "relu3_2",
            '15': "relu3_3",
            '17': "relu3_4",
            '20': "relu4_1",
            '22': "relu4_2",
            '24': "relu4_3",
            '26': "relu4_4",
            '29': "relu5_1"
        }
        
        # Freeze the network
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x, layers=None):
        """Extract features from input tensor.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            layers: List of layer names to extract features from
                   If None, extract from all layers
        
        Returns:
            Dictionary of features {layer_name: feature_maps}
        """
        if layers is None:
            layers = list(self.layer_name_mapping.values())
        
        features = {}
        for name, module in self.vgg19_features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping and self.layer_name_mapping[name] in layers:
                features[self.layer_name_mapping[name]] = x
        
        return features


class WaveletTransform(nn.Module):
    """Wavelet transform for feature decomposition."""
    
    def __init__(self, wavelet='db1', mode='symmetric', device='cpu'):
        """Initialize wavelet transform module.
        
        Args:
            wavelet: Wavelet type (db1, db2, sym2, etc.)
            mode: Signal extension mode
            device: Device to run computation
        """
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.device = device
    
    def forward(self, x):
        """Perform 2D wavelet transform.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            List of wavelet coefficients [approximation, (horizontal, vertical, diagonal)]
        """
        batch_size, channels, height, width = x.shape
        
        # Process each batch and channel separately
        all_approx = []
        all_detail_h = []
        all_detail_v = []
        all_detail_d = []
        
        for b in range(batch_size):
            batch_approx = []
            batch_detail_h = []
            batch_detail_v = []
            batch_detail_d = []
            
            for c in range(channels):
                # Get numpy array from tensor
                img_np = x[b, c].cpu().numpy()
                
                # Perform wavelet decomposition
                coeffs = pywt.dwt2(img_np, self.wavelet, mode=self.mode)
                approximation, (detail_h, detail_v, detail_d) = coeffs
                
                # Convert to tensors
                batch_approx.append(torch.from_numpy(approximation).to(self.device))
                batch_detail_h.append(torch.from_numpy(detail_h).to(self.device))
                batch_detail_v.append(torch.from_numpy(detail_v).to(self.device))
                batch_detail_d.append(torch.from_numpy(detail_d).to(self.device))
            
            # Stack channels for this batch
            all_approx.append(torch.stack(batch_approx))
            all_detail_h.append(torch.stack(batch_detail_h))
            all_detail_v.append(torch.stack(batch_detail_v))
            all_detail_d.append(torch.stack(batch_detail_d))
        
        # Stack batches
        result_approx = torch.stack(all_approx)
        result_detail_h = torch.stack(all_detail_h)
        result_detail_v = torch.stack(all_detail_v)
        result_detail_d = torch.stack(all_detail_d)
        
        # Return as a tuple: approximation and detail coefficients
        return result_approx, (result_detail_h, result_detail_v, result_detail_d)

class InverseWaveletTransform(nn.Module):
    """Inverse wavelet transform to reconstruct features."""
    
    def __init__(self, wavelet='db1', mode='symmetric', device='cpu'):
        """Initialize inverse wavelet transform module.
        
        Args:
            wavelet: Wavelet type (db1, db2, sym2, etc.)
            mode: Signal extension mode
            device: Device to run computation
        """
        super(InverseWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.device = device
    
    def forward(self, coeffs):
        """Perform inverse 2D wavelet transform.
        
        Args:
            coeffs: Tuple of (approximation, (detail_h, detail_v, detail_d))
        
        Returns:
            Reconstructed tensor [B, C, H, W]
        """
        approximation, (detail_h, detail_v, detail_d) = coeffs
        batch_size, channels = approximation.shape[:2]
        
        # Process each batch and channel separately
        all_reconstructed = []
        
        for b in range(batch_size):
            batch_reconstructed = []
            
            for c in range(channels):
                # Get numpy arrays from tensors
                approx_np = approximation[b, c].cpu().numpy()
                detail_h_np = detail_h[b, c].cpu().numpy()
                detail_v_np = detail_v[b, c].cpu().numpy()
                detail_d_np = detail_d[b, c].cpu().numpy()
                
                # Perform inverse wavelet transform
                coeffs = (approx_np, (detail_h_np, detail_v_np, detail_d_np))
                reconstructed = pywt.idwt2(coeffs, self.wavelet, mode=self.mode)
                
                # Add to batch results
                batch_reconstructed.append(torch.from_numpy(reconstructed).to(self.device))
            
            # Stack channels for this batch
            all_reconstructed.append(torch.stack(batch_reconstructed))
        
        # Stack batches
        result = torch.stack(all_reconstructed)
        
        return result


class SpectralAttention(nn.Module):
    """Frequency-selective attention module for adaptive style manipulation."""
    
    def __init__(self, channels, reduction=16):
        """Initialize spectral attention module.
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction factor for attention
        """
        super(SpectralAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Create separate MLPs for each coefficient type
        self.approx_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.detail_h_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.detail_v_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.detail_d_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, approx, details):
        """Apply frequency-selective attention.
        
        Args:
            approx: Wavelet approximation coefficients [B, C, H, W]
            details: Tuple of wavelet detail coefficients (H, V, D)
        
        Returns:
            Tuple of attention-weighted coefficients
        """
        detail_h, detail_v, detail_d = details
        
        # Calculate attention weights
        # For approximation coefficients (low frequency)
        approx_attn = self.avg_pool(approx)
        detail_h_attn = self.avg_pool(detail_h)
        detail_v_attn = self.avg_pool(detail_v)
        detail_d_attn = self.avg_pool(detail_d)
        
        # Apply individual MLPs
        approx_attn = self.approx_mlp(approx_attn)
        detail_h_attn = self.detail_h_mlp(detail_h_attn)
        detail_v_attn = self.detail_v_mlp(detail_v_attn)
        detail_d_attn = self.detail_d_mlp(detail_d_attn)
        
        # Apply sigmoid and multiply with original coefficients
        approx_out = approx * self.sigmoid(approx_attn)
        detail_h_out = detail_h * self.sigmoid(detail_h_attn)
        detail_v_out = detail_v * self.sigmoid(detail_v_attn)
        detail_d_out = detail_d * self.sigmoid(detail_d_attn)
        
        return approx_out, (detail_h_out, detail_v_out, detail_d_out)


class AdaptiveFrequencyModulation(nn.Module):
    """Adaptive frequency modulation for style transfer."""
    
    def __init__(self, device='cpu'):
        """Initialize adaptive frequency modulation module.
        
        Args:
            device: Device to run computation
        """
        super(AdaptiveFrequencyModulation, self).__init__()
        self.device = device
    
    def histogram_matching(self, source, target):
        """Match histograms between source and target tensors.
        
        Args:
            source: Source tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
        
        Returns:
            Source tensor with histogram matched to target
        """
        batch_size, channels, height, width = source.shape
        result = torch.zeros_like(source)
        
        for b in range(batch_size):
            for c in range(channels):
                src = source[b, c].view(-1).detach().cpu().numpy()
                tgt = target[b, c].view(-1).detach().cpu().numpy()
                
                # Get sorted indices
                src_sorted = np.sort(src)
                tgt_sorted = np.sort(tgt)
                
                # Map source values to target values using histogram matching
                src_ranks = np.searchsorted(src_sorted, src)
                src_ranks = np.clip(src_ranks, 0, len(src_sorted) - 1)
                
                # Calculate mapped values
                mapped_values = np.interp(src_ranks, 
                                          np.linspace(0, len(tgt_sorted) - 1, len(src_ranks)),
                                          tgt_sorted)
                
                # Reshape back to original shape
                result[b, c] = torch.from_numpy(mapped_values.reshape(height, width)).to(self.device)
        
        return result
    
    def phase_preserving_style_transfer(self, content_coeffs, style_coeffs, alpha_low=0.8, alpha_high=0.4):
        content_approx, content_details = content_coeffs
        style_approx, style_details = style_coeffs
        content_detail_h, content_detail_v, content_detail_d = content_details
        style_detail_h, style_detail_v, style_detail_d = style_details
        
        content_approx_mag = torch.abs(content_approx)
        style_approx_mag = torch.abs(style_approx)
        content_approx_phase = torch.angle(content_approx.to(torch.complex64))
        style_approx_phase = torch.angle(style_approx.to(torch.complex64))
        
        matched_approx_mag = self.histogram_matching(content_approx_mag, style_approx_mag)
        
        blend_ratio = 0.2
        blended_phase = (1 - blend_ratio) * content_approx_phase + blend_ratio * style_approx_phase
        
        stylized_approx = matched_approx_mag * torch.exp(1j * blended_phase).real
        
        content_detail_h_mag = torch.abs(content_detail_h)
        content_detail_v_mag = torch.abs(content_detail_v)
        content_detail_d_mag = torch.abs(content_detail_d)
        
        style_detail_h_mag = torch.abs(style_detail_h)
        style_detail_v_mag = torch.abs(style_detail_v)
        style_detail_d_mag = torch.abs(style_detail_d)
        
        style_scale = 1.8
        detail_h_ratio = torch.mean(style_detail_h_mag) / torch.mean(content_detail_h_mag)
        detail_v_ratio = torch.mean(style_detail_v_mag) / torch.mean(content_detail_v_mag)
        detail_d_ratio = torch.mean(style_detail_d_mag) / torch.mean(content_detail_d_mag)
        
        scaled_h_mag = content_detail_h_mag * detail_h_ratio * style_scale
        scaled_v_mag = content_detail_v_mag * detail_v_ratio * style_scale
        scaled_d_mag = content_detail_d_mag * detail_d_ratio * style_scale
        
        content_detail_h_phase = torch.angle(content_detail_h.to(torch.complex64))
        content_detail_v_phase = torch.angle(content_detail_v.to(torch.complex64))
        content_detail_d_phase = torch.angle(content_detail_d.to(torch.complex64))
        
        style_detail_h_phase = torch.angle(style_detail_h.to(torch.complex64))
        style_detail_v_phase = torch.angle(style_detail_v.to(torch.complex64))
        style_detail_d_phase = torch.angle(style_detail_d.to(torch.complex64))
        
        detail_blend = 0.3
        h_phase = (1 - detail_blend) * content_detail_h_phase + detail_blend * style_detail_h_phase
        v_phase = (1 - detail_blend) * content_detail_v_phase + detail_blend * style_detail_v_phase
        d_phase = (1 - detail_blend) * content_detail_d_phase + detail_blend * style_detail_d_phase
        
        stylized_detail_h = scaled_h_mag * torch.exp(1j * h_phase).real
        stylized_detail_v = scaled_v_mag * torch.exp(1j * v_phase).real
        stylized_detail_d = scaled_d_mag * torch.exp(1j * d_phase).real
        
        stylized_details = (stylized_detail_h, stylized_detail_v, stylized_detail_d)
        
        return stylized_approx, stylized_details
    
    def process_detail_coeff(self, content_detail, style_detail, alpha):
        matched_detail = self.histogram_matching(content_detail, style_detail)
        
        content_magnitude = torch.abs(content_detail)
        style_magnitude = torch.abs(style_detail)
        
        magnitude_ratio = style_magnitude / (content_magnitude + 1e-6)
        enhanced_style = matched_detail * magnitude_ratio * 2.0
        
        stylized_detail = enhanced_style
        
        return stylized_detail


class FrequencyConsistencyLoss(nn.Module):
    """Loss function to enforce consistency between frequency components."""
    
    def __init__(self):
        super(FrequencyConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, stylized_coeffs, content_coeffs, lambda_approx=1.0, lambda_detail=0.5):
        """Calculate frequency consistency loss.
        
        Args:
            stylized_coeffs: Stylized wavelet coefficients (approx, (detail_h, detail_v, detail_d))
            content_coeffs: Content wavelet coefficients (approx, (detail_h, detail_v, detail_d))
            lambda_approx: Weight for approximation consistency
            lambda_detail: Weight for detail consistency
        
        Returns:
            Consistency loss value
        """
        stylized_approx, stylized_details = stylized_coeffs
        content_approx, content_details = content_coeffs
        
        # Calculate approximation consistency loss
        approx_loss = self.mse(stylized_approx, content_approx)
        
        # Calculate detail consistency loss
        s_detail_h, s_detail_v, s_detail_d = stylized_details
        c_detail_h, c_detail_v, c_detail_d = content_details
        
        detail_h_loss = self.mse(s_detail_h, c_detail_h)
        detail_v_loss = self.mse(s_detail_v, c_detail_v)
        detail_d_loss = self.mse(s_detail_d, c_detail_d)
        
        detail_loss = (detail_h_loss + detail_v_loss + detail_d_loss) / 3.0
        
        # Combine losses
        total_loss = lambda_approx * approx_loss + lambda_detail * detail_loss
        
        return total_loss


class WaveletNeuralStyleTransfer(nn.Module):
    """Wavelet Neural Style Transfer model combining pretrained features and wavelet processing."""
    
    def __init__(self, device='cpu'):
        """Initialize the model.
        
        Args:
            device: Device to run computation
        """
        super(WaveletNeuralStyleTransfer, self).__init__()
        self.device = device
        
        # Feature extractor
        self.vgg = VGG19FeatureExtractor(device=device)
        
        # Content and style feature layers
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        # Wavelet transform modules
        self.wavelet = 'db2'  # Daubechies wavelets
        self.wavelet_transform = WaveletTransform(wavelet=self.wavelet, device=device)
        self.inverse_wavelet_transform = InverseWaveletTransform(wavelet=self.wavelet, device=device)
        
        # Spectral attention for different channel sizes
        self.spectral_attention_64 = SpectralAttention(channels=64).to(device)
        self.spectral_attention_128 = SpectralAttention(channels=128).to(device)
        self.spectral_attention_256 = SpectralAttention(channels=256).to(device)
        self.spectral_attention_512 = SpectralAttention(channels=512).to(device)
        
        # Adaptive frequency modulation
        self.adaptive_freq_mod = AdaptiveFrequencyModulation(device=device)
        
        # Consistency loss
        self.consistency_loss = FrequencyConsistencyLoss()
        
        # Preprocessing transform
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def preprocess_input(self, x):
        """Preprocess input images to match VGG expected range.
        
        Args:
            x: Input tensor in range [0, 1]
            
        Returns:
            Preprocessed tensor
        """
        return self.preprocess(x)
    
    def extract_features(self, x, layers=None):
        """Extract features from input using VGG.
        
        Args:
            x: Input tensor
            layers: List of layer names to extract features from
            
        Returns:
            Dictionary of features {layer_name: feature_maps}
        """
        # Preprocess input
        x = self.preprocess_input(x)
        
        # Extract features
        return self.vgg(x, layers)
    
    def wavelet_decompose_features(self, features):
        """Apply wavelet transform to feature maps.
        
        Args:
            features: Dictionary of feature maps {layer_name: feature_maps}
            
        Returns:
            Dictionary of wavelet coefficients {layer_name: (approx, details)}
        """
        wavelet_features = {}
        
        for layer_name, feature in features.items():
            wavelet_features[layer_name] = self.wavelet_transform(feature)
        
        return wavelet_features
    
    def apply_style_transfer(self, content_wavelet_features, style_wavelet_features, 
                           alpha_low=0.8, alpha_high=0.4):
        """Apply style transfer in wavelet domain.
        
        Args:
            content_wavelet_features: Content wavelet coefficients
            style_wavelet_features: Style wavelet coefficients
            alpha_low: Weight for style in low frequencies
            alpha_high: Weight for style in high frequencies
            
        Returns:
            Dictionary of stylized wavelet coefficients
        """
        stylized_wavelet_features = {}
        
        # Process each layer
        for layer_name in content_wavelet_features.keys():
            if layer_name in self.content_layers:
                # For content layers, apply mild style transfer
                alpha_l = alpha_low * 0.5
                alpha_h = alpha_high * 0.5
            else:
                # For style layers, apply stronger style transfer
                alpha_l = alpha_low
                alpha_h = alpha_high
            
            # Apply phase-preserving style transfer
            stylized_coeffs = self.adaptive_freq_mod.phase_preserving_style_transfer(
                content_wavelet_features[layer_name],
                style_wavelet_features[layer_name],
                alpha_l, alpha_h
            )
            
            # Apply spectral attention if it's a style layer
            if layer_name in self.style_layers:
                # Choose the appropriate spectral attention based on channel count
                channels = stylized_coeffs[0].shape[1]
                if channels == 64:
                    attention = self.spectral_attention_64
                elif channels == 128:
                    attention = self.spectral_attention_128
                elif channels == 256:
                    attention = self.spectral_attention_256
                elif channels == 512:
                    attention = self.spectral_attention_512
                else:
                    # Skip attention if channel count doesn't match
                    continue
                    
                stylized_coeffs = attention(
                    stylized_coeffs[0], stylized_coeffs[1])
            
            stylized_wavelet_features[layer_name] = stylized_coeffs
        
        return stylized_wavelet_features
    
    def reconstruct_features(self, wavelet_features):
        """Reconstruct feature maps from wavelet coefficients.
        
        Args:
            wavelet_features: Dictionary of wavelet coefficients
            
        Returns:
            Dictionary of reconstructed feature maps
        """
        reconstructed_features = {}
        
        for layer_name, coeffs in wavelet_features.items():
            reconstructed_features[layer_name] = self.inverse_wavelet_transform(coeffs)
        
        return reconstructed_features
    
    def stylize(self, content_img, style_img, alpha_low=0.8, alpha_high=0.4, num_steps=1000, 
               content_weight=1.0, style_weight=1e5, consistency_weight=1e-3, 
               lr=0.01, show_every=100):
        """Stylize content image with style image.
        
        Args:
            content_img: Content image tensor [1, 3, H, W] in range [0, 1]
            style_img: Style image tensor [1, 3, H, W] in range [0, 1]
            alpha_low: Weight for style in low frequencies
            alpha_high: Weight for style in high frequencies
            num_steps: Number of optimization steps
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            consistency_weight: Weight for frequency consistency loss
            lr: Learning rate for optimization
            show_every: How often to show intermediate results
            
        Returns:
            Stylized image tensor [1, 3, H, W] in range [0, 1]
        """
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        # Initialize with content image
        stylized_img = content_img.clone().requires_grad_(True)
        
        # Extract features
        content_features = self.extract_features(content_img, 
                                             self.content_layers + self.style_layers)
        style_features = self.extract_features(style_img, self.style_layers)
        
        # Decompose features with wavelet transform
        content_wavelet_features = self.wavelet_decompose_features(content_features)
        style_wavelet_features = self.wavelet_decompose_features(style_features)
        
        # Setup optimizer
        optimizer = torch.optim.LBFGS([stylized_img], lr=lr)
        
        # Track losses
        content_losses = []
        style_losses = []
        consistency_losses = []
        total_losses = []
        
        step = [0]  # Use list to allow modification in closure
        
        def get_stylized_features():
            """Get features from current stylized image."""
            return self.extract_features(stylized_img, 
                                     self.content_layers + self.style_layers)
        
        while step[0] <= num_steps:
            def closure():
                optimizer.zero_grad()
                
                # Get current stylized features
                stylized_features = get_stylized_features()
                
                # Decompose with wavelet transform
                stylized_wavelet_features = self.wavelet_decompose_features(stylized_features)
                
                # Calculate content loss
                content_loss = 0
                for layer in self.content_layers:
                    content_loss += F.mse_loss(
                        stylized_features[layer], content_features[layer])
                content_loss *= content_weight
                
                # Calculate style loss (gram matrix)
                style_loss = 0
                for layer in self.style_layers:
                    s_feature = stylized_features[layer]
                    c_feature = style_features[layer]
                    
                    # Reshape and calculate gram matrices
                    b, c, h, w = s_feature.size()
                    s_view = s_feature.view(b, c, h * w)
                    s_gram = torch.bmm(s_view, s_view.transpose(1, 2)) / (c * h * w)
                    
                    c_view = c_feature.view(b, c, h * w)
                    c_gram = torch.bmm(c_view, c_view.transpose(1, 2)) / (c * h * w)
                    
                    style_loss += F.mse_loss(s_gram, c_gram)
                style_loss *= style_weight
                
                # Calculate frequency consistency loss
                consistency_loss = 0
                for layer in self.content_layers:
                    consistency_loss += self.consistency_loss(
                        stylized_wavelet_features[layer],
                        content_wavelet_features[layer]
                    )
                consistency_loss *= consistency_weight
                
                # Total loss
                total_loss = content_loss + style_loss + consistency_loss
                
                # Backward pass
                total_loss.backward()
                
                # Store losses
                content_losses.append(content_loss.item())
                style_losses.append(style_loss.item())
                consistency_losses.append(consistency_loss.item())
                total_losses.append(total_loss.item())
                
                # Print progress
                if step[0] % show_every == 0:
                    print(f"Step {step[0]}/{num_steps}")
                    print(f"Content Loss: {content_loss.item():.4f}")
                    print(f"Style Loss: {style_loss.item():.4f}")
                    print(f"Consistency Loss: {consistency_loss.item():.4f}")
                    print(f"Total Loss: {total_loss.item():.4f}")
                    print()
                
                step[0] += 1
                
                return total_loss
            
            optimizer.step(closure)
            
            # Break if completed all steps
            if step[0] > num_steps:
                break
        
        # Ensure output is in valid range
        stylized_img_output = torch.clamp(stylized_img.detach(), 0, 1)
        
        return stylized_img_output, {
            'content_losses': content_losses,
            'style_losses': style_losses,
            'consistency_losses': consistency_losses,
            'total_losses': total_losses
        }
    
    def fast_stylize(self, content_img, style_img, alpha_low=1.0, alpha_high=1.0):
        """Fast stylization without optimization (direct feature manipulation).
        
        Args:
            content_img: Content image tensor [1, 3, H, W] in range [0, 1]
            style_img: Style image tensor [1, 3, H, W] in range [0, 1]
            alpha_low: Weight for style in low frequencies
            alpha_high: Weight for style in high frequencies
            
        Returns:
            Stylized image tensor [1, 3, H, W] in range [0, 1]
        """
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        with torch.no_grad():
            content_img_wavelet = self.wavelet_transform(content_img)
            style_img_wavelet = self.wavelet_transform(style_img)
            
            stylized_img_wavelet = self.adaptive_freq_mod.phase_preserving_style_transfer(
                content_img_wavelet, style_img_wavelet, 0.8, 0.6)
            
            stylized_img = self.inverse_wavelet_transform(stylized_img_wavelet)
            
            # Ensure output is in valid range
            stylized_img = torch.clamp(stylized_img, 0, 1)
        
        return stylized_img