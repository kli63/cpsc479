# Style transfer model implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pywt
import matplotlib.pyplot as plt
import copy

# Feature extraction and wavelet transform modules
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19FeatureExtractor, self).__init__()
        # Ensure VGG is loaded to the correct device
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        # Use standard ReLU layer names for loss calculation
        self.relu_layer_indices = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu4_1': 20, 'relu5_1': 29}
        # Keep only layers up to the maximum needed index
        max_needed_index = max(self.relu_layer_indices.values())
        self.vgg_features = nn.Sequential(*list(vgg19.children())[:max_needed_index + 1])

        # Freeze the network
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layers_to_extract):
        features = {}
        current_index = -1
        # Map layer names to indices for easier iteration
        target_indices = {self.relu_layer_indices[name] for name in layers_to_extract if name in self.relu_layer_indices}
        layer_name_map = {v: k for k, v in self.relu_layer_indices.items()}

        for module in self.vgg_features:
            current_index += 1
            x = module(x)
            if current_index in target_indices:
                layer_name = layer_name_map[current_index]
                features[layer_name] = x
            # Early exit if all features are collected
            if len(features) == len(layers_to_extract):
                break
        return features


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', mode='symmetric', device='cpu'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.device = device

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        all_approx, all_detail_h, all_detail_v, all_detail_d = [], [], [], []

        for b in range(batch_size):
            batch_approx, batch_detail_h, batch_detail_v, batch_detail_d = [], [], [], []
            for c in range(channels):
                img_np = x[b, c].detach().cpu().numpy()
                try:
                    coeffs = pywt.dwt2(img_np, self.wavelet, mode=self.mode)
                except ValueError as e:
                    # print(f"Error during pywt.dwt2: {e}, Input shape: {img_np.shape}")
                    h_half, w_half = (height + 1) // 2, (width + 1) // 2
                    approximation = np.zeros((h_half, w_half), dtype=img_np.dtype)
                    detail_h = np.zeros_like(approximation)
                    detail_v = np.zeros_like(approximation)
                    detail_d = np.zeros_like(approximation)
                    coeffs = approximation, (detail_h, detail_v, detail_d)

                approximation, (detail_h, detail_v, detail_d) = coeffs
                batch_approx.append(torch.from_numpy(approximation.copy()).to(self.device))
                batch_detail_h.append(torch.from_numpy(detail_h.copy()).to(self.device))
                batch_detail_v.append(torch.from_numpy(detail_v.copy()).to(self.device))
                batch_detail_d.append(torch.from_numpy(detail_d.copy()).to(self.device))

            all_approx.append(torch.stack(batch_approx))
            all_detail_h.append(torch.stack(batch_detail_h))
            all_detail_v.append(torch.stack(batch_detail_v))
            all_detail_d.append(torch.stack(batch_detail_d))

        result_approx = torch.stack(all_approx)
        result_detail_h = torch.stack(all_detail_h)
        result_detail_v = torch.stack(all_detail_v)
        result_detail_d = torch.stack(all_detail_d)
        # Return: LL, (LH, HL, HH)
        return result_approx, (result_detail_h, result_detail_v, result_detail_d)


class InverseWaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', mode='symmetric', device='cpu'):
        super(InverseWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.device = device

    def forward(self, coeffs):
        approximation, (detail_h, detail_v, detail_d) = coeffs
        # Handle potential tensors that require grad - detach before numpy
        approximation_d = approximation.detach()
        detail_h_d = detail_h.detach()
        detail_v_d = detail_v.detach()
        detail_d_d = detail_d.detach()

        batch_size, channels = approximation_d.shape[:2]
        all_reconstructed = []

        for b in range(batch_size):
            batch_reconstructed = []
            for c in range(channels):
                approx_np = approximation_d[b, c].cpu().numpy()
                detail_h_np = detail_h_d[b, c].cpu().numpy()
                detail_v_np = detail_v_d[b, c].cpu().numpy()
                detail_d_np = detail_d_d[b, c].cpu().numpy()

                expected_shape = approx_np.shape
                if detail_h_np.shape != expected_shape: detail_h_np = detail_h_np[:expected_shape[0], :expected_shape[1]]
                if detail_v_np.shape != expected_shape: detail_v_np = detail_v_np[:expected_shape[0], :expected_shape[1]]
                if detail_d_np.shape != expected_shape: detail_d_np = detail_d_np[:expected_shape[0], :expected_shape[1]]

                coeffs_np = (approx_np, (detail_h_np, detail_v_np, detail_d_np))
                try:
                    reconstructed = pywt.idwt2(coeffs_np, self.wavelet, mode=self.mode)
                except ValueError as e:
                    # print(f"Error during pywt.idwt2: {e}")
                    target_h, target_w = approx_np.shape[0] * 2, approx_np.shape[1] * 2
                    reconstructed = np.zeros((target_h, target_w), dtype=approx_np.dtype)

                batch_reconstructed.append(torch.from_numpy(reconstructed.copy()).to(self.device))
            all_reconstructed.append(torch.stack(batch_reconstructed))
        result = torch.stack(all_reconstructed)
        return result


class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.channels = channels
        # Prevent reduction factor making channels < 1
        internal_channels = max(1, channels // reduction if reduction > 0 else channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.approx_mlp = nn.Sequential(
            nn.Conv2d(self.channels, internal_channels, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, self.channels, 1, bias=False)
        )
        self.detail_h_mlp = nn.Sequential(
            nn.Conv2d(self.channels, internal_channels, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, self.channels, 1, bias=False)
        )
        self.detail_v_mlp = nn.Sequential(
            nn.Conv2d(self.channels, internal_channels, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, self.channels, 1, bias=False)
        )
        self.detail_d_mlp = nn.Sequential(
            nn.Conv2d(self.channels, internal_channels, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, self.channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, approx, details):
        detail_h, detail_v, detail_d = details
        approx_attn = self.sigmoid(self.approx_mlp(self.avg_pool(approx)))
        detail_h_attn = self.sigmoid(self.detail_h_mlp(self.avg_pool(detail_h)))
        detail_v_attn = self.sigmoid(self.detail_v_mlp(self.avg_pool(detail_v)))
        detail_d_attn = self.sigmoid(self.detail_d_mlp(self.avg_pool(detail_d)))
        approx_out = approx * approx_attn
        detail_h_out = detail_h * detail_h_attn
        detail_v_out = detail_v * detail_v_attn
        detail_d_out = detail_d * detail_d_attn
        return approx_out, (detail_h_out, detail_v_out, detail_d_out)


class AdaptiveFrequencyModulation(nn.Module):
    def __init__(self, device='cpu'):
        super(AdaptiveFrequencyModulation, self).__init__()
        self.device = device

    def histogram_matching(self, source, target):
        batch_size, channels, height, width = source.shape
        result = torch.zeros_like(source)
        # Detach source and target from graph before numpy conversion for hist matching
        source_cpu = source.detach().cpu()
        target_cpu = target.detach().cpu()

        for b in range(batch_size):
            for c in range(channels):
                src = source_cpu[b, c].view(-1).numpy()
                tgt = target_cpu[b, c].view(-1).numpy()

                # Avoid issues with empty arrays
                if src.size == 0 or tgt.size == 0:
                    mapped_values = np.zeros_like(src)
                else:
                    src_sorted_indices = np.argsort(src)
                    tgt_sorted = np.sort(tgt)
                    mapped_values = np.zeros_like(src)
                    # Interpolate target values onto source ranks
                    mapped_values_sorted = np.interp(np.linspace(0, 1, len(src)), np.linspace(0, 1, len(tgt)), tgt_sorted)
                    # Place mapped values back in original order
                    mapped_values[src_sorted_indices] = mapped_values_sorted

                result[b, c] = torch.from_numpy(mapped_values.reshape(height, width)).to(self.device)
        return result

    def phase_preserving_style_transfer(self, content_coeffs, style_coeffs, alpha_low=0.8, alpha_high=0.4):
        content_approx, content_details = content_coeffs
        style_approx, style_details = style_coeffs
        content_detail_h, content_detail_v, content_detail_d = content_details
        style_detail_h, style_detail_v, style_detail_d = style_details

        # Approx: Histogram matching on magnitude, phase blending
        content_approx_mag = torch.abs(content_approx)
        style_approx_mag = torch.abs(style_approx)
        matched_approx_mag = self.histogram_matching(content_approx_mag, style_approx_mag)
        content_approx_complex = content_approx.to(torch.complex64) if not torch.is_complex(content_approx) else content_approx
        style_approx_complex = style_approx.to(torch.complex64) if not torch.is_complex(style_approx) else style_approx
        content_approx_phase = torch.angle(content_approx_complex)
        style_approx_phase = torch.angle(style_approx_complex)
        phase_blend = alpha_low
        blended_phase = phase_blend * content_approx_phase + (1 - phase_blend) * style_approx_phase
        stylized_approx_complex = torch.polar(matched_approx_mag, blended_phase)
        stylized_approx = stylized_approx_complex.real

        # Detail: Magnitude and phase blending
        def process_detail(content_d, style_d, alpha):
            content_mag = torch.abs(content_d)
            style_mag = torch.abs(style_d)
            mag_blend = alpha
            blended_mag = mag_blend * content_mag + (1 - mag_blend) * style_mag
            content_complex = content_d.to(torch.complex64) if not torch.is_complex(content_d) else content_d
            style_complex = style_d.to(torch.complex64) if not torch.is_complex(style_d) else style_d
            content_phase = torch.angle(content_complex)
            style_phase = torch.angle(style_complex)
            phase_blend_d = alpha
            blended_phase_d = phase_blend_d * content_phase + (1 - phase_blend_d) * style_phase
            stylized_complex = torch.polar(blended_mag, blended_phase_d)
            return stylized_complex.real

        stylized_detail_h = process_detail(content_detail_h, style_detail_h, alpha=alpha_high)
        stylized_detail_v = process_detail(content_detail_v, style_detail_v, alpha=alpha_high)
        stylized_detail_d = process_detail(content_detail_d, style_detail_d, alpha=alpha_high)

        stylized_details = (stylized_detail_h, stylized_detail_v, stylized_detail_d)
        return stylized_approx, stylized_details

# Wavelet pooling and unpooling modules

class WaveletPool(nn.Module):
    def __init__(self, channels, wavelet='db1', device='cpu'):
        super(WaveletPool, self).__init__()
        self.wavelet = wavelet
        self.device = device
        self.dwt = WaveletTransform(wavelet=self.wavelet, device=self.device)

    def forward(self, x):
        ll, hf_details = self.dwt(x)
        return ll, hf_details # Returns LL, (LH, HL, HH)

class WaveletUnpool(nn.Module):
    def __init__(self, channels, wavelet='db1', device='cpu'):
        super(WaveletUnpool, self).__init__()
        self.wavelet = wavelet
        self.device = device
        self.idwt = InverseWaveletTransform(wavelet=self.wavelet, device=self.device)

    def forward(self, ll, hf_details):
        if hf_details is None:
            b, c, h_half, w_half = ll.shape
            zeros = torch.zeros_like(ll)
            hf_details = (zeros, zeros, zeros)

        reconstructed = self.idwt((ll, hf_details))

        # Adjust size if IDWT output size mismatch
        b, c, h_in, w_in = ll.shape
        b, c, h_out, w_out = reconstructed.shape
        target_h, target_w = h_in * 2, w_in * 2
        if h_out != target_h or w_out != target_w:
             reconstructed = F.interpolate(reconstructed, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return reconstructed


# Main style transfer model

class WaveletOptimizedStyleTransfer(nn.Module):
    """Style transfer model with wavelet-integrated optimization."""
    def __init__(self, device='cpu', wavelet='db1', modulation_wavelet='db2'):
        super(WaveletOptimizedStyleTransfer, self).__init__()
        self.device = device
        self.wavelet = wavelet
        self.modulation_wavelet_name = modulation_wavelet

        # Model architecture components
        vgg19_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.enc_1 = nn.Sequential(*list(vgg19_features.children())[:2])
        self.enc_2 = nn.Sequential(*list(vgg19_features.children())[2:7])
        self.enc_3 = nn.Sequential(*list(vgg19_features.children())[7:12])
        self.enc_4 = nn.Sequential(*list(vgg19_features.children())[12:21])
        # Wavelet pooling
        self.pool_1 = WaveletPool(64, wavelet=self.wavelet, device=device)
        self.pool_2 = WaveletPool(128, wavelet=self.wavelet, device=device)
        self.pool_3 = WaveletPool(256, wavelet=self.wavelet, device=device)
        # Decoder
        self.dec_4 = self._make_decoder_layer(list(vgg19_features.children())[12:21])
        self.dec_3 = self._make_decoder_layer(list(vgg19_features.children())[7:12])
        self.dec_2 = self._make_decoder_layer(list(vgg19_features.children())[2:7])
        self.dec_1 = self._make_decoder_layer(list(vgg19_features.children())[:2])
        # Wavelet unpooling
        self.unpool_3 = WaveletUnpool(256, wavelet=self.wavelet, device=device)
        self.unpool_2 = WaveletUnpool(128, wavelet=self.wavelet, device=device)
        self.unpool_1 = WaveletUnpool(64, wavelet=self.wavelet, device=device)
        # Frequency modulation
        self.adaptive_freq_mod = AdaptiveFrequencyModulation(device=device)
        self.spectral_attention_modules = nn.ModuleDict({
            '64': SpectralAttention(channels=64, reduction=4).to(device),
            '128': SpectralAttention(channels=128, reduction=8).to(device),
            '256': SpectralAttention(channels=256, reduction=16).to(device),
            '512': SpectralAttention(channels=512, reduction=16).to(device)
        })
        # Loss calculation components
        self.loss_feature_extractor = VGG19FeatureExtractor(device=device)
        self.content_layers = ['relu4_1']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mod_wavelet_transform = WaveletTransform(wavelet=self.modulation_wavelet_name, device=device)
        self.gram_matrix = self._gram_matrix


    def _gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
        return gram

    def _make_decoder_layer(self, layer_group_list):
        modules = []
        for module in reversed(layer_group_list):
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv2d(module.out_channels, module.in_channels,
                                     kernel_size=module.kernel_size, padding=module.padding)
                modules.append(new_conv)
            elif isinstance(module, nn.ReLU):
                 modules.append(nn.ReLU(inplace=False))
        return nn.Sequential(*modules)

    def _calculate_mean_std(self, tensor, eps=1e-5):
        b, c, h, w = tensor.shape
        tensor_flat = tensor.view(b, c, -1)
        mean = tensor_flat.mean(dim=2, keepdim=True)
        std = tensor_flat.std(dim=2, keepdim=True) + eps
        return mean.squeeze(2), std.squeeze(2) # Return shape B, C


    def stylize_optimize(self, content_img, style_img, num_steps=500,
                         lr=1.0,
                         content_weight=1.0, style_weight=1e5,
                         wavelet_content_weight=1e0, wavelet_style_weight=1e2,
                         show_every=100):
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        stylized_img = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([stylized_img], lr=lr, max_iter=1, line_search_fn="strong_wolfe")

        # Precompute targets
        with torch.no_grad():
            style_img_p = self.preprocess(style_img)
            target_style_features = self.loss_feature_extractor(style_img_p, self.style_layers)
            target_style_grams = {layer: self.gram_matrix(feat).detach() for layer, feat in target_style_features.items()}

            content_img_p = self.preprocess(content_img)
            target_content_features = self.loss_feature_extractor(content_img_p, self.content_layers)
            target_content_features = {layer: feat.detach() for layer, feat in target_content_features.items()}

        # Wavelet targets
        with torch.no_grad():
            target_content_coeffs = self.mod_wavelet_transform(content_img) # (LL, (LH, HL, HH))
            target_style_coeffs = self.mod_wavelet_transform(style_img)     # (LL, (LH, HL, HH))
            target_style_hf_stats = []
            for hf_band in target_style_coeffs[1]:
                mean, std = self._calculate_mean_std(hf_band)
                target_style_hf_stats.append((mean.detach(), std.detach()))

        print(f"Starting optimization for {num_steps} steps (VGG + Wavelet Losses)...")
        step = [0]
        losses_log = {'content': [], 'style': [], 'wavelet_content': [], 'wavelet_style': [], 'total': []}

        while step[0] < num_steps:
            def closure():
                optimizer.zero_grad()
                stylized_img.data.clamp_(0, 1)

                # VGG losses
                stylized_img_p = self.preprocess(stylized_img)
                current_features = self.loss_feature_extractor(stylized_img_p, self.content_layers + self.style_layers)
                content_loss = 0
                for layer in self.content_layers:
                    content_loss += F.mse_loss(current_features[layer], target_content_features[layer])
                content_loss *= content_weight
                style_loss = 0
                for layer in self.style_layers:
                    style_loss += F.mse_loss(self.gram_matrix(current_features[layer]), target_style_grams[layer])
                style_loss *= style_weight

                # Wavelet losses
                wavelet_content_loss = torch.tensor(0.0, device=self.device)
                wavelet_style_loss = torch.tensor(0.0, device=self.device)
                try:
                    current_coeffs = self.mod_wavelet_transform(stylized_img)
                    current_ll, current_hf_bands = current_coeffs
                    # Wavelet Content Loss (LL Band)
                    wavelet_content_loss = F.mse_loss(current_ll, target_content_coeffs[0].detach()) * wavelet_content_weight
                    # Wavelet Style Loss (HF Bands - match mean/std)
                    temp_wave_style_loss = 0
                    for i in range(3): # LH, HL, HH
                        current_mean, current_std = self._calculate_mean_std(current_hf_bands[i])
                        target_mean, target_std = target_style_hf_stats[i]
                        temp_wave_style_loss += F.mse_loss(current_mean, target_mean)
                        temp_wave_style_loss += F.mse_loss(current_std, target_std)
                    wavelet_style_loss = (temp_wave_style_loss / 3.0) * wavelet_style_weight # Average over 3 bands

                except Exception as e:
                    print(f"\nWarning: Error calculating wavelet loss: {e}")

                # --- Total Loss ---
                total_loss = content_loss + style_loss + wavelet_content_loss + wavelet_style_loss

                # Check for NaN/Inf loss
                if not torch.isfinite(total_loss):
                     print(f"\nWarning: Non-finite total loss detected at step {step[0]}. Stopping closure.")
                     # Optionally return None or a previous loss to prevent optimizer step with bad grads
                     # Returning 0 loss to prevent crash, but indicates an issue
                     return torch.tensor(0.0, device=self.device, requires_grad=True)


                total_loss.backward()

                # Log losses only if loss is finite
                losses_log['content'].append(content_loss.item())
                losses_log['style'].append(style_loss.item())
                losses_log['wavelet_content'].append(wavelet_content_loss.item())
                losses_log['wavelet_style'].append(wavelet_style_loss.item())
                losses_log['total'].append(total_loss.item())

                if step[0] % show_every == 0:
                    print(f"Step {step[0]}/{num_steps} | Total: {total_loss.item():.2f} "
                          f"| VGG_C: {content_loss.item():.2f} | VGG_S: {style_loss.item():.2f} "
                          f"| Wav_C: {wavelet_content_loss.item():.2f} | Wav_S: {wavelet_style_loss.item():.2f}")

                step[0] += 1
                if step[0] >= num_steps:
                     print("Optimization finished.")

                return total_loss

            optimizer.step(closure)
            # Add a check to break if loss becomes NaN/Inf outside closure as well
            if not np.isfinite(losses_log['total'][-1]):
                 print(f"Error: Non-finite loss encountered at step {step[0]-1}. Stopping optimization.")
                 break

        stylized_img_final = torch.clamp(stylized_img.detach(), 0, 1)
        losses_dict_viz = {
            'VGG Content Loss': losses_log['content'],
            'VGG Style Loss': losses_log['style'],
            'Wavelet Content Loss': losses_log['wavelet_content'],
            'Wavelet Style Loss': losses_log['wavelet_style'],
            'Total Loss': losses_log['total']
        }
        return stylized_img_final, losses_dict_viz

    # Note: The 'forward', 'encode', 'decode', 'apply_hybrid_transfer' methods
    # are now unused by the style_transfer.py script but are kept here
    # as they define the feed-forward architecture if needed later (e.g., for training).