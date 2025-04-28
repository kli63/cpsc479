import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image
import cv2
import os

def load_image(image_path, target_size=None, device='cpu'):
    """Load an image and convert to tensor.
    
    Args:
        image_path: Path to the image file
        target_size: Optional size to resize to (W, H)
        device: Device to load tensor to
        
    Returns:
        Image tensor [1, 3, H, W] in range [0, 1]
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize if needed
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Add batch dimension
    tensor = transform(img).unsqueeze(0).to(device)
    
    return tensor

def tensor_to_image(tensor):
    """Convert tensor to numpy image.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        
    Returns:
        Numpy array [H, W, 3] in range [0, 255] (uint8)
    """
    # Remove batch dimension and move channels to last dimension
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Ensure range [0, 1]
    image = np.clip(image, 0, 1)
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image

def save_image(tensor, output_path):
    """Save tensor as image.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        output_path: Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert to numpy and save
    image = tensor_to_image(tensor)
    Image.fromarray(image).save(output_path)
    
    print(f"Saved image to {output_path}")

def visualize_wavelet_decomposition(tensor, wavelet='db2', level=1, figsize=(15, 10)):
    """Visualize wavelet decomposition of an image tensor.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        wavelet: Wavelet type
        level: Decomposition level
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    
    # Process each channel
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        # Get channel
        channel = img[:, :, i]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        
        # Approximation coefficients
        axes[i, 0].imshow(coeffs[0], cmap='gray')
        axes[i, 0].set_title(f"{color} Approximation")
        axes[i, 0].axis('off')
        
        # Detail coefficients
        for j, (name, array) in enumerate(zip(['Horizontal', 'Vertical', 'Diagonal'], coeffs[1])):
            axes[i, j+1].imshow(array, cmap='gray')
            axes[i, j+1].set_title(f"{color} {name} Detail")
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    return fig

def create_comparison_image(content_tensor, style_tensor, stylized_tensor, output_path=None):
    """Create a comparison image with content, style, and result.
    
    Args:
        content_tensor: Content image tensor [1, 3, H, W]
        style_tensor: Style image tensor [1, 3, H, W]
        stylized_tensor: Stylized image tensor [1, 3, H, W]
        output_path: Optional path to save the comparison image
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    content_img = tensor_to_image(content_tensor)
    style_img = tensor_to_image(style_tensor)
    stylized_img = tensor_to_image(stylized_tensor)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(content_img)
    axes[0].set_title("Content Image")
    axes[0].axis('off')
    
    axes[1].imshow(style_img)
    axes[1].set_title("Style Image")
    axes[1].axis('off')
    
    axes[2].imshow(stylized_img)
    axes[2].set_title("Stylized Result")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved comparison image to {output_path}")
    
    return fig

def visualize_frequency_spectrum(tensor, log_scale=True, figsize=(15, 5)):
    """Visualize frequency spectrum of an image tensor.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        log_scale: Whether to use log scale for magnitude
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Process each channel
    for i, (color, ax) in enumerate(zip(['Red', 'Green', 'Blue'], axes)):
        # Get channel
        channel = img[:, :, i]
        
        # Apply FFT
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude = np.abs(f_shift)
        
        # Apply log scale if needed
        if log_scale:
            magnitude = np.log1p(magnitude)
        
        # Display magnitude spectrum
        im = ax.imshow(magnitude, cmap='viridis')
        ax.set_title(f"{color} Channel Frequency Spectrum")
        ax.axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def visualize_loss_history(losses_dict, figsize=(10, 6)):
    """Visualize loss history during optimization.
    
    Args:
        losses_dict: Dictionary with loss arrays
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each loss type
    for loss_name, values in losses_dict.items():
        if values:  # Check if list is not empty
            ax.plot(values, label=loss_name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss History During Stylization')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def get_frequency_bands(tensor, n_bands=3):
    """Split image tensor into frequency bands.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        n_bands: Number of frequency bands
        
    Returns:
        List of frequency band tensors
    """
    bands = []
    batch_size, channels, height, width = tensor.shape
    
    # Process each channel separately
    for c in range(channels):
        # Get channel
        channel = tensor[:, c]
        
        # Apply FFT
        fft = torch.fft.fft2(channel)
        fft_shift = torch.fft.fftshift(fft)
        
        # Create distance matrix from center
        y, x = torch.meshgrid(
            torch.arange(height, device=tensor.device),
            torch.arange(width, device=tensor.device),
            indexing='ij'
        )
        center_y, center_x = height // 2, width // 2
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Maximum distance (radius)
        max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2, 
                                               device=tensor.device))
        
        # Create band masks
        band_masks = []
        for i in range(n_bands):
            inner_radius = max_distance * i / n_bands
            outer_radius = max_distance * (i + 1) / n_bands
            
            mask = ((distance >= inner_radius) & (distance < outer_radius)).float()
            band_masks.append(mask)
        
        # Apply masks and reconstruct
        band_tensors = []
        for mask in band_masks:
            # Apply mask to FFT
            masked_fft = fft_shift * mask.unsqueeze(0)
            
            # Inverse FFT
            inv_fft_shift = torch.fft.ifftshift(masked_fft)
            inverse = torch.fft.ifft2(inv_fft_shift)
            
            # Get real part
            result = torch.real(inverse)
            band_tensors.append(result.unsqueeze(1))  # Add channel dimension
        
        # Stack bands
        bands.append(band_tensors)
    
    # Reorganize to have n_bands tensors, each with all channels
    result_bands = []
    for i in range(n_bands):
        # Stack channels for this band
        band_channels = [bands[c][i] for c in range(channels)]
        result_bands.append(torch.cat(band_channels, dim=1))
    
    return result_bands

def create_multi_scale_visualization(tensor, n_scales=3, output_path=None, figsize=(15, 5)):
    """Create multi-scale visualization of an image using wavelet decomposition.
    
    Args:
        tensor: Image tensor [1, 3, H, W]
        n_scales: Number of scales to visualize
        output_path: Optional path to save the visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(n_scales, 4, figsize=(15, 5 * n_scales))
    
    # Apply wavelet decomposition
    coeffs = pywt.wavedecn(img, 'db2', level=n_scales)
    
    # Plot approximation
    axes[0, 0].imshow(pywt.waverecn([coeffs[0], {}], 'db2'))
    axes[0, 0].set_title("Approximation")
    axes[0, 0].axis('off')
    
    # Plot reconstructions at different scales
    for i in range(n_scales):
        # Create coefficient set with only one level
        coeff_set = [None] * (n_scales + 1)
        coeff_set[0] = np.zeros_like(coeffs[0])
        coeff_set[i + 1] = coeffs[i + 1]
        
        # Reconstruct
        scale_img = pywt.waverecn(coeff_set, 'db2')
        
        # Plot
        axes[i, 1].imshow(scale_img)
        axes[i, 1].set_title(f"Scale {i+1}")
        axes[i, 1].axis('off')
        
        # Plot frequency spectrum
        f_transform = np.fft.fft2(np.mean(scale_img, axis=2))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        
        axes[i, 2].imshow(magnitude, cmap='viridis')
        axes[i, 2].set_title(f"Frequency Spectrum Scale {i+1}")
        axes[i, 2].axis('off')
        
        # Plot edge detection (gradient magnitude)
        grad_x = cv2.Sobel(scale_img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(scale_img, cv2.CV_32F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.mean(grad_mag, axis=2)
        
        axes[i, 3].imshow(grad_mag, cmap='gray')
        axes[i, 3].set_title(f"Edge Detection Scale {i+1}")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved multi-scale visualization to {output_path}")
    
    return fig