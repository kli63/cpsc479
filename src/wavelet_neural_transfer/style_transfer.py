import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from models import WaveletNeuralStyleTransfer
from utils import (
    load_image, 
    save_image, 
    create_comparison_image, 
    visualize_wavelet_decomposition,
    visualize_frequency_spectrum,
    visualize_loss_history,
    create_multi_scale_visualization
)

def parse_args():
    parser = argparse.ArgumentParser(description='Wavelet Neural Style Transfer')
    
    parser.add_argument('--content', type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to style image')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--size', type=int, default=512,
                        help='Size of output image')
    parser.add_argument('--mode', type=str, default='fast', choices=['fast', 'optimize'],
                        help='Style transfer mode (fast or optimize)')
    
    # Transfer parameters
    parser.add_argument('--alpha_low', type=float, default=1.0,
                        help='Style weight for low frequencies (0-1)')
    parser.add_argument('--alpha_high', type=float, default=1.0,
                        help='Style weight for high frequencies (0-1)')
    
    # Optimization parameters (for optimize mode)
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of optimization steps')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=1e5,
                        help='Weight for style loss')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimization')
    
    # Device
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate wavelet and frequency visualizations')
    
    return parser.parse_args()

def stylize_image(args):
    # Set device
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"wavelet_style_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    print("Loading images...")
    content_img = load_image(args.content, target_size=(args.size, args.size), device=device)
    style_img = load_image(args.style, target_size=(args.size, args.size), device=device)
    
    # Create model
    print("Creating model...")
    model = WaveletNeuralStyleTransfer(device=device)
    
    # Apply style transfer
    if args.mode == 'fast':
        print("Applying fast stylization...")
        start_time = time.time()
        stylized_img = model.fast_stylize(
            content_img, style_img, 
            alpha_low=args.alpha_low, 
            alpha_high=args.alpha_high
        )
        end_time = time.time()
        print(f"Stylization completed in {end_time - start_time:.2f} seconds")
        
        # Losses dictionary (empty for fast mode)
        losses = {}
        
    else:  # optimize mode
        print(f"Optimizing style transfer for {args.steps} steps...")
        start_time = time.time()
        stylized_img, losses = model.stylize(
            content_img, style_img,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            num_steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            lr=args.lr,
            show_every=args.steps // 10 or 1
        )
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    print("Saving results...")
    content_name = os.path.splitext(os.path.basename(args.content))[0]
    style_name = os.path.splitext(os.path.basename(args.style))[0]
    
    # Save stylized image
    result_path = os.path.join(output_dir, f"{content_name}_styled_with_{style_name}.jpg")
    save_image(stylized_img, result_path)
    
    # Create and save comparison
    comparison_path = os.path.join(output_dir, f"{content_name}_styled_with_{style_name}_comparison.jpg")
    create_comparison_image(content_img, style_img, stylized_img, comparison_path)
    
    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Wavelet decomposition visualizations
        content_wavelet_viz = visualize_wavelet_decomposition(content_img)
        plt.savefig(os.path.join(viz_dir, "content_wavelet_decomposition.jpg"))
        plt.close()
        
        style_wavelet_viz = visualize_wavelet_decomposition(style_img)
        plt.savefig(os.path.join(viz_dir, "style_wavelet_decomposition.jpg"))
        plt.close()
        
        result_wavelet_viz = visualize_wavelet_decomposition(stylized_img)
        plt.savefig(os.path.join(viz_dir, "result_wavelet_decomposition.jpg"))
        plt.close()
        
        # Frequency spectrum visualizations
        content_freq_viz = visualize_frequency_spectrum(content_img)
        plt.savefig(os.path.join(viz_dir, "content_frequency_spectrum.jpg"))
        plt.close()
        
        style_freq_viz = visualize_frequency_spectrum(style_img)
        plt.savefig(os.path.join(viz_dir, "style_frequency_spectrum.jpg"))
        plt.close()
        
        result_freq_viz = visualize_frequency_spectrum(stylized_img)
        plt.savefig(os.path.join(viz_dir, "result_frequency_spectrum.jpg"))
        plt.close()
        
        # Multi-scale visualizations
        create_multi_scale_visualization(
            content_img, 
            output_path=os.path.join(viz_dir, "content_multi_scale.jpg")
        )
        plt.close()
        
        create_multi_scale_visualization(
            stylized_img, 
            output_path=os.path.join(viz_dir, "result_multi_scale.jpg")
        )
        plt.close()
        
        # Loss history visualization (if available)
        if losses:
            loss_viz = visualize_loss_history(losses)
            plt.savefig(os.path.join(viz_dir, "loss_history.jpg"))
            plt.close()
    
    print(f"Results saved to {output_dir}")
    return result_path, comparison_path

def main():
    args = parse_args()
    result_path, comparison_path = stylize_image(args)
    print(f"Stylized image: {result_path}")
    print(f"Comparison image: {comparison_path}")

if __name__ == "__main__":
    main()