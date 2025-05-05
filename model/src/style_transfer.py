# Style transfer using wavelets

import os
import argparse
import torch
import matplotlib.pyplot as plt
import time

from models import WaveletOptimizedStyleTransfer
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
    parser = argparse.ArgumentParser(description='Style Transfer using Wavelets')

    parser.add_argument('--content', type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to style image')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--size', type=int, default=512,
                        help='Size of output image (resized square)')

    parser.add_argument('--steps', type=int, default=500,
                        help='Number of optimization steps')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Weight for VGG content loss')
    parser.add_argument('--style_weight', type=float, default=1e5,
                        help='Weight for VGG style loss')
    parser.add_argument('--wavelet_content_weight', type=float, default=1e0,
                        help='Weight for wavelet content loss')
    parser.add_argument('--wavelet_style_weight', type=float, default=1e2,
                        help='Weight for wavelet style loss')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate for LBFGS optimization')

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
    output_dir = os.path.join(args.output, f"style_transfer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    print("Loading images...")
    content_img = load_image(args.content, target_size=(args.size, args.size), device=device)
    style_img = load_image(args.style, target_size=(args.size, args.size), device=device)

    # Create model
    print("Creating model...")
    model = WaveletOptimizedStyleTransfer(device=device)
    model.to(device)

    # Apply style transfer
    print(f"Optimizing for {args.steps} steps...")
    start_time = time.time()

    stylized_img, losses = model.stylize_optimize(
        content_img, style_img,
        num_steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        wavelet_content_weight=args.wavelet_content_weight,
        wavelet_style_weight=args.wavelet_style_weight,
        lr=args.lr,
        show_every=max(1, args.steps // 10)
    )
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")

    # Save results
    print("Saving results...")
    content_name = os.path.splitext(os.path.basename(args.content))[0]
    style_name = os.path.splitext(os.path.basename(args.style))[0]

    result_path = os.path.join(output_dir, f"{content_name}_styled_with_{style_name}.jpg")
    save_image(stylized_img, result_path)

    comparison_path = os.path.join(output_dir, f"{content_name}_styled_with_{style_name}_comparison.jpg")
    create_comparison_image(content_img, style_img, stylized_img, comparison_path)

    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        content_cpu = content_img.cpu()
        style_cpu = style_img.cpu()
        stylized_cpu = stylized_img.cpu()

        # Wavelet decomposition visualizations
        try:
            content_wavelet_viz = visualize_wavelet_decomposition(content_cpu)
            plt.savefig(os.path.join(viz_dir, "content_wavelet_decomposition.jpg"))
            plt.close(content_wavelet_viz)
            style_wavelet_viz = visualize_wavelet_decomposition(style_cpu)
            plt.savefig(os.path.join(viz_dir, "style_wavelet_decomposition.jpg"))
            plt.close(style_wavelet_viz)
            result_wavelet_viz = visualize_wavelet_decomposition(stylized_cpu)
            plt.savefig(os.path.join(viz_dir, "result_wavelet_decomposition.jpg"))
            plt.close(result_wavelet_viz)
        except Exception as e: print(f"Could not generate wavelet decomposition plots: {e}")

        # Frequency spectrum visualizations
        try:
            content_freq_viz = visualize_frequency_spectrum(content_cpu)
            plt.savefig(os.path.join(viz_dir, "content_frequency_spectrum.jpg"))
            plt.close(content_freq_viz)
            style_freq_viz = visualize_frequency_spectrum(style_cpu)
            plt.savefig(os.path.join(viz_dir, "style_frequency_spectrum.jpg"))
            plt.close(style_freq_viz)
            result_freq_viz = visualize_frequency_spectrum(stylized_cpu)
            plt.savefig(os.path.join(viz_dir, "result_frequency_spectrum.jpg"))
            plt.close(result_freq_viz)
        except Exception as e: print(f"Could not generate frequency spectrum plots: {e}")

        # Multi-scale visualizations
        try:
            multi_scale_content_fig = create_multi_scale_visualization(content_cpu, output_path=os.path.join(viz_dir, "content_multi_scale.jpg"))
            plt.close(multi_scale_content_fig)
            multi_scale_result_fig = create_multi_scale_visualization(stylized_cpu, output_path=os.path.join(viz_dir, "result_multi_scale.jpg"))
            plt.close(multi_scale_result_fig)
        except Exception as e: print(f"Could not generate multi-scale plots: {e}")

        # Loss history visualization (should always be available now)
        if losses and 'Total Loss' in losses and losses['Total Loss']:
            try:
                loss_viz = visualize_loss_history(losses)
                plt.savefig(os.path.join(viz_dir, "loss_history.jpg"))
                plt.close(loss_viz)
            except Exception as e: print(f"Could not generate loss history plot: {e}")
        else:
             print("No loss history data found to plot.")

    print("-" * 40)
    print(f"Results saved to: {output_dir}")
    print("-" * 40)
    return result_path, comparison_path

def main():
    args = parse_args()
    result_path, comparison_path = stylize_image(args)

if __name__ == "__main__":
    main()