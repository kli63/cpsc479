import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

def create_gaussian_pyramid(img, levels=4):
    G = img.copy()
    gaussian_pyramid = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gaussian_pyramid.append(G)
    return gaussian_pyramid


def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        current_level = gaussian_pyramid[i]
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        
        if expanded.shape != current_level.shape:
            expanded = cv2.resize(expanded, (current_level.shape[1], current_level.shape[0]))
        
        # Compute the Laplacian as the difference
        laplacian = cv2.subtract(current_level, expanded)
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def estimate_depth(img):
    # Convert to grayscale if the image has multiple channels
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    kernel_size = 15
    laplacian_abs = np.abs(laplacian)
    focus_map = cv2.GaussianBlur(laplacian_abs, (kernel_size, kernel_size), 0)
    
    depth_map = cv2.normalize(focus_map, None, 0, 1, cv2.NORM_MINMAX)
    
    return depth_map


def process_image(img_path, output_dir):
  
    base_name = os.path.basename(img_path).split('.')[0]
    
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    # Generate Gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(img, levels=4)
    
    # Generate Laplacian pyramid
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
    
    # Estimate depth
    depth_map = estimate_depth(img)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the depth map
    depth_path = os.path.join(output_dir, f"{base_name}_depth_map.png")
    plt.imsave(depth_path, depth_map, cmap='viridis')
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    gaussian_resized = [cv2.resize(level, (img.shape[1] // 5, img.shape[0] // 5)) for level in gaussian_pyramid]
    gaussian_display = np.hstack(gaussian_resized)
    axs[1].imshow(cv2.cvtColor(gaussian_display, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Gaussian Pyramid')
    axs[1].axis('off')
    
    laplacian_resized = []
    for level in laplacian_pyramid:
        level_norm = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian_resized.append(cv2.resize(level_norm, (img.shape[1] // 5, img.shape[0] // 5)))
    
    laplacian_display = np.hstack(laplacian_resized)
    axs[2].imshow(laplacian_display, cmap='viridis')
    axs[2].set_title('Laplacian Pyramid')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(viz_path)
    plt.close(fig)
    
    # Save individual pyramid levels for further processing
    for i, level in enumerate(gaussian_pyramid):
        level_path = os.path.join(output_dir, f"{base_name}_gaussian_level_{i}.png")
        cv2.imwrite(level_path, level)
    
    for i, level in enumerate(laplacian_pyramid):
        level_norm = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        level_path = os.path.join(output_dir, f"{base_name}_laplacian_level_{i}.png")
        cv2.imwrite(level_path, level_norm)
    
    print(f"Processed {img_path} and saved results to {output_dir}")
    
    return {
        'gaussian_pyramid': gaussian_pyramid,
        'laplacian_pyramid': laplacian_pyramid,
        'depth_map': depth_map
    }


def process_all_images(input_dir, output_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files in {input_dir}")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        process_image(img_path, output_dir)
    
    print(f"All images processed. Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images to create Gaussian and Laplacian pyramids with depth estimation.')
    parser.add_argument('--input', default='Final\\input', 
                        help='Path to input directory containing images')
    parser.add_argument('--output', default='Final\\output', 
                        help='Path to output directory for results')
    args = parser.parse_args()
    
    process_all_images(args.input, args.output)
