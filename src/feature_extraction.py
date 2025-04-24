"""
Image Feature Extraction for Photo-to-Music Conversion
---------------------------------------------------

This script extracts comprehensive visual features from images to enable mapping 
to musical parameters. The following features are extracted:

1. Multi-scale Analysis
   - Gaussian pyramids: Multi-resolution representations at different scales
   - Laplacian pyramids: Band-pass filtered versions highlighting details at different scales
   - Depth map estimation: Using focus/defocus cues to create depth information

2. Color Features
   - Color statistics: Mean, standard deviation in BGR, HSV, and LAB color spaces
   - Dominant colors: Using k-means clustering to identify main colors
   - Color proportions: Distribution of colors across the image
   - Hue, saturation, and value statistics: For emotional mapping

3. Texture Features
   - GLCM (Gray Level Co-occurrence Matrix): Contrast, homogeneity, energy
   - LBP (Local Binary Pattern): Texture pattern histograms
   - Gabor filter responses: At different orientations for texture analysis

4. Shape Features
   - Edge density: Measure of detail and complexity
   - Contour analysis: Number, size, and complexity of shapes
   - Hu moments: Shape descriptors invariant to translation, rotation, and scale

5. Spatial Features
   - Grid-based analysis: 3×3 grid with statistics for each cell
   - Spatial frequency: Using FFT (Fast Fourier Transform)
   - Balance measurements: Horizontal and vertical distribution of elements

Mapping
- Color features → Harmonic structure, chord progressions, and emotional qualities
- Texture features → Timbre selection and instrumental choices
- Shape features → Melodic contours and phrase structures
- Spatial features → Rhythmic patterns and time signatures
- Depth information → Musical layering and spatial characteristics

The extracted features are saved as JSON data for subsequent musical parameter mapping.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import json

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
        # Get the current level and the expanded version of the next level
        current_level = gaussian_pyramid[i]
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        
        # Resize the expanded image to match the current level
        if expanded.shape != current_level.shape:
            expanded = cv2.resize(expanded, (current_level.shape[1], current_level.shape[0]))
        
        # Compute the Laplacian as the difference
        laplacian = cv2.subtract(current_level, expanded)
        laplacian_pyramid.append(laplacian)
    
    # Add the smallest Gaussian level as the last Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid

def estimate_depth(img):
    # Convert to grayscale if the image has multiple channels
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Laplacian to detect edges and measure focus
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate local variance of Laplacian as a focus measure
    kernel_size = 15
    laplacian_abs = np.abs(laplacian)
    focus_map = cv2.GaussianBlur(laplacian_abs, (kernel_size, kernel_size), 0)
    
    # Normalize to get a depth map (higher values = more in focus = closer)
    depth_map = cv2.normalize(focus_map, None, 0, 1, cv2.NORM_MINMAX)
    
    return depth_map

def extract_color_features(img):
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split channels
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(hsv)
    l, a, b_lab = cv2.split(lab)
    
    # Calculate color histograms
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Find dominant colors using k-means clustering
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5  # Number of dominant colors to extract
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixels in each cluster to get color proportions
    unique_labels, counts = np.unique(labels, return_counts=True)
    color_proportions = counts / counts.sum()
    
    # Convert centers to BGRs for color visualization
    centers = centers.astype(np.uint8)
    
    # Calculate color moments
    color_features = {
        # Color statistics
        'mean_hue': float(np.mean(h)),
        'std_hue': float(np.std(h)),
        'mean_saturation': float(np.mean(s)),
        'std_saturation': float(np.std(s)),
        'mean_value': float(np.mean(v)),
        'std_value': float(np.std(v)),
        
        # BGR statistics
        'mean_blue': float(np.mean(b)),
        'std_blue': float(np.std(b)),
        'mean_green': float(np.mean(g)),
        'std_green': float(np.std(g)),
        'mean_red': float(np.mean(r)),
        'std_red': float(np.std(r)),
        
        # LAB statistics (perceptual)
        'mean_lightness': float(np.mean(l)),
        'std_lightness': float(np.std(l)),
        
        # Dominant hues
        'dominant_hue': float(np.argmax(h_hist)),
        
        # Dominant colors as BGR values
        'dominant_colors': [centers[i].tolist() for i in range(k)],
        'color_proportions': color_proportions.tolist()
    }
    
    return color_features

def extract_texture_features(img):
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1. Calculate Haralick texture features using GLCM
    # We'll use a simplified approach here
    glcm = np.zeros((256, 256), dtype=np.uint32)
    h, w = gray.shape
    
    # Simple GLCM computation (horizontal adjacency)
    for i in range(h):
        for j in range(w-1):
            glcm[gray[i,j], gray[i,j+1]] += 1
    
    # Normalize the GLCM
    glcm = glcm.astype(np.float32)
    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        glcm /= glcm_sum
    
    # Calculate some basic GLCM statistics
    # Contrast
    contrast = 0
    for i in range(256):
        for j in range(256):
            contrast += glcm[i,j] * (i-j)**2
    
    # Homogeneity
    homogeneity = 0
    for i in range(256):
        for j in range(256):
            homogeneity += glcm[i,j] / (1 + abs(i-j))
    
    # Energy
    energy = np.sum(glcm**2)
    
    # 2. Local Binary Pattern (simplified)
    lbp = np.zeros_like(gray)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            
            # Compute 8-bit LBP code
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            
            lbp[i, j] = code
    
    # Calculate LBP histogram
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist = lbp_hist.flatten() / (h * w)  # Normalize
    
    # 3. Gabor filters for different orientations
    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_features.append(float(np.mean(filtered)))
        gabor_features.append(float(np.std(filtered)))
    
    texture_features = {
        # GLCM features
        'contrast': float(contrast),
        'homogeneity': float(homogeneity),
        'energy': float(energy),
        
        # LBP stats (first few bins of histogram)
        'lbp_stats': lbp_hist[:10].tolist(),
        
        # Gabor features
        'gabor_features': gabor_features
    }
    
    return texture_features

def extract_shape_features(img):
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Edge detection with Canny
    edges = cv2.Canny(gray, 100, 200)
    
    # Count edge pixels and calculate edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate contour features
    num_contours = len(contours)
    
    # Average contour size and complexity
    avg_contour_points = 0
    avg_contour_area = 0
    
    if num_contours > 0:
        total_points = sum(len(contour) for contour in contours)
        avg_contour_points = total_points / num_contours
        
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        avg_contour_area = total_area / num_contours
    
    # Calculate image moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    shape_features = {
        'edge_density': float(edge_density),
        'num_contours': num_contours,
        'avg_contour_points': float(avg_contour_points),
        'avg_contour_area': float(avg_contour_area),
        'hu_moments': [float(moment) for moment in hu_moments]
    }
    
    return shape_features

def extract_spatial_features(img):
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Divide the image into a 3x3 grid and compute statistics for each cell
    h, w = gray.shape
    cell_h, cell_w = h // 3, w // 3
    
    cell_stats = []
    for i in range(3):
        for j in range(3):
            # Extract cell
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Compute statistics
            cell_stats.append({
                'mean': float(np.mean(cell)),
                'std': float(np.std(cell)),
                'position': [i, j]
            })
    
    # Calculate spatial frequency using FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # Add 1 to avoid log(0)
    
    # Calculate statistics from frequency domain
    freq_mean = float(np.mean(magnitude_spectrum))
    freq_std = float(np.std(magnitude_spectrum))
    
    # Calculate horizontal and vertical imbalance
    left_half = np.mean(gray[:, :w//2])
    right_half = np.mean(gray[:, w//2:])
    h_imbalance = float(abs(left_half - right_half) / 255.0)
    
    top_half = np.mean(gray[:h//2, :])
    bottom_half = np.mean(gray[h//2:, :])
    v_imbalance = float(abs(top_half - bottom_half) / 255.0)
    
    spatial_features = {
        'cell_stats': cell_stats,
        'frequency_mean': freq_mean,
        'frequency_std': freq_std,
        'horizontal_imbalance': h_imbalance,
        'vertical_imbalance': v_imbalance
    }
    
    return spatial_features

def extract_all_features(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    # Basic image info
    height, width, channels = img.shape if len(img.shape) == 3 else (*img.shape, 1)
    aspect_ratio = width / height
    
    # Generate Gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(img, levels=4)
    
    # Generate Laplacian pyramid
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
    
    # Estimate depth
    depth_map = estimate_depth(img)
    depth_stats = {
        'mean_depth': float(np.mean(depth_map)),
        'std_depth': float(np.std(depth_map)),
        'min_depth': float(np.min(depth_map)),
        'max_depth': float(np.max(depth_map))
    }
    
    # Extract various features
    color_features = extract_color_features(img)
    texture_features = extract_texture_features(img)
    shape_features = extract_shape_features(img)
    spatial_features = extract_spatial_features(img)
    
    # Compile all features
    all_features = {
        'image_info': {
            'filename': os.path.basename(img_path),
            'height': height,
            'width': width,
            'channels': channels,
            'aspect_ratio': float(aspect_ratio)
        },
        'depth_features': depth_stats,
        'color_features': color_features,
        'texture_features': texture_features,
        'shape_features': shape_features,
        'spatial_features': spatial_features
    }
    
    return all_features

def process_image(img_path, output_dir):
    # Create output filename based on input filename
    base_name = os.path.basename(img_path).split('.')[0]
    
    # Read the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    # Extract features
    features = extract_all_features(img_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the features as JSON
    features_path = os.path.join(output_dir, f"{base_name}_features.json")
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    # Generate Gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(img, levels=4)
    
    # Generate Laplacian pyramid
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
    
    # Estimate depth
    depth_map = estimate_depth(img)
    
    # Save the depth map
    depth_path = os.path.join(output_dir, f"{base_name}_depth_map.png")
    plt.imsave(depth_path, depth_map, cmap='viridis')
    
    # Create and save visualization
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    # Display original image
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Display Gaussian pyramid levels
    gaussian_resized = [cv2.resize(level, (img.shape[1] // 5, img.shape[0] // 5)) for level in gaussian_pyramid]
    gaussian_display = np.hstack(gaussian_resized)
    axs[1].imshow(cv2.cvtColor(gaussian_display, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Gaussian Pyramid')
    axs[1].axis('off')
    
    laplacian_resized = []
    for level in laplacian_pyramid:
        # Normalize for better visualization
        level_norm = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian_resized.append(cv2.resize(level_norm, (img.shape[1] // 5, img.shape[0] // 5)))
    
    laplacian_display = np.hstack(laplacian_resized)
    axs[2].imshow(laplacian_display, cmap='viridis')
    axs[2].set_title('Laplacian Pyramid')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(viz_path)
    plt.close(fig)
    
    # Save individual pyramid levels for further processing
    for i, level in enumerate(gaussian_pyramid):
        level_path = os.path.join(output_dir, f"{base_name}_gaussian_level_{i}.png")
        cv2.imwrite(level_path, level)
    
    for i, level in enumerate(laplacian_pyramid):
        # Normalize for proper saving
        level_norm = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        level_path = os.path.join(output_dir, f"{base_name}_laplacian_level_{i}.png")
        cv2.imwrite(level_path, level_norm)
    
    print(f"Processed {img_path} and saved results to {output_dir}")
    
    return features

def process_all_images(input_dir, output_dir):
    # Get all image files in the input directory
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
    all_features = {}
    for img_path in tqdm(image_files, desc="Processing images"):
        features = process_image(img_path, output_dir)
        if features:
            all_features[os.path.basename(img_path)] = features
    
    # Save all features in a single JSON file
    all_features_path = os.path.join(output_dir, "all_features.json")
    with open(all_features_path, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print(f"All images processed. Results saved to {output_dir}")
    print(f"Combined features saved to {all_features_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract visual features from images using OpenCV.')
    parser.add_argument('--input', default='assets/input', 
                        help='Path to input directory containing images')
    parser.add_argument('--output', default='assets/_results', 
                        help='Path to output directory for results')
    args = parser.parse_args()
    
    process_all_images(args.input, args.output)
