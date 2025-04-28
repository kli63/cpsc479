import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import random

class StyleTransfer:
    def __init__(self):
        """Initialize the style transfer."""
        pass
        
    def load_and_preprocess(self, content_path, style_path, target_size=(512, 512)):
        content_img = cv2.imread(content_path)
        style_img = cv2.imread(style_path)
        
        if content_img is None:
            raise ValueError(f"Could not load content image from {content_path}")
        if style_img is None:
            raise ValueError(f"Could not load style image from {style_path}")
            
        # Convert from BGR to RGB
        content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        
        # Resize images
        content_img = cv2.resize(content_img, target_size)
        style_img = cv2.resize(style_img, target_size)
        
        # Convert to float and normalize
        content_img = content_img.astype(np.float32) / 255.0
        style_img = style_img.astype(np.float32) / 255.0
        
        return content_img, style_img
    
    def simulate_brush_strokes(self, image, brush_size=5, num_strokes=5000, stroke_length=10):
        # Create a copy of the image to work with
        canvas = image.copy()
        h, w = image.shape[:2]
        
        # Calculate gradient magnitude and direction
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Normalize gradient magnitude
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        # Create a probability map based on gradient magnitude
        # Higher gradient areas (edges) have higher probability
        prob_map = magnitude
        
        # Flatten probability map for sampling
        flat_prob = prob_map.flatten()
        if np.sum(flat_prob) > 0:
            flat_prob = flat_prob / np.sum(flat_prob)
        else:
            flat_prob = np.ones_like(flat_prob) / len(flat_prob)
        
        # Indices for sampling
        indices = np.arange(h * w)
        
        # Sample stroke positions based on probability
        stroke_indices = np.random.choice(indices, size=num_strokes, p=flat_prob)
        stroke_positions = np.column_stack((stroke_indices % w, stroke_indices // w))
        
        # Apply brush strokes
        for pos in stroke_positions:
            x, y = pos
            
            # Skip if out of bounds
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            
            # Get color at this position
            color = image[y, x].copy()
            
            # Get stroke direction (perpendicular to gradient)
            angle = direction[y, x] + 90
            if random.random() < 0.5:  # Randomly flip direction
                angle += 180
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Calculate stroke endpoints
            x1 = int(x - stroke_length * np.cos(angle_rad))
            y1 = int(y - stroke_length * np.sin(angle_rad))
            x2 = int(x + stroke_length * np.cos(angle_rad))
            y2 = int(y + stroke_length * np.sin(angle_rad))
            
            # Draw stroke
            cv2.line(canvas, (x1, y1), (x2, y2), color.tolist(), brush_size, cv2.LINE_AA)
        
        return canvas
    
    def anisotropic_diffusion(self, image, iterations=10, k=20, gamma=0.1):
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.copy()
        
        filtered = image.copy()
        
        for _ in range(iterations):
            north = np.roll(filtered, -1, axis=0) - filtered
            south = np.roll(filtered, 1, axis=0) - filtered
            east = np.roll(filtered, 1, axis=1) - filtered
            west = np.roll(filtered, -1, axis=1) - filtered
            
            edge_strength = np.sqrt(np.mean(gray**2, axis=-1) if len(gray.shape) == 3 else gray**2)
            
            c_north = np.exp(-(edge_strength**2) / (k**2))
            c_south = np.exp(-(edge_strength**2) / (k**2))
            c_east = np.exp(-(edge_strength**2) / (k**2))
            c_west = np.exp(-(edge_strength**2) / (k**2))
            
            if len(filtered.shape) == 3:
                for i in range(3):  # Process each channel
                    filtered[:,:,i] = filtered[:,:,i] + gamma * (
                        c_north * north[:,:,i] +
                        c_south * south[:,:,i] +
                        c_east * east[:,:,i] +
                        c_west * west[:,:,i]
                    )
            else:
                filtered = filtered + gamma * (
                    c_north * north +
                    c_south * south +
                    c_east * east +
                    c_west * west
                )
        
        return np.clip(filtered, 0, 1)
    
    def create_brush_texture(self, size, density=0.1, brush_size=3):
        h, w = size
        texture = np.ones((h, w), dtype=np.float32)
        
        num_strokes = int(h * w * density)
        for _ in range(num_strokes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(5, 20)
            angle = np.random.rand() * 2 * np.pi
            
            x1 = int(x - length * np.cos(angle))
            y1 = int(y - length * np.sin(angle))
            x2 = int(x + length * np.cos(angle))
            y2 = int(y + length * np.sin(angle))
            
            intensity = 0.8 + 0.2 * np.random.rand()
            
            cv2.line(texture, (max(0, x1), max(0, y1)), 
                     (min(w-1, x2), min(h-1, y2)), 
                     intensity, brush_size, cv2.LINE_AA)
        
        texture = cv2.GaussianBlur(texture, (5, 5), 1.0)
        
        # Expand to 3 channels
        texture_rgb = np.stack([texture] * 3, axis=2)
        return texture_rgb
    
    def apply_texture(self, image, texture_strength=0.2):
        # Create brush texture
        texture = self.create_brush_texture(image.shape[:2], density=0.2, brush_size=2)
        

        textured = image * (1 - texture_strength + texture_strength * texture)
        
        return np.clip(textured, 0, 1)
    
    def color_transfer(self, content, style, strength=0.7):
        content_lab = cv2.cvtColor((content * 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
        style_lab = cv2.cvtColor((style * 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
        
        # Split channels
        l_content, a_content, b_content = cv2.split(content_lab)
        l_style, a_style, b_style = cv2.split(style_lab)
        
        for c_idx, (c_content, c_style) in enumerate([(a_content, a_style), (b_content, b_style)]):
            mean_content = np.mean(c_content)
            std_content = np.std(c_content)
            
            mean_style = np.mean(c_style)
            std_style = np.std(c_style)
            
            c_content[:] = (1 - strength) * c_content + strength * ((c_content - mean_content) / (std_content + 1e-5) * std_style + mean_style)
        
        mean_l_content = np.mean(l_content)
        std_l_content = np.std(l_content)
        mean_l_style = np.mean(l_style)
        std_l_style = np.std(l_style)
        
        l_content = (1 - strength*0.5) * l_content + strength*0.5 * ((l_content - mean_l_content) / (std_l_content + 1e-5) * std_l_style + mean_l_style)
        
        output_lab = cv2.merge([l_content, a_content, b_content])
        
        output_rgb = cv2.cvtColor(output_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
        
        return output_rgb.astype(np.float32) / 255.0
    
    def add_vignette(self, image, strength=0.25):
        h, w = image.shape[:2]
        
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist / max_dist
        
        vignette = 1 - norm_dist**2
        vignette = np.clip(vignette, 0, 1)
        
        vignette = 1 - strength * (1 - vignette)
        
        vignette = np.stack([vignette] * 3, axis=2)
        result = image * vignette
        
        return result
    
    def enhance_contrast(self, image, saturation=1.3, contrast=1.2):
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        mean = np.mean(rgb, axis=(0, 1), keepdims=True)
        rgb = mean + (rgb - mean) * contrast
        
        return np.clip(rgb, 0, 1)
    
    def process(self, content_path, style_path, target_size=(512, 512), preset="balanced"):
        print("Loading and preprocessing images...")
        content_rgb, style_rgb = self.load_and_preprocess(content_path, style_path, target_size)
        
        if preset == "balanced":
            color_strength = 0.6
            texture_strength = 0.3
            brush_density = 5000
            brush_size = 3
            vignette_strength = 0.2
            contrast = 1.2
            saturation = 1.3
        elif preset == "painterly":
            color_strength = 0.7
            texture_strength = 0.4
            brush_density = 8000
            brush_size = 4
            vignette_strength = 0.25
            contrast = 1.3
            saturation = 1.4
        elif preset == "classical":
            color_strength = 0.5
            texture_strength = 0.35
            brush_density = 6000
            brush_size = 3
            vignette_strength = 0.3
            contrast = 1.1
            saturation = 1.2
        elif preset == "dramatic":
            color_strength = 0.8
            texture_strength = 0.45
            brush_density = 10000
            brush_size = 5
            vignette_strength = 0.35
            contrast = 1.4
            saturation = 1.5
        else:  # Custom preset
            color_strength = 0.6
            texture_strength = 0.3
            brush_density = 5000
            brush_size = 3
            vignette_strength = 0.2
            contrast = 1.2
            saturation = 1.3
        
        print(f"Applying {preset} style transfer...")
        
        print("Applying painterly smoothing...")
        smoothed = self.anisotropic_diffusion(content_rgb, iterations=15, k=25, gamma=0.1)
        
        print("Transferring color palette...")
        color_transferred = self.color_transfer(smoothed, style_rgb, strength=color_strength)
        
        print("Enhancing color vibrancy...")
        enhanced = self.enhance_contrast(color_transferred, saturation=saturation, contrast=contrast)
        
        print("Simulating brush strokes...")
        with_strokes = self.simulate_brush_strokes(
            enhanced, 
            brush_size=brush_size,
            num_strokes=brush_density,
            stroke_length=8
        )
        
        print("Adding canvas texture...")
        textured = self.apply_texture(with_strokes, texture_strength=texture_strength)
        
        print("Adding classical vignette...")
        with_vignette = self.add_vignette(textured, strength=vignette_strength)
        
        print("Applying final smoothing...")
        result_uint8 = (with_vignette * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(result_uint8, 3, 10, 10)
        final_result = smoothed.astype(np.float32) / 255.0
        
        return content_rgb, style_rgb, final_result


if __name__ == "__main__":
    style_transfer = StyleTransfer()
    
    # Updated paths to match new directory structure
    content_path = "assets/input/download.jpg"
    style_path = "assets/reference/classical_art.jpg"
    
    output_dir = "assets/_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\nStarting stylization with 'balanced' preset...")
        content, style, result1 = style_transfer.process(
            content_path, style_path, 
            target_size=(512, 512),
            preset="balanced"
        )
        
        print("\nStarting stylization with 'painterly' preset...")
        content, style, result2 = style_transfer.process(
            content_path, style_path, 
            target_size=(512, 512),
            preset="painterly"
        )
        
        print("\nStarting stylization with 'classical' preset...")
        content, style, result3 = style_transfer.process(
            content_path, style_path, 
            target_size=(512, 512),
            preset="classical"
        )
        
        print("\nStarting stylization with 'dramatic' preset...")
        content, style, result4 = style_transfer.process(
            content_path, style_path, 
            target_size=(512, 512),
            preset="dramatic"
        )
        
        print("\nSaving results...")
        preset_names = ["balanced", "painterly", "classical", "dramatic"]
        for idx, result in enumerate([result1, result2, result3, result4], 1):
            output_path = os.path.join(output_dir, f"painterly_{preset_names[idx-1]}.png")
            result_display = (result * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(result_display, cv2.COLOR_RGB2BGR))
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(content)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(2, 3, 2)
        plt.imshow(style)
        plt.title("Style Reference")
        plt.axis("off")
        
        plt.subplot(2, 3, 3)
        plt.imshow(result1)
        plt.title("Balanced")
        plt.axis("off")
        
        plt.subplot(2, 3, 4)
        plt.imshow(result2)
        plt.title("Painterly")
        plt.axis("off")
        
        plt.subplot(2, 3, 5)
        plt.imshow(result3)
        plt.title("Classical")
        plt.axis("off")
        
        plt.subplot(2, 3, 6)
        plt.imshow(result4)
        plt.title("Dramatic")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "painterly_comparison.png"))
        print(f"Results saved to {output_dir}")
        print("Opening comparison image...")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"Content image exists: {os.path.exists(content_path)}")
        print(f"Style image exists: {os.path.exists(style_path)}")
        
        import traceback
        traceback.print_exc()
