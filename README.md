# Wavelet-Based Neural Style Transfer with 3D Gallery

This project combines wavelet-based neural style transfer with a 3D gallery to display results.

## Project Components

- `/model` - Neural style transfer implementation
  - Style transfer using wavelet transforms
  - Command-line interface for processing images
  - Image assets and style references

- `/gallery` - 3D gallery viewer
  - Three.js-based virtual environment
  - First-person navigation
  - Display of original and styled images

## Quick Start

To run the gallery:

```bash
# Go to gallery directory
cd path/to/CPSC479/FP/gallery

# Start
./start-gallery.sh

# Open in browser
http://localhost:8000/gallery/
```

## Gallery Features

- **Navigation**: WASD keys or arrow keys to move, mouse to look around
- **Interaction**: Click on artwork to view details
- **Content Display**: See original content, style reference, and result side-by-side
- **Dynamic Loading**: Automatically discovers and displays new style transfer results
- **Responsive Design**: Works on different screen sizes and devices

## Using the Style Transfer Model

To run the style transfer model:

```bash
# Navigate to the model directory
cd path/to/CPSC479/FP/model

# Run style transfer with default parameters
python -m src.style_transfer --content assets/input/0001.jpg --style assets/reference/0022.jpg --output results/my_output.jpg
```

The results will automatically appear in the gallery on the next launch.

## Project Structure

- `model/src/` - Core neural style transfer implementation
- `model/assets/` - Content and style images
- `model/results/` - Generated output images
- `gallery/js/` - 3D gallery implementation
- `gallery/css/` - Styling for the gallery interface

See README files in individual directories for more detailed information.
