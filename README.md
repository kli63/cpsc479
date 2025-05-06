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

## Environment Setup

To set up the environment for this project:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

To run the gallery:

```bash
# Start the gallery
./start-gallery.sh

# Open in browser (automatically opens)
http://localhost:8000/
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
cd model

# Run style transfer with default parameters
python -m src.style_transfer --content assets/input/0001.jpg --style assets/reference/0022.jpg
```

The results will automatically be saved in a timestamped directory under `model/results/` and will appear in the gallery on the next launch.

### Key Style Transfer Parameters

- `--content`: Path to content image (required)
- `--style`: Path to style image (required)
- `--size`: Output image size (default: 512)
- `--steps`: Number of optimization steps (default: 500)
- `--visualize`: Generate additional visualizations

For additional parameters and options, see the model README in the `/model` directory.

## Finding Results

Style transfer results are saved in the following location:

```
model/results/style_transfer_YYYYMMDD_HHMMSS/
```

Each result directory contains:
- The stylized image
- A comparison image showing the content, style, and result
- Visualizations (if enabled with the `--visualize` flag)

## Project Structure

- `model/src/` - Core neural style transfer implementation
- `model/assets/` - Content and style images
- `model/results/` - Generated output images
- `gallery/js/` - 3D gallery implementation
- `gallery/css/` - Styling for the gallery interface

See README files in individual directories for more detailed information.