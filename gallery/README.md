# 3D Virtual Gallery for Style Transfer Project

This is a Three.js-based virtual gallery application designed to showcase the style transfer images from the computational photography project. 

> **Note:** The main `index.html` entry point is located in the root directory for GitHub Pages compatibility. The gallery code itself is in this directory.

## Structure

- `/js` - JavaScript files for the application
  - `main.js` - Main entry point and scene setup
  - `room-builder.js` - Creates the 3D gallery environment
  - `artwork-manager.js` - Handles loading and displaying artwork
  - `image-mapper.js` - Maps style transfer images to their sources
  - `collision-manager.js` - Handles collision detection for navigation
- `/css` - Stylesheets
- `/data` - Contains the gallery manifest file tracking available images
- `/assets/best` - Contains the best style transfer results for priority display

## Running the Gallery

To run the gallery:

```bash
# Make the scripts executable (if needed)
chmod +x start-gallery.sh gallery/update-gallery-manifest.sh

# Run the gallery
./start-gallery.sh
```

This will:
1. Generate a manifest file of all available images
2. Start a local web server on port 8000
3. Navigate to: http://localhost:8000/

### GitHub Pages Hosting

The gallery is deployed on GitHub Pages for easy sharing and viewing:

1. Visit: [https://kli63.github.io/cpsc479/](https://kli63.github.io/cpsc479/)
2. The manifest file is included in the repository for proper GitHub Pages deployment

## Features

- First-person navigation using WASD keys and mouse
- Gallery environment with wall-mounted frames
- Dynamic loading of style-transferred images
- Customizable gallery layout
- Comparison views for original content, style reference, and result

## Integration with Style Transfer Project

The gallery dynamically loads style transfer images from the `model/results/` directory. When you run style transfer to create new images, they will automatically appear in the gallery on the next reload.

### Gallery Navigation

- **Move**: WASD keys or arrow keys
- **Look around**: Mouse movement
- **Select artwork**: Click on any framed image
- **Close details**: ESC key or click the close button
- **Start gallery**: Click anywhere in the center of the screen

### How Images Are Organized

The gallery shows:
- Content images from `model/assets/input/`
- Style images from `model/assets/reference/`
- Style transfer results from `model/results/`
- Best results from `gallery/assets/best/`

The most interesting and visually appealing results are featured prominently in the central gallery wall.

## Updating the Gallery Manifest

After generating new style transfer results, run the update script to refresh the gallery manifest:

```bash
./gallery/update-gallery-manifest.sh
```