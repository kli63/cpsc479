# 3D Virtual Gallery for Style Transfer Project

This is a Three.js-based virtual gallery application designed to showcase the style transfer images from the computational photography project.

## Structure

- `/js` - JavaScript files for the application
  - `main.js` - Main entry point and scene setup
  - `room-builder.js` - Creates the 3D gallery environment
  - `artwork-manager.js` - Handles loading and displaying artwork
  - `image-mapper.js` - Maps style transfer images to their sources
  - `collision-manager.js` - Handles collision detection for navigation
- `/css` - Stylesheets

## Running the Gallery

To run the gallery:

```bash
# Navigate to the gallery directory
cd path/to/CPSC479/FP/gallery

# Make the scripts executable (if needed)
chmod +x start-gallery.sh update-gallery-manifest.sh

# Run the gallery
./start-gallery.sh
```

This will:
1. Generate a manifest file of all available images
2. Start a local web server on port 8000
3. Navigate to: http://localhost:8000/gallery/

### GitHub Pages Hosting

The gallery works with GitHub Pages:

1. Run `./update-gallery-manifest.sh` to generate the manifest
2. Add all files to your repository
3. Enable GitHub Pages in your repository settings

## Features

- First-person navigation using WASD keys and mouse
- Gallery environment with wall-mounted frames
- Dynamic loading of style-transferred images
- Customizable gallery layout

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

## Future Enhancements

- Interactive elements for each artwork (zoom, information panel)
- Different gallery layouts/rooms
- Ambient audio
- Information panels about the style transfer technique
- Before/after comparison views
