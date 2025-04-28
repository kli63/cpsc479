# 3D Virtual Gallery for Style Transfer Project

This is a Three.js-based virtual gallery application designed to showcase the style transfer images from the computational photography project.

## Structure

- `/js` - JavaScript files for the application
  - `main.js` - Main entry point and scene setup
  - `gallery-loader.js` - Handles loading and displaying artwork
- `/css` - Stylesheets
- `/assets` - Additional assets needed for the gallery (textures, models, etc.)
- `/models` - 3D models for the gallery environment

## Getting Started

To run this application locally, you need to serve it through a web server due to browser security restrictions when loading files via JavaScript.

### Using Python's built-in HTTP server:

```bash
python -m http.server
```

Then open your browser and navigate to `http://localhost:8000/gallery`

### Using Node.js and http-server:

1. Install http-server if you haven't already:
   ```bash
   npm install -g http-server
   ```

2. Run the server:
   ```bash
   http-server
   ```

3. Open your browser and navigate to `http://localhost:8080/gallery`

## Features

- First-person navigation using WASD keys and mouse
- Gallery environment with wall-mounted frames
- Dynamic loading of style-transferred images
- Customizable gallery layout

## Integration with Style Transfer Project

The gallery is designed to display the style transfer images from the `model/assets/_results` directory. To add new images to the gallery, add them to the appropriate directory and update the application to include them in the gallery.

## Future Enhancements

- Interactive elements for each artwork (zoom, information panel)
- Different gallery layouts/rooms
- Ambient audio
- Information panels about the style transfer technique
- Before/after comparison views
