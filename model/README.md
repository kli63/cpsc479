# Style Transfer Model

This repository contains a wavelet-based style transfer model for image enhancement and artistic stylization.

## Viewing Results in the Gallery

To view style transfer results in the 3D gallery:

```bash
# Start from the project root directory
cd path/to/CPSC479/FP
./start-gallery.sh

# Open in browser
http://localhost:8000/
```

## Directory Structure

- `src/` - Core model implementation
  - `models.py` - WaveletOptimizedStyleTransfer implementation
  - `style_transfer.py` - Command-line interface for running style transfer
  - `feature_extraction.py` - Feature extraction utilities

- `assets/` - Image assets
  - `input/` - Content images
  - `reference/` - Style reference images
  - `_results/` - Saved style transfer results

- `results/` - Generated style transfer outputs
  - Output is organized by date/time of generation

## Running Style Transfer

```bash
# Navigate to the model directory
cd path/to/CPSC479/FP/model

# Run style transfer with basic parameters
python -m src.style_transfer --content assets/input/0001.jpg --style assets/reference/0005.jpg
```

## Bulk Style Transfer Generator

To generate multiple style transfers in parallel:

```bash
# Run with default settings (50 random combinations, 4 workers)
./generate_combinations.py

# Run with custom settings
./generate_combinations.py --max_combinations 20 --workers 8 --steps 500 --size 768 --visualize
```

The script will:
- Track progress in a JSON file
- Resume from where it left off if interrupted
- Select random combinations of content and style images
- Run transfers in parallel
- Update the gallery manifest when complete

## Command-line Arguments

The style transfer model supports the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--content` | str | *required* | Path to content image |
| `--style` | str | *required* | Path to style image |
| `--output` | str | `./results` | Directory to save results |
| `--size` | int | 512 | Size of output image (resized square) |
| `--steps` | int | 500 | Number of optimization steps |
| `--content_weight` | float | 1.0 | Weight for VGG content loss |
| `--style_weight` | float | 1e5 | Weight for VGG style loss |
| `--wavelet_content_weight` | float | 1.0 | Weight for wavelet content loss |
| `--wavelet_style_weight` | float | 100.0 | Weight for wavelet style loss |
| `--lr` | float | 1.0 | Learning rate for LBFGS optimization |
| `--cpu` | flag | False | Force CPU usage (instead of GPU) |
| `--visualize` | flag | False | Generate wavelet and frequency visualizations |

## Example Usage

### Basic Style Transfer

```bash
python -m src.style_transfer --content assets/input/0001.jpg --style assets/reference/0005.jpg
```

### Advanced Style Transfer with Custom Parameters

```bash
python -m src.style_transfer \
  --content assets/input/0002.jpg \
  --style assets/reference/0022.jpg \
  --size 768 \
  --steps 1000 \
  --content_weight 1.5 \
  --style_weight 2e5 \
  --wavelet_content_weight 2.0 \
  --wavelet_style_weight 150.0 \
  --visualize
```

### Controlling Style Intensity

To create a more subtle style transfer effect, decrease the style weights:

```bash
python -m src.style_transfer \
  --content assets/input/0031.jpg \
  --style assets/reference/0022.jpg \
  --style_weight 5e4 \
  --wavelet_style_weight 50.0
```

To create a stronger style effect, increase the style weights:

```bash
python -m src.style_transfer \
  --content assets/input/0031.jpg \
  --style assets/reference/0022.jpg \
  --style_weight 3e5 \
  --wavelet_style_weight 200.0
```

### Running on CPU (if no GPU is available)

```bash
python -m src.style_transfer \
  --content assets/input/0001.jpg \
  --style assets/reference/0005.jpg \
  --cpu
```

### Generating Visualizations

To generate detailed wavelet and frequency domain visualizations:

```bash
python -m src.style_transfer \
  --content assets/input/0075.jpg \
  --style assets/reference/0005.jpg \
  --visualize
```

This will create additional visualizations in a subdirectory showing wavelet decompositions, frequency spectra, and multi-scale representations of the images.
