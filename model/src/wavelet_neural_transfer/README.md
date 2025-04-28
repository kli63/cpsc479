# Wavelet Neural Style Transfer

A novel approach to neural style transfer that combines pretrained VGG features with advanced wavelet domain processing for high-quality results without requiring extensive training.

## Key Features

- **Wavelet Domain Processing**: Uses discrete wavelet transforms instead of simple FFT for multi-resolution frequency analysis
- **Pretrained Feature Extraction**: Leverages VGG19 pretrained features for high-level semantic understanding
- **Phase-Preserving Style Transfer**: Maintains content structure while transferring style characteristics
- **Spectral Attention**: Frequency-selective attention mechanism that operates differently on high vs. low frequency components
- **Adaptive Frequency Modulation**: Content-dependent frequency response adjustment for better stylization
- **Multi-scale Analysis**: Processes different frequency bands with appropriate style weights

## Technical Approach

This implementation combines several advanced techniques:

1. **Multi-scale Wavelet Domain Feature Alignment**
   - Decompose features into frequency subbands using discrete wavelet transform
   - Apply different style transfer approaches to each subband
   - Novel recombination strategy with content-adaptive weighting

2. **Frequency-Selective Attention Mechanism**
   - Channel attention mechanism that operates differently on high vs. low frequency components
   - Analyze spectral statistics to determine where style should be emphasized

3. **Phase-Preserving Histogram Matching**
   - Match amplitude statistics in wavelet domain while preserving phase information
   - Novel approach to maintain content structure while transferring style characteristics

4. **Spectral Feature Consistency Regularization**
   - Enforce consistency between frequency components across scales
   - Novel regularization approach specific to wavelet domain

## Usage

```bash
python style_transfer.py --content path/to/content.jpg --style path/to/style.jpg
```

### Options

- `--content`: Path to content image
- `--style`: Path to style image
- `--output`: Directory to save results (default: ./results)
- `--size`: Size of output image (default: 512)
- `--mode`: Style transfer mode ('fast' or 'optimize')
- `--alpha_low`: Style weight for low frequencies (0-1)
- `--alpha_high`: Style weight for high frequencies (0-1)
- `--steps`: Number of optimization steps (for optimize mode)
- `--content_weight`: Weight for content loss (for optimize mode)
- `--style_weight`: Weight for style loss (for optimize mode)
- `--lr`: Learning rate for optimization (for optimize mode)
- `--cpu`: Force CPU usage
- `--visualize`: Generate wavelet and frequency visualizations

## Examples

### Fast Mode (Direct Feature Manipulation)

```bash
python style_transfer.py --content assets/input/0001.jpg --style assets/reference/0001.jpg --mode fast
```

### Optimize Mode (Gradient-Based Optimization)

```bash
python style_transfer.py --content assets/input/0001.jpg --style assets/reference/0001.jpg --mode optimize --steps 500
```

### With Visualizations

```bash
python style_transfer.py --content assets/input/0001.jpg --style assets/reference/0001.jpg --visualize
```

## Technical Details

### Wavelet Transform

Unlike FFT-based methods that only separate frequency components globally, wavelet transforms provide:

- Multi-resolution analysis
- Localized frequency information
- Better preservation of edges and textures
- Separate handling of approximation and detail coefficients

### Frequency-Selective Processing

The model applies different processing to different frequency bands:

- Low frequencies (approximation coefficients): Carry structural information, processed with lower style weights
- High frequencies (detail coefficients): Carry texture information, processed with higher style weights

### Pretrained Feature Extraction

VGG19 pretrained on ImageNet provides:

- Content features from higher layers (relu4_2)
- Style features from multiple layers (relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)

These features are then processed in the wavelet domain for style transfer.

## Results

The results directory contains:
- Stylized image
- Comparison of content, style, and result
- Visualizations of wavelet decomposition, frequency spectrum, and multi-scale analysis (if --visualize is used)
- Loss history for optimize mode