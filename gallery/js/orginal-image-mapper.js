// Helper module to map styled images to their original source images and inspiration art
export class OriginalImageMapper {
    constructor() {
        this.mappings = new Map();
        this.inspirationMappings = new Map();
        this.styleResultsPath = '../model/assets/_results';
        this.inputImagesPath = '../model/assets/input';
        this.inspirationPath = '../model/assets/inspiration';
        
        // Initialize mappings
        this.initializeMappings();
    }
    
    // Initialize known image mappings
    initializeMappings() {
        // Style variation examples
        this.addMapping('style_balanced.png', 'base_image.jpg', 'balanced_style.jpg');
        this.addMapping('style_classical.png', 'base_image.jpg', 'classical_style.jpg');
        this.addMapping('style_dramatic.png', 'base_image.jpg', 'dramatic_style.jpg');
        this.addMapping('style_painterly.png', 'base_image.jpg', 'painterly_style.jpg');
        
        // Comparison examples (already contain both images)
        this.addMapping('style_transfer_comparison.png', 'style_transfer_comparison.png', 'style_transfer_comparison.png');
        this.addMapping('best_result.jpg', 'best_result.jpg', 'best_result.jpg');
        
        // Specific examples
        this.addMapping('portrait_enhanced.jpg', 'portrait.jpg', 'enhanced_style.jpg');
        this.addMapping('portrait_starry_night.jpg', 'portrait.jpg', 'starry_night.jpg');
        this.addMapping('landscape_enhanced.jpg', 'landscape.jpg', 'enhanced_style.jpg');
        this.addMapping('landscape_abstract.jpg', 'landscape.jpg', 'abstract_style.jpg');
        this.addMapping('cityscape_enhanced.jpg', 'cityscape.jpg', 'enhanced_style.jpg');
        this.addMapping('cityscape_starry_night.jpg', 'cityscape.jpg', 'starry_night.jpg');
        
        // Wavelet results
        this.addWaveletMappings();
    }
    
    // Add mappings for wavelet style transfers
    addWaveletMappings() {
        // For wavelet results, the style ID is in the filename
        // Example: 0085_styled_with_0022.jpg
        // Original is 0085.jpg, style source is 0022.jpg
        
        // Standard style IDs
        const styleIds = ['0001', '0002', '0022', '0024', '0075'];
        
        // Add a generic mapping for each style
        styleIds.forEach(styleId => {
            this.addInspirationMapping(
                `_styled_with_${styleId}`,
                `${this.inspirationPath}/style_${styleId}.jpg`
            );
        });
    }
    
    // Add a mapping from styled to original and inspiration
    addMapping(styledFilename, originalFilename, inspirationFilename) {
        const styledPath = `${this.styleResultsPath}/${styledFilename}`;
        const originalPath = `${this.inputImagesPath}/${originalFilename}`;
        const inspirationPath = `${this.inspirationPath}/${inspirationFilename}`;
        
        this.mappings.set(styledPath, originalPath);
        this.inspirationMappings.set(styledPath, inspirationPath);
    }
    
    // Add just an inspiration mapping by pattern
    addInspirationMapping(pattern, inspirationPath) {
        this.inspirationMappings.set(pattern, inspirationPath);
    }
    
    // Get the original image path for a styled image
    getOriginalImagePath(styledImagePath) {
        // Check if we have a direct mapping
        if (this.mappings.has(styledImagePath)) {
            return this.mappings.get(styledImagePath);
        }
        
        // Check if it's a comparison file
        if (styledImagePath.includes('comparison')) {
            return styledImagePath; // Already contains both images
        }
        
        // Extract the base filename
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        
        // Check if it's a wavelet styled file
        if (filename.includes('styled_with')) {
            const baseImageId = filename.split('_styled_with_')[0];
            return `${this.inputImagesPath}/${baseImageId}.jpg`;
        }
        
        // For other style transfer results with format pattern_style.jpg
        const nameParts = filename.split('_');
        if (nameParts.length >= 2) {
            const basePattern = nameParts[0];
            return `${this.inputImagesPath}/${basePattern}.jpg`;
        }
        
        // Default fallback
        return `${this.inputImagesPath}/original.jpg`;
    }
    
    // Get the inspiration/style source image for a styled image
    getInspirationImagePath(styledImagePath) {
        // Check if we have a direct mapping
        if (this.inspirationMappings.has(styledImagePath)) {
            return this.inspirationMappings.get(styledImagePath);
        }
        
        // Check if it's a comparison file
        if (styledImagePath.includes('comparison')) {
            return styledImagePath; // Already contains style information
        }
        
        // Extract the base filename
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        
        // Check if it's a wavelet styled file
        if (filename.includes('styled_with')) {
            // Check for pattern matches first
            for (const [pattern, path] of this.inspirationMappings.entries()) {
                if (filename.includes(pattern)) {
                    return path;
                }
            }
            
            // Extract style ID if no pattern match
            const styleId = filename.split('_styled_with_')[1].split('.')[0];
            return `${this.inspirationPath}/style_${styleId}.jpg`;
        }
        
        // Look for style name in filename
        const stylePatterns = {
            'starry_night': 'starry_night.jpg',
            'abstract': 'abstract_style.jpg',
            'enhanced': 'enhanced_style.jpg',
            'classical': 'classical_style.jpg',
            'dramatic': 'dramatic_style.jpg',
            'painterly': 'painterly_style.jpg',
            'balanced': 'balanced_style.jpg'
        };
        
        for (const [pattern, file] of Object.entries(stylePatterns)) {
            if (filename.includes(pattern)) {
                return `${this.inspirationPath}/${file}`;
            }
        }
        
        // Default fallback
        return `${this.inspirationPath}/default_style.jpg`;
    }
}