// Helper module to map styled images to their original source images
export class ImageMapper {
    constructor() {
        this.mappings = new Map();
        this.styleResultsPath = '../model/assets/_results';
        this.inputImagesPath = '../model/assets/input';
        this.referencePath = '../model/assets/reference';
        
        this.initializeMappings();
    }
    
    initializeMappings() {
        // Pre-defined mappings
        const stylePatterns = [
            { pattern: '_balanced', type: 'preset' },
            { pattern: '_classical', type: 'preset' },
            { pattern: '_dramatic', type: 'preset' },
            { pattern: '_painterly', type: 'preset' },
            { pattern: '_all_presets', type: 'preset' },
            { pattern: 'starry_night', type: 'specific' },
            { pattern: 'enhanced', type: 'specific' },
            { pattern: 'abstract', type: 'specific' }
        ];
        
        // Identify image base and style from filename patterns
        this.stylePatterns = stylePatterns;
    }
    
    // Get the original image path for a styled image
    getOriginalImagePath(styledImagePath) {
        if (!styledImagePath) return '';
        
        // Check if it's a comparison image
        if (styledImagePath.includes('comparison') || 
            styledImagePath.includes('best_result')) {
            return styledImagePath;
        }
        
        // Extract the base ID from various naming patterns
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        
        // Pattern: 0000_0000_style.png - Extract the first ID
        if (/^\d{4}_\d{4}/.test(filename)) {
            const baseId = filename.split('_')[0];
            return `${this.inputImagesPath}/${baseId}.jpg`;
        }
        
        // Pattern: 0000_styled_with_0000.jpg or similar
        if (filename.includes('styled_with')) {
            const baseId = filename.split('_styled_with_')[0];
            return `${this.inputImagesPath}/${baseId}.jpg`;
        }
        
        // Pattern: subject_style.jpg (e.g., portrait_enhanced.jpg)
        if (filename.includes('_')) {
            const subject = filename.split('_')[0];
            return `${this.inputImagesPath}/${subject}.jpg`;
        }
        
        // Default fallback - return the same image
        return styledImagePath;
    }
    
    // Get the style reference image if available
    getStyleImagePath(styledImagePath) {
        if (!styledImagePath) return '';
        
        // Check if it's a comparison image
        if (styledImagePath.includes('comparison') || 
            styledImagePath.includes('best_result')) {
            return '';
        }
        
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        
        // Pattern: 0000_0000_style.png - Extract the second ID
        if (/^\d{4}_\d{4}/.test(filename)) {
            const styleId = filename.split('_')[1];
            return `${this.referencePath}/${styleId}.jpg`;
        }
        
        // Pattern: 0000_styled_with_0000.jpg or similar
        if (filename.includes('styled_with')) {
            const styleId = filename.split('_styled_with_')[1].split('.')[0];
            return `${this.referencePath}/${styleId}.jpg`;
        }
        
        // Pattern: subject_style.jpg (e.g., portrait_enhanced.jpg)
        for (const { pattern, type } of this.stylePatterns) {
            if (filename.includes(pattern)) {
                return `${this.referencePath}/style_${pattern.replace('_', '')}.jpg`;
            }
        }
        
        // Default fallback - no style reference
        return '';
    }
}