export class ImageMapper {
    constructor() {
        this.mappings = new Map();
        this.styleResultsPath = '../model/results';
        this.inputImagesPath = '../model/assets/input';
        this.referencePath = '../model/assets/reference';
        
        this.initializeMappings();
    }
    
    initializeMappings() {
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
        
        this.stylePatterns = stylePatterns;
    }
    
    getOriginalImagePath(styledImagePath) {
        if (!styledImagePath) return '';
        
        if (styledImagePath.includes('comparison') || 
            styledImagePath.includes('best_result')) {
            return styledImagePath;
        }
        
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        if (/^\d{4}_\d{4}/.test(filename)) {
            const baseId = filename.split('_')[0];
            return `${this.inputImagesPath}/${baseId}.jpg`;
        }
        
        if (filename.includes('styled_with')) {
            const baseId = filename.split('_styled_with_')[0];
            return `${this.inputImagesPath}/${baseId}.jpg`;
        }
        
        if (filename.includes('_')) {
            const subject = filename.split('_')[0];
            return `${this.inputImagesPath}/${subject}.jpg`;
        }
        
        return styledImagePath;
    }
    
    getStyleImagePath(styledImagePath) {
        if (!styledImagePath) return '';
        
        if (styledImagePath.includes('comparison') || 
            styledImagePath.includes('best_result')) {
            return '';
        }
        
        const parts = styledImagePath.split('/');
        const filename = parts[parts.length - 1];
        
        if (/^\d{4}_\d{4}/.test(filename)) {
            const styleId = filename.split('_')[1];
            return `${this.referencePath}/${styleId}.jpg`;
        }
        
        if (filename.includes('styled_with')) {
            const styleId = filename.split('_styled_with_')[1].split('.')[0];
            return `${this.referencePath}/${styleId}.jpg`;
        }
        
        for (const { pattern, type } of this.stylePatterns) {
            if (filename.includes(pattern)) {
                return `${this.referencePath}/style_${pattern.replace('_', '')}.jpg`;
            }
        }
        
        return '';
    }
}