// Utility script to handle loading style transfer images into the gallery
import { GalleryLoader } from './gallery-loader.js';

// This class will handle loading images from the style transfer project
export class StyleTransferImageLoader {
    constructor(scene, loadingManager) {
        this.galleryLoader = new GalleryLoader(scene, loadingManager);
        this.styleTransferPath = '../model/assets/_results';
        this.resultsPath = '../model/results';
    }

    // Get a subset of images from the style transfer results
    async getStyleTransferImages() {        
        const styleTransferImages = [
            // Style comparisons
            `${this.styleTransferPath}/style_balanced.png`,
            `${this.styleTransferPath}/style_classical.png`,
            `${this.styleTransferPath}/style_dramatic.png`,
            `${this.styleTransferPath}/style_painterly.png`,
            
            // Full examples
            `${this.styleTransferPath}/style_transfer_comparison.png`,
            `${this.styleTransferPath}/best_result.jpg`,
            
            // Specific examples
            `${this.styleTransferPath}/portrait_enhanced.jpg`,
            `${this.styleTransferPath}/portrait_starry_night.jpg`,
            `${this.styleTransferPath}/landscape_enhanced.jpg`,
            `${this.styleTransferPath}/landscape_abstract.jpg`,
            `${this.styleTransferPath}/cityscape_enhanced.jpg`,
            `${this.styleTransferPath}/cityscape_starry_night.jpg`,
            
            // Wavelet results
            `${this.resultsPath}/wavelet_style_20250428_143134/0085_styled_with_0022.jpg`,
            `${this.resultsPath}/wavelet_style_20250428_143134/0085_styled_with_0022_comparison.jpg`,
        ];
        
        return styleTransferImages;
    }

    // Load all style transfer images into the gallery
    async loadGallery(roomDimensions) {
        try {
            const images = await this.getStyleTransferImages();
            return await this.galleryLoader.arrangeArtworks(images, roomDimensions);
        } catch (error) {
            console.error('Error loading gallery:', error);
            return [];
        }
    }

    // Get visualization images to show the style transfer process
    async getVisualizationImages(imageId) {
        // Image IDs that have visualizations: 0002, 0024, 0075
        const validIds = ['0002', '0024', '0075'];
        
        if (!validIds.includes(imageId)) {
            console.warn(`No visualization available for image ID: ${imageId}`);
            return [];
        }
        
        return [
            `${this.styleTransferPath}/${imageId}_visualization.png`,
            `${this.styleTransferPath}/${imageId}_depth_map.png`,
            
            // Gaussian pyramid
            `${this.styleTransferPath}/${imageId}_gaussian_level_0.png`,
            `${this.styleTransferPath}/${imageId}_gaussian_level_1.png`,
            `${this.styleTransferPath}/${imageId}_gaussian_level_2.png`,
            `${this.styleTransferPath}/${imageId}_gaussian_level_3.png`,
            `${this.styleTransferPath}/${imageId}_gaussian_level_4.png`,
            
            // Laplacian pyramid
            `${this.styleTransferPath}/${imageId}_laplacian_level_0.png`,
            `${this.styleTransferPath}/${imageId}_laplacian_level_1.png`,
            `${this.styleTransferPath}/${imageId}_laplacian_level_2.png`,
            `${this.styleTransferPath}/${imageId}_laplacian_level_3.png`,
            `${this.styleTransferPath}/${imageId}_laplacian_level_4.png`,
        ];
    }
}