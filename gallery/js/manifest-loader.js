export class ManifestLoader {
    constructor() {
        this.manifestPath = './data/gallery-manifest.json';
        this.manifest = null;
    }

    async loadManifest() {
        if (this.manifest) {
            return this.manifest;
        }

        try {
            const response = await fetch(this.manifestPath);
            if (!response.ok) {
                throw new Error(`Failed to load manifest: ${response.status} ${response.statusText}`);
            }
            this.manifest = await response.json();
            console.log('Gallery manifest loaded successfully');
            return this.manifest;
        } catch (error) {
            console.error('Error loading gallery manifest:', error);
            this.manifest = {
                lastUpdated: new Date().toISOString(),
                contentImages: [],
                styleImages: [],
                presetResults: [],
                customResults: []
            };
            return this.manifest;
        }
    }

    async getContentImages() {
        const manifest = await this.loadManifest();
        return manifest.contentImages || [];
    }

    async getStyleImages() {
        const manifest = await this.loadManifest();
        return manifest.styleImages || [];
    }

    async getAllStyledImages() {
        const manifest = await this.loadManifest();
        const styledImages = [
            ...(manifest.presetResults || []),
            ...(manifest.customResults || [])
        ];
        
        return styledImages.filter(path => !path.includes('_comparison.jpg'));
    }
    
    async getBestImages() {
        const manifest = await this.loadManifest();
        const bestImages = manifest.bestResults || [];
        return bestImages.filter(path => !path.includes('_comparison.jpg'));
    }

    async getPresetResults() {
        const manifest = await this.loadManifest();
        return manifest.presetResults || [];
    }

    async getCustomResults() {
        const manifest = await this.loadManifest();
        return manifest.customResults || [];
    }
}