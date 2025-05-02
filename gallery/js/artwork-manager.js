import * as THREE from 'three';

export class ArtworkManager {
    constructor(scene, loadingManager) {
        this.scene = scene;
        this.loadingManager = loadingManager || new THREE.LoadingManager();
        this.textureLoader = new THREE.TextureLoader(this.loadingManager);
        this.styleTransferPath = '../model/assets/_results';
        this.artworks = [];
    }

    // Get all style transfer images from the results folder
    async getAllStyleTransferImages() {
        // Filter images that are clearly result images
        const patternMatchers = [
            /_painterly\.png$/,
            /_classical\.png$/,
            /_dramatic\.png$/,
            /_balanced\.png$/,
            /_styled_with_.*\.jpg$/,
            /portrait_.*\.jpg$/,
            /landscape_.*\.jpg$/,
            /cityscape_.*\.jpg$/
        ];
        
        // Featured images for the central wall
        const featuredImages = [
            `${this.styleTransferPath}/style_transfer_comparison.png`,
            `${this.styleTransferPath}/best_result.jpg`,
            `${this.styleTransferPath}/0028_0011_all_presets.png`,
            `${this.styleTransferPath}/0031_0022_all_presets.png`,
            `${this.styleTransferPath}/0032_0024_all_presets.png`,
            `${this.styleTransferPath}/0029_0009_all_presets.png`
        ];
        
        // Regular style transfer images
        const styleTransferImages = [
            `${this.styleTransferPath}/0028_0011_painterly.png`,
            `${this.styleTransferPath}/0028_0011_classical.png`,
            `${this.styleTransferPath}/0028_0011_dramatic.png`,
            `${this.styleTransferPath}/0028_0011_balanced.png`,
            `${this.styleTransferPath}/0031_0022_painterly.png`,
            `${this.styleTransferPath}/0031_0022_classical.png`,
            `${this.styleTransferPath}/0031_0022_dramatic.png`,
            `${this.styleTransferPath}/0031_0022_balanced.png`,
            `${this.styleTransferPath}/0032_0024_painterly.png`,
            `${this.styleTransferPath}/0032_0024_classical.png`,
            `${this.styleTransferPath}/0032_0024_dramatic.png`,
            `${this.styleTransferPath}/0032_0024_balanced.png`,
            `${this.styleTransferPath}/0029_0009_painterly.png`,
            `${this.styleTransferPath}/0029_0009_classical.png`,
            `${this.styleTransferPath}/0029_0009_dramatic.png`,
            `${this.styleTransferPath}/0029_0009_balanced.png`,
            `${this.styleTransferPath}/0075_0022_painterly.png`,
            `${this.styleTransferPath}/0075_0005_painterly.png`,
            `${this.styleTransferPath}/0002_0022_painterly.png`,
            `${this.styleTransferPath}/0002_0005_painterly.png`,
            `${this.styleTransferPath}/0024_0022_painterly.png`,
            `${this.styleTransferPath}/0024_0005_painterly.png`,
            `${this.styleTransferPath}/portrait_enhanced.jpg`,
            `${this.styleTransferPath}/portrait_starry_night.jpg`,
            `${this.styleTransferPath}/landscape_enhanced.jpg`,
            `${this.styleTransferPath}/landscape_abstract.jpg`,
            `${this.styleTransferPath}/cityscape_enhanced.jpg`,
            `${this.styleTransferPath}/cityscape_starry_night.jpg`
        ];
        
        return {
            featured: featuredImages,
            regular: this.shuffleArray(styleTransferImages)
        };
    }
    
    // Shuffle array for random selection
    shuffleArray(array) {
        const shuffled = array.slice();
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // Create a frame with artwork
    createArtworkFrame(texture, position, rotation, size = { width: 2, height: 1.5 }, imagePath) {
        const frameWidth = size.width + 0.1;
        const frameHeight = size.height + 0.1;
        const frameDepth = 0.05;
        
        // Regular frame material
        const frameMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x8B4513,
            roughness: 0.5,
            metalness: 0.2
        });
        
        // Glow material for hover effect
        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0xC3A066,
            roughness: 0.3,
            metalness: 0.4,
            emissive: 0xC3A066,
            emissiveIntensity: 0.5
        });
        
        // Create frame group
        const frame = new THREE.Group();
        frame.userData = {
            isArtwork: true,
            imagePath: imagePath,
            originalMaterial: frameMaterial,
            glowMaterial: glowMaterial,
            frameParts: [] // To store frame parts for hover effect
        };

        // Frame borders
        const topBorder = new THREE.Mesh(
            new THREE.BoxGeometry(frameWidth, 0.05, frameDepth),
            frameMaterial
        );
        topBorder.position.y = frameHeight / 2 - 0.025;
        frame.add(topBorder);
        frame.userData.frameParts.push(topBorder);

        const bottomBorder = new THREE.Mesh(
            new THREE.BoxGeometry(frameWidth, 0.05, frameDepth),
            frameMaterial
        );
        bottomBorder.position.y = -frameHeight / 2 + 0.025;
        frame.add(bottomBorder);
        frame.userData.frameParts.push(bottomBorder);

        const leftBorder = new THREE.Mesh(
            new THREE.BoxGeometry(0.05, frameHeight, frameDepth),
            frameMaterial
        );
        leftBorder.position.x = -frameWidth / 2 + 0.025;
        frame.add(leftBorder);
        frame.userData.frameParts.push(leftBorder);

        const rightBorder = new THREE.Mesh(
            new THREE.BoxGeometry(0.05, frameHeight, frameDepth),
            frameMaterial
        );
        rightBorder.position.x = frameWidth / 2 - 0.025;
        frame.add(rightBorder);
        frame.userData.frameParts.push(rightBorder);

        // Background plane behind artwork for highlight effect
        const highlightMaterial = new THREE.MeshBasicMaterial({
            color: 0xFFFFFF,
            opacity: 0,
            transparent: true
        });
        
        const highlight = new THREE.Mesh(
            new THREE.PlaneGeometry(size.width + 0.02, size.height + 0.02),
            highlightMaterial
        );
        highlight.position.z = frameDepth / 2;
        frame.add(highlight);
        frame.userData.highlight = highlight;

        // Artwork plane
        const artworkMaterial = new THREE.MeshBasicMaterial({ 
            map: texture,
            side: THREE.FrontSide
        });
        
        const artwork = new THREE.Mesh(
            new THREE.PlaneGeometry(size.width, size.height),
            artworkMaterial
        );
        artwork.position.z = frameDepth / 2 + 0.002;
        frame.add(artwork);

        frame.position.copy(position);
        frame.rotation.copy(rotation);
        
        // Add hover and unhover methods
        frame.userData.setHover = function(isHovered) {
            // Apply glow effect to frame parts
            const material = isHovered ? this.glowMaterial : this.originalMaterial;
            for (const part of this.frameParts) {
                part.material = material;
            }
            
            // Apply highlight effect
            if (this.highlight) {
                this.highlight.material.opacity = isHovered ? 0.2 : 0;
            }
        };

        return frame;
    }

    loadArtwork(imagePath, position, rotation, size) {
        return new Promise((resolve, reject) => {
            this.textureLoader.load(
                imagePath,
                (texture) => {
                    const frame = this.createArtworkFrame(
                        texture,
                        position,
                        rotation,
                        size,
                        imagePath
                    );
                    this.scene.add(frame);
                    this.artworks.push(frame);
                    resolve(frame);
                },
                undefined,
                (error) => {
                    console.error(`Error loading texture: ${imagePath}`, error);
                    reject(error);
                }
            );
        });
    }

    // Layout images on the central wall
    arrangeCentralWallArtworks(centralWall, artworkPaths, isBackSide = false) {
        const promises = [];
        const wallWidth = centralWall.geometry.parameters.width;
        const wallHeight = centralWall.geometry.parameters.height;
        
        // Create a grid of images
        const rows = 2;
        const cols = 3;
        const frameWidth = wallWidth / (cols + 1) * 0.6;
        const frameHeight = frameWidth * 0.75;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const index = row * cols + col;
                if (index >= artworkPaths.length) break;
                
                const x = (col - (cols - 1) / 2) * (frameWidth * 1.2);
                
                // Fixed: Ensure images are placed at proper height on the wall
                // Calculate position between 25% and 75% of wall height based on row
                const heightPercent = 0.7 - (row * 0.4);
                const y = wallHeight * heightPercent;
                
                // Calculate z-offset based on which side, to avoid z-fighting
                const zOffset = isBackSide ? -0.05 : 0.05;
                
                const worldY = centralWall.parent ? 
                    centralWall.parent.position.y + y : y;
                
                const position = new THREE.Vector3(
                    centralWall.position.x + x,
                    worldY,
                    centralWall.position.z + zOffset
                );
                
                // Rotation needs to account for which side of the wall
                const rotation = isBackSide ? 
                    new THREE.Euler(0, Math.PI, 0) : 
                    new THREE.Euler(0, 0, 0);
                
                promises.push(
                    this.loadArtwork(
                        artworkPaths[index],
                        position,
                        rotation,
                        {width: frameWidth, height: frameHeight}
                    )
                );
            }
        }
        
        return promises;
    }

    // Layout images on outer walls
    arrangeOuterWallArtworks(wall, artworkPaths, count = 5) {
        const promises = [];
        const imagesToUse = artworkPaths.slice(0, count);
        const wallWidth = wall.geometry.parameters.width;
        const wallHeight = wall.geometry.parameters.height;
        
        // Find wall orientation
        let wallDirection;
        if (Math.abs(wall.rotation.y - Math.PI/2) < 0.1) {
            wallDirection = 'west';
        } else if (Math.abs(wall.rotation.y + Math.PI/2) < 0.1) {
            wallDirection = 'east';
        } else if (Math.abs(wall.rotation.y) < 0.1) {
            wallDirection = 'north';
        } else if (Math.abs(Math.abs(wall.rotation.y) - Math.PI) < 0.1) {
            wallDirection = 'south';
        }
        
        // Calculate positions based on wall direction
        const spacing = wallWidth / (count + 1);
        const frameWidth = Math.min(spacing * 0.8, 2);
        const frameHeight = frameWidth * 0.75;
        
        const isHorizontalWall = wallDirection === 'north' || wallDirection === 'south';
        
        for (let i = 0; i < count; i++) {
            if (i >= imagesToUse.length) break;
            
            let position, rotation;
            
            // Fix: Place artwork at consistent height on walls (eye level)
            const artworkHeight = wallHeight * 0.55; // Position at ~55% of wall height for good viewing
            
            if (isHorizontalWall) {
                const x = (i - (count - 1) / 2) * spacing;
                position = new THREE.Vector3(
                    wall.position.x + x,
                    artworkHeight, // Fixed height
                    wall.position.z + (wallDirection === 'north' ? 0.05 : -0.05)
                );
                rotation = new THREE.Euler(0, wallDirection === 'south' ? Math.PI : 0, 0);
            } else {
                const z = (i - (count - 1) / 2) * spacing;
                position = new THREE.Vector3(
                    wall.position.x + (wallDirection === 'east' ? -0.05 : 0.05),
                    artworkHeight, // Fixed height
                    wall.position.z + z
                );
                rotation = new THREE.Euler(0, wallDirection === 'east' ? -Math.PI/2 : Math.PI/2, 0);
            }
            
            promises.push(
                this.loadArtwork(
                    imagesToUse[i],
                    position,
                    rotation,
                    {width: frameWidth, height: frameHeight}
                )
            );
        }
        
        return promises;
    }

    // Load all artworks into the gallery
    async populateGallery(wallsMap) {
        try {
            const { featured, regular } = await this.getAllStyleTransferImages();
            const promises = [];
            
            // Central wall - different featured images on each side
            if (wallsMap.central.front) {
                const frontSideImages = featured.slice(0, 6);
                const backSideImages = featured.slice(0, 6).reverse();
                
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.front, frontSideImages, false));
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.back, backSideImages, true));
            }
            
            // Outer walls - random selection of regular images
            let startIndex = 0;
            const imagesPerWall = 5;
            
            // North wall
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.north, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
            // South wall
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.south, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
            // East wall
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.east, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
            // West wall
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.west, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            
            const artworks = await Promise.all(promises);
            console.log(`Loaded ${artworks.length} artworks into the gallery`);
            
            return artworks;
        } catch (error) {
            console.error('Failed to load artworks:', error);
            return [];
        }
    }
}