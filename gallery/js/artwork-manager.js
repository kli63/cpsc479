import * as THREE from 'three';
import { ManifestLoader } from './manifest-loader.js';

export class ArtworkManager {
    constructor(scene, loadingManager) {
        this.scene = scene;
        this.loadingManager = loadingManager || new THREE.LoadingManager();
        this.textureLoader = new THREE.TextureLoader(this.loadingManager);
        this.styleTransferPath = '../model/results';
        this.contentImagesPath = '../model/assets/input';
        this.styleImagesPath = '../model/assets/reference';
        this.artworks = [];
        this.allowDuplicates = true;
        this.manifestLoader = new ManifestLoader();
    }

    async getAllStyleTransferImages() {
        const uniqueImagesMap = new Map();
        
        try {
            // Use the manifest loader instead of server request
            const styledImages = await this.manifestLoader.getAllStyledImages();
            
            styledImages.forEach(imagePath => {
                // Skip comparison images
                if (imagePath.includes('_comparison')) {
                    return;
                }
                
                const relPath = imagePath.startsWith('/') ? `..${imagePath}` : imagePath;
                const parts = relPath.split('/');
                const filename = parts[parts.length - 1];
                const match = filename.match(/(\d+)_styled_with_(\d+)/) || 
                             filename.match(/(\d+)_(\d+)_/);
                
                if (match) {
                    const key = `${match[1]}_${match[2]}`;
                    uniqueImagesMap.set(key, relPath);
                }
            });
        } catch (error) {
            console.error("Error loading styled images from manifest:", error);
            
            // Fallback images if manifest fails
            const fallbackImages = [
                `${this.styleTransferPath}/style_transfer_20250505_104324/0002_styled_with_0022.jpg`, 
                `${this.styleTransferPath}/style_transfer_20250505_104421/0001_styled_with_0022.jpg`
            ];
            
            fallbackImages.forEach(imagePath => {
                const parts = imagePath.split('/');
                const filename = parts[parts.length - 1];
                const match = filename.match(/(\d+)_styled_with_(\d+)/);
                
                if (match) {
                    const key = `${match[1]}_${match[2]}`;
                    uniqueImagesMap.set(key, imagePath);
                }
            });
        }
        
        const uniqueImages = Array.from(uniqueImagesMap.values());
        console.log(`Loaded ${uniqueImages.length} styled images from manifest`);
        
        const requiredFeaturedCount = 6; 
        const requiredRegularCount = 20;
        const totalRequired = requiredFeaturedCount + requiredRegularCount;
        
        let featuredImages, regularImages;
        
        if (this.allowDuplicates && uniqueImages.length < totalRequired) {
            const duplicatedSet = [];
            
            while (duplicatedSet.length < totalRequired) {
                duplicatedSet.push(...uniqueImages);
            }
            
            const shuffled = this.shuffleArray(duplicatedSet);
            featuredImages = shuffled.slice(0, requiredFeaturedCount);
            regularImages = shuffled.slice(requiredFeaturedCount, totalRequired);
        } else {
            const shuffled = this.shuffleArray([...uniqueImages]);
            featuredImages = shuffled.slice(0, Math.min(requiredFeaturedCount, shuffled.length));
            
            const remainingForRegular = Math.max(0, shuffled.length - requiredFeaturedCount);
            regularImages = shuffled.slice(requiredFeaturedCount, requiredFeaturedCount + remainingForRegular);
        }
        
        return {
            featured: featuredImages,
            regular: regularImages
        };
    }
    
    getImageMetadata(imagePath) {
        const filename = imagePath.split('/').pop();
        const match = filename.match(/(\d+)_styled_with_(\d+)/);
        
        if (match) {
            const contentId = match[1];
            const styleId = match[2];
            
            return {
                contentImage: `../model/assets/input/${contentId}.jpg`,
                styleImage: `../model/assets/reference/${styleId}.jpg`,
                comparisonImage: null
            };
        }
        
        return null;
    }
    
    shuffleArray(array) {
        const shuffled = array.slice();
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    createArtworkFrame(texture, position, rotation, size = { width: 2, height: 1.5 }, imagePath) {
        const frameWidth = size.width + 0.1;
        const frameHeight = size.height + 0.1;
        const frameDepth = 0.05;
        
        const frameMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x8B4513,
            roughness: 0.5,
            metalness: 0.2
        });
        
        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0xC3A066,
            roughness: 0.3,
            metalness: 0.4,
            emissive: 0xC3A066,
            emissiveIntensity: 0.5
        });
        
        const frame = new THREE.Group();
        
        const metadata = this.getImageMetadata(imagePath);
        
        frame.userData = {
            isArtwork: true,
            imagePath: imagePath,
            originalMaterial: frameMaterial,
            glowMaterial: glowMaterial,
            frameParts: [],
            metadata: metadata
        };

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
        
        frame.userData.setHover = function(isHovered) {
            const material = isHovered ? this.glowMaterial : this.originalMaterial;
            for (const part of this.frameParts) {
                part.material = material;
            }
            
            if (this.highlight) {
                this.highlight.material.opacity = isHovered ? 0.2 : 0;
            }
        };

        return frame;
    }

    loadArtwork(imagePath, position, rotation, size) {
        return new Promise((resolve, reject) => {
            console.log(`Loading image: ${imagePath}`);
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

    arrangeCentralWallArtworks(centralWall, artworkPaths, isBackSide = false) {
        const promises = [];
        const wallWidth = centralWall.geometry.parameters.width;
        const wallHeight = centralWall.geometry.parameters.height;
        
        const rows = 2;
        const cols = 3;
        const frameWidth = wallWidth / (cols + 1) * 0.6;
        const frameHeight = frameWidth * 0.75;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const index = row * cols + col;
                if (index >= artworkPaths.length) break;
                
                const x = (col - (cols - 1) / 2) * (frameWidth * 1.2);
                
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

    arrangeOuterWallArtworks(wall, artworkPaths, count = 5) {
        const promises = [];
        const imagesToUse = artworkPaths.slice(0, count);
        const wallWidth = wall.geometry.parameters.width;
        const wallHeight = wall.geometry.parameters.height;
        
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
        
        const spacing = wallWidth / (count + 1);
        const frameWidth = Math.min(spacing * 0.8, 2);
        const frameHeight = frameWidth * 0.75;
        
        const isHorizontalWall = wallDirection === 'north' || wallDirection === 'south';
        
        for (let i = 0; i < count; i++) {
            if (i >= imagesToUse.length) break;
            
            let position, rotation;
            
            const artworkHeight = wallHeight * 0.55;
            
            if (isHorizontalWall) {
                const x = (i - (count - 1) / 2) * spacing;
                position = new THREE.Vector3(
                    wall.position.x + x,
                    artworkHeight,
                    wall.position.z + (wallDirection === 'north' ? 0.05 : -0.05)
                );
                rotation = new THREE.Euler(0, wallDirection === 'south' ? Math.PI : 0, 0);
            } else {
                const z = (i - (count - 1) / 2) * spacing;
                position = new THREE.Vector3(
                    wall.position.x + (wallDirection === 'east' ? -0.05 : 0.05),
                    artworkHeight,
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

    async populateGallery(wallsMap) {
        try {
            const { featured, regular } = await this.getAllStyleTransferImages();
            const promises = [];
            
            if (wallsMap.central.front) {
                const frontSideImages = featured.slice(0, 6);
                const backSideImages = featured.slice(0, 6).reverse();
                
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.front, frontSideImages, false));
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.back, backSideImages, true));
            }
            
            let startIndex = 0;
            const imagesPerWall = 5;
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.north, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.south, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
            promises.push(...this.arrangeOuterWallArtworks(
                wallsMap.east, 
                regular.slice(startIndex, startIndex + imagesPerWall), 
                imagesPerWall
            ));
            startIndex += imagesPerWall;
            
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