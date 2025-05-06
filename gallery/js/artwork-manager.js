import * as THREE from 'three';
import { ManifestLoader } from './manifest-loader.js';

export class ArtworkManager {
    constructor(scene, loadingManager) {
        this.scene = scene;
        this.loadingManager = loadingManager || new THREE.LoadingManager();
        this.textureLoader = new THREE.TextureLoader(this.loadingManager);
        this.basePath = window.location.hostname.includes('github.io') ? '/cpsc479/FP/' : '';
        this.styleTransferPath = `${this.basePath}model/results`;
        this.contentImagesPath = `${this.basePath}model/assets/input`;
        this.styleImagesPath = `${this.basePath}model/assets/reference`;
        this.bestImagesPath = `${this.basePath}gallery/assets/best`;
        this.artworks = [];
        this.allowDuplicates = false;
        this.manifestLoader = new ManifestLoader();
    }
    
    adjustPath(path) {
        if (!path) return path;
        
        if (this.basePath && !path.startsWith(this.basePath) && !path.startsWith('/')) {
            return `${this.basePath}${path}`;
        }
        
        if (!this.basePath && path.startsWith('/')) {
            return path.substring(1);
        }
        
        return path;
    }

    async getAllStyleTransferImages() {
        const uniqueImagesMap = new Map();
        
        try {
            const styledImages = await this.manifestLoader.getAllStyledImages();
            
            styledImages.forEach(imagePath => {
                if (imagePath.includes('_comparison')) {
                    return;
                }
                
                const relPath = imagePath.startsWith('/') ? imagePath.substring(1) : imagePath;
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
        console.log(`Loaded ${uniqueImages.length} styled images from manifest and directory structure`);
        
        const requiredFeaturedCount = 8; 
        const requiredRegularCount = 20;
        const totalRequired = requiredFeaturedCount + requiredRegularCount;
        
        let featuredImages, regularImages;
        
        const shuffled = this.shuffleArray([...uniqueImages]);
        featuredImages = shuffled.slice(0, Math.min(requiredFeaturedCount, shuffled.length));
        
        const remainingForRegular = Math.max(0, shuffled.length - requiredFeaturedCount);
        regularImages = shuffled.slice(requiredFeaturedCount, requiredFeaturedCount + remainingForRegular);
        
        if (featuredImages.length < requiredFeaturedCount || regularImages.length < requiredRegularCount) {
            console.warn("Not enough unique images found, keeping unique images only");
            this.allowDuplicates = false;
            
            if (this.allowDuplicates) {
                const duplicatedSet = [];
                
                while (duplicatedSet.length < totalRequired) {
                    duplicatedSet.push(...uniqueImages);
                }
                
                const shuffled = this.shuffleArray(duplicatedSet);
                featuredImages = shuffled.slice(0, requiredFeaturedCount);
                regularImages = shuffled.slice(requiredFeaturedCount, totalRequired);
            }
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
                contentImage: `model/assets/input/${contentId}.jpg`,
                styleImage: `model/assets/reference/${styleId}.jpg`,
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
            const adjustedPath = this.adjustPath(imagePath);
            console.log(`Loading image: ${adjustedPath}`);
            this.textureLoader.load(
                adjustedPath,
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
                    console.error(`Error loading texture: ${adjustedPath}`, error);
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
        const cols = 4;
        
        const frameWidth = wallWidth * 0.17;
        const frameHeight = frameWidth * 0.75;
        const spacing = frameWidth * 0.45;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const index = row * cols + col;
                if (index >= artworkPaths.length) break;
                
                const totalUsedWidth = (cols * frameWidth) + ((cols - 1) * spacing);
                const startX = -totalUsedWidth / 2;
                const x = startX + (col * (frameWidth + spacing)) + (frameWidth / 2);
                
                const heightPercent = row === 0 ? 0.735 : 0.265;
                const y = wallHeight * heightPercent;
                
                const zOffset = isBackSide ? -0.05 : 0.05;
                
                const worldY = centralWall.parent ? 
                    centralWall.parent.position.y + y : y;
                
                const position = new THREE.Vector3(
                    centralWall.position.x + x,
                    worldY,
                    centralWall.position.z + zOffset
                );
                
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
            let bestImages = [];
            try {
                bestImages = await this.manifestLoader.getBestImages();
            } catch (error) {
                console.warn("Error loading best images, falling back to featured images");
            }
            
            const { featured, regular } = await this.getAllStyleTransferImages();
            
            let processedBestImages = bestImages.map(path => {
                if (path.startsWith('/gallery/')) {
                    return path.substring(1);
                } else if (path.startsWith('/')) {
                    return path.substring(1);
                }
                return path;
            });
            
            const promises = [];
            
            if (wallsMap.central.front) {
                const extractKey = (path) => {
                    const filename = path.split('/').pop();
                    const match = filename.match(/(\d+)_styled_with_(\d+)/);
                    if (match) {
                        return `${match[1]}_${match[2]}`;
                    }
                    return null;
                };
                
                const frontWallMax = 8;
                const backWallMax = 8;
                
                const filteredBestImages = processedBestImages.filter(path => !path.includes('_comparison'));
                const filteredFeatured = featured.filter(path => !path.includes('_comparison'));
                const filteredRegular = regular.filter(path => !path.includes('_comparison'));
                
                const allImages = [...filteredBestImages, ...filteredFeatured, ...filteredRegular];
                console.log(`Total available unique images: ${allImages.length}`);
                
                const frontWallImages = [];
                const backWallImages = [];
                
                const usedKeys = new Set();
                
                if (filteredBestImages.length > 0) {
                    const bestImagesShuffled = this.shuffleArray([...filteredBestImages]);
                    
                    for (const path of bestImagesShuffled) {
                        if (frontWallImages.length >= frontWallMax) break;
                        
                        const key = extractKey(path);
                        if (key && !usedKeys.has(key)) {
                            frontWallImages.push(path);
                            usedKeys.add(key);
                        }
                    }
                    
                    for (const path of bestImagesShuffled) {
                        if (backWallImages.length >= backWallMax) break;
                        
                        const key = extractKey(path);
                        if (key && !usedKeys.has(key)) {
                            backWallImages.push(path);
                            usedKeys.add(key);
                        }
                    }
                }
                
                const regularImagesShuffled = this.shuffleArray([...filteredFeatured, ...filteredRegular]);
                
                if (frontWallImages.length < frontWallMax) {
                    for (const path of regularImagesShuffled) {
                        if (frontWallImages.length >= frontWallMax) break;
                        
                        const key = extractKey(path);
                        if (key && !usedKeys.has(key)) {
                            frontWallImages.push(path);
                            usedKeys.add(key);
                        }
                    }
                }
                
                if (backWallImages.length < backWallMax) {
                    for (const path of regularImagesShuffled) {
                        if (backWallImages.length >= backWallMax) break;
                        
                        const key = extractKey(path);
                        if (key && !usedKeys.has(key)) {
                            backWallImages.push(path);
                            usedKeys.add(key);
                        }
                    }
                }
                
                if (frontWallImages.length < frontWallMax || backWallImages.length < backWallMax) {
                    const allImagesShuffled = this.shuffleArray([...allImages]);
                    
                    if (frontWallImages.length < frontWallMax) {
                        let index = 0;
                        while (frontWallImages.length < frontWallMax && index < 1000) {
                            const path = allImagesShuffled[index % allImagesShuffled.length];
                            frontWallImages.push(path);
                            index++;
                        }
                    }
                    
                    if (backWallImages.length < backWallMax) {
                        let index = 0;
                        while (backWallImages.length < backWallMax && index < 1000) {
                            const path = allImagesShuffled[index % allImagesShuffled.length];
                            backWallImages.push(path);
                            index++;
                        }
                    }
                }
                
                console.log(`Front wall: ${frontWallImages.length}/${frontWallMax}, Back wall: ${backWallImages.length}/${backWallMax}`);
                
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.front, frontWallImages, false));
                promises.push(...this.arrangeCentralWallArtworks(wallsMap.central.back, backWallImages, true));
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