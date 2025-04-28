import * as THREE from 'three';

// This module will be responsible for loading artwork into the gallery
export class GalleryLoader {
    constructor(scene, loadingManager) {
        this.scene = scene;
        this.loadingManager = loadingManager || new THREE.LoadingManager();
        this.textureLoader = new THREE.TextureLoader(this.loadingManager);
        this.artworks = [];
    }

    // Load artworks from a directory
    async loadArtworksFromDirectory(directoryPath) {
        console.log(`Would load artworks from: ${directoryPath}`);
        return "Artwork loading placeholder";
    }

    // Create a frame with the given artwork texture
    createArtworkFrame(texture, position, rotation, size = { width: 2, height: 1.5 }) {
        // Frame
        const frameWidth = size.width + 0.1;
        const frameHeight = size.height + 0.1;
        const frameDepth = 0.05;
        const frameMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x8B4513,
            roughness: 0.5,
            metalness: 0.2
        });
        
        // Create frame borders
        const frame = new THREE.Group();

        // Top border
        const topBorder = new THREE.Mesh(
            new THREE.BoxGeometry(frameWidth, 0.05, frameDepth),
            frameMaterial
        );
        topBorder.position.y = frameHeight / 2 - 0.025;
        frame.add(topBorder);

        // Bottom border
        const bottomBorder = new THREE.Mesh(
            new THREE.BoxGeometry(frameWidth, 0.05, frameDepth),
            frameMaterial
        );
        bottomBorder.position.y = -frameHeight / 2 + 0.025;
        frame.add(bottomBorder);

        // Left border
        const leftBorder = new THREE.Mesh(
            new THREE.BoxGeometry(0.05, frameHeight, frameDepth),
            frameMaterial
        );
        leftBorder.position.x = -frameWidth / 2 + 0.025;
        frame.add(leftBorder);

        // Right border
        const rightBorder = new THREE.Mesh(
            new THREE.BoxGeometry(0.05, frameHeight, frameDepth),
            frameMaterial
        );
        rightBorder.position.x = frameWidth / 2 - 0.025;
        frame.add(rightBorder);

        // Artwork plane
        const artworkMaterial = new THREE.MeshBasicMaterial({ 
            map: texture,
            side: THREE.FrontSide
        });
        
        const artwork = new THREE.Mesh(
            new THREE.PlaneGeometry(size.width, size.height),
            artworkMaterial
        );
        artwork.position.z = frameDepth / 2 + 0.001;
        frame.add(artwork);

        frame.position.copy(position);
        frame.rotation.copy(rotation);

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
                        size
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

    arrangeArtworks(artworkPaths, roomDimensions) {
        const promises = [];
        const spacing = 3;
        const height = roomDimensions.height / 2;
        const wallOffset = 0.1;

        const calculatePositions = (count, length) => {
            const positions = [];
            const availableSpace = length - 2;
            
            if (count <= 1) {
                positions.push(0);
                return positions;
            }
            
            const step = Math.min(spacing, availableSpace / (count - 1));
            const start = -step * (count - 1) / 2;
            
            for (let i = 0; i < count; i++) {
                positions.push(start + i * step);
            }
            
            return positions;
        };

        const artworksPerWall = Math.ceil(artworkPaths.length / 3);
        let pathIndex = 0;

        // Back wall
        const backWallPositions = calculatePositions(
            Math.min(artworksPerWall, artworkPaths.length - pathIndex),
            roomDimensions.width
        );
        for (const x of backWallPositions) {
            if (pathIndex >= artworkPaths.length) break;
            promises.push(
                this.loadArtwork(
                    artworkPaths[pathIndex++],
                    new THREE.Vector3(x, height, -roomDimensions.depth / 2 + wallOffset),
                    new THREE.Euler(0, 0, 0)
                )
            );
        }

        // Left wall
        const leftWallPositions = calculatePositions(
            Math.min(artworksPerWall, artworkPaths.length - pathIndex),
            roomDimensions.depth
        );
        for (const z of leftWallPositions) {
            if (pathIndex >= artworkPaths.length) break;
            promises.push(
                this.loadArtwork(
                    artworkPaths[pathIndex++],
                    new THREE.Vector3(-roomDimensions.width / 2 + wallOffset, height, z),
                    new THREE.Euler(0, Math.PI / 2, 0)
                )
            );
        }

        // Right wall
        const rightWallPositions = calculatePositions(
            Math.min(artworksPerWall, artworkPaths.length - pathIndex),
            roomDimensions.depth
        );
        for (const z of rightWallPositions) {
            if (pathIndex >= artworkPaths.length) break;
            promises.push(
                this.loadArtwork(
                    artworkPaths[pathIndex++],
                    new THREE.Vector3(roomDimensions.width / 2 - wallOffset, height, z),
                    new THREE.Euler(0, -Math.PI / 2, 0)
                )
            );
        }

        return Promise.all(promises);
    }
}