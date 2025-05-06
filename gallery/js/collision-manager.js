import * as THREE from 'three';

export class CollisionManager {
    constructor(scene, camera, config) {
        this.scene = scene;
        this.camera = camera;
        this.config = config;
        this.colliders = [];
        this.playerRadius = 0.8;
    }
    
    addBoxCollider(position, dimensions, rotation = { x: 0, y: 0, z: 0 }) {
        const collider = {
            type: 'box',
            position: position,
            dimensions: dimensions,
            rotation: rotation
        };
        
        this.colliders.push(collider);
        
        if (this.config.debug) {
            const boxGeometry = new THREE.BoxGeometry(
                dimensions.width, 
                dimensions.height, 
                dimensions.depth
            );
            const boxMaterial = new THREE.MeshBasicMaterial({ 
                color: 0xff0000, 
                wireframe: true,
                transparent: true,
                opacity: 0.3
            });
            const boxMesh = new THREE.Mesh(boxGeometry, boxMaterial);
            boxMesh.position.set(position.x, position.y, position.z);
            boxMesh.rotation.set(rotation.x, rotation.y, rotation.z);
            this.scene.add(boxMesh);
        }
        
        return collider;
    }
    
    addWallColliders(roomDimensions) {
        const wallThickness = 0.5;
        const halfWidth = roomDimensions.width / 2;
        const halfDepth = roomDimensions.depth / 2;
        const height = roomDimensions.height;
        
        this.addBoxCollider(
            { x: 0, y: height/2, z: -halfDepth },
            { width: roomDimensions.width, height: height, depth: wallThickness }
        );
        
        this.addBoxCollider(
            { x: 0, y: height/2, z: halfDepth },
            { width: roomDimensions.width, height: height, depth: wallThickness }
        );
        
        this.addBoxCollider(
            { x: halfWidth, y: height/2, z: 0 },
            { width: wallThickness, height: height, depth: roomDimensions.depth },
        );
        
        this.addBoxCollider(
            { x: -halfWidth, y: height/2, z: 0 },
            { width: wallThickness, height: height, depth: roomDimensions.depth },
        );
        
        const centralWallWidth = roomDimensions.width * 0.6;
        const centralWallHeight = roomDimensions.height * 0.9;
        const centralWallThickness = 0.3;
        
        this.addBoxCollider(
            { x: 0, y: centralWallHeight/2, z: 0 },
            { width: centralWallWidth, height: centralWallHeight, depth: centralWallThickness }
        );
    }
    
    checkCollision(newPosition) {
        for (const collider of this.colliders) {
            if (collider.type === 'box') {
                if (this.checkBoxCollision(newPosition, collider)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    checkBoxCollision(position, boxCollider) {
        const playerX = position.x;
        const playerZ = position.z;
        
        let boxMinX, boxMaxX, boxMinZ, boxMaxZ;
        
        if (Math.abs(boxCollider.rotation.y) < 0.01) {
            boxMinX = boxCollider.position.x - boxCollider.dimensions.width/2;
            boxMaxX = boxCollider.position.x + boxCollider.dimensions.width/2;
            boxMinZ = boxCollider.position.z - boxCollider.dimensions.depth/2;
            boxMaxZ = boxCollider.position.z + boxCollider.dimensions.depth/2;
        } else {
            const maxDim = Math.max(boxCollider.dimensions.width, boxCollider.dimensions.depth);
            boxMinX = boxCollider.position.x - maxDim/2;
            boxMaxX = boxCollider.position.x + maxDim/2;
            boxMinZ = boxCollider.position.z - maxDim/2;
            boxMaxZ = boxCollider.position.z + maxDim/2;
        }
        
        const closestX = Math.max(boxMinX, Math.min(playerX, boxMaxX));
        const closestZ = Math.max(boxMinZ, Math.min(playerZ, boxMaxZ));
        
        const distanceX = playerX - closestX;
        const distanceZ = playerZ - closestZ;
        const distanceSquared = distanceX * distanceX + distanceZ * distanceZ;
        return distanceSquared < (this.playerRadius * this.playerRadius);
    }
    
    resolveCollision(currentPosition, newPosition) {
        if (!this.checkCollision(newPosition)) {
            return newPosition;
        }
        
        const slideX = { x: newPosition.x, y: currentPosition.y, z: currentPosition.z };
        if (!this.checkCollision(slideX)) {
            return slideX;
        }
        
        const slideZ = { x: currentPosition.x, y: currentPosition.y, z: newPosition.z };
        if (!this.checkCollision(slideZ)) {
            return slideZ;
        }
        
        return currentPosition;
    }
}