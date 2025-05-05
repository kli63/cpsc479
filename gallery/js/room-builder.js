import * as THREE from 'three';

export class RoomBuilder {
    constructor(scene) {
        this.scene = scene;
        this.textureLoader = new THREE.TextureLoader();
        this.wallsMap = {
            central: { front: null, back: null },
            north: null,
            south: null,
            east: null,
            west: null
        };
    }

    buildRoom(config) {
        this.config = config;
        this.createFloor();
        this.createCeiling();
        this.createWalls();
        this.addLighting();
        
        return this.wallsMap;
    }

    createFloor() {
        const floorGeometry = new THREE.PlaneGeometry(this.config.width, this.config.depth);
        
        const woodTexture = this.textureLoader.load('https://threejs.org/examples/textures/hardwood2_diffuse.jpg', 
            (texture) => {
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.repeat.set(10, 10);
            }
        );
        
        const floorMaterial = new THREE.MeshStandardMaterial({ 
            map: woodTexture,
            roughness: 0.8,
            metalness: 0.1
        });
        
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);
    }

    createCeiling() {
        const ceilingGeometry = new THREE.PlaneGeometry(this.config.width, this.config.depth);
        const ceilingMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xFFFFFF,
            roughness: 0.85,
            metalness: 0.1
        });
        
        const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
        ceiling.rotation.x = Math.PI / 2;
        ceiling.position.y = this.config.height;
        ceiling.receiveShadow = true;
        this.scene.add(ceiling);
    }

    createWalls() {
        const wallMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xF5F5F5,
            roughness: 0.9,
            metalness: 0.0
        });

        const northWallGeometry = new THREE.PlaneGeometry(this.config.width, this.config.height);
        const northWall = new THREE.Mesh(northWallGeometry, wallMaterial);
        northWall.position.set(0, this.config.height/2, -this.config.depth/2);
        this.scene.add(northWall);
        this.wallsMap.north = northWall;
        
        const southWall = new THREE.Mesh(northWallGeometry, wallMaterial);
        southWall.position.set(0, this.config.height/2, this.config.depth/2);
        southWall.rotation.y = Math.PI;
        this.scene.add(southWall);
        this.wallsMap.south = southWall;
        
        const eastWallGeometry = new THREE.PlaneGeometry(this.config.depth, this.config.height);
        const eastWall = new THREE.Mesh(eastWallGeometry, wallMaterial);
        eastWall.position.set(this.config.width/2, this.config.height/2, 0);
        eastWall.rotation.y = -Math.PI/2;
        this.scene.add(eastWall);
        this.wallsMap.east = eastWall;
        
        const westWall = new THREE.Mesh(eastWallGeometry, wallMaterial);
        westWall.position.set(-this.config.width/2, this.config.height/2, 0);
        westWall.rotation.y = Math.PI/2;
        this.scene.add(westWall);
        this.wallsMap.west = westWall;
        
        const centralWallWidth = this.config.width * 0.6;
        const centralWallHeight = this.config.height * 0.9;
        const centralWallThickness = 0.3;
        
        const centralWallGroup = new THREE.Group();
        
        const centralWallGeometry = new THREE.BoxGeometry(
            centralWallWidth, 
            centralWallHeight, 
            centralWallThickness
        );
        
        const centralWall = new THREE.Mesh(centralWallGeometry, wallMaterial.clone());
        centralWall.position.set(0, centralWallHeight/2, 0);
        centralWallGroup.add(centralWall);
        this.scene.add(centralWallGroup);
        
        const frontPlaneGeometry = new THREE.PlaneGeometry(centralWallWidth, centralWallHeight);
        const centralWallFront = new THREE.Mesh(frontPlaneGeometry, wallMaterial.clone());
        centralWallFront.position.set(0, 0, centralWallThickness/2 + 0.01);
        centralWallFront.visible = false;
        centralWallGroup.add(centralWallFront);
        this.wallsMap.central.front = centralWallFront;
        
        const centralWallBack = new THREE.Mesh(frontPlaneGeometry, wallMaterial.clone());
        centralWallBack.position.set(0, 0, -centralWallThickness/2 - 0.01);
        centralWallBack.rotation.y = Math.PI;
        centralWallBack.visible = false;
        centralWallGroup.add(centralWallBack);
        this.wallsMap.central.back = centralWallBack;
    }

    addLighting() {
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        // Add directional lights for better illumination
        const directions = [
            { x: 1, y: 1, z: 0.5 },
            { x: -1, y: 1, z: 0.5 },
            { x: 0, y: 1, z: -1 }
        ];
        
        directions.forEach(dir => {
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
            directionalLight.position.set(dir.x * 5, dir.y * 5, dir.z * 5);
            directionalLight.castShadow = true;
            this.scene.add(directionalLight);
        });
    }
}