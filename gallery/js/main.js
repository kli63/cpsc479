import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// Gallery configuration
const CONFIG = {
    room: {
        width: 20,
        height: 5,
        depth: 20
    },
    camera: {
        fov: 75,
        near: 0.1,
        far: 1000,
        position: { x: 0, y: 1.6, z: 0 }  // Average human height
    },
    controls: {
        speed: 3.0,
        lookSpeed: 0.1
    },
    lights: {
        ambient: { intensity: 0.5 },
        directional: { intensity: 0.8, position: { x: 5, y: 10, z: 5 } }
    }
};

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

// Camera setup
const camera = new THREE.PerspectiveCamera(
    CONFIG.camera.fov,
    window.innerWidth / window.innerHeight,
    CONFIG.camera.near,
    CONFIG.camera.far
);
camera.position.set(
    CONFIG.camera.position.x,
    CONFIG.camera.position.y,
    CONFIG.camera.position.z
);

// Renderer setup
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Controls setup
const controls = new PointerLockControls(camera, renderer.domElement);
scene.add(controls.getObject());

// Movement
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let moveForward = false;
let moveBackward = false;
let moveLeft = false;
let moveRight = false;

// Key events
const onKeyDown = function (event) {
    switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
            moveForward = true;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            moveLeft = true;
            break;
        case 'ArrowDown':
        case 'KeyS':
            moveBackward = true;
            break;
        case 'ArrowRight':
        case 'KeyD':
            moveRight = true;
            break;
    }
};

const onKeyUp = function (event) {
    switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
            moveForward = false;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            moveLeft = false;
            break;
        case 'ArrowDown':
        case 'KeyS':
            moveBackward = false;
            break;
        case 'ArrowRight':
        case 'KeyD':
            moveRight = false;
            break;
    }
};

document.addEventListener('keydown', onKeyDown);
document.addEventListener('keyup', onKeyUp);

// Click to lock controls
document.addEventListener('click', function () {
    controls.lock();
});

controls.addEventListener('lock', function () {
    document.getElementById('info').style.display = 'none';
});

controls.addEventListener('unlock', function () {
    document.getElementById('info').style.display = 'block';
});

// Lights
const ambientLight = new THREE.AmbientLight(
    0xffffff,
    CONFIG.lights.ambient.intensity
);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(
    0xffffff,
    CONFIG.lights.directional.intensity
);
directionalLight.position.set(
    CONFIG.lights.directional.position.x,
    CONFIG.lights.directional.position.y,
    CONFIG.lights.directional.position.z
);
directionalLight.castShadow = true;
scene.add(directionalLight);

// Create gallery room
function createRoom() {
    // Floor
    const floorGeometry = new THREE.PlaneGeometry(CONFIG.room.width, CONFIG.room.depth);
    const floorMaterial = new THREE.MeshStandardMaterial({ 
        color: 0x444444,
        roughness: 0.5,
        metalness: 0.2
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Walls
    const wallMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xffffff,
        roughness: 0.8,
        metalness: 0.2
    });

    // Back wall
    const backWallGeometry = new THREE.PlaneGeometry(CONFIG.room.width, CONFIG.room.height);
    const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
    backWall.position.z = -CONFIG.room.depth / 2;
    backWall.position.y = CONFIG.room.height / 2;
    scene.add(backWall);

    // Front wall
    const frontWall = new THREE.Mesh(backWallGeometry, wallMaterial);
    frontWall.position.z = CONFIG.room.depth / 2;
    frontWall.position.y = CONFIG.room.height / 2;
    frontWall.rotation.y = Math.PI;
    scene.add(frontWall);

    // Left wall
    const sideWallGeometry = new THREE.PlaneGeometry(CONFIG.room.depth, CONFIG.room.height);
    const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    leftWall.position.x = -CONFIG.room.width / 2;
    leftWall.position.y = CONFIG.room.height / 2;
    leftWall.rotation.y = Math.PI / 2;
    scene.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    rightWall.position.x = CONFIG.room.width / 2;
    rightWall.position.y = CONFIG.room.height / 2;
    rightWall.rotation.y = -Math.PI / 2;
    scene.add(rightWall);
}

// Add image frames on walls (placeholders for actual artwork)
function addArtworkFrames() {
    const frameGeometry = new THREE.BoxGeometry(2, 1.5, 0.1);
    const frameMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513 });

    // Add frames to back wall
    for (let i = -2; i <= 2; i++) {
        if (i === 0) continue; // Skip center position
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.set(i * 3, CONFIG.room.height / 2, -CONFIG.room.depth / 2 + 0.1);
        scene.add(frame);
    }

    // Add frames to side walls
    for (let i = -2; i <= 2; i++) {
        // Left wall
        const leftFrame = new THREE.Mesh(frameGeometry, frameMaterial);
        leftFrame.position.set(-CONFIG.room.width / 2 + 0.1, CONFIG.room.height / 2, i * 3);
        leftFrame.rotation.y = Math.PI / 2;
        scene.add(leftFrame);

        // Right wall
        const rightFrame = new THREE.Mesh(frameGeometry, frameMaterial);
        rightFrame.position.set(CONFIG.room.width / 2 - 0.1, CONFIG.room.height / 2, i * 3);
        rightFrame.rotation.y = -Math.PI / 2;
        scene.add(rightFrame);
    }
}

// Loading manager to track progress
const loadingManager = new THREE.LoadingManager();
loadingManager.onProgress = function (url, itemsLoaded, itemsTotal) {
    const progressBar = document.querySelector('.progress-bar-fill');
    progressBar.style.width = (itemsLoaded / itemsTotal * 100) + '%';
};

loadingManager.onLoad = function () {
    document.getElementById('loading-screen').style.display = 'none';
};

// Initialize the scene
createRoom();
addArtworkFrames();

// Load artwork from style transfer project
import { StyleTransferImageLoader } from './image-loader.js';

// Initialize gallery with images
const imageLoader = new StyleTransferImageLoader(scene, loadingManager);
imageLoader.loadGallery(CONFIG.room).then(artworks => {
    console.log(`Loaded ${artworks.length} artworks into the gallery`);
}).catch(error => {
    console.error('Failed to load artworks:', error);
});

// Animation loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    if (controls.isLocked) {
        const delta = Math.min(clock.getDelta(), 0.1);

        velocity.x -= velocity.x * 10.0 * delta;
        velocity.z -= velocity.z * 10.0 * delta;

        direction.z = Number(moveForward) - Number(moveBackward);
        direction.x = Number(moveRight) - Number(moveLeft);
        direction.normalize();

        if (moveForward || moveBackward) velocity.z -= direction.z * CONFIG.controls.speed * delta;
        if (moveLeft || moveRight) velocity.x -= direction.x * CONFIG.controls.speed * delta;

        controls.moveRight(-velocity.x * delta);
        controls.moveForward(-velocity.z * delta);
    }

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();