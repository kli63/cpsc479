import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { RoomBuilder } from './room-builder.js';
import { ArtworkManager } from './artwork-manager.js';
import { ImageMapper } from './image-mapper.js';
import { CollisionManager } from './collision-manager.js';

// Gallery configuration
const CONFIG = {
    room: {
        width: 30,
        height: 6,
        depth: 30
    },
    debug: false, // Set to true to visualize colliders
    camera: {
        fov: 75,
        near: 0.1,
        far: 1000,
        position: { x: 0, y: 2.4, z: 5 }
    },
    controls: {
        speed: 40.0,
        lookSpeed: 0.5
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

// Raycaster for detecting hover and clicks on artworks
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredObject = null;

// Movement variables
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let moveForward = false;
let moveBackward = false;
let moveLeft = false;
let moveRight = false;
let controlsActive = true;

// Key events
const onKeyDown = function (event) {
    switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
            if (controlsActive) moveForward = true;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            if (controlsActive) moveLeft = true;
            break;
        case 'ArrowDown':
        case 'KeyS':
            if (controlsActive) moveBackward = true;
            break;
        case 'ArrowRight':
        case 'KeyD':
            if (controlsActive) moveRight = true;
            break;
        case 'Escape':
            if (document.getElementById('image-detail-overlay').style.display === 'flex') {
                closeImageDetail();
            }
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

// Mouse events for raycasting
document.addEventListener('mousemove', onMouseMove);
document.addEventListener('click', onClick);

function onMouseMove(event) {
    if (!controls.isLocked) return;
    
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
    
    checkIntersection();
}

function onClick(event) {
    if (controls.isLocked && hoveredObject) {
        showImageDetail(hoveredObject);
    } else if (!controls.isLocked) {
        controls.lock();
    }
}

function checkIntersection() {
    raycaster.setFromCamera(mouse, camera);
    
    // Get all artwork in the scene
    const artworks = scene.children.filter(child => 
        child.userData && child.userData.isArtwork
    );
    
    const intersects = raycaster.intersectObjects(artworks, true);
    
    document.body.style.cursor = 'default';
    
    // Clear previous hover state
    if (hoveredObject) {
        // Remove hover effect from previously hovered object
        if (hoveredObject.userData && hoveredObject.userData.setHover) {
            hoveredObject.userData.setHover(false);
        }
        hoveredObject = null;
    }
    
    if (intersects.length > 0) {
        let object = intersects[0].object;
        while (object.parent && !object.userData.isArtwork) {
            object = object.parent;
        }
        
        if (object.userData.isArtwork) {
            document.body.style.cursor = 'pointer';
            hoveredObject = object;
            
            // Apply hover effect
            if (hoveredObject.userData && hoveredObject.userData.setHover) {
                hoveredObject.userData.setHover(true);
            }
        }
    }
}

// Create image detail overlay
function createImageDetailOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'image-detail-overlay';
    overlay.style.display = 'none';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    overlay.style.zIndex = '100';
    overlay.style.flexDirection = 'column';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'row';
    container.style.flexWrap = 'wrap';
    container.style.justifyContent = 'center';
    container.style.maxWidth = '95%';
    container.style.maxHeight = '85%';
    container.style.gap = '20px';
    
    // Title
    const title = document.createElement('h2');
    title.textContent = 'Style Transfer Details';
    title.style.color = 'white';
    title.style.fontFamily = 'Arial, sans-serif';
    title.style.width = '100%';
    title.style.textAlign = 'center';
    title.style.marginBottom = '20px';
    container.appendChild(title);
    
    // Create the three image containers
    // 1. Original image
    const originalImageContainer = createImageContainer('Original Image');
    const originalImage = createImageElement('original-image');
    originalImageContainer.appendChild(originalImage);
    container.appendChild(originalImageContainer);
    
    // 2. Style reference image
    const styleImageContainer = createImageContainer('Style Reference');
    const styleImage = createImageElement('style-image');
    styleImageContainer.appendChild(styleImage);
    container.appendChild(styleImageContainer);
    
    // 3. Final styled result
    const styledImageContainer = createImageContainer('Final Result');
    const styledImage = createImageElement('styled-image');
    styledImageContainer.appendChild(styledImage);
    container.appendChild(styledImageContainer);
    
    // Close button
    const closeButton = document.createElement('button');
    closeButton.textContent = 'Close';
    closeButton.style.padding = '12px 24px';
    closeButton.style.marginTop = '20px';
    closeButton.style.backgroundColor = '#5588ff';
    closeButton.style.color = 'white';
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '5px';
    closeButton.style.cursor = 'pointer';
    closeButton.style.fontFamily = 'Arial, sans-serif';
    closeButton.style.fontSize = '16px';
    closeButton.style.fontWeight = 'bold';
    
    closeButton.addEventListener('click', closeImageDetail);
    
    overlay.appendChild(container);
    overlay.appendChild(closeButton);
    
    document.body.appendChild(overlay);
    
    // Helper function to create image containers
    function createImageContainer(labelText) {
        const container = document.createElement('div');
        container.className = 'image-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
        container.style.padding = '15px';
        container.style.borderRadius = '8px';
        container.style.transition = 'transform 0.2s ease';
        
        const label = document.createElement('h3');
        label.textContent = labelText;
        label.style.color = 'white';
        label.style.fontFamily = 'Arial, sans-serif';
        label.style.marginTop = '10px';
        
        container.appendChild(label);
        
        return container;
    }
    
    // Helper function to create image elements
    function createImageElement(id) {
        const img = document.createElement('img');
        img.id = id;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '50vh';
        img.style.objectFit = 'contain';
        img.style.border = '2px solid white';
        img.style.boxShadow = '0 0 15px rgba(0, 0, 0, 0.5)';
        
        return img;
    }
}

// Create image mapper
const imageMapper = new ImageMapper();

// Show image detail
function showImageDetail(artwork) {
    const styledImagePath = artwork.userData.imagePath;
    const originalImagePath = imageMapper.getOriginalImagePath(styledImagePath);
    const styleImagePath = imageMapper.getStyleImagePath(styledImagePath);
    
    const styledImage = document.getElementById('styled-image');
    const originalImage = document.getElementById('original-image');
    const styleImage = document.getElementById('style-image');
    
    styledImage.src = styledImagePath;
    originalImage.src = originalImagePath;
    
    if (styleImagePath && styleImagePath !== '') {
        styleImage.src = styleImagePath;
        styleImage.parentElement.style.display = 'flex';
    } else {
        styleImage.parentElement.style.display = 'none';
    }
    
    const overlay = document.getElementById('image-detail-overlay');
    overlay.style.display = 'flex';
    
    controls.unlock();
    controlsActive = false;
}

// Close image detail
function closeImageDetail() {
    const overlay = document.getElementById('image-detail-overlay');
    overlay.style.display = 'none';
    
    controls.lock();
    controlsActive = true;
}

// Click to lock controls
document.addEventListener('click', function () {
    if (!controls.isLocked && controlsActive) {
        controls.lock();
    }
});

controls.addEventListener('lock', function () {
    document.getElementById('info').style.display = 'none';
});

controls.addEventListener('unlock', function () {
    if (controlsActive) {
        document.getElementById('info').style.display = 'block';
    }
});

// Loading manager to track progress
const loadingManager = new THREE.LoadingManager();
loadingManager.onProgress = function (url, itemsLoaded, itemsTotal) {
    const progressBar = document.querySelector('.progress-bar-fill');
    progressBar.style.width = (itemsLoaded / itemsTotal * 100) + '%';
};

loadingManager.onLoad = function () {
    document.getElementById('loading-screen').style.display = 'none';
};

// Initialize the gallery
async function initGallery() {
    // Build the room
    const roomBuilder = new RoomBuilder(scene);
    const wallsMap = roomBuilder.buildRoom(CONFIG.room);
    
    // Setup collision detection
    const collisionManager = new CollisionManager(scene, camera, CONFIG);
    collisionManager.addWallColliders(CONFIG.room);
    
    // Assign to global scope for use in animate loop
    window.collisionManager = collisionManager;
    
    // Create image detail overlay
    createImageDetailOverlay();
    
    // Load and place artwork
    const artworkManager = new ArtworkManager(scene, loadingManager);
    await artworkManager.populateGallery(wallsMap);
}

// Animation loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    if (controls.isLocked && controlsActive) {
        const delta = Math.min(clock.getDelta(), 0.1);

        velocity.x -= velocity.x * 10.0 * delta;
        velocity.z -= velocity.z * 10.0 * delta;

        direction.z = Number(moveForward) - Number(moveBackward);
        direction.x = Number(moveRight) - Number(moveLeft);
        direction.normalize();

        if (moveForward || moveBackward) velocity.z -= direction.z * CONFIG.controls.speed * delta;
        if (moveLeft || moveRight) velocity.x -= direction.x * CONFIG.controls.speed * delta;

        const currentPosition = controls.getObject().position.clone();
        const moveX = -velocity.x * delta;
        const moveZ = -velocity.z * delta;
        
        const potentialPosition = {
            x: currentPosition.x + moveX,
            y: currentPosition.y,
            z: currentPosition.z + moveZ
        };
        if (window.collisionManager) {
            const resolvedPosition = window.collisionManager.resolveCollision(
                { x: currentPosition.x, y: currentPosition.y, z: currentPosition.z },
                potentialPosition
            );
            
            if (resolvedPosition.x !== currentPosition.x) {
                controls.moveRight(resolvedPosition.x - currentPosition.x);
            }
            
            if (resolvedPosition.z !== currentPosition.z) {
                controls.moveForward(resolvedPosition.z - currentPosition.z);
            }
        } else {
            controls.moveRight(moveX);
            controls.moveForward(moveZ);
        }
    }

    renderer.render(scene, camera);
}

function updateOverlayLayout() {
    const overlay = document.getElementById('image-detail-overlay');
    if (overlay) {
        const container = overlay.querySelector('div');
        if (container) {
            if (window.innerWidth < 768) {
                container.style.flexDirection = 'column';
                container.style.maxWidth = '95%';
            } else {
                container.style.flexDirection = 'row';
                container.style.maxWidth = '90%';
            }
        }
    }
}

window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    updateOverlayLayout();
});

// Start the gallery
initGallery();
animate();