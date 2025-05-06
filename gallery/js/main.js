import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { RoomBuilder } from './room-builder.js';
import { ArtworkManager } from './artwork-manager.js';
import { ImageMapper } from './image-mapper.js';
import { CollisionManager } from './collision-manager.js';

const CONFIG = {
    room: {
        width: 30,
        height: 6,
        depth: 30
    },
    debug: false,
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

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

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

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('canvas-container').appendChild(renderer.domElement);

const controls = new PointerLockControls(camera, renderer.domElement);
scene.add(controls.getObject());

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredObject = null;

const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let moveForward = false;
let moveBackward = false;
let moveLeft = false;
let moveRight = false;
let controlsActive = true;

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

document.addEventListener('mousemove', onMouseMove);
document.addEventListener('click', onClick);

function onMouseMove(event) {
    if (!controls.isLocked) return;
    
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
    
    checkIntersection();
}

function onClick(event) {
    if (controls.isLocked && hoveredObject && hoveredObject.userData && hoveredObject.userData.isArtwork) {
        showImageDetail(hoveredObject);
    } else if (!controls.isLocked && controlsActive) {
        controls.lock();
    }
}

function checkIntersection() {
    raycaster.setFromCamera(mouse, camera);
    
    const artworks = scene.children.filter(child => 
        child.userData && child.userData.isArtwork
    );
    
    const intersects = raycaster.intersectObjects(artworks, true);
    
    document.body.style.cursor = 'default';
    
    if (hoveredObject) {
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
            
            if (hoveredObject.userData && hoveredObject.userData.setHover) {
                hoveredObject.userData.setHover(true);
            }
        }
    }
}

function createImageDetailOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'image-detail-overlay';
    overlay.style.display = 'none';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
    overlay.style.zIndex = '100';
    overlay.style.flexDirection = 'column';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.padding = '20px';
    overlay.style.boxSizing = 'border-box';
    
    const container = document.createElement('div');
    container.id = 'image-detail-container';
    container.style.display = 'flex';
    container.style.flexDirection = window.innerWidth < 768 ? 'column' : 'row';
    container.style.flexWrap = 'wrap';
    container.style.justifyContent = 'center';
    container.style.alignItems = 'center';
    container.style.maxWidth = '95%';
    container.style.maxHeight = '85vh';
    container.style.gap = '20px';
    container.style.overflowY = 'auto';
    container.style.borderRadius = '10px';
    container.style.padding = '20px';
    container.style.backgroundColor = 'rgba(30, 30, 30, 0.7)';
    
    const title = document.createElement('h2');
    title.textContent = 'Style Transfer Details';
    title.style.color = 'white';
    title.style.fontFamily = 'Arial, sans-serif';
    title.style.width = '100%';
    title.style.textAlign = 'center';
    title.style.marginBottom = '20px';
    title.style.fontSize = window.innerWidth < 768 ? '24px' : '28px';
    title.style.fontWeight = '600';
    title.style.textShadow = '0 2px 4px rgba(0, 0, 0, 0.5)';
    container.appendChild(title);
    
    const originalImageContainer = createImageContainer('Original Image');
    const originalImage = createImageElement('original-image');
    originalImageContainer.appendChild(originalImage);
    container.appendChild(originalImageContainer);
    
    const styleImageContainer = createImageContainer('Style Reference');
    const styleImage = createImageElement('style-image');
    styleImageContainer.appendChild(styleImage);
    container.appendChild(styleImageContainer);
    
    const styledImageContainer = createImageContainer('Final Result');
    const styledImage = createImageElement('styled-image');
    styledImageContainer.appendChild(styledImage);
    container.appendChild(styledImageContainer);
    
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
    closeButton.style.transition = 'background-color 0.2s ease';
    closeButton.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.2)';
    
    closeButton.addEventListener('mouseover', function() {
        closeButton.style.backgroundColor = '#4477ee';
    });
    
    closeButton.addEventListener('mouseout', function() {
        closeButton.style.backgroundColor = '#5588ff';
    });
    
    closeButton.addEventListener('click', closeImageDetail);
    
    const keyListener = function(e) {
        if (e.key === 'Escape') {
            closeImageDetail();
        }
    };
    
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            closeImageDetail();
        }
    });
    
    document.addEventListener('keydown', keyListener);
    
    overlay.appendChild(container);
    overlay.appendChild(closeButton);
    
    function closeImageDetailInternal() {
        document.removeEventListener('keydown', keyListener);
        closeImageDetail();
    }
    
    document.body.appendChild(overlay);
    
    function createImageContainer(labelText) {
        const container = document.createElement('div');
        container.className = 'image-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
        container.style.padding = '15px';
        container.style.borderRadius = '8px';
        container.style.transition = 'all 0.3s ease';
        container.style.flexGrow = '1';
        container.style.flexBasis = window.innerWidth < 768 ? '100%' : '30%';
        container.style.minWidth = window.innerWidth < 768 ? '100%' : '250px';
        container.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
        container.style.margin = '5px';
        
        const label = document.createElement('h3');
        label.textContent = labelText;
        label.style.color = 'white';
        label.style.fontFamily = 'Arial, sans-serif';
        label.style.margin = '10px 0';
        label.style.fontSize = window.innerWidth < 768 ? '16px' : '18px';
        label.style.textAlign = 'center';
        label.style.width = '100%';
        
        container.appendChild(label);
        
        return container;
    }
    
    function createImageElement(id) {
        const img = document.createElement('img');
        img.id = id;
        img.style.width = '100%';
        img.style.maxWidth = '100%';
        img.style.maxHeight = window.innerWidth < 768 ? '30vh' : '50vh';
        img.style.objectFit = 'contain';
        img.style.border = '2px solid rgba(255, 255, 255, 0.7)';
        img.style.borderRadius = '4px';
        img.style.boxShadow = '0 0 15px rgba(0, 0, 0, 0.5)';
        img.style.transition = 'transform 0.3s ease';
        img.style.backgroundColor = 'rgba(0, 0, 0, 0.3)';
        
        img.addEventListener('load', function() {
            img.style.opacity = '1';
        });
        
        img.style.opacity = '0';
        
        return img;
    }
}

const imageMapper = new ImageMapper();

function showImageDetail(artwork) {
    if (!artwork || !artwork.userData) return;
    
    const styledImagePath = artwork.userData.imagePath;
    if (!styledImagePath) return;
    
    let originalImagePath, styleImagePath;
    
    if (artwork.userData.metadata) {
        originalImagePath = artwork.userData.metadata.contentImage;
        styleImagePath = artwork.userData.metadata.styleImage;
    } else {
        originalImagePath = imageMapper.getOriginalImagePath(styledImagePath);
        styleImagePath = imageMapper.getStyleImagePath(styledImagePath);
    }
    
    const styledImage = document.getElementById('styled-image');
    const originalImage = document.getElementById('original-image');
    const styleImage = document.getElementById('style-image');
    
    if (!styledImage || !originalImage || !styleImage) return;
    
    styledImage.src = '';
    originalImage.src = '';
    styleImage.src = '';
    
    const basePath = window.location.hostname.includes('github.io') ? '/cpsc479/' : '';
    const adjustPath = (path) => {
        if (!path) return '';
        if (basePath && !path.startsWith(basePath) && !path.startsWith('/')) {
            return `${basePath}${path}`;
        }
        if (!basePath && path.startsWith('/')) {
            return path.substring(1);
        }
        return path;
    };
    
    setTimeout(() => {
        styledImage.src = adjustPath(styledImagePath);
        originalImage.src = adjustPath(originalImagePath);
        
        if (styleImagePath && styleImagePath !== '') {
            styleImage.src = adjustPath(styleImagePath);
            styleImage.parentElement.style.display = 'flex';
        } else {
            styleImage.parentElement.style.display = 'none';
        }
        
        const overlay = document.getElementById('image-detail-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            updateOverlayLayout();
            
            setTimeout(() => {
                overlay.classList.add('visible');
            }, 10);
        }
        
        controls.unlock();
        controlsActive = false;
    }, 10);
}

function closeImageDetail() {
    const overlay = document.getElementById('image-detail-overlay');
    if (!overlay) return;
    
    overlay.classList.remove('visible');
    
    setTimeout(() => {
        overlay.style.display = 'none';
        
        const images = overlay.querySelectorAll('img');
        images.forEach(img => {
            img.src = '';
            img.style.opacity = '0';
        });
        
        controls.lock();
        controlsActive = true;
    }, 300);
}

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

const loadingManager = new THREE.LoadingManager();
loadingManager.onProgress = function (url, itemsLoaded, itemsTotal) {
    const progressBar = document.querySelector('.progress-bar-fill');
    progressBar.style.width = (itemsLoaded / itemsTotal * 100) + '%';
};

loadingManager.onLoad = function () {
    document.getElementById('loading-screen').style.display = 'none';
};

async function initGallery() {
    try {
        const roomBuilder = new RoomBuilder(scene);
        const wallsMap = roomBuilder.buildRoom(CONFIG.room);
        
        const collisionManager = new CollisionManager(scene, camera, CONFIG);
        collisionManager.addWallColliders(CONFIG.room);
        
        window.collisionManager = collisionManager;
        
        createImageDetailOverlay();
        
        const artworkManager = new ArtworkManager(scene, loadingManager);
        await artworkManager.populateGallery(wallsMap).catch(error => {
            console.error('Error populating gallery:', error);
            document.getElementById('loading-screen').innerHTML = 
                '<div style="color: white; text-align: center; padding: 20px;">' +
                '<h2>Failed to load gallery</h2>' +
                '<p>Error: ' + error.message + '</p>' +
                '<button onclick="window.location.reload()" style="padding: 10px; margin-top: 20px; cursor: pointer;">Retry</button>' +
                '</div>';
        });
    } catch (error) {
        console.error('Gallery initialization error:', error);
        document.getElementById('loading-screen').innerHTML = 
            '<div style="color: white; text-align: center; padding: 20px;">' +
            '<h2>Failed to initialize gallery</h2>' +
            '<p>Error: ' + error.message + '</p>' +
            '<button onclick="window.location.reload()" style="padding: 10px; margin-top: 20px; cursor: pointer;">Retry</button>' +
            '</div>';
    }
}

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
                container.style.maxHeight = '90vh';
                container.style.overflowY = 'auto';
                
                const imageContainers = container.querySelectorAll('.image-container');
                imageContainers.forEach(imgContainer => {
                    imgContainer.style.width = '100%';
                    imgContainer.style.maxWidth = '100%';
                    imgContainer.style.marginBottom = '20px';
                });
                
                const images = container.querySelectorAll('img');
                images.forEach(img => {
                    img.style.maxHeight = '30vh';
                    img.style.objectFit = 'contain';
                });
            } else {
                container.style.flexDirection = 'row';
                container.style.maxWidth = '95%';
                container.style.flexWrap = 'wrap';
                container.style.maxHeight = '85vh';
                container.style.overflowY = 'auto';
                
                const imageContainers = container.querySelectorAll('.image-container');
                imageContainers.forEach(imgContainer => {
                    imgContainer.style.width = window.innerWidth < 1200 ? '45%' : '30%';
                    imgContainer.style.marginBottom = '10px';
                });
                
                const images = container.querySelectorAll('img');
                images.forEach(img => {
                    img.style.maxHeight = '50vh';
                    img.style.objectFit = 'contain';
                });
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

initGallery();
animate();