const ws = new WebSocket("ws://localhost:8000/ws");
const edgesSvg = document.getElementById("edges-svg");
const nodesContainer = document.getElementById("uav-nodes");
const pauseBtn = document.getElementById("pauseBtn");
const pauseIcon = pauseBtn.querySelector("i");
const baseStation = document.getElementById("base-station");

let isPaused = false;
let uavNodes = {}; // Keep track of rendered nodes
let uavPositions = {}; // Keep track of calculated positions

const updateCenterVars = () => {
    return {
        centerX: window.innerWidth / 2,
        centerY: window.innerHeight / 2,
        radius: Math.min(window.innerWidth, window.innerHeight) * 0.35 // 35% of min dimension
    };
};

let { centerX, centerY, radius } = updateCenterVars();

async function togglePause() {
    await fetch("/toggle_pause", { method: "POST" });
}

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.is_paused !== undefined) {
        if (data.is_paused !== isPaused) {
            isPaused = data.is_paused;
            if (isPaused) {
                pauseBtn.classList.add("paused");
                pauseIcon.className = "fas fa-play";
            } else {
                pauseBtn.classList.remove("paused");
                pauseIcon.className = "fas fa-pause";
            }
        }
    }
    
    if (data.uavs) {
        renderGraph(data.uavs);
    }
};

function renderGraph(uavs) {
    const uavNames = Object.keys(uavs);
    const numUAVs = uavNames.length;
    
    const dims = updateCenterVars();
    centerX = dims.centerX;
    centerY = dims.centerY;
    radius = dims.radius;
    
    uavNames.sort().forEach((name, index) => {
        const uav = uavs[name];
        let nodeData = uavNodes[name];
        
        // Only layout circularly if position is not defined yet
        if (!uavPositions[name]) {
            const angle = (index / numUAVs) * 2 * Math.PI - Math.PI / 2;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            uavPositions[name] = { x, y };
        }
        
        const pos = uavPositions[name];
        
        if (!nodeData) {
            // Create Card
            const card = document.createElement("div");
            card.className = "uav-card";
            card.style.left = `${pos.x}px`;
            card.style.top = `${pos.y}px`;
            
            // Create Edge
            const edge = document.createElementNS("http://www.w3.org/2000/svg", "line");
            edge.setAttribute("x1", centerX);
            edge.setAttribute("y1", centerY);
            edge.setAttribute("x2", pos.x);
            edge.setAttribute("y2", pos.y);
            edge.className.baseVal = "edge";
            
            nodesContainer.appendChild(card);
            edgesSvg.insertBefore(edge, edgesSvg.firstChild);
            
            nodeData = { card, edge };
            uavNodes[name] = nodeData;
            
            makeDraggable(card, name);
        } else {
            // Adjust line base to center if center changes
            nodeData.edge.setAttribute("x1", centerX);
            nodeData.edge.setAttribute("y1", centerY);
        }
        
        updateCardContent(nodeData.card, uav);
    });
    
    Object.keys(uavNodes).forEach(name => {
        if (!uavs[name]) {
            nodesContainer.removeChild(uavNodes[name].card);
            edgesSvg.removeChild(uavNodes[name].edge);
            delete uavNodes[name];
            delete uavPositions[name];
        }
    });
}

function makeDraggable(card, name) {
    let isDragging = false;
    let offsetX, offsetY;

    card.addEventListener('mousedown', (e) => {
        isDragging = true;
        // Keep offset from cursor to transform origin
        offsetX = e.clientX - uavPositions[name].x;
        offsetY = e.clientY - uavPositions[name].y;
        card.style.zIndex = 1000;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        const newX = e.clientX - offsetX;
        const newY = e.clientY - offsetY;
        
        uavPositions[name].x = newX;
        uavPositions[name].y = newY;
        
        card.style.left = `${newX}px`;
        card.style.top = `${newY}px`;
        
        const edge = uavNodes[name].edge;
        edge.setAttribute('x2', newX);
        edge.setAttribute('y2', newY);
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            card.style.zIndex = '';
        }
    });
}

function updateCardContent(card, uav) {
    if (!card.hasChildNodes()) {
        card.innerHTML = `
            <div class="uav-header">
                <h3 class="uav-name"></h3>
                <i class="fas fa-plane uav-icon"></i>
            </div>
            <div class="uav-body">
                <p>Model <span class="uav-model value"></span></p>
                <p>State <span class="uav-state value"></span></p>
                <p>Battery <span class="uav-battery"></span></p>
                <p>Energy <span class="uav-energy value"></span></p>
                <p>Edge <span class="uav-edge value"></span></p>
            </div>
            <div class="inference-result"></div>
            <div class="uav-image-container"></div>
        `;
    }

    card.querySelector('.uav-name').textContent = uav.name;
    card.querySelector('.uav-model').textContent = uav.model;
    card.querySelector('.uav-state').textContent = uav.flight_state;
    
    const battery = uav.battery_pct.toFixed(1);
    const isBatteryLow = battery < 20;
    const battSpan = card.querySelector('.uav-battery');
    battSpan.textContent = `${battery}%`;
    battSpan.className = `uav-battery ${isBatteryLow ? 'low' : 'ok'} value`;
    
    card.querySelector('.uav-energy').textContent = `${uav.battery_j.toFixed(0)} J`;
    card.querySelector('.uav-edge').textContent = uav.edge_device || "None";

    const infResult = card.querySelector('.inference-result');
    let imgClass = '';
    
    if (uav.inference_result === true) {
        if (infResult.textContent !== "FIRE DETECTED") {
            infResult.textContent = "FIRE DETECTED";
            infResult.className = "inference-result inference-fire";
        }
        imgClass = 'fire-border';
    } else if (uav.inference_result === false) {
        if (infResult.textContent !== "SAFE") {
            infResult.textContent = "SAFE";
            infResult.className = "inference-result inference-safe";
        }
        imgClass = 'safe-border';
    } else {
        if (infResult.textContent !== "NO DATA") {
            infResult.textContent = "NO DATA";
            infResult.className = "inference-result";
        }
    }

    const imgContainer = card.querySelector('.uav-image-container');
    if (uav.current_image) {
        let img = imgContainer.querySelector('img');
        if (!img) {
            imgContainer.innerHTML = `<img src="${uav.current_image}" class="${imgClass}" />`;
        } else {
            if (img.getAttribute('src') !== uav.current_image) {
                img.setAttribute('src', uav.current_image);
            }
            if (img.className !== imgClass) {
                img.className = imgClass;
            }
        }
    } else {
        if (!imgContainer.innerHTML.includes("No image")) {
            imgContainer.innerHTML = `<p style="opacity:0.5; text-align:center; font-size:12px; margin-top:10px;">No image</p>`;
        }
    }
}
