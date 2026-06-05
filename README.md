# UAV Edge Simulation

A comprehensive simulation environment designed to mimic Unmanned Aerial Vehicles (UAVs) acting as edge infrastructure nodes. The simulation features real-time edge inference, visual telemetry reporting, and a federated learning architecture.

## Key Features

- **Real-Time Edge Inference:** UAV nodes consume streams of images and process them on the edge using an ONNX model (specifically tuned for early fire detection).
- **Centralized Live Dashboard:** A lightweight FastAPI backend powers a real-time WebSocket dashboard. Track UAV telemetry, battery/joules status, current flight state, and live compressed camera feeds marked directly with dynamic inference boundaries.
- **Fleet Control:** Features a centralized global pause toggle directly on the web application permitting instant suspension of all connected UAV operations. 
- **Federated Learning:** Integrates with the Flower framework (`flwr`) to experiment with collaborative federated model training across the simulated UAV fleet asynchronously.
- **Dockerized Setup:** Fully configured for isolated deployment with `docker-compose`, enabling swift orchestration of the controller alongside multi-agent UAV swarms.

## Project Structure

- `controller/`: Houses the central FastAPI backend API (`api.py`), the Flower central server node (`flower_server.py`), and the live web frontend visualizer (`static/index.html`).
- `uav/`: Contains the core UAV physics/simulator class (`uav.py`), main lifecycle edge inference worker (`main.py`, `onnx_inference.py`), and the decentralized federated learning client (`fl_client.py`).
- `configs/`: JSON configuration maps determining device parameters including initial battery levels, payload sizes, compute strengths, and integrated sensor layouts.
- `Model/`: Directory meant for hosting the foundational ONNX prediction models (e.g., `fire_model.onnx`).

## Prerequisites

- **Docker & Docker Compose** (ensure these are properly installed based on your OS).
- Image feeds and Datasets to supply to the simulation paths.

## Getting Started

1. **Set Local Environment Variables:**
   The `docker-compose` environment relies on volume mounts. Define the directories housing your simulation images and FL datasets:
   ```bash
   export IMG_PATH="/path/to/your/camera/feed/images"
   export DATASET_PATH="/path/to/your/federated/dataset"
   ```

2. **Boot the Simulation:**
   Deploy the cluster simply using Docker Compose. This starts up the controller backend, sets up networking, and brings the frontend dashboard to life alongside the UAV edges.
   ```bash
   docker-compose up --build
   ```

3. **Monitor from the Dashboard:**
   Launch a browser and navigate to [`http://localhost:8000/dashboard`](http://localhost:8000/dashboard) to oversee live telemetry streams and edge inferences directly from the swarming UAVs.
