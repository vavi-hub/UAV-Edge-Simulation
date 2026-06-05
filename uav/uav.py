import random
from typing import List
import threading

# ----------------------------
# Define Task and UAV Classes
# ----------------------------
class Task:
    def __init__(self, data_size: float):
        self.data_size = data_size  # size of the image/frame in MB
        self.processed = False
        self.result = None

class UAV:
    def __init__(self, name, model_specs, edge_device=None, curr_battery=100):
        self.name = name
        self._lock = threading.Lock()

        self.flight_state = "hover" 

        # --- Metadata ---
        self.model = model_specs["model"]

        # --- ENERGY ---
        energy_specs = model_specs["energy"]
        self.max_energy = energy_specs["battery_energy_j"]
        self.battery_energy = (curr_battery / 100) * self.max_energy

        # --- FLIGHT ---
        flight_specs = model_specs["flight"]
        self.hover_power = flight_specs["hover_power_w"]
        self.cruise_power = flight_specs["cruise_power_w"]
        self.max_speed = flight_specs["max_speed_mps"]

        # --- COMPUTE ---
        compute_specs = model_specs["compute"]
        self.compute_power = compute_specs["base_power_w"]
        self.inference_energy = compute_specs["inference_energy_j"]

        # optional edge device
        self.edge_device = edge_device

        # --- COMMUNICATION ---
        comm_specs = model_specs["communication"]
        self.tx_energy_per_mb = comm_specs["tx_energy_per_mb"]
        self.image_size = comm_specs["image_size_mb"]

        # --- SENSORS ---
        self.sensors = model_specs.get("sensors", [])

    # ========================
    # ENERGY METHODS
    # ========================

    def _consume_energy(self, energy):
        with self._lock:
            self.battery_energy -= energy
            if self.battery_energy < 0:
                self.battery_energy = 0

    def hover(self, dt):
        energy = self.hover_power * dt
        self._consume_energy(energy)

    def cruise(self, dt):
        energy = self.cruise_power * dt
        self._consume_energy(energy)

    def run_inference(self,dt):
        if self.edge_device:
            self._consume_energy(self.edge_device["power"]["load_w"]*dt)
        else:
            self._consume_energy(self.inference_energy*dt)
    
    def run_training(self, dt):
        # Training draws max compute power
        self._consume_energy(self.compute_power * dt * 2.0)

    #need work
    def transmit_image(self,dt=0, size_mb=None):
        if size_mb is None:
            size_mb = self.image_size
        energy = self.tx_energy_per_mb * size_mb
        self._consume_energy(energy)

    # ========================
    # STEP SYSTEM
    # ========================

    def step(self, dt, movement, tasks=[None]):
        if movement == "hover":
            self.hover(dt)
            self.flight_state = "hover"
        elif movement == "cruise":
            self.cruise(dt)
            self.flight_state = "cruise"

        # Tasks 
        if "infer" in tasks:
            self.run_inference(dt)

        if "transmit" in tasks:
            self.transmit_image(dt)


    # ========================
    # STATUS / TELEMETRY
    # ========================

    def get_telemetry(self):
        return {
            "name": self.name,
            "model": self.model,
            "flight_state": self.flight_state,
            "battery_j": self.battery_energy,
            "battery_pct": (self.battery_energy / self.max_energy) * 100,
            "edge_device": self.edge_device["name"] if self.edge_device else None,
            "sensors": self.sensors
        }

    def status(self):
        t = self.get_telemetry()
        print(f"{t['name']} ({t['model']}) | Battery: {t['battery_pct']:.2f}%")

