import os
import torch
import torch.nn as nn
import flwr as fl
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

# Define the MobileNetV2 architecture (same as training)
def get_model():
    model = models.mobilenet_v2(pretrained=False) # Or pretrained depending on how you initialize
    model.classifier[1] = nn.Linear(1280, 1)
    return model

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class UAVFlowerClient(fl.client.NumPyClient):
    def __init__(self, uav_obj, dataset_path="/dataset"):
        self.uav = uav_obj
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # We need a small batch size for edge device
        self.batch_size = 8
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load local data
        if os.path.exists(self.dataset_path):
            self.dataset = datasets.ImageFolder(self.dataset_path, transform=self.transform)
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        else:
            self.dataset = None
            self.train_loader = None
            print(f"Warning: Dataset path {self.dataset_path} does not exist.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        if self.train_loader is None:
            return self.get_parameters(config={}), 0, {}

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        epochs = 1
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Simulate energy consumption per step
                self.uav.run_training(dt=0.1) # Simulate that this batch took 0.1s to compute

        # simulate communication energy for uploading parameters Model = ~13MB for mobilenet
        self.uav.transmit_image(dt=0.5, size_mb=13.0) 

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = 0.0, 0.0
        
        if self.train_loader is None:
            return float(loss), 0, {"accuracy": float(accuracy)}
            
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        return float(loss), total, {"accuracy": float(accuracy)}

def start_fl_client(uav_obj):
    server_address = os.getenv("FLOWER_SERVER", "Controller:8080")
    print(f"[{uav_obj.name}] Starting FLClient connected to {server_address}")
    fl.client.start_client(
        server_address=server_address,
        client=UAVFlowerClient(uav_obj)
    )
