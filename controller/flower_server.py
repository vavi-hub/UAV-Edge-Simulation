import flwr as fl

def main():
    print("[FLWR Server] Starting Flower Server on port 8080...")
    # Define aggregation strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # All available clients will be requested to train
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
