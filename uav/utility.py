import json

def load_specs(model, spec_file):
    with open(spec_file, "r") as f:
        all_specs = json.load(f)

    if model not in all_specs:
        raise ValueError(f"UAV model '{model}' not found in spec file")

    return all_specs[model]

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        return json.load(f)