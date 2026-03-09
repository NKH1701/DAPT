import os
import json
import yaml
import pickle


def load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def save_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)


def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(obj, file)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
