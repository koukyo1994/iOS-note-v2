import argparse
import json


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser


def load_config(namespace: argparse.Namespace) -> dict:
    with open(namespace.config, "r") as f:
        config = json.load(f)
    return config
