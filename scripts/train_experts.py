"""Train or configure ToM and Social expert models.

For v1, uses OLMo-3 with role-specific prompting (no fine-tuning).
Saves expert configurations for use in routed inference.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experts.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # For v1: save expert configs (OLMo-3 with role-specific prompts)
    # No separate training needed — experts are prompt-differentiated

    tom_config = {
        "type": "olmo_prompted",
        "model_name": "allenai/Olmo-3-7B-Instruct",
        "role": "tom",
        "system_prompt": (
            "You are an expert at Theory of Mind reasoning. "
            "Focus on characters' hidden beliefs, false beliefs, knowledge states, "
            "intentions, and perspectives that are not directly observable."
        ),
    }

    social_config = {
        "type": "olmo_prompted",
        "model_name": "allenai/Olmo-3-7B-Instruct",
        "role": "social",
        "system_prompt": (
            "You are an expert at social reasoning. "
            "Focus on social norms, emotional reactions, relationship dynamics, "
            "and observable social behaviors."
        ),
    }

    # Save configs
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    tom_dir = out_dir / "tom_expert"
    tom_dir.mkdir(exist_ok=True)
    with open(tom_dir / "config.json", "w") as f:
        json.dump(tom_config, f, indent=2)

    social_dir = out_dir / "social_expert"
    social_dir.mkdir(exist_ok=True)
    with open(social_dir / "config.json", "w") as f:
        json.dump(social_config, f, indent=2)

    print("Expert configurations saved:")
    print(f"  ToM expert: {tom_dir / 'config.json'}")
    print(f"  Social expert: {social_dir / 'config.json'}")
    print("\nV1 experts use OLMo-3-7B-Instruct with role-specific prompting.")
    print("No separate training required — loaded at inference time.")


if __name__ == "__main__":
    main()
