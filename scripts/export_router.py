"""Export the trained router for deployment."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.router_student import StudentRouter, get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/router_student/best_f1")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output", default="outputs/checkpoints/router_exported")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = StudentRouter(model_name=args.model_name)
    state_dict = torch.load(
        f"{args.checkpoint}/model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Save full model state
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save tokenizer
    tokenizer = get_tokenizer(args.model_name)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "input_format": "[CONTEXT] {context} [QUESTION] {question}",
        "output": "single logit -> sigmoid for P(requires_tom)",
    }

    # Load threshold info if available
    metrics_path = Path("outputs/reports/router_eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            eval_data = json.load(f)
            metadata["thresholds"] = eval_data.get("thresholds", {})
            metadata["test_metrics"] = eval_data.get("main_metrics", {})

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Router exported to {output_dir}")
    print(f"  Model: {output_dir / 'model.pt'}")
    print(f"  Tokenizer: {output_dir}")
    print(f"  Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
