#!/bin/bash
# Full pipeline: data -> teacher -> student -> eval
# Run from project root: bash run_all.sh

set -e

echo "=== Phase 1: Data Preparation ==="
python scripts/prepare_simpletom.py
python scripts/prepare_kokomind.py
python scripts/prepare_theory_of_mind.py
python scripts/prepare_tomi_nli.py
python scripts/prepare_social_iqa.py
python scripts/prepare_cicero.py
python scripts/build_router_dataset.py

echo ""
echo "=== Phase 2: Teacher Labeling ==="
TRANSFORMERS_VERBOSITY=error python scripts/generate_teacher_labels.py --batch-size 4

echo ""
echo "=== Phase 3: Student Training ==="
python scripts/train_student_router.py --config configs/router_student.yaml

echo ""
echo "=== Phase 4: Evaluation ==="
python scripts/eval_router.py --config configs/eval_router.yaml

echo ""
echo "=== Phase 5: Expert Setup & Routed System ==="
python scripts/train_experts.py --config configs/experts.yaml
python scripts/eval_routed_system.py --config configs/eval_routed.yaml

echo ""
echo "=== Phase 6: Export ==="
python scripts/export_router.py

echo ""
echo "=== Done! ==="
echo "Reports: outputs/reports/"
echo "Checkpoints: outputs/checkpoints/"
