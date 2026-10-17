#!/usr/bin/env python3
"""Check that the public forecaster trainers match the versions that produced the canonical fits.

The canonical XGBoost and LSTM fits record the SHA-256 of train_xgb.py and train_lstm.py as the
files were when the fits ran. The public files in experiments/src differ from those versions only
in comments and docstrings. historical_trainers/ keeps the exact files of the runs. This script
checks them against the recorded hashes and checks that, with docstrings removed, the historical
and the public version of each trainer parse to the same syntax tree.

Usage (from the repository root):  python3 analysis/reproducibility/check_trainer_provenance.py
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = Path(__file__).resolve().parent / "historical_trainers"
RECORDED = {
    "train_xgb.py": "f093701060ce038eb74b777ee556fbf35300a1c44c80138f2fb689facabf19cc",
    "train_lstm.py": "3102657dd42171c0f12dccc5b834fe923931be8cfa034e60963f02e9693b9567",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_without_docstrings(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def main() -> int:
    ok = True
    for name, recorded in RECORDED.items():
        old = HISTORICAL / name
        new = ROOT / "experiments" / "src" / name
        same_bytes = sha256(old) == recorded
        same_tree = tree_without_docstrings(old) == tree_without_docstrings(new)
        ok &= same_bytes and same_tree
        print(f"[{'OK' if same_bytes else 'MISMATCH'}] {name} run version {sha256(old)} (recorded {recorded})")
        print(f"[{'OK' if same_tree else 'MISMATCH'}] {name} public version {sha256(new)}: same code without docstrings")
    print("all checks passed" if ok else "some checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
