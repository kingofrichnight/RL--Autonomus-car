"""Validate core source/config structure and optionally local artifact fingerprints."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def check(root: Path, artifacts: bool = False) -> list[str]:
    errors: list[str] = []
    required = [
        "README.md", "pyproject.toml", "requirements.txt", "configs/intersection.yaml",
        "safeintent_rl/envs/intersection.py", "scripts/train_ppo.py",
        "scripts/evaluate_policy.py", "tests/test_ttc.py", "artifacts/manifest.json",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing required file: {name}")
    for directory in ("safeintent_rl", "scripts", "tests", "tools"):
        for path in (root / directory).rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError) as error:
                errors.append(f"{path.relative_to(root)}: {error}")
    for path in (root / "configs").glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    try:
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"pyproject.toml: {error}")
    if artifacts:
        try:
            manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
            for item in manifest["artifacts"]:
                path = (root / item["path"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    errors.append(f"Artifact path escapes repository: {item['path']}")
                elif not path.is_file():
                    errors.append(f"Missing artifact: {item['path']}")
                else:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != item["sha256"]:
                        errors.append(f"Artifact checksum mismatch: {item['path']}")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Artifact manifest: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", action="store_true", help="Require all recorded local artifacts")
    args = parser.parse_args()
    errors = check(ROOT, args.artifacts)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Repository checks: {'FAIL' if errors else 'PASS'} ({len(errors)} errors)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
