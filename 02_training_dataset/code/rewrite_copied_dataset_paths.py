#!/usr/bin/env python3
"""Rewrite only old-root string prefixes inside an already copied dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".log", ".yaml", ".yml", ".urdf"}


def replace_nested_strings(value: Any, replacements: list[tuple[str, str]]) -> tuple[Any, int]:
    """Replace decoded path prefixes in YAML values, including escaped Unicode."""
    if isinstance(value, str):
        updated = value
        count = 0
        for old, new in replacements:
            count += updated.count(old)
            updated = updated.replace(old, new)
        return updated, count
    if isinstance(value, list):
        output = []
        count = 0
        for item in value:
            rewritten, item_count = replace_nested_strings(item, replacements)
            output.append(rewritten)
            count += item_count
        return output, count
    if isinstance(value, dict):
        output = {}
        count = 0
        for key, item in value.items():
            rewritten, item_count = replace_nested_strings(item, replacements)
            output[key] = rewritten
            count += item_count
        return output, count
    return value, 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-scene-root", required=True)
    parser.add_argument("--new-scene-root", required=True)
    parser.add_argument("--old-single-root", required=True)
    parser.add_argument("--new-single-root", required=True)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Additional literal prefix replacement; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = args.root.expanduser().resolve()
    replacements = [
        (args.old_scene_root.rstrip("/"), args.new_scene_root.rstrip("/")),
        (args.old_single_root.rstrip("/"), args.new_single_root.rstrip("/")),
    ]
    for specification in args.replace:
        if "=" not in specification:
            raise ValueError(f"--replace requires OLD=NEW, got {specification!r}")
        old, new = specification.split("=", 1)
        replacements.append((old.rstrip("/"), new.rstrip("/")))
    scanned = changed = replacements_made = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for old, new in replacements:
            count = updated.count(old)
            if count:
                replacements_made += count
                updated = updated.replace(old, new)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml

                payload = yaml.safe_load(updated)
                rewritten, structured_count = replace_nested_strings(payload, replacements)
                if structured_count:
                    replacements_made += structured_count
                    updated = yaml.safe_dump(
                        rewritten,
                        allow_unicode=True,
                        sort_keys=False,
                        width=1000,
                    )
            except (ImportError, ValueError, TypeError):
                # Literal replacement above remains valid even in a minimal
                # migration environment without PyYAML.
                pass
        if updated == original:
            continue
        temporary = path.with_name(path.name + ".migration-tmp")
        temporary.write_text(updated, encoding="utf-8")
        os.replace(temporary, path)
        changed += 1
    print(
        f"scanned_text_files={scanned} changed_files={changed} "
        f"path_replacements={replacements_made}"
    )


if __name__ == "__main__":
    main()
