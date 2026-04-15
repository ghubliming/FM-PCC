import argparse
import json
from pathlib import Path


def flush_cell(cells, cell_type, lines):
    if cell_type is None:
        return

    if cell_type == "markdown":
        source = []
        for line in lines:
            if line.startswith("# "):
                source.append(line[2:])
            elif line == "#":
                source.append("")
            else:
                source.append(line)
    else:
        source = lines

    source = [f"{line}\n" for line in source]
    cells.append(
        {
            "cell_type": "markdown" if cell_type == "markdown" else "code",
            "metadata": {"language": "markdown" if cell_type == "markdown" else "python"},
            "source": source,
            "outputs": [] if cell_type == "code" else None,
            "execution_count": None if cell_type == "code" else None,
        }
    )


def convert_py_to_ipynb(py_path: Path, ipynb_path: Path) -> None:
    lines = py_path.read_text(encoding="utf-8").splitlines()

    cells = []
    current_type = None
    buffer = []

    for line in lines:
        if line.startswith("# %%"):
            flush_cell(cells, current_type, buffer)
            buffer = []
            current_type = "markdown" if "[markdown]" in line else "code"
            continue

        if current_type is None:
            continue

        buffer.append(line)

    flush_cell(cells, current_type, buffer)

    for cell in cells:
        if cell["cell_type"] == "markdown":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    ipynb_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert #%% Python notebook source to .ipynb")
    parser.add_argument("py_file", type=Path, help="Input .py file")
    parser.add_argument("ipynb_file", type=Path, help="Output .ipynb file")
    args = parser.parse_args()

    convert_py_to_ipynb(args.py_file, args.ipynb_file)
    print(f"Wrote notebook: {args.ipynb_file}")
