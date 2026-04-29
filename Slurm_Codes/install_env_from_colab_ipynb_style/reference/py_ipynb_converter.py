import sys
import os
import json
import argparse

def ipynb_to_py(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(py_path, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type')
            source = "".join(cell.get('source', []))
            
            if cell_type == 'code':
                f.write('# %% [code]\n')
                f.write(source + '\n\n')
            elif cell_type == 'markdown':
                f.write('# %% [markdown]\n')
                for line in source.splitlines():
                    f.write('# ' + line + '\n')
                f.write('\n\n')
    print(f"Converted {ipynb_path} -> {py_path}")

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cells = []
    current_cell = []
    cell_type = "code"

    def add_cell(ctype, clines):
        if not clines: return
        while clines and clines[-1].strip() == '':
            clines.pop()
        if not clines: return
        
        if ctype == "markdown":
            source = []
            for l in clines:
                if l.startswith("# "):
                    source.append(l[2:])
                elif l.startswith("#"):
                    source.append(l[1:])
                else:
                    source.append(l)
        else:
            source = clines
            
        cell = {
            "cell_type": ctype,
            "metadata": {},
            "source": [s if s.endswith('\n') else s + '\n' for s in source]
        }
        if ctype == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    for line in lines:
        if line.startswith("# %%"):
            if current_cell:
                add_cell(cell_type, current_cell)
                current_cell = []
            if "[markdown]" in line:
                cell_type = "markdown"
            else:
                cell_type = "code"
        else:
            current_cell.append(line)
    
    if current_cell:
        add_cell(cell_type, current_cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Converted {py_path} -> {ipynb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between .ipynb and .py files.")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", nargs="?", help="Output file path (optional)")
    
    args = parser.parse_args()
    input_path = args.input
    
    if input_path.endswith(".ipynb"):
        output_path = args.output or input_path.replace(".ipynb", ".py")
        ipynb_to_py(input_path, output_path)
    elif input_path.endswith(".py"):
        output_path = args.output or input_path.replace(".py", ".ipynb")
        py_to_ipynb(input_path, output_path)
    else:
        print("Error: Input file must be .ipynb or .py")
        sys.exit(1)
