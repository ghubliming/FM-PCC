import sys
import os
import json

def migrate_and_convert(src_path, dest_py_path, dest_ipynb_path):
    if not os.path.exists(src_path):
        print(f"Error: Source file {src_path} not found.")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    skip_section = False
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Path and Logic Replacements
        if "# ## 1) Mount Google Drive" in line:
            new_lines.append("# ## 1) Setup Workspace\n")
            i += 1
            continue
        if "from google.colab import drive" in line or "drive.mount(" in line or "print('Drive mounted')" in line:
            i += 1
            continue
        if "FMPCC_ROOT = '/content/drive/MyDrive/FMPCC'" in line:
            new_lines.append("FMPCC_ROOT = '/workspaces'\n")
            i += 1
            continue
        if 'ROOT="/content/drive/MyDrive/FMPCC"' in line:
            new_lines.append('ROOT="/workspaces"\n')
            i += 1
            continue
        if "FMPCC = '/content/drive/MyDrive/FMPCC/FM-PCC'" in line:
            new_lines.append("FMPCC = '/workspaces/FM-PCC'\n")
            i += 1
            continue
        if 'REPO="/content/drive/MyDrive/FMPCC/FM-PCC"' in line:
            new_lines.append('REPO="/workspaces/FM-PCC"\n')
            i += 1
            continue
        if "KEY_FILE = Path('/content/drive/MyDrive/FMPCC/.wandb_api_key')" in line:
            new_lines.append("KEY_FILE = Path('/workspaces/.wandb_api_key')\n")
            i += 1
            continue
        if 'DATA_ZIP="/content/drive/MyDrive/DPCC/dpcc/d3il/environments/dataset/data/dataset.zip"' in line:
            new_lines.append('DATA_ZIP="/workspaces/dataset.zip"\n')
            i += 1
            continue
        if 'OUT="/content/drive/MyDrive/FMPCC/runs_snapshot/$STAMP"' in line:
            new_lines.append('OUT="/workspaces/runs_snapshot/$STAMP"\n')
            i += 1
            continue
        if "/content/miniconda3" in line:
            new_lines.append(line.replace("/content/miniconda3", "/workspaces/miniconda3"))
            i += 1
            continue
        if "/content/miniconda.sh" in line:
            new_lines.append(line.replace("/content/miniconda.sh", "/tmp/miniconda.sh"))
            i += 1
            continue

        # Skip sections like Cache Wiring
        if "# ## 3) Boost Startup Cache Wiring" in line:
            skip_section = True
        
        if skip_section:
            if "# ## 4) Install Miniconda and Create Env" in line:
                skip_section = False
                new_lines.append("# ## 4) Install Miniconda and Create Env\n#\n# Keeps Python pinned to 3.10 for compatibility.\n\n# %%\n%%bash\nset -e\n\nROOT=\"/workspaces\"\nPERSIST_CONDA=\"$ROOT/miniconda3\"\nCONDA_BIN=\"$PERSIST_CONDA/bin/conda\"\nFORCE_REINSTALL_CONDA=\"${FORCE_REINSTALL_CONDA:-0}\"\n\nif [ \"$FORCE_REINSTALL_CONDA\" = \"1\" ]; then\n  rm -rf \"$PERSIST_CONDA\"\nfi\n\nif [ ! -x \"$CONDA_BIN\" ]; then\n  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh\n  bash /tmp/miniconda.sh -b -p \"$PERSIST_CONDA\" -u\nfi\n\n\"$CONDA_BIN\" env list | grep -q \"^FMPCC \" || \"$CONDA_BIN\" create -n FMPCC python=3.10 -y -q\n\"$PERSIST_CONDA/envs/FMPCC/bin/python\" -V\n")
                j = i
                while j < len(lines):
                    if "# ## 5) Install D3IL" in lines[j]:
                        i = j
                        break
                    j += 1
                continue
        
        # Simplify Requirements
        if "# ## 6) Install Requirements" in line:
            new_lines.append("# ## 6) Install Requirements\n# %%\n%%bash\nset -e\nREPO=\"/workspaces/FM-PCC\"\nPIP=\"/workspaces/miniconda3/envs/FMPCC/bin/pip\"\nPY=\"/workspaces/miniconda3/envs/FMPCC/bin/python\"\ncd \"$REPO\"\n\"$PIP\" install -r requirements.txt\n")
            j = i
            while j < len(lines):
                if "# ## 7) Runtime Environment Variables" in lines[j]:
                    i = j
                    break
                j += 1
            continue

        if not skip_section:
            new_lines.append(line)
        i += 1

    # Write .py
    with open(dest_py_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Convert to .ipynb
    cells = []
    current_cell = []
    cell_type = "code"

    def add_cell(ctype, clines):
        if not clines: return
        while clines and clines[-1].strip() == '': clines.pop()
        if not clines: return
        if ctype == "markdown":
            source = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in clines]
        else:
            source = clines
        cell = {"cell_type": ctype, "metadata": {}, "source": source}
        if ctype == "code": cell["outputs"] = []; cell["execution_count"] = None
        cells.append(cell)

    for line in new_lines:
        if line.startswith("# %%"):
            if current_cell: add_cell(cell_type, current_cell); current_cell = []
            cell_type = "markdown" if "[markdown]" in line else "code"
        else:
            current_cell.append(line)
    if current_cell: add_cell(cell_type, current_cell)

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(dest_ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Done. Created {dest_py_path} and {dest_ipynb_path}")

if __name__ == "__main__":
    src = "/workspaces/FM-PCC/ipynbs/develop_the_ipynb/until_gen2/Colab_GPU_FMPCC_MuJoCo_boosted.py"
    out_py = "/workspaces/FM-PCC/ipynbs/develop_the_ipynb/until_gen2/Remote_Local_FMPCC_MuJoCo.py"
    out_nb = "/workspaces/FM-PCC/ipynbs/develop_the_ipynb/until_gen2/Remote_Local_FMPCC_MuJoCo.ipynb"
    migrate_and_convert(src, out_py, out_nb)
