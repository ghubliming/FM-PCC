import argparse
import os
import re

def clean_log(input_path):
    # Determine output path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_cleaned{ext}"

    print(f"Reading: {input_path}")
    print(f"Writing: {output_path}")

    # Pattern for tqdm progress bar: " 0%|          | 0/1000" or similar
    # We look for the percentage followed by the | bar | or the time estimate [00:00<...]
    pbar_pattern = re.compile(r'\d{1,3}%\||\|\s+\d+/\d+|\[\d+:\d+<\d+:\d+')

    lines_kept = 0
    lines_removed = 0

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # If it looks like a progress bar
                is_pbar = pbar_pattern.search(line)
                if is_pbar:
                    # Keep it ONLY if it is a completion line (100%)
                    if '100%' in line:
                        f_out.write(line)
                        lines_kept += 1
                    else:
                        lines_removed += 1
                else:
                    # Keep all other lines (errors, prints, etc.)
                    f_out.write(line)
                    lines_kept += 1

    print(f"Done! Kept {lines_kept} lines, removed {lines_removed} spam lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean tqdm progress bar spam from Slurm logs.")
    parser.add_argument("log_file", help="Path to the .log file to clean.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: File not found: {args.log_file}")
    else:
        clean_log(args.log_file)
