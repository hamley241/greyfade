#!/usr/bin/env python3
"""
Download WebArena execution traces from official releases.

Data Sources:
- WebArena GitHub: https://github.com/web-arena-x/webarena
- Paper: https://arxiv.org/abs/2307.13854 (Zhou et al., 2023)
- Resources README: https://github.com/web-arena-x/webarena/blob/main/resources/README.md

Available Trace Datasets:
1. V2 Execution Traces (Nov 2023): GPT-4 and GPT-3.5 agent runs on 812 tasks
   - Google Drive: https://drive.google.com/drive/folders/1H4wkzDkY2ufiC63DISMXllri0j-ipWcs
   
2. Human Trajectories (Dec 2023): ~179 tasks performed by human annotators
   - Google Drive: https://drive.google.com/drive/folders/1NrN_sawtYK2V_uHnmmS8ugmGIKUAsPgt

V2 Trace Files:
- v2_919_gpt4_8k_cot.zip          (~1.5GB) - GPT-4 with Chain-of-Thought
- v2_919_gpt35_16k_cot_na.zip     (~1.5GB) - GPT-3.5-16k CoT without action hints
- v2_919_gpt35_16k_cot.zip        (~1.7GB) - GPT-3.5-16k CoT with action hints
- v2_919_gpt35_16k_direct_na.zip  (~1.5GB) - GPT-3.5-16k Direct without action hints
- v2_919_gpt35_16k_direct.zip     (~1.5GB) - GPT-3.5-16k Direct with action hints
- v2_919_text_bison_001_cot.zip   (~1.5GB) - text-bison-001 with CoT

Version: 2026-03-25
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Google Drive folder IDs
V2_TRACES_FOLDER_ID = "1H4wkzDkY2ufiC63DISMXllri0j-ipWcs"
HUMAN_TRAJECTORIES_FOLDER_ID = "1NrN_sawtYK2V_uHnmmS8ugmGIKUAsPgt"

# Individual file IDs (for targeted downloads)
V2_FILES = {
    "v2_919_gpt4_8k_cot.zip": "1rJzu7edOhbZ36p1I-gYhPw1qP3l0MfRI",
    "v2_919_gpt35_16k_cot_na.zip": "1WI3aBqLaX18vANjb5fmRNOe50JyOxNg3",
    "v2_919_gpt35_16k_cot.zip": "1YRc_8unBHkLrl505K2eeIMIQwOTYkaRe",
    "v2_919_gpt35_16k_direct_na.zip": "1jixSKbVQv6RUv1x628DN_wQhiq8SJ8lQ",
    "v2_919_gpt35_16k_direct.zip": "1m7vS8y4XOcYYukQOl3x7uzw4wCXzCV77",
    "v2_919_text_bison_001_cot.zip": "1Etp6YoeJZ7qChHNhboYZBsEO_qqI1Bv4",
}

# Output directories
BASE_DIR = Path(__file__).parent.parent.parent / "data" / "webarena" / "traces"


def download_with_gdown(file_id: str, output_path: Path) -> bool:
    """Download a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    cmd = ["pipx", "run", "gdown", url, "-O", str(output_path)]
    print(f"Downloading to {output_path}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        return False


def download_folder_with_gdown(folder_id: str, output_dir: Path) -> bool:
    """Download entire folder from Google Drive using gdown."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    cmd = ["pipx", "run", "gdown", "--folder", url, "-O", str(output_dir)]
    print(f"Downloading folder to {output_dir}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading folder: {e}")
        return False


def download_v2_traces(output_dir: Path = None, files: list = None):
    """
    Download V2 execution traces.
    
    Args:
        output_dir: Where to save files (default: BASE_DIR/v2_traces)
        files: List of specific files to download (default: all)
    """
    output_dir = output_dir or BASE_DIR / "v2_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = files or list(V2_FILES.keys())
    
    for filename in files_to_download:
        if filename not in V2_FILES:
            print(f"Unknown file: {filename}")
            continue
            
        output_path = output_dir / filename
        if output_path.exists():
            print(f"Skipping {filename} (already exists)")
            continue
            
        download_with_gdown(V2_FILES[filename], output_path)


def download_human_trajectories(output_dir: Path = None):
    """Download human trajectory recordings."""
    output_dir = output_dir or BASE_DIR / "human_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    download_folder_with_gdown(HUMAN_TRAJECTORIES_FOLDER_ID, output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download WebArena execution traces")
    parser.add_argument("--v2", action="store_true", help="Download V2 agent traces")
    parser.add_argument("--human", action="store_true", help="Download human trajectories")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--file", type=str, help="Download specific file from V2")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if not any([args.v2, args.human, args.all, args.file]):
        parser.print_help()
        print("\nAvailable V2 files:")
        for f in V2_FILES:
            print(f"  - {f}")
        return
    
    output_dir = Path(args.output) if args.output else None
    
    if args.all or args.v2:
        download_v2_traces(output_dir)
    
    if args.all or args.human:
        download_human_trajectories(output_dir)
    
    if args.file:
        download_v2_traces(output_dir, files=[args.file])


if __name__ == "__main__":
    main()
