#!/usr/bin/env python3
"""
Extract WebArena execution traces from HTML render files and convert to JSON format.

Input: WebArena V2 trace zip files (from Google Drive)
Output: webarena-traces.json with format:
[{
  "trace_id": "webarena-0001",
  "task_id": "...",
  "turns": [{"action": "...", "observation": "..."}],
  "success": true/false,
  "reward": 0.0-1.0
}]

Version: 2026-03-25
"""

import os
import re
import json
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent / "data" / "webarena" / "traces"
OUTPUT_DIR = BASE_DIR
V2_DIR = BASE_DIR / "v2_traces"


def parse_log_file(log_content: str) -> Dict[int, bool]:
    """Parse merged_log.txt to get pass/fail status per task."""
    results = {}
    for line in log_content.split("\n"):
        if "[Result]" in line:
            # Extract task ID from config path
            match = re.search(r"(\d+)\.json", line)
            if match:
                task_id = int(match.group(1))
                success = "(PASS)" in line
                results[task_id] = success
    return results


def extract_turns_from_html(html_content: str) -> List[Dict[str, str]]:
    """Extract action-observation turns from an HTML render file."""
    soup = BeautifulSoup(html_content, "html.parser")
    turns = []
    
    # Get observations
    observations = soup.find_all("div", {"class": "state_obv"})
    # Get raw predictions
    raw_predictions = soup.find_all("div", {"class": "raw_parsed_prediction"})
    # Get parsed actions
    parsed_actions = soup.find_all("div", {"class": "parsed_action"})
    
    # Pair observations with actions
    for i, obs_div in enumerate(observations):
        obs_text = obs_div.get_text(strip=True) if obs_div else ""
        
        action_text = ""
        if i < len(parsed_actions):
            action_text = parsed_actions[i].get_text(strip=True)
        elif i < len(raw_predictions):
            action_text = raw_predictions[i].get_text(strip=True)
        
        if obs_text or action_text:
            turns.append({
                "observation": obs_text[:2000] if obs_text else "",  # Truncate long observations
                "action": action_text
            })
    
    return turns


def extract_task_info(html_content: str) -> Dict[str, Any]:
    """Extract task metadata from HTML header."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find the first <pre> which contains task info
    pre_tags = soup.find_all("pre")
    if not pre_tags:
        return {}
    
    first_pre = pre_tags[0].get_text()
    
    info = {}
    
    # Parse YAML-like structure
    for line in first_pre.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key in ["task_id", "intent", "intent_template", "intent_template_id"]:
                info[key] = value
            elif key == "eval":
                # Try to parse eval dict
                try:
                    # This is a simplified extraction
                    if "reference_answers" in value:
                        info["eval_type"] = "string_match" if "string_match" in value else "other"
                except:
                    pass
    
    return info


def process_trace_zip(zip_path: Path, model_name: str) -> List[Dict[str, Any]]:
    """Process a V2 trace zip file and extract all traces."""
    traces = []
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the log file
        log_file = None
        html_files = []
        
        for name in zf.namelist():
            if name.endswith("merged_log.txt"):
                log_file = name
            elif name.endswith(".html") and "render_" in name:
                html_files.append(name)
        
        # Parse results from log
        results = {}
        if log_file:
            log_content = zf.read(log_file).decode("utf-8", errors="ignore")
            results = parse_log_file(log_content)
        
        # Process each HTML file
        for html_file in sorted(html_files):
            # Extract task ID from filename
            match = re.search(r"render_(\d+)\.html", html_file)
            if not match:
                continue
            task_id = int(match.group(1))
            
            try:
                html_content = zf.read(html_file).decode("utf-8", errors="ignore")
                
                task_info = extract_task_info(html_content)
                turns = extract_turns_from_html(html_content)
                
                success = results.get(task_id, False)
                
                trace = {
                    "trace_id": f"webarena-{model_name}-{task_id:04d}",
                    "task_id": str(task_id),
                    "model": model_name,
                    "intent": task_info.get("intent", ""),
                    "intent_template": task_info.get("intent_template", ""),
                    "turns": turns,
                    "success": success,
                    "reward": 1.0 if success else 0.0
                }
                
                traces.append(trace)
                
            except Exception as e:
                print(f"Error processing {html_file}: {e}")
                continue
    
    return traces


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract WebArena traces from HTML files")
    parser.add_argument("--input", type=str, help="Input zip file or directory")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--model", type=str, default="gpt4", help="Model name for trace IDs")
    
    args = parser.parse_args()
    
    # Default paths
    input_path = Path(args.input) if args.input else V2_DIR / "v2_919_gpt4_8k_cot.zip"
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "webarena-traces.json"
    
    all_traces = []
    
    if input_path.is_file() and input_path.suffix == ".zip":
        print(f"Processing {input_path}...")
        traces = process_trace_zip(input_path, args.model)
        all_traces.extend(traces)
        print(f"  Extracted {len(traces)} traces")
    elif input_path.is_dir():
        # Process all zip files in directory
        for zip_file in input_path.glob("*.zip"):
            # Extract model name from filename
            model_name = zip_file.stem.replace("v2_919_", "").replace("_", "-")
            print(f"Processing {zip_file.name} ({model_name})...")
            traces = process_trace_zip(zip_file, model_name)
            all_traces.extend(traces)
            print(f"  Extracted {len(traces)} traces")
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_traces, f, indent=2)
    
    print(f"\nTotal traces: {len(all_traces)}")
    print(f"Saved to: {output_path}")
    
    # Stats
    success_count = sum(1 for t in all_traces if t["success"])
    print(f"Success rate: {success_count}/{len(all_traces)} ({100*success_count/len(all_traces):.1f}%)")


if __name__ == "__main__":
    main()
