#!/usr/bin/env python3
"""
Extract WebArena execution traces from HTML render files (no BeautifulSoup dependency).

Uses regex to parse HTML files. For more robust parsing, use extract_webarena_traces.py
with BeautifulSoup installed.

Version: 2026-03-25
"""

import os
import re
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent / "data" / "webarena" / "traces"
OUTPUT_DIR = BASE_DIR
V2_DIR = BASE_DIR / "v2_traces"


def parse_log_file(log_content: str) -> Dict[int, bool]:
    """Parse merged_log.txt to get pass/fail status per task."""
    results = {}
    for line in log_content.split("\n"):
        if "[Result]" in line:
            match = re.search(r"(\d+)\.json", line)
            if match:
                task_id = int(match.group(1))
                success = "(PASS)" in line
                results[task_id] = success
    return results


def extract_text(content: str, class_name: str) -> List[str]:
    """Extract text from div elements with a specific class."""
    pattern = rf"<div class=['\"]?{class_name}['\"]?[^>]*><pre>(.*?)</pre></div>"
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]


def extract_turns_from_html(html_content: str) -> List[Dict[str, str]]:
    """Extract action-observation turns from an HTML render file using regex."""
    turns = []
    
    # Extract observations
    obs_pattern = r"<div class=['\"]?state_obv['\"]?><pre>(.*?)</pre>"
    observations = re.findall(obs_pattern, html_content, re.DOTALL)
    
    # Extract parsed actions
    action_pattern = r"<div class=['\"]?parsed_action['\"]?[^>]*><pre>(.*?)</pre></div>"
    actions = re.findall(action_pattern, html_content, re.DOTALL)
    
    # Pair observations with actions
    for i in range(max(len(observations), len(actions))):
        obs = observations[i].strip()[:2000] if i < len(observations) else ""
        action = actions[i].strip() if i < len(actions) else ""
        
        if obs or action:
            turns.append({
                "observation": obs,
                "action": action
            })
    
    return turns


def extract_task_info(html_content: str) -> Dict[str, Any]:
    """Extract task metadata from HTML header using regex."""
    info = {}
    
    # Find the first <pre> block which contains task info
    pre_match = re.search(r"<pre>(.*?)</pre>", html_content, re.DOTALL)
    if not pre_match:
        return info
    
    first_pre = pre_match.group(1)
    
    # Extract key fields
    patterns = {
        "task_id": r"task_id:\s*(\d+)",
        "intent": r"intent:\s*(.+?)(?:\n|$)",
        "intent_template": r"intent_template:\s*(.+?)(?:\n|$)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, first_pre)
        if match:
            info[key] = match.group(1).strip()
    
    return info


def process_trace_zip(zip_path: Path, model_name: str) -> List[Dict[str, Any]]:
    """Process a V2 trace zip file and extract all traces."""
    traces = []
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the log file and HTML files
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
        for zip_file in sorted(input_path.glob("*.zip")):
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
    if all_traces:
        success_count = sum(1 for t in all_traces if t["success"])
        print(f"Success rate: {success_count}/{len(all_traces)} ({100*success_count/len(all_traces):.1f}%)")


if __name__ == "__main__":
    main()
