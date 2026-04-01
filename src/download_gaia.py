#!/usr/bin/env python3
"""
Download GAIA Benchmark Traces from HAL Leaderboard

This script downloads and decrypts agent execution traces from the Holistic Agent
Leaderboard (HAL) for the GAIA benchmark. These traces contain full agent execution
logs with success/failure labels for grey failure analysis.

Data Sources:
- HAL Leaderboard: https://hal.cs.princeton.edu/gaia
- Trace Storage: https://huggingface.co/datasets/agent-evals/hal_traces
- HAL Harness: https://github.com/princeton-pli/hal-harness

GAIA Benchmark:
- Paper: "GAIA: a benchmark for General AI Assistants" (Mialon et al., 2023)
- arXiv: https://arxiv.org/abs/2311.12983
- 165 validation questions across 3 difficulty levels

Encryption:
- HAL traces are encrypted to prevent benchmark contamination
- Decryption key: "hal1234" (published in hal-harness repo)
- Uses Fernet encryption with PBKDF2 key derivation

Version: 1.0.0
Date: 2026-03-25
Requirements: cryptography (pip install cryptography)
"""

import os
import json
import base64
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from typing import Dict, List, Optional, Any

# Encryption dependencies - install with: pip install cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: cryptography not installed. Run: pip install cryptography")


# HAL trace encryption key (published in hal-harness repo)
HAL_ENCRYPTION_KEY = "hal1234"

# Available GAIA traces from HAL leaderboard (as of 2026-03-25)
AVAILABLE_TRACES = [
    {
        "name": "Claude Sonnet 4.5",
        "filename": "gaia_hal_generalist_agent_claudesonnet45_1759276006_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_claudesonnet45_1759276006_UPLOAD.zip",
        "accuracy": 0.7455,
        "level1": 0.8207,
        "level2": 0.7268,
        "level3": 0.6539,
        "cost_usd": 178.20,
        "scaffold": "HAL Generalist Agent",
    },
    {
        "name": "GPT-4.1",
        "filename": "gaia_hal_generalist_agent_gpt4120250414_1744652581_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_gpt4120250414_1744652581_UPLOAD.zip",
        "accuracy": 0.4970,
        "level1": 0.5283,
        "level2": 0.5581,
        "level3": 0.2308,
        "cost_usd": 74.19,
        "scaffold": "HAL Generalist Agent",
    },
    {
        "name": "Gemini 2.0 Flash",
        "filename": "gaia_hal_generalist_agent_gemini20flash_1744828175_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_gemini20flash_1744828175_UPLOAD.zip",
        "accuracy": 0.3273,
        "level1": 0.4340,
        "level2": 0.3256,
        "level3": 0.1154,
        "cost_usd": 7.80,
        "scaffold": "HAL Generalist Agent",
    },
    {
        "name": "DeepSeek V3",
        "filename": "gaia_hal_generalist_agent_deepseekaideepseekv3_1744673872_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_deepseekaideepseekv3_1744673872_UPLOAD.zip",
        "accuracy": 0.2939,
        "level1": 0.3868,
        "level2": 0.3197,
        "level3": 0.0193,
        "cost_usd": 17.40,
        "scaffold": "HAL Generalist Agent",
    },
    {
        "name": "Claude 3.7 Sonnet",
        "filename": "gaia_hal_generalist_agent_claude37sonnet20250219_1744772193_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_claude37sonnet20250219_1744772193_UPLOAD.zip",
        "accuracy": 0.5636,
        "level1": 0.6226,
        "level2": 0.5581,
        "level3": 0.4615,
        "cost_usd": 130.68,
        "scaffold": "HAL Generalist Agent",
    },
    {
        "name": "o4-mini High",
        "filename": "gaia_hal_generalist_agent_o4mini20250416_high_1745167285_UPLOAD.zip",
        "url": "https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/gaia_hal_generalist_agent_o4mini20250416_high_1745167285_UPLOAD.zip",
        "accuracy": 0.5455,
        "level1": 0.6038,
        "level2": 0.5349,
        "level3": 0.4615,
        "cost_usd": 59.39,
        "scaffold": "HAL Generalist Agent",
    },
]


def generate_key(password: bytes, salt: bytes) -> bytes:
    """Generate Fernet key from password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password))


def decrypt_hal_traces(encrypted_data: str, salt: str, password: str = HAL_ENCRYPTION_KEY) -> dict:
    """
    Decrypt HAL trace data.
    
    Args:
        encrypted_data: Base64-encoded encrypted data
        salt: Base64-encoded salt
        password: Encryption password (default: hal1234)
    
    Returns:
        Decrypted JSON data as dict
    """
    if not HAS_CRYPTO:
        raise ImportError("cryptography package required. Install with: pip install cryptography")
    
    salt_bytes = base64.b64decode(salt.encode("utf-8"))
    key = generate_key(password.encode(), salt_bytes)
    cipher = Fernet(key)
    encrypted_bytes = base64.b64decode(encrypted_data.encode("utf-8"))
    decrypted_data = cipher.decrypt(encrypted_bytes)
    return json.loads(decrypted_data.decode("utf-8"))


def decrypt_hal_zip(zip_path: Path) -> dict:
    """Extract and decrypt HAL traces from zip file."""
    with ZipFile(zip_path, 'r') as zf:
        json_filename = zf.namelist()[0]
        with zf.open(json_filename) as f:
            encrypted = json.load(f)
    
    return decrypt_hal_traces(encrypted["encrypted_data"], encrypted["salt"])


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> None:
    """Download file with progress indicator."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")
    
    def progress_hook(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest_path, progress_hook if show_progress else None)
    if show_progress:
        print()  # newline after progress


def extract_traces_from_hal(hal_data: dict, model_name: str, metadata: dict) -> List[dict]:
    """
    Extract conversation traces from HAL format.
    
    HAL trace structure:
    - raw_logging_results: List of Weave log entries (LLM calls)
    - raw_eval_results: Dict mapping task_id -> {score, explanation}
    """
    raw_logs = hal_data.get('raw_logging_results', [])
    raw_eval = hal_data.get('raw_eval_results', {})
    
    # Group logs by weave_task_id (internal task identifier)
    task_logs = {}
    for log in raw_logs:
        if log is None:
            continue
        attrs = log.get('attributes') or {}
        task_id = attrs.get('weave_task_id') or log.get('weave_task_id')
        if task_id:
            if task_id not in task_logs:
                task_logs[task_id] = []
            task_logs[task_id].append(log)
    
    traces = []
    
    for internal_task_id, logs in task_logs.items():
        # Sort logs chronologically
        logs.sort(key=lambda x: x.get('started_at', '') or '')
        
        # Extract conversation turns
        turns = []
        for log in logs:
            if log is None:
                continue
            inputs = log.get('inputs') or {}
            messages = inputs.get('messages') or []
            output = log.get('output') or {}
            
            # Process input messages
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content is None:
                    content = ''
                # Handle multimodal content (list of content blocks)
                if isinstance(content, list):
                    text_parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get('type') == 'text':
                            text_parts.append(p.get('text', ''))
                    content = ' '.join(text_parts) if text_parts else str(content)
                
                turns.append({
                    'role': role,
                    'content': content,
                    'timestamp': log.get('started_at')
                })
            
            # Process output (assistant response)
            if isinstance(output, dict):
                choices = output.get('choices') or []
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        assistant_msg = first_choice.get('message') or {}
                        if isinstance(assistant_msg, dict):
                            content = assistant_msg.get('content')
                            if content:
                                turns.append({
                                    'role': 'assistant',
                                    'content': content,
                                    'timestamp': log.get('ended_at')
                                })
        
        trace = {
            'internal_task_id': internal_task_id,
            'turns': turns,
            'model': model_name,
            'scaffold': metadata.get('scaffold', 'unknown'),
            'num_llm_calls': len(logs),
        }
        traces.append(trace)
    
    # Match evaluation results to traces
    eval_results = list(raw_eval.items())
    
    if len(traces) == len(eval_results):
        for i, trace in enumerate(traces):
            gaia_task_id, result = eval_results[i]
            trace['task_id'] = gaia_task_id
            if isinstance(result, dict):
                trace['success'] = result.get('score', False)
                trace['eval_explanation'] = result.get('explanation', '')
            else:
                trace['success'] = False
                trace['eval_explanation'] = str(result) if result else ''
    else:
        print(f"  Warning: {len(traces)} traces but {len(eval_results)} eval results")
        for i, trace in enumerate(traces):
            if i < len(eval_results):
                gaia_task_id, result = eval_results[i]
                trace['task_id'] = gaia_task_id
                if isinstance(result, dict):
                    trace['success'] = result.get('score', False)
                    trace['eval_explanation'] = result.get('explanation', '')
    
    return traces


def download_gaia_traces(
    output_dir: Path,
    models: Optional[List[str]] = None,
    keep_zips: bool = False
) -> Dict[str, List[dict]]:
    """
    Download and extract GAIA traces from HAL leaderboard.
    
    Args:
        output_dir: Directory to save traces
        models: List of model names to download (None = all)
        keep_zips: Keep original zip files after extraction
    
    Returns:
        Dict mapping model names to list of traces
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter models if specified
    traces_to_download = AVAILABLE_TRACES
    if models:
        models_lower = [m.lower() for m in models]
        traces_to_download = [
            t for t in AVAILABLE_TRACES 
            if any(m in t["name"].lower() for m in models_lower)
        ]
    
    print(f"Downloading {len(traces_to_download)} trace files to {output_dir}")
    print("-" * 60)
    
    all_traces = {}
    
    for trace_info in traces_to_download:
        name = trace_info["name"]
        url = trace_info["url"]
        filename = trace_info["filename"]
        
        print(f"\n[{name}]")
        print(f"  Accuracy: {trace_info['accuracy']*100:.1f}%")
        print(f"  Level 1/2/3: {trace_info['level1']*100:.0f}%/{trace_info['level2']*100:.0f}%/{trace_info['level3']*100:.0f}%")
        
        zip_path = output_dir / filename
        
        # Download if not exists
        if not zip_path.exists():
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"  ERROR downloading: {e}")
                continue
        else:
            print(f"  Already downloaded: {filename}")
        
        # Decrypt and extract traces
        try:
            print(f"  Decrypting...")
            hal_data = decrypt_hal_zip(zip_path)
            traces = extract_traces_from_hal(hal_data, name, trace_info)
            
            success_count = sum(1 for t in traces if t.get('success'))
            print(f"  Extracted {len(traces)} traces, {success_count} successes ({100*success_count/len(traces):.1f}%)")
            
            all_traces[name] = traces
            
            if not keep_zips:
                zip_path.unlink()
                print(f"  Removed zip file")
        except Exception as e:
            print(f"  ERROR extracting: {e}")
            import traceback
            traceback.print_exc()
    
    return all_traces


def create_unified_dataset(
    traces_by_model: Dict[str, List[dict]],
    output_path: Path
) -> dict:
    """
    Combine traces from multiple models into unified dataset.
    
    Output format matches task specification:
    {
        "trace_id": "gaia-XXXX",
        "task": "question text" (from first user turn),
        "turns": [...],
        "success": true/false,
        "difficulty": "level1/level2/level3" (if known),
        ...
    }
    """
    all_traces = []
    
    for model_name, traces in traces_by_model.items():
        for trace in traces:
            # Extract task question from first substantive user turn
            task_text = ""
            for turn in trace.get('turns', []):
                if turn.get('role') == 'user' and len(turn.get('content', '')) > 50:
                    task_text = turn['content'][:500]  # First 500 chars
                    break
            
            trace['task'] = task_text
            all_traces.append(trace)
    
    # Assign global trace IDs
    for i, trace in enumerate(all_traces):
        trace['trace_id'] = f"gaia-{i:04d}"
    
    # Build dataset
    dataset = {
        'version': '1.0.0',
        'created': '2026-03-25',
        'source': 'HAL Leaderboard (https://hal.cs.princeton.edu/gaia)',
        'benchmark': 'GAIA',
        'paper': 'https://arxiv.org/abs/2311.12983',
        'description': 'Agent execution traces from GAIA benchmark evaluations',
        'num_traces': len(all_traces),
        'models': list(traces_by_model.keys()),
        'success_rate': sum(1 for t in all_traces if t.get('success')) / len(all_traces) if all_traces else 0,
        'traces': all_traces
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved unified dataset: {output_path}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Success rate: {dataset['success_rate']*100:.1f}%")
    print(f"File size: {size_mb:.1f} MB")
    
    return dataset


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download GAIA benchmark traces from HAL leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all available traces
  python download_gaia.py -o ./data/gaia
  
  # Download specific models only
  python download_gaia.py -o ./data/gaia -m "Claude" "GPT"
  
  # List available traces
  python download_gaia.py --list
  
  # Create unified dataset
  python download_gaia.py -o ./data/gaia --unified ./data/gaia/gaia-traces.json
"""
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data/gaia"),
        help="Output directory for traces (default: ./data/gaia)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Specific models to download (default: all). Partial matches allowed."
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep original encrypted zip files after extraction"
    )
    parser.add_argument(
        "--unified", "-u",
        type=Path,
        help="Path for unified dataset output (combines all models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available traces and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available GAIA traces from HAL Leaderboard:")
        print("-" * 70)
        for t in AVAILABLE_TRACES:
            print(f"  {t['name']}")
            print(f"    Accuracy: {t['accuracy']*100:.1f}% | Cost: ${t['cost_usd']:.2f}")
            print(f"    L1: {t['level1']*100:.0f}% | L2: {t['level2']*100:.0f}% | L3: {t['level3']*100:.0f}%")
            print()
        return
    
    # Check dependencies
    if not HAS_CRYPTO:
        print("ERROR: cryptography package required for decryption")
        print("Install with: pip install cryptography")
        return 1
    
    # Download traces
    traces_by_model = download_gaia_traces(
        output_dir=args.output_dir,
        models=args.models,
        keep_zips=args.keep_zips
    )
    
    if not traces_by_model:
        print("No traces downloaded!")
        return 1
    
    # Create unified dataset
    if args.unified:
        create_unified_dataset(traces_by_model, args.unified)
    else:
        # Default: create unified dataset in output dir
        unified_path = args.output_dir / "gaia-traces.json"
        create_unified_dataset(traces_by_model, unified_path)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
