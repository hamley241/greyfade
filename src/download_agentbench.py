#!/usr/bin/env python3
"""
Download and process AgentBench execution traces from THUDM/AgentInstruct dataset.

Data Source:
- HuggingFace: https://huggingface.co/datasets/THUDM/AgentInstruct
- GitHub: https://github.com/THUDM/AgentBench
- Paper: https://arxiv.org/abs/2308.03688 (AgentBench: Evaluating LLMs as Agents, ICLR'24)

The AgentInstruct dataset contains interaction trajectories from the AgentBench benchmark,
covering 6 domains: OS interaction, Database (SQL), ALFWorld, WebShop, Knowledge Graph, and Mind2Web.

Requirements:
    pip install datasets

Note on Data:
    The THUDM/AgentInstruct dataset contains TRAINING trajectories, which are curated
    successful runs used for behavior cloning. As such, most traces will be successful.
    Use --add-failures to generate synthetic failure traces by truncating successful ones.

Output:
    agentbench-traces.json - Formatted traces with the following schema:
    [
        {
            "trace_id": "agentbench-0001",
            "domain": "os|db|alfworld|webshop|knowledgegraph|mind2web",
            "task_id": "os_0",
            "turns": [
                {"role": "agent|environment", "content": "...", "loss": true/false/null}
            ],
            "success": true/false/null,
            "score": 0.0-1.0,
            "num_turns": 18,
            "source": "THUDM/AgentInstruct"
        }
    ]

Version: 1.0.0
Date: 2026-03-25
Author: Generated for grey-failures-paper research
"""

import json
import argparse
import os
import re
from pathlib import Path

def load_agentinstruct_dataset():
    """Load the THUDM/AgentInstruct dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print("Loading THUDM/AgentInstruct dataset from HuggingFace...")
    print("Source: https://huggingface.co/datasets/THUDM/AgentInstruct")
    
    ds = load_dataset('THUDM/AgentInstruct')
    return ds


def parse_conversations(conversations):
    """Parse conversation data which may be a list or JSON string."""
    if isinstance(conversations, str):
        try:
            return json.loads(conversations)
        except json.JSONDecodeError:
            return []
    return conversations if conversations else []


def determine_success(turns, domain=None):
    """
    Determine if the trace was successful based on the final agent action.
    
    Different domains have different success patterns:
    - OS: "Act: answer(" or "Act: finish"
    - DB: "Action: Answer" followed by "Final Answer:"
    - ALFWorld: Task completion indicated by environment response
    - WebShop: "click[Buy Now]" action
    - KG: "Final Answer: #" pattern
    - Mind2Web: "Answer:" pattern
    
    Note: These are training trajectories from successful runs, so most should be successful.
    We mark success based on whether the trace reaches a terminal action.
    """
    if not turns:
        return None, None
    
    # Find last agent turn
    last_agent_content = None
    for turn in reversed(turns):
        if turn.get("role") == "agent":
            last_agent_content = turn.get("content", "")
            break
    
    if not last_agent_content:
        return None, None
    
    # Domain-specific success patterns
    success_patterns = {
        'os': [r'Act:\s*answer\(', r'Act:\s*finish'],
        'db': [r'Action:\s*Answer', r'Final Answer:'],
        'alfworld': [r'ACTION:', r'put .* in/on'],  # Terminal actions
        'webshop': [r'click\[Buy Now\]', r'Action:\s*click'],
        'knowledgegraph': [r'Final Answer:', r'#\d+'],
        'kg': [r'Final Answer:', r'#\d+'],
        'mind2web': [r'Answer:', r'click\[', r'type\['],
    }
    
    # Check for domain-specific patterns
    patterns = success_patterns.get(domain, [])
    for pattern in patterns:
        if re.search(pattern, last_agent_content, re.IGNORECASE):
            return True, 1.0
    
    # Generic success patterns (fallback)
    generic_patterns = [
        r'Act:\s*answer\(',
        r'Act:\s*finish',
        r'Final Answer:',
        r'Action:\s*Answer',
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, last_agent_content, re.IGNORECASE):
            return True, 1.0
    
    # If has loss=True on final turn, it's part of training data (likely successful)
    for turn in reversed(turns):
        if turn.get("role") == "agent" and turn.get("loss") is True:
            return True, 1.0
        break
    
    return False, 0.0


def process_dataset(ds, max_content_length=2000):
    """
    Process the dataset into standardized trace format.
    
    Args:
        ds: HuggingFace dataset object
        max_content_length: Maximum length for turn content (truncates longer content)
    
    Returns:
        List of trace dictionaries
    """
    # Domain name mapping for consistency
    domain_map = {
        'os': 'os',
        'db': 'db', 
        'alfworld': 'alfworld',
        'webshop': 'webshop',
        'kg': 'knowledgegraph',
        'mind2web': 'mind2web'
    }
    
    traces = []
    trace_id_counter = 0
    
    for domain in ds.keys():
        print(f"\nProcessing {domain} ({len(ds[domain])} samples)...")
        domain_stats = {'success': 0, 'failure': 0, 'unknown': 0}
        
        for idx, item in enumerate(ds[domain]):
            trace_id_counter += 1
            
            # Parse conversations
            conversations = parse_conversations(item.get('conversations', []))
            
            # Extract and format turns
            turns = []
            for turn in conversations:
                role_raw = turn.get('from', 'unknown')
                content = turn.get('value', '')
                loss = turn.get('loss', None)
                
                # Normalize role names
                if role_raw == 'gpt':
                    role = 'agent'
                elif role_raw == 'human':
                    role = 'environment'
                else:
                    role = role_raw
                
                turn_entry = {
                    "role": role,
                    "content": content[:max_content_length] if len(content) > max_content_length else content
                }
                
                # Include loss field if present (used for training)
                if loss is not None:
                    turn_entry["loss"] = loss
                
                turns.append(turn_entry)
            
            # Determine success (pass domain for domain-specific patterns)
            success, score = determine_success(turns, domain=domain_map.get(domain, domain))
            
            # Update stats
            if success is True:
                domain_stats['success'] += 1
            elif success is False:
                domain_stats['failure'] += 1
            else:
                domain_stats['unknown'] += 1
            
            # Build trace object
            trace = {
                "trace_id": f"agentbench-{trace_id_counter:04d}",
                "domain": domain_map.get(domain, domain),
                "task_id": item.get('id', f"{domain}_{idx}"),
                "turns": turns,
                "success": success,
                "score": score,
                "num_turns": len(turns),
                "source": "THUDM/AgentInstruct"
            }
            
            traces.append(trace)
        
        print(f"  Stats: {domain_stats['success']} success, {domain_stats['failure']} failure, {domain_stats['unknown']} unknown")
    
    return traces


def save_traces(traces, output_path):
    """Save traces to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(traces, f, indent=2)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nSaved {len(traces)} traces to {output_path}")
    print(f"File size: {file_size_kb:.1f} KB")


def print_summary(traces):
    """Print summary statistics of the traces."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nTotal traces: {len(traces)}")
    
    # Domain breakdown
    domains = {}
    for t in traces:
        d = t['domain']
        if d not in domains:
            domains[d] = {'total': 0, 'success': 0, 'failure': 0}
        domains[d]['total'] += 1
        if t['success'] is True:
            domains[d]['success'] += 1
        elif t['success'] is False:
            domains[d]['failure'] += 1
    
    print("\nBy domain:")
    for domain, stats in sorted(domains.items()):
        print(f"  {domain:15} {stats['total']:4} traces ({stats['success']} success, {stats['failure']} failure)")
    
    # Sample trace
    print("\nSample trace structure:")
    sample = traces[0]
    sample_display = {k: v if k != 'turns' else f"[{len(v)} turns]" for k, v in sample.items()}
    print(json.dumps(sample_display, indent=2))


def add_synthetic_failures(traces, num_failures=50, seed=42):
    """
    Generate synthetic failure traces by truncating successful trajectories.
    
    This simulates cases where agents got stuck, gave up, or failed to complete tasks.
    Useful for creating a more balanced dataset for analysis.
    
    Args:
        traces: List of trace dictionaries
        num_failures: Number of synthetic failures to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of synthetic failure traces
    """
    import random
    random.seed(seed)
    
    # Filter traces suitable for truncation (need enough turns)
    suitable = [t for t in traces if t['num_turns'] >= 6 and t['success'] is True]
    
    if len(suitable) < num_failures:
        print(f"Warning: Only {len(suitable)} traces suitable for truncation, generating {len(suitable)} failures")
        num_failures = len(suitable)
    
    # Sample traces to truncate
    sampled = random.sample(suitable, min(num_failures, len(suitable)))
    
    failure_traces = []
    trace_id_offset = max(int(t['trace_id'].split('-')[1]) for t in traces)
    
    for i, trace in enumerate(sampled):
        # Truncate to half the turns
        truncate_point = len(trace['turns']) // 2
        truncated_turns = trace['turns'][:truncate_point]
        
        failure_trace = {
            "trace_id": f"agentbench-{trace_id_offset + i + 1:04d}",
            "domain": trace['domain'],
            "task_id": trace['task_id'] + "_truncated",
            "turns": truncated_turns,
            "success": False,
            "score": 0.0,
            "num_turns": len(truncated_turns),
            "source": "THUDM/AgentInstruct (synthetic failure)",
            "failure_mode": "incomplete_trajectory"
        }
        failure_traces.append(failure_trace)
    
    return failure_traces


def main():
    parser = argparse.ArgumentParser(
        description='Download and process AgentBench traces from THUDM/AgentInstruct',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  - HuggingFace: https://huggingface.co/datasets/THUDM/AgentInstruct
  - GitHub: https://github.com/THUDM/AgentBench
  - Paper: https://arxiv.org/abs/2308.03688

Examples:
  %(prog)s                          # Download to current directory
  %(prog)s -o /path/to/output.json  # Specify output path
  %(prog)s --max-content 5000       # Allow longer content in turns
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        default='agentbench-traces.json',
        help='Output file path (default: agentbench-traces.json)'
    )
    
    parser.add_argument(
        '--max-content',
        type=int,
        default=2000,
        help='Maximum content length per turn (default: 2000 chars)'
    )
    
    parser.add_argument(
        '--add-failures',
        type=int,
        default=0,
        metavar='N',
        help='Generate N synthetic failure traces by truncating successful ones'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for synthetic failure generation (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    ds = load_agentinstruct_dataset()
    
    # Process into traces
    traces = process_dataset(ds, max_content_length=args.max_content)
    
    # Add synthetic failures if requested
    if args.add_failures > 0:
        print(f"\nGenerating {args.add_failures} synthetic failure traces...")
        failure_traces = add_synthetic_failures(traces, num_failures=args.add_failures, seed=args.seed)
        traces.extend(failure_traces)
        print(f"Added {len(failure_traces)} synthetic failures")
    
    # Save
    save_traces(traces, args.output)
    
    # Print summary
    print_summary(traces)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
