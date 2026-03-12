#!/usr/bin/env python3
"""
Timing Analyzer - Extract and analyze timing information from simulation logs.
"""
import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Tuple


def parse_timing_log(log_path: str, pattern: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Parse simulation log file and extract timing entries.
    
    Args:
        log_path: Path to log file
        pattern: Custom regex pattern (optional)
    
    Returns:
        List of (phase_name, time_seconds) tuples
    """
    # Default patterns for common log formats
    default_patterns = [
        r'Phase:\s*([^,]+),\s*Time:\s*([\d.]+)s',
        r'(\w+(?:\s+\w+)*)\s+took\s+([\d.]+)\s*s',
        r'\[([^\]]+)\]\s*:\s*([\d.]+)\s*s',
        r'Time\s+for\s+([^:]+):\s*([\d.]+)',
    ]
    
    patterns_to_try = [pattern] if pattern else default_patterns
    
    entries = []
    malformed_count = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                matched = False
                for pat in patterns_to_try:
                    match = re.search(pat, line)
                    if match:
                        try:
                            phase = match.group(1).strip()
                            time_val = float(match.group(2))
                            if time_val >= 0:
                                entries.append((phase, time_val))
                                matched = True
                                break
                        except (ValueError, IndexError):
                            malformed_count += 1
                
                if not matched and any(keyword in line.lower() for keyword in ['time', 'phase', 'took']):
                    malformed_count += 1
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {log_path}")
    except Exception as e:
        raise ValueError(f"Error reading log file: {e}")
    
    if malformed_count > 0:
        print(f"Warning: Skipped {malformed_count} malformed timing entries", file=sys.stderr)
    
    return entries


def aggregate_timings(entries: List[Tuple[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate timing entries by phase.
    
    Args:
        entries: List of (phase_name, time_seconds) tuples
    
    Returns:
        Dictionary mapping phase names to aggregated statistics
    """
    if not entries:
        return {}
    
    phase_times: Dict[str, List[float]] = {}
    for phase, time_val in entries:
        if phase not in phase_times:
            phase_times[phase] = []
        phase_times[phase].append(time_val)
    
    total_time = sum(time_val for _, time_val in entries)
    
    aggregated = {}
    for phase, times in phase_times.items():
        aggregated[phase] = {
            'total_time': sum(times),
            'count': len(times),
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'percentage': (sum(times) / total_time * 100) if total_time > 0 else 0.0
        }
    
    return aggregated


def identify_slowest_phases(aggregated: Dict[str, Dict[str, float]], top_n: int = 5) -> List[str]:
    """
    Identify the slowest computational phases.
    
    Args:
        aggregated: Aggregated timing data
        top_n: Number of slowest phases to return
    
    Returns:
        List of phase names sorted by total time (descending)
    """
    if not aggregated:
        return []
    
    sorted_phases = sorted(aggregated.items(), key=lambda x: x[1]['total_time'], reverse=True)
    return [phase for phase, _ in sorted_phases[:top_n]]


def main():
    parser = argparse.ArgumentParser(
        description='Extract and analyze timing information from simulation logs'
    )
    parser.add_argument('--log', required=True, help='Path to simulation log file')
    parser.add_argument('--pattern', help='Custom regex pattern for timing entries')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Parse log file
        entries = parse_timing_log(args.log, args.pattern)
        
        # Aggregate timings
        aggregated = aggregate_timings(entries)
        
        # Identify slowest phases
        slowest = identify_slowest_phases(aggregated)
        
        # Calculate total time
        total_time = sum(data['total_time'] for data in aggregated.values())
        
        # Format output
        if args.json:
            output = {
                'inputs': {
                    'log_file': args.log,
                    'pattern': args.pattern or 'default'
                },
                'results': {
                    'phases': [
                        {
                            'name': phase,
                            **aggregated[phase]
                        }
                        for phase in aggregated
                    ],
                    'total_time': total_time,
                    'slowest_phase': slowest[0] if slowest else None
                }
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Timing Analysis Results")
            print(f"=" * 60)
            print(f"Total time: {total_time:.2f}s\n")
            
            if aggregated:
                print(f"{'Phase':<30} {'Total (s)':<12} {'Count':<8} {'Mean (s)':<12} {'%':<8}")
                print("-" * 60)
                for phase in slowest:
                    data = aggregated[phase]
                    print(f"{phase:<30} {data['total_time']:<12.2f} {data['count']:<8} "
                          f"{data['mean_time']:<12.2f} {data['percentage']:<8.1f}")
            else:
                print("No timing data found in log file")
    
    except (FileNotFoundError, ValueError) as e:
        if args.json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({'error': f'Unexpected error: {e}'}))
        else:
            print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
