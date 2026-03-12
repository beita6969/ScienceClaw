#!/usr/bin/env python3
"""
Scaling Analyzer - Analyze strong and weak scaling from multi-run data.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional


def load_scaling_data(path: str) -> List[Dict]:
    """
    Load scaling data from JSON file.
    
    Args:
        path: Path to JSON file with scaling data
    
    Returns:
        List of run configurations
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'runs' not in data:
            raise ValueError("JSON must contain 'runs' array")
        
        runs = data['runs']
        
        if len(runs) < 2:
            raise ValueError("At least 2 runs required for scaling analysis")
        
        # Validate required fields
        for i, run in enumerate(runs):
            if 'processors' not in run:
                raise ValueError(f"Run {i} missing 'processors' field")
            if 'time' not in run:
                raise ValueError(f"Run {i} missing 'time' field")
            if run['time'] <= 0:
                raise ValueError(f"Run {i} has invalid time: {run['time']}")
            if run['processors'] <= 0:
                raise ValueError(f"Run {i} has invalid processor count: {run['processors']}")
        
        return runs
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaling data file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def compute_strong_scaling(runs: List[Dict]) -> Dict:
    """
    Compute strong scaling efficiency (fixed problem size, varying processors).
    
    Args:
        runs: List of run configurations
    
    Returns:
        Dictionary with scaling analysis results
    """
    # Sort by processor count
    sorted_runs = sorted(runs, key=lambda x: x['processors'])
    
    # Use smallest processor count as baseline
    baseline = sorted_runs[0]
    baseline_time = baseline['time']
    baseline_procs = baseline['processors']
    
    results = []
    for run in sorted_runs:
        procs = run['processors']
        time = run['time']
        
        # Speedup = T_baseline / T_N
        speedup = baseline_time / time
        
        # Efficiency = Speedup / (N / N_baseline) = T_baseline / (N * T_N / N_baseline)
        efficiency = speedup / (procs / baseline_procs)
        
        results.append({
            'processors': procs,
            'time': time,
            'speedup': speedup,
            'efficiency': efficiency
        })
    
    # Find efficiency threshold (first point where efficiency < 0.70)
    threshold_procs = None
    for result in results:
        if result['efficiency'] < 0.70:
            threshold_procs = result['processors']
            break
    
    # Calculate average efficiency
    avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
    
    return {
        'type': 'strong',
        'baseline': {
            'processors': baseline_procs,
            'time': baseline_time
        },
        'results': results,
        'efficiency_threshold_processors': threshold_procs,
        'average_efficiency': avg_efficiency
    }


def compute_weak_scaling(runs: List[Dict]) -> Dict:
    """
    Compute weak scaling efficiency (constant work per processor, varying processors).
    
    Args:
        runs: List of run configurations
    
    Returns:
        Dictionary with scaling analysis results
    """
    # Sort by processor count
    sorted_runs = sorted(runs, key=lambda x: x['processors'])
    
    # Use smallest processor count as baseline
    baseline = sorted_runs[0]
    baseline_time = baseline['time']
    baseline_procs = baseline['processors']
    
    results = []
    for run in sorted_runs:
        procs = run['processors']
        time = run['time']
        
        # For weak scaling, efficiency = T_baseline / T_N
        efficiency = baseline_time / time
        
        # Speedup is not meaningful for weak scaling (problem size changes)
        speedup = efficiency * (procs / baseline_procs)
        
        results.append({
            'processors': procs,
            'time': time,
            'speedup': speedup,
            'efficiency': efficiency
        })
    
    # Find efficiency threshold
    threshold_procs = None
    for result in results:
        if result['efficiency'] < 0.70:
            threshold_procs = result['processors']
            break
    
    # Calculate average efficiency
    avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
    
    return {
        'type': 'weak',
        'baseline': {
            'processors': baseline_procs,
            'time': baseline_time
        },
        'results': results,
        'efficiency_threshold_processors': threshold_procs,
        'average_efficiency': avg_efficiency
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze strong and weak scaling from multi-run data'
    )
    parser.add_argument('--data', required=True, help='Path to JSON file with scaling data')
    parser.add_argument('--type', required=True, choices=['strong', 'weak'],
                       help='Scaling type: strong or weak')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Load scaling data
        runs = load_scaling_data(args.data)
        
        # Compute scaling analysis
        if args.type == 'strong':
            analysis = compute_strong_scaling(runs)
        else:
            analysis = compute_weak_scaling(runs)
        
        # Format output
        if args.json:
            output = {
                'inputs': {
                    'data_file': args.data,
                    'scaling_type': args.type
                },
                'results': analysis
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"{args.type.capitalize()} Scaling Analysis")
            print(f"=" * 60)
            print(f"Baseline: {analysis['baseline']['processors']} processors, "
                  f"{analysis['baseline']['time']:.2f}s\n")
            
            print(f"{'Procs':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
            print("-" * 60)
            for result in analysis['results']:
                print(f"{result['processors']:<8} {result['time']:<12.2f} "
                      f"{result['speedup']:<12.2f} {result['efficiency']:<12.3f}")
            
            print(f"\nAverage efficiency: {analysis['average_efficiency']:.3f}")
            if analysis['efficiency_threshold_processors']:
                print(f"Efficiency drops below 0.70 at {analysis['efficiency_threshold_processors']} processors")
            else:
                print("Efficiency remains above 0.70 for all configurations")
    
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
