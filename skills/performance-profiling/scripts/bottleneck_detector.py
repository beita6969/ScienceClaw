#!/usr/bin/env python3
"""
Bottleneck Detector - Identify performance bottlenecks and recommend optimizations.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional


def load_analysis_results(timing_path: str, scaling_path: Optional[str] = None,
                          memory_path: Optional[str] = None) -> Dict:
    """
    Load analysis results from JSON files.
    
    Args:
        timing_path: Path to timing analysis JSON
        scaling_path: Path to scaling analysis JSON (optional)
        memory_path: Path to memory profile JSON (optional)
    
    Returns:
        Combined analysis data
    """
    results = {}
    
    # Load timing data (required)
    try:
        with open(timing_path, 'r', encoding='utf-8') as f:
            results['timing'] = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Timing analysis file not found: {timing_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid timing JSON: {e}")
    
    # Load scaling data (optional)
    if scaling_path:
        try:
            with open(scaling_path, 'r', encoding='utf-8') as f:
                results['scaling'] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Scaling file not found: {scaling_path}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid scaling JSON: {e}", file=sys.stderr)
    
    # Load memory data (optional)
    if memory_path:
        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                results['memory'] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Memory file not found: {memory_path}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid memory JSON: {e}", file=sys.stderr)
    
    return results


def detect_timing_bottlenecks(timing_data: Dict, threshold: float = 50.0) -> List[Dict]:
    """
    Detect timing bottlenecks from timing analysis.
    
    Args:
        timing_data: Timing analysis results
        threshold: Percentage threshold for bottleneck (default: 50%)
    
    Returns:
        List of bottleneck dictionaries
    """
    bottlenecks = []
    
    if 'timing_data' not in timing_data:
        return bottlenecks
    
    phases = timing_data['timing_data'].get('phases', [])
    
    for phase in phases:
        percentage = phase.get('percentage', 0)
        if percentage > threshold:
            severity = 'high' if percentage > 70 else 'medium'
            bottlenecks.append({
                'type': 'timing',
                'phase': phase['name'],
                'severity': severity,
                'metric': 'percentage',
                'value': percentage,
                'threshold': threshold
            })
    
    return bottlenecks


def detect_scaling_bottlenecks(scaling_data: Dict, threshold: float = 0.70) -> List[Dict]:
    """
    Detect scaling bottlenecks from scaling analysis.
    
    Args:
        scaling_data: Scaling analysis results
        threshold: Efficiency threshold (default: 0.70)
    
    Returns:
        List of bottleneck dictionaries
    """
    bottlenecks = []
    
    if 'scaling_analysis' not in scaling_data:
        return bottlenecks
    
    analysis = scaling_data['scaling_analysis']
    avg_efficiency = analysis.get('average_efficiency', 1.0)
    
    if avg_efficiency < threshold:
        bottlenecks.append({
            'type': 'scaling',
            'phase': 'parallel_efficiency',
            'severity': 'high' if avg_efficiency < 0.5 else 'medium',
            'metric': 'efficiency',
            'value': avg_efficiency,
            'threshold': threshold
        })
    
    return bottlenecks


def detect_memory_bottlenecks(memory_data: Dict, threshold: float = 0.80) -> List[Dict]:
    """
    Detect memory bottlenecks from memory profile.
    
    Args:
        memory_data: Memory profile results
        threshold: Memory usage threshold (default: 0.80 = 80%)
    
    Returns:
        List of bottleneck dictionaries
    """
    bottlenecks = []
    
    if 'memory_profile' not in memory_data:
        return bottlenecks
    
    profile = memory_data['memory_profile']
    
    # Check if warnings exist (indicates high memory usage)
    if profile.get('warnings'):
        total_memory = profile.get('total_memory_gb', 0)
        bottlenecks.append({
            'type': 'memory',
            'phase': 'memory_usage',
            'severity': 'high',
            'metric': 'total_memory_gb',
            'value': total_memory,
            'threshold': threshold
        })
    
    return bottlenecks


def generate_recommendations(bottlenecks: List[Dict], timing_data: Optional[Dict] = None) -> List[Dict]:
    """
    Generate optimization recommendations based on bottlenecks.
    
    Args:
        bottlenecks: List of detected bottlenecks
        timing_data: Optional timing data for context
    
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    if not bottlenecks:
        recommendations.append({
            'priority': 'low',
            'category': 'general',
            'issue': 'No significant bottlenecks detected',
            'strategies': ['Performance appears balanced', 'Consider profiling at larger scale']
        })
        return recommendations
    
    # Process timing bottlenecks
    for bottleneck in bottlenecks:
        if bottleneck['type'] == 'timing':
            phase = bottleneck['phase'].lower()
            
            # Solver-related bottlenecks
            if any(keyword in phase for keyword in ['solve', 'linear', 'iteration', 'cg', 'gmres']):
                recommendations.append({
                    'priority': 'high',
                    'category': 'solver',
                    'issue': f"{bottleneck['phase']} dominates runtime ({bottleneck['value']:.1f}%)",
                    'strategies': [
                        'Use algebraic multigrid (AMG) preconditioner',
                        'Tighten solver tolerance if over-solving',
                        'Consider direct solver for small problems',
                        'Profile matrix assembly vs solve time'
                    ]
                })
            
            # Assembly bottlenecks
            elif any(keyword in phase for keyword in ['assembly', 'assemble', 'build']):
                recommendations.append({
                    'priority': 'high',
                    'category': 'assembly',
                    'issue': f"{bottleneck['phase']} dominates runtime ({bottleneck['value']:.1f}%)",
                    'strategies': [
                        'Cache element matrices if geometry is static',
                        'Use vectorized assembly routines',
                        'Consider matrix-free methods',
                        'Parallelize assembly with coloring'
                    ]
                })
            
            # I/O bottlenecks
            elif any(keyword in phase for keyword in ['io', 'write', 'output', 'save']):
                recommendations.append({
                    'priority': 'medium',
                    'category': 'io',
                    'issue': f"{bottleneck['phase']} dominates runtime ({bottleneck['value']:.1f}%)",
                    'strategies': [
                        'Reduce output frequency',
                        'Use parallel I/O (HDF5, MPI-IO)',
                        'Write to fast scratch storage',
                        'Compress output data'
                    ]
                })
            
            # Generic timing bottleneck
            else:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'general',
                    'issue': f"{bottleneck['phase']} dominates runtime ({bottleneck['value']:.1f}%)",
                    'strategies': [
                        'Profile this phase in detail',
                        'Look for algorithmic improvements',
                        'Consider parallelization opportunities'
                    ]
                })
        
        # Process scaling bottlenecks
        elif bottleneck['type'] == 'scaling':
            recommendations.append({
                'priority': 'high',
                'category': 'parallel',
                'issue': f"Poor parallel efficiency ({bottleneck['value']:.2f})",
                'strategies': [
                    'Investigate communication overhead',
                    'Check for load imbalance',
                    'Reduce synchronization points',
                    'Use asynchronous communication',
                    'Consider hybrid MPI+OpenMP'
                ]
            })
        
        # Process memory bottlenecks
        elif bottleneck['type'] == 'memory':
            recommendations.append({
                'priority': 'high',
                'category': 'memory',
                'issue': f"High memory usage ({bottleneck['value']:.2f} GB)",
                'strategies': [
                    'Reduce mesh resolution',
                    'Use iterative solver (lower memory than direct)',
                    'Enable out-of-core computation',
                    'Increase number of processors',
                    'Use single precision where appropriate'
                ]
            })
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Identify performance bottlenecks and recommend optimizations'
    )
    parser.add_argument('--timing', required=True, help='Path to timing analysis JSON')
    parser.add_argument('--scaling', help='Path to scaling analysis JSON (optional)')
    parser.add_argument('--memory', help='Path to memory profile JSON (optional)')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Load analysis results
        results = load_analysis_results(args.timing, args.scaling, args.memory)
        
        # Detect bottlenecks
        bottlenecks = []
        
        if 'timing' in results:
            bottlenecks.extend(detect_timing_bottlenecks(results['timing']))
        
        if 'scaling' in results:
            bottlenecks.extend(detect_scaling_bottlenecks(results['scaling']))
        
        if 'memory' in results:
            bottlenecks.extend(detect_memory_bottlenecks(results['memory']))
        
        # Generate recommendations
        recommendations = generate_recommendations(bottlenecks, results.get('timing'))
        
        # Format output
        if args.json:
            output = {
                'inputs': {
                    'timing_file': args.timing,
                    'scaling_file': args.scaling,
                    'memory_file': args.memory
                },
                'results': {
                    'bottlenecks': bottlenecks,
                    'recommendations': recommendations
                }
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Bottleneck Analysis")
            print(f"=" * 60)
            
            if bottlenecks:
                print(f"\nDetected Bottlenecks:")
                for bottleneck in bottlenecks:
                    print(f"  [{bottleneck['severity'].upper()}] {bottleneck['phase']}: "
                          f"{bottleneck['metric']} = {bottleneck['value']:.2f}")
            else:
                print("\nNo significant bottlenecks detected")
            
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"\n  [{rec['priority'].upper()}] {rec['category'].upper()}")
                print(f"  Issue: {rec['issue']}")
                print(f"  Strategies:")
                for strategy in rec['strategies']:
                    print(f"    - {strategy}")
    
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
