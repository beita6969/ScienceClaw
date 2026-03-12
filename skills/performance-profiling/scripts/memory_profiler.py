#!/usr/bin/env python3
"""
Memory Profiler - Estimate memory requirements from simulation parameters.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional


def load_parameters(path: str) -> Dict:
    """
    Load simulation parameters from JSON file.
    
    Args:
        path: Path to JSON file with parameters
    
    Returns:
        Parameter dictionary
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # Validate required fields
        missing = []
        if 'mesh' not in params:
            missing.append('mesh')
        if 'fields' not in params:
            missing.append('fields')
        
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        return params
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def estimate_field_memory(mesh: Dict, fields: Dict) -> float:
    """
    Estimate memory for field variables.

    Args:
        mesh: Mesh parameters (nx, ny, nz)
        fields: Field definitions

    Returns:
        Memory in GB
    """
    # Calculate total mesh points
    nx = mesh.get('nx', 1)
    ny = mesh.get('ny', 1)
    nz = mesh.get('nz', 1)
    # Validate mesh dimensions
    for name, val in [('nx', nx), ('ny', ny), ('nz', nz)]:
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Mesh dimension '{name}' must be a positive integer, got {val}")
    mesh_points = nx * ny * nz

    # Calculate memory for all fields
    total_bytes = 0
    for field_name, field_spec in fields.items():
        components = field_spec.get('components', 1)
        bytes_per_value = field_spec.get('bytes_per_value', 8)  # default: double precision
        if not isinstance(components, int) or components <= 0:
            raise ValueError(f"Field '{field_name}' components must be a positive integer, got {components}")
        if not isinstance(bytes_per_value, (int, float)) or bytes_per_value <= 0:
            raise ValueError(f"Field '{field_name}' bytes_per_value must be positive, got {bytes_per_value}")
        total_bytes += mesh_points * components * bytes_per_value
    
    # Convert to GB
    return total_bytes / (1024 ** 3)


def estimate_solver_memory(mesh: Dict, solver: Dict) -> float:
    """
    Estimate memory for solver workspace.
    
    Args:
        mesh: Mesh parameters
        solver: Solver configuration
    
    Returns:
        Memory in GB
    """
    # Calculate total mesh points
    nx = mesh.get('nx', 1)
    ny = mesh.get('ny', 1)
    nz = mesh.get('nz', 1)
    mesh_points = nx * ny * nz
    
    # Get workspace multiplier based on solver type
    solver_type = solver.get('type', 'iterative')
    workspace_multiplier = solver.get('workspace_multiplier', 5)
    
    # Estimate workspace (typically several vectors of size mesh_points)
    bytes_per_value = 8  # double precision
    workspace_bytes = mesh_points * workspace_multiplier * bytes_per_value
    
    # Convert to GB
    return workspace_bytes / (1024 ** 3)


def compute_total_memory(params: Dict, available_gb: Optional[float] = None) -> Dict:
    """
    Compute total memory requirements.
    
    Args:
        params: Simulation parameters
        available_gb: Available system memory (optional)
    
    Returns:
        Memory profile dictionary
    """
    mesh = params['mesh']
    fields = params['fields']
    solver = params.get('solver', {'type': 'iterative', 'workspace_multiplier': 5})
    processors = params.get('processors', 1)
    if not isinstance(processors, int) or processors <= 0:
        raise ValueError(f"processors must be a positive integer, got {processors}")
    
    # Calculate mesh points
    nx = mesh.get('nx', 1)
    ny = mesh.get('ny', 1)
    nz = mesh.get('nz', 1)
    mesh_points = nx * ny * nz
    
    # Estimate memory components
    field_memory_gb = estimate_field_memory(mesh, fields)
    solver_workspace_gb = estimate_solver_memory(mesh, solver)
    total_memory_gb = field_memory_gb + solver_workspace_gb
    per_process_gb = total_memory_gb / processors
    
    # Generate warnings
    warnings = []
    if available_gb is not None:
        if total_memory_gb > available_gb:
            warnings.append(f"Total memory ({total_memory_gb:.2f} GB) exceeds available memory ({available_gb:.2f} GB)")
        elif total_memory_gb > 0.8 * available_gb:
            warnings.append(f"Memory usage ({total_memory_gb:.2f} GB) is high (>80% of available {available_gb:.2f} GB)")
    
    return {
        'mesh_points': mesh_points,
        'field_memory_gb': field_memory_gb,
        'solver_workspace_gb': solver_workspace_gb,
        'total_memory_gb': total_memory_gb,
        'per_process_gb': per_process_gb,
        'warnings': warnings
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate memory requirements from simulation parameters'
    )
    parser.add_argument('--params', required=True, help='Path to JSON file with simulation parameters')
    parser.add_argument('--available-gb', type=float, help='Available system memory in GB')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        params = load_parameters(args.params)
        
        # Compute memory profile
        profile = compute_total_memory(params, args.available_gb)
        
        # Format output
        if args.json:
            output = {
                'inputs': {
                    'params_file': args.params,
                    'available_gb': args.available_gb
                },
                'results': profile
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Memory Profile")
            print(f"=" * 60)
            print(f"Mesh points: {profile['mesh_points']:,}")
            print(f"Field memory: {profile['field_memory_gb']:.3f} GB")
            print(f"Solver workspace: {profile['solver_workspace_gb']:.3f} GB")
            print(f"Total memory: {profile['total_memory_gb']:.3f} GB")
            print(f"Per-process memory: {profile['per_process_gb']:.3f} GB")
            
            if profile['warnings']:
                print(f"\nWarnings:")
                for warning in profile['warnings']:
                    print(f"  - {warning}")
    
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
        sys.exit(1)


if __name__ == '__main__':
    main()
