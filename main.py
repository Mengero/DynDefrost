"""
Dynamic Defrost Model - Main Script

1-D Dynamic Defrost Model for simulating frost layer behavior during defrost cycles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_defrost_data, get_frost_properties, parse_case_filename
from model_init import initialize_model
from solver import DefrostSolver
from temperature_interpolation import interpolate_temperature, print_interpolation_info


def main():
    """Main entry point for the dynamic defrost model."""
    # ===== User Parameters =====
    data_file = "55min_60deg_83%_12C.txt"
    n_layers = 4
    dt = 0.1  # Time step [s]
    method = 'explicit'  # Solver method: 'explicit' or 'implicit'
    # ===========================
    
    # Load data
    loader, time_raw, temperature_raw = load_defrost_data(data_file)
    frost_props = get_frost_properties(data_file)
    
    # Extract surface type from filename (e.g., "60deg" from "55min_60deg_83%_12C.txt")
    case_params = parse_case_filename(data_file)
    contact_angle_str = None
    # Extract contact angle from filename parts
    name = Path(data_file).stem
    parts = name.split('_')
    if len(parts) > 1:
        contact_angle_str = parts[1]  # e.g., "60deg" or "160deg"
    
    # Get frost properties from experiment file - REQUIRED, no defaults
    if not isinstance(frost_props, dict):
        raise ValueError(
            f"ERROR: No matching experiment data found for {data_file}.\n"
            f"  Cannot proceed without experiment thickness and porosity data.\n"
            f"  Please ensure the case exists in defrost_sloughing_experiment_data.csv"
        )
    
    # Extract thickness and porosity from experiment data
    frost_thickness = frost_props['thickness']
    porosity = frost_props['porosity']
    
    # Validate that thickness is valid
    if frost_thickness is None or frost_thickness <= 0:
        raise ValueError(
            f"ERROR: Invalid thickness from experiment data: {frost_thickness}\n"
            f"  Thickness must be a positive number."
        )
    
    print(f"\nUsing experiment data:")
    print(f"  Thickness: {frost_thickness*1000:.2f} mm ({frost_thickness:.4e} m)")
    print(f"  Porosity: {porosity:.3f}")
    
    # Interpolate temperature data to match simulation time step
    # This smooths out large gradients and prevents temperature jumps
    time, temperature = interpolate_temperature(time_raw, temperature_raw, dt=dt, kind='linear')
    print_interpolation_info(time_raw, temperature_raw, time, temperature, dt)
    
    # Initialize model with surface type from filename
    model = initialize_model(
        n_layers=n_layers,
        frost_thickness=frost_thickness,
        porosity=porosity,
        T_initial=temperature[0],
        surface_type=contact_angle_str  # e.g., "60deg" or "160deg"
    )
    
    # Create solver
    solver = DefrostSolver(model, dt=dt, method=method)
    
    # Solve
    print(f"\nSolving defrost problem...")
    print(f"  Time steps: {len(time)}")
    print(f"  Duration: {time[-1]:.1f} s")
    print(f"  Method: {method}")
    print(f"  Time step: {dt} s")
    
    results = solver.solve(time, temperature, save_history=True)
    
    print(f"\nSolution completed!")
    print(f"  Final time: {results['time'][-1]:.1f} s")
    print(f"  Final temperature range: {np.min(results['temperature'][:, -1]):.1f}°C to {np.max(results['temperature'][:, -1]):.1f}°C")
    
    # Plot thickness vs time
    if results['h_total'] is not None and len(results['h_total']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(results['time'], results['h_total'] * 1000, 'b-', linewidth=2, label='Total frost thickness')
        
        # Add critical thickness if available
        if results['h_crit'] is not None and len(results['h_crit']) > 0:
            plt.plot(results['time'], results['h_crit'] * 1000, 'r--', linewidth=2, label='Critical detachment thickness')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Thickness [mm]', fontsize=12)
        plt.title('Frost Layer Thickness vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: No thickness data available for plotting")
    
    return time, temperature, frost_props, model, results


if __name__ == "__main__":
    time, temperature, frost_props, model, results = main()
 