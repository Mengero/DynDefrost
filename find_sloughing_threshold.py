"""
Find Sloughing Threshold for Different Porosities

This script finds the threshold frost thickness that causes sloughing
for different porosity values, using the representative temperature curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from model_init import initialize_model
from solver import DefrostSolver
from stability_criterion import calculate_max_stable_dt, get_initial_frost_properties
from surface_retention import calculate_surface_retention
from temperature_interpolation import interpolate_temperature
from scipy.interpolate import interp1d


def load_representative_temperature(filepath='figure/representative_temperature_data.csv'):
    """Load representative temperature data."""
    filepath = Path(filepath)
    
    time_s = []
    temperature = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_s.append(float(row['Time_s']))
            temperature.append(float(row['Temperature_C']))
    
    return np.array(time_s), np.array(temperature)


def load_hydrophilic_porosities(data_file='exp_data/defrost_sloughing_experiment_data.csv'):
    """Get unique porosity values for hydrophilic surfaces."""
    filepath = Path(data_file)
    porosities = []
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Surface Type'].strip() == 'Hydrophilic':
                porosity = float(row['porosity (-)'])
                if porosity not in porosities:
                    porosities.append(porosity)
    
    return sorted(porosities)


def run_simulation(frost_thickness, porosity, time_raw, temperature_raw, T_ambient=12.0, 
                   method='explicit', dt_safety_factor=0.8):
    """
    Run a defrost simulation and check for sloughing.
    Follows the same pattern as main.py.
    
    Parameters:
    -----------
    frost_thickness : float
        Initial frost thickness [m]
    porosity : float
        Initial frost porosity
    time_raw : array
        Raw time array [s] (will be interpolated)
    temperature_raw : array
        Raw temperature array [°C] (will be interpolated and offset applied)
    T_ambient : float
        Ambient temperature [°C]
    method : str
        Solver method ('explicit' or 'implicit')
    dt_safety_factor : float
        Safety factor for time step (for explicit method)
        
    Returns:
    --------
    dict
        Results including sloughing status
    """
    # Apply temperature offset (same as main.py)
    temperature_offset = -1  # [°C]
    temperature_raw = temperature_raw + temperature_offset
    
    # Calculate surface retention thickness to determine number of layers (same as main.py)
    theta_receding = 60.0
    theta_advancing = 70.0
    retention_result = calculate_surface_retention(theta_receding, theta_advancing)
    surface_retention_thickness = retention_result['thickness']
    
    # This ensures each layer is approximately half the retention thickness
    n_layers = int(np.round(frost_thickness / 3e-5))
    n_layers = max(1, n_layers)
    
    # Calculate dt based on method (same as main.py)
    if method == 'explicit':
        # Get initial frost properties for stability calculation
        props = get_initial_frost_properties(porosity=porosity)
        dt_max = calculate_max_stable_dt(
            n_layers=n_layers,
            frost_thickness=frost_thickness,
            k_eff=props['k_eff'],
            rho_eff=props['rho_eff'],
            cp_eff=props['cp_eff'],
            safety_factor=0.4  # CFL condition uses 0.5
        )
        # Auto-select dt as a fraction of max stable dt
        dt = dt_max * dt_safety_factor
    else:
        # For implicit methods, use a reasonable default
        dt = 0.1  # Default for implicit

    # Check if we need interpolation or can use data directly
    # If dt is close to or larger than the original time step, use data directly
    if len(time_raw) > 1:
        original_dt = np.mean(np.diff(time_raw))
        if dt >= original_dt * 0.9:  # If dt is close to original, use original data
            time = time_raw.copy()
            temperature = temperature_raw.copy()
        else:
            # Interpolate temperature data to match simulation time step (same as main.py)
            time, temperature = interpolate_temperature(time_raw, temperature_raw, dt=dt, kind='linear')
    else:
        # Fallback: use raw data
        time = time_raw.copy()
        temperature = temperature_raw.copy()
    
    if len(time) == 0:
        return {'sloughing': False, 'sloughing_time': None, 'completely_melted': False}
    
    # Initialize model
    model = initialize_model(
        n_layers=n_layers,
        frost_thickness=frost_thickness,
        porosity=porosity,
        T_initial=temperature[0],
        surface_type='60deg'  # Hydrophilic
    )
    
    # Create solver
    h_conv = 4.0  # Natural convection heat transfer coefficient [W/(m²·K)]
    solver = DefrostSolver(model, dt=dt, method=method, h_conv=h_conv, T_ambient=T_ambient)
    
    # Solve
    results = solver.solve(time, temperature, save_history=True, history_save_interval=1.0)
    
    # Check for sloughing
    sloughing = False
    sloughing_time = None
    completely_melted = False
    
    if results['sloughing'] is not None and len(results['sloughing']) > 0:
        sloughing_indices = np.where(results['sloughing'] == True)[0]
        if len(sloughing_indices) > 0:
            sloughing = True
            sloughing_time = results['time'][sloughing_indices[0]]
    
    # Check if completely melted (no sloughing)
    # If simulation completed without sloughing, check if all ice melted
    if not sloughing and results['alpha_ice'] is not None:
        # Check final state - if all ice volume fractions are essentially zero
        final_alpha_ice = results['alpha_ice'][-1] if len(results['alpha_ice']) > 0 else None
        if final_alpha_ice is not None:
            total_ice = np.sum(final_alpha_ice)
            if total_ice < 1e-6:  # Essentially no ice left
                completely_melted = True
    
    # Also check the solver's internal flag
    if hasattr(solver, '_all_layers_water') and solver._all_layers_water:
        completely_melted = True
    
    return {
        'sloughing': sloughing,
        'sloughing_time': sloughing_time,
        'completely_melted': completely_melted,
        'results': results
    }


def find_threshold_thickness(porosity, time, temperature, T_ambient=12.0,
                             thickness_min=0.001, thickness_max=0.005,
                             n_cases=8):
    """
    Find the threshold frost thickness for sloughing by testing multiple thickness values.
    
    Parameters:
    -----------
    porosity : float
        Frost porosity
    time : array
        Time array [s]
    temperature : array
        Temperature array [°C]
    T_ambient : float
        Ambient temperature [°C]
    thickness_min : float
        Minimum thickness to test [m]
    thickness_max : float
        Maximum thickness to test [m]
    n_cases : int
        Number of thickness values to test
        
    Returns:
    --------
    float
        Threshold thickness [m], or None if not found
    """
    # Use explicit method (same as main.py)
    method = 'explicit'
    dt_safety_factor = 0.8
    
    # Create list of thickness values to test (from smallest to largest)
    thickness_values = np.linspace(thickness_min, thickness_max, n_cases)
    
    print(f"      Testing {n_cases} thickness values from {thickness_min*1000:.1f} to {thickness_max*1000:.1f} mm")
    
    # Run simulations sequentially, stop at first sloughing
    threshold = None
    for thickness in thickness_values:
        result = run_simulation(thickness, porosity, time, temperature, T_ambient, method, dt_safety_factor)
        
        status = "Sloughing!" if result['sloughing'] else ("Completely melted" if result['completely_melted'] else "No sloughing")
        print(f"        {thickness*1000:.3f} mm: {status}")
        
        if result['sloughing']:
            # Found sloughing - this is the threshold (smallest thickness tested that causes sloughing)
            threshold = thickness
            print(f"      Threshold found: {threshold*1000:.3f} mm (first thickness with sloughing)")
            break  # Stop testing once we find sloughing
    
    if threshold is None:
        print(f"      No sloughing detected in tested range")
    
    return threshold




def plot_threshold_curve(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                         output_dir='figure', figsize=(10, 8)):
    """
    Find and plot the threshold thickness curve for different porosities.
    """
    print("=" * 60)
    print("Finding Sloughing Threshold for Different Porosities")
    print("=" * 60)
    
    # Load representative temperature data
    print("\nLoading representative temperature data...")
    time, temperature = load_representative_temperature()
    print(f"  Time range: {time[0]:.1f} to {time[-1]:.1f} s")
    print(f"  Temperature range: {np.min(temperature):.1f} to {np.max(temperature):.1f} °C")
    
    # Get unique porosity values for hydrophilic surfaces
    print("\nLoading hydrophilic porosity values...")
    all_porosities = load_hydrophilic_porosities(data_file)
    print(f"  Found {len(all_porosities)} unique porosity values")
    
    # Select a subset for testing (every other one, or specific range)
    # Focus on the range where sloughing typically occurs (lower porosities)
    porosities = [p for p in all_porosities if p <= 0.95]  # Focus on lower porosities
    if len(porosities) > 8:
        # Take evenly spaced subset
        indices = np.linspace(0, len(porosities)-1, 8, dtype=int)
        porosities = [porosities[i] for i in indices]
    
    print(f"  Testing {len(porosities)} porosity values:")
    for p in porosities:
        print(f"    {p:.3f}")
    
    # Find threshold thickness for each porosity (in parallel)
    print("\nFinding threshold thickness for each porosity...")
    T_ambient = 12.0  # 12°C ambient condition
    thickness_min = 0.001  # 1 mm
    thickness_max = 0.005  # 5 mm
    n_cases = 8  # Number of thickness values to test per porosity
    
    threshold_thicknesses = []
    threshold_porosities = []
    
    # Find threshold thickness for each porosity (sequential)
    print(f"  Each porosity will test {n_cases} thickness values")
    for i, porosity in enumerate(porosities):
        print(f"\n  Porosity {i+1}/{len(porosities)}: {porosity:.3f}")
        
        threshold = find_threshold_thickness(
            porosity, time, temperature, T_ambient,
            thickness_min=thickness_min,
            thickness_max=thickness_max,
            n_cases=n_cases
        )
        
        if threshold is not None:
            threshold_thicknesses.append(threshold * 1000)  # Convert to mm
            threshold_porosities.append(porosity)
            print(f"    Threshold thickness: {threshold*1000:.3f} mm")
        else:
            print(f"    Warning: Could not find threshold in range {thickness_min*1000:.1f}-{thickness_max*1000:.1f} mm")
    
    if len(threshold_thicknesses) == 0:
        print("\nERROR: No threshold values found!")
        return None, None
    
    # Load experimental data for plotting
    print("\nLoading experimental data for comparison...")
    filepath = Path(data_file)
    exp_thickness = []
    exp_porosity = []
    exp_behavior = []
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Surface Type'].strip() == 'Hydrophilic':
                exp_thickness.append(float(row['t (mm)']))
                exp_porosity.append(float(row['porosity (-)']))
                exp_behavior.append(row['Behavior'].strip())
    
    # Separate by behavior
    slough_indices = [i for i, b in enumerate(exp_behavior) if b == 'Slough']
    other_indices = [i for i, b in enumerate(exp_behavior) if b != 'Slough']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot experimental data
    if slough_indices:
        slough_thickness = [exp_thickness[i] for i in slough_indices]
        slough_porosity = [exp_porosity[i] for i in slough_indices]
        ax.scatter(slough_porosity, slough_thickness, 
                  marker='x', s=150, linewidths=3, 
                  color='red', label='Slough (experimental)', zorder=3)
    
    if other_indices:
        other_thickness = [exp_thickness[i] for i in other_indices]
        other_porosity = [exp_porosity[i] for i in other_indices]
        ax.scatter(other_porosity, other_thickness, 
                  marker='o', s=100, linewidths=2, 
                  edgecolors='blue', facecolors='none', 
                  label='Other behaviors (experimental)', zorder=2)
    
    # Plot threshold curve
    if len(threshold_porosities) > 1:
        # Sort by porosity for plotting
        sort_idx = np.argsort(threshold_porosities)
        sorted_porosities = [threshold_porosities[i] for i in sort_idx]
        sorted_thicknesses = [threshold_thicknesses[i] for i in sort_idx]
        
        ax.plot(sorted_porosities, sorted_thicknesses, 
               'k--', linewidth=2.5, label='Sloughing threshold (simulated)', zorder=4)
    else:
        # Single point
        ax.scatter(threshold_porosities, threshold_thicknesses,
                  marker='s', s=200, color='black', 
                  label='Sloughing threshold (simulated)', zorder=4)
    
    # Customize plot
    ax.set_xlabel('Frost Porosity (-)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frost Thickness (mm)', fontsize=14, fontweight='bold')
    ax.set_title('Frost Thickness vs Porosity - Hydrophilic Surface\n(Sloughing Threshold)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Set axis limits
    all_porosities = exp_porosity + threshold_porosities
    all_thicknesses = exp_thickness + threshold_thicknesses
    x_min, x_max = np.min(all_porosities), np.max(all_porosities)
    y_min, y_max = np.min(all_thicknesses), np.max(all_thicknesses)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'thickness_vs_porosity_with_threshold.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Save threshold data
    threshold_file = output_path / 'sloughing_threshold_data.csv'
    with open(threshold_file, 'w') as f:
        f.write("Porosity,Threshold_Thickness_mm\n")
        for p, t in zip(threshold_porosities, threshold_thicknesses):
            f.write(f"{p:.6f},{t:.6f}\n")
    print(f"Threshold data saved to: {threshold_file}")
    
    return fig, ax


if __name__ == '__main__':
    # Protect main block for multiprocessing
    import sys
    try:
        fig, ax = plot_threshold_curve()
        print("\nDone!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
