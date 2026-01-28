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
                   method='explicit', dt_safety_factor=0.8, dt_fixed=None):
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
        Safety factor for time step (for explicit method, only used if dt_fixed is None)
    dt_fixed : float, optional
        Fixed time step to use [s]. If provided, this dt is used directly instead of calculating.
        This ensures consistency across simulations. Default: None (calculate dt)
        
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
    # Use consistent layer density to avoid numerical inconsistencies
    # Calculate based on a reference thickness to maintain consistent resolution
    layer_density = 20000  # layers per meter (50 μm per layer)
    n_layers = int(np.round(frost_thickness * layer_density))
    n_layers = max(10, n_layers)  # Minimum 10 layers for stability
    
    # Calculate dt based on method (same as main.py)
    # If dt_fixed is provided, use it directly for consistency across simulations
    if dt_fixed is not None:
        dt = dt_fixed
    elif method == 'explicit':
        # Get initial frost properties for stability calculation
        props = get_initial_frost_properties(porosity=porosity)
        dt_max = calculate_max_stable_dt(
            n_layers=n_layers,
            frost_thickness=frost_thickness,
            k_eff=props['k_eff'],
            rho_eff=props['rho_eff'],
            cp_eff=props['cp_eff'],
            safety_factor=0.5  # CFL condition uses 0.5
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
    h_conv = 1.5  # Natural convection heat transfer coefficient [W/(m²·K)]
    solver = DefrostSolver(model, dt=dt, method=method, h_conv=h_conv, T_ambient=T_ambient)
    
    # Solve
    results = solver.solve(time, temperature, save_history=True, history_save_interval=1.0)
    
    # Check for sloughing
    sloughing = False
    sloughing_time = None
    completely_melted = False
    
    # First check the solver's internal sloughing info (most reliable when simulation stops early)
    # This has the h_crit value from BEFORE the state was modified
    if hasattr(solver, '_latest_sloughing_info') and solver._latest_sloughing_info is not None:
        if solver._latest_sloughing_info.get('sloughing', False):
            sloughing = True
            # Get time from results if available, otherwise use last time
            if results['time'] is not None and len(results['time']) > 0:
                sloughing_time = results['time'][-1]
            # Store h_crit, h_total, rho_eff from solver's internal info (BEFORE state modification)
            # This is more reliable than the history which might have invalid values after sloughing
            if 'h_crit_at_sloughing' not in results:
                results['h_crit_at_sloughing'] = solver._latest_sloughing_info.get('h_crit')
                results['h_total_at_sloughing'] = solver._latest_sloughing_info.get('h_total')
                results['rho_eff_at_sloughing'] = solver._latest_sloughing_info.get('rho_eff')
    
    # Also check results history (in case sloughing was saved to history)
    if not sloughing and results['sloughing'] is not None and len(results['sloughing']) > 0:
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
                             start_thickness=0.0019, increment=0.0002, max_thickness=0.01):
    """
    Find the threshold frost thickness for sloughing by running simulations
    with increasing initial thickness until sloughing occurs.
    
    For a given porosity:
    1. Start with start_thickness (default: 1.9 mm for hydrophilic surface)
    2. Run simulation
    3. If sloughing occurs, check validity criteria:
       - h_total > 1 mm (if <= 1 mm, too thin - invalid)
       - Thickness increasing rate >= 0.2 mm/s (if < 0.2 mm/s, too slow - invalid)
    4. If sloughing is invalid or no sloughing, increase thickness by increment and try again
    5. Repeat until valid sloughing occurs or max_thickness is reached
    6. When valid sloughing occurs, record the thickness at that moment
    7. That thickness becomes the threshold for this porosity
    
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
    start_thickness : float
        Initial frost thickness to start testing [m]
        Default: 0.0019 m (1.9 mm for hydrophilic surface)
    increment : float
        Thickness increment to add if no sloughing [m]
        Default: 0.0002 m (0.2 mm)
    max_thickness : float
        Maximum thickness to test [m]
        Default: 0.01 m (10 mm)
        
    Returns:
    --------
    float
        Threshold thickness [m] at moment of sloughing, or None if no sloughing occurred
    """
    # Use explicit method (same as main.py)
    method = 'explicit'
    dt_safety_factor = 0.9  # Safety factor for calculating fixed dt (can be higher since we use fixed dt)
    
    # Calculate fixed dt once for maximum thickness to ensure stability for all cases
    # This ensures all simulations use the same dt, improving consistency and speed
    print(f"      Calculating fixed time step for stability...")
    layer_density = 20000  # layers per meter (50 μm per layer)
    n_layers_max = int(np.round(max_thickness * layer_density))
    n_layers_max = max(10, n_layers_max)
    
    props = get_initial_frost_properties(porosity=porosity)
    dt_max = calculate_max_stable_dt(
        n_layers=n_layers_max,
        frost_thickness=max_thickness,
        k_eff=props['k_eff'],
        rho_eff=props['rho_eff'],
        cp_eff=props['cp_eff'],
        safety_factor=0.4  # CFL condition uses 0.5
    )
    dt_fixed = dt_max * dt_safety_factor
    print(f"      Fixed time step: dt = {dt_fixed:.6f} s (based on max thickness {max_thickness*1000:.1f} mm)")
    
    current_thickness = start_thickness
    iteration = 0
    
    print(f"      Starting from {start_thickness*1000:.2f} mm, increment: {increment*1000:.2f} mm")
    
    while current_thickness <= max_thickness:
        iteration += 1
        print(f"      Iteration {iteration}: Testing thickness = {current_thickness*1000:.2f} mm")
        
        # Run simulation with current thickness using fixed dt for consistency
        result = run_simulation(current_thickness, porosity, time, temperature, T_ambient, method, 
                               dt_safety_factor=dt_safety_factor, dt_fixed=dt_fixed)
        
        if result['sloughing']:
            # Sloughing occurred - check if it's valid (h_crit >= 0.5 mm)
            # Try to get sloughing index from history first
            sloughing_indices = np.where(result['results']['sloughing'] == True)[0] if result['results']['sloughing'] is not None else []
            
            # If not in history, use last index (simulation stopped early)
            if len(sloughing_indices) > 0:
                sloughing_idx = sloughing_indices[0]
            else:
                # Simulation stopped early, use last saved index
                sloughing_idx = -1
            
            # Check rho_eff at the moment of sloughing
            # IMPORTANT: Get rho_eff from solver's internal info (before state was modified)
            # Also calculate smoothed/averaged value from history for stability
            rho_eff_at_sloughing = None
            if 'rho_eff_at_sloughing' in result['results']:
                rho_eff_at_sloughing = result['results']['rho_eff_at_sloughing']
            
            # Calculate smoothed density from history for additional stability check
            # Use average of last few points to reduce noise
            rho_eff_smoothed = None
            if result['results']['h_total'] is not None and len(result['results']['h_total']) > 0:
                # Try to get density history if available (might need to calculate from model state)
                # For now, use the single value but could be extended to use history
                pass
            
            # Also get h_crit for logging purposes
            h_crit_at_sloughing = None
            if 'h_crit_at_sloughing' in result['results']:
                h_crit_at_sloughing = result['results']['h_crit_at_sloughing']
            elif result['results']['h_crit'] is not None and len(result['results']['h_crit']) > 0:
                if len(result['results']['h_crit']) > 1:
                    h_crit_at_sloughing = result['results']['h_crit'][-2]
                else:
                    h_crit_at_sloughing = result['results']['h_crit'][-1]
            
            # Check criteria for valid sloughing:
            # 1. Thickness > 1 mm
            # 2. Thickness increasing rate >= 0.2 mm/s
            
            # Get h_total at the moment of sloughing
            # IMPORTANT: Get h_total from solver's internal info (before state was modified)
            threshold = None
            if 'h_total_at_sloughing' in result['results']:
                threshold = result['results']['h_total_at_sloughing']
            elif result['results']['h_total'] is not None and len(result['results']['h_total']) > 0:
                # Fallback: use second-to-last value (before sloughing modified state)
                # The last value might be invalid after layers were set to zero
                if len(result['results']['h_total']) > 1:
                    threshold = result['results']['h_total'][-2]  # Second-to-last
                else:
                    threshold = result['results']['h_total'][-1]
            
            if threshold is None:
                print(f"      Sloughing detected but h_total data not available")
                return None
            
            
            
            # All criteria met - valid sloughing
            sloughing_time = result['sloughing_time'] if result['sloughing_time'] is not None else (result['results']['time'][-1] if result['results']['time'] is not None and len(result['results']['time']) > 0 else None)
            h_crit_str = f", h_crit = {h_crit_at_sloughing*1000:.2f} mm" if h_crit_at_sloughing is not None else ""
            rho_eff_str = f", rho_eff = {rho_eff_at_sloughing:.2f} kg/m³" if rho_eff_at_sloughing is not None else ""
            time_str = f"t = {sloughing_time:.1f} s" if sloughing_time is not None else "final time"
            print(f"      ✓ Valid sloughing detected at {time_str}{h_crit_str}{rho_eff_str}")
            print(f"      ✓ Threshold thickness: {threshold*1000:.3f} mm (thickness at sloughing)")
            return threshold
        elif result['completely_melted']:
            print(f"      → Frost completely melted (no sloughing), increasing thickness...")
            # Increase thickness and try again
            current_thickness += increment
        else:
            print(f"      → No sloughing detected, increasing thickness...")
            # Increase thickness and try again
            current_thickness += increment
    
    # If we reach here, we've exceeded max_thickness without finding sloughing
    print(f"      ✗ No sloughing detected up to {max_thickness*1000:.1f} mm")
    return None




def plot_threshold_curve(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                         output_dir='figure', figsize=(10, 8)):
    """
    Find and plot the threshold thickness curve for different porosities.
    """
    print("=" * 60)
    print("Finding Sloughing Threshold for Different Porosities")
    print("=" * 60)
    
    # Load temperature data from specific file: 120min_60deg_55%_22C.txt
    print("\nLoading temperature data from '120min_60deg_55%_22C.txt'...")
    from data_loader import load_defrost_data
    loader, time, temperature = load_defrost_data("120min_60deg_55%_22C.txt")
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
    
    # Find threshold thickness for each porosity
    print("\nFinding threshold thickness for each porosity...")
    T_ambient = 12.0  # 12°C ambient condition
    start_thickness = 0.0024  # 1.9 mm starting thickness (for hydrophilic surface)
    increment = 0.0002  # 0.2 mm increment
    max_thickness = 0.01  # 10 mm maximum thickness
    
    threshold_thicknesses = []
    threshold_porosities = []
    
    # Find threshold thickness for each porosity (sequential)
    print(f"  For each porosity:")
    print(f"    - Start with {start_thickness*1000:.1f} mm initial thickness")
    print(f"    - If no sloughing, increase by {increment*1000:.1f} mm and try again")
    print(f"    - Continue until sloughing occurs or reach {max_thickness*1000:.1f} mm")
    print(f"    - Record thickness at moment of sloughing")
    for i, porosity in enumerate(porosities):
        print(f"\n  Porosity {i+1}/{len(porosities)}: {porosity:.3f}")
        
        threshold = find_threshold_thickness(
            porosity, time, temperature, T_ambient,
            start_thickness=start_thickness,
            increment=increment,
            max_thickness=max_thickness
        )
        
        if threshold is not None:
            threshold_thicknesses.append(threshold * 1000)  # Convert to mm
            threshold_porosities.append(porosity)
            print(f"    Threshold thickness: {threshold*1000:.3f} mm")
        else:
            print(f"    Warning: No sloughing detected for this porosity")
    
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
