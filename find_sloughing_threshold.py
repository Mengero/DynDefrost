"""
Find Sloughing Threshold for Different Porosities

This script finds the critical frost thickness for sloughing per porosity:
- For porosities that have at least one "Slough" case: use the lowest-sloughing-thickness
  case's temperature profile, then reduce thickness by 0.1 mm until "all layers have
  become water"; that thickness is the critical sloughing thickness.
- For porosities with only "no sloughing" (e.g. Drain) cases: use one case's temperature
  profile and increase thickness by 0.1 mm until sloughing occurs; that thickness is
  the critical sloughing thickness.
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
from data_loader import load_defrost_data
from scipy.interpolate import interp1d


def _data_filename_from_row(row, surface_type='Hydrophilic'):
    """Build temperature data filename from experiment CSV row (e.g. 120min_60deg_55%_22C.txt)."""
    # Hydrophilic -> 60deg, Superhydrophobic -> 160deg
    deg = '60deg' if surface_type.strip() == 'Hydrophilic' else '160deg'
    t_min = int(float(row['frosting time (min)']))
    rh_pct = int(round(float(row['RH']) * 100))
    T_air = int(float(row['Air Dry Bulb [C]']))
    return f"{t_min}min_{deg}_{rh_pct}%_{T_air}C.txt"


def load_experiment_data_grouped_by_porosity(
    data_file='exp_data/defrost_sloughing_experiment_data.csv',
    surface_type='Hydrophilic',
    exp_data_dir='exp_data',
):
    """
    Load experiment CSV and group rows by porosity. For each porosity, attach only
    rows that have a matching temperature data file in exp_data_dir.
    
    Returns
    -------
    dict
        porosity -> list of dicts: thickness (m), thickness_mm, behavior, row (for filename), data_file
    """
    filepath = Path(data_file)
    exp_dir = Path(exp_data_dir)
    groups = {}  # porosity -> list of case dicts
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Surface Type', '').strip() != surface_type:
                continue
            try:
                porosity = float(row['porosity (-)'])
                thickness_m = float(row['t (m)'])
                thickness_mm = float(row['t (mm)'])
                behavior = row.get('Behavior', '').strip()
            except (ValueError, KeyError):
                continue
            data_file = _data_filename_from_row(row, surface_type)
            data_path = exp_dir / data_file
            if not data_path.exists():
                continue
            case = {
                'porosity': porosity,
                'thickness': thickness_m,
                'thickness_mm': thickness_mm,
                'behavior': behavior,
                'row': row,
                'data_file': data_file,
            }
            if porosity not in groups:
                groups[porosity] = []
            groups[porosity].append(case)
    
    return groups


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


def _get_fixed_dt(porosity, max_thickness=0.01, layer_density=20000, dt_safety_factor=0.9):
    """Compute a fixed time step for stability (used across thickness sweeps)."""
    n_layers_max = max(10, int(np.round(max_thickness * layer_density)))
    props = get_initial_frost_properties(porosity=porosity)
    dt_max = calculate_max_stable_dt(
        n_layers=n_layers_max,
        frost_thickness=max_thickness,
        k_eff=props['k_eff'],
        rho_eff=props['rho_eff'],
        cp_eff=props['cp_eff'],
        safety_factor=0.4
    )
    return dt_max * dt_safety_factor


def find_critical_thickness_sloughing_case(
    porosity, time, temperature, start_thickness_m,
    T_ambient=12.0, decrement_m=0.0001, min_thickness_m=0.0005,
    method='explicit', dt_safety_factor=0.9,
):
    """
    For a porosity that has at least one Slough case: start at the lowest sloughing
    case's thickness, use that case's temperature profile, and reduce thickness by
    0.1 mm until "all layers have become water". That thickness is the critical
    sloughing thickness (below it, no sloughing).

    Parameters
    ----------
    porosity : float
    time : array [s]
    temperature : array [°C]
    start_thickness_m : float
        Starting thickness [m] (lowest sloughing case thickness).
    T_ambient : float [°C]
    decrement_m : float
        Thickness step [m], default 0.0001 (0.1 mm).
    min_thickness_m : float
        Stop if thickness goes below this [m].
    method, dt_safety_factor : passed to run_simulation.

    Returns
    -------
    float or None
        Critical thickness [m] (first thickness at which all layers become water), or None.
    """
    max_thickness = start_thickness_m + 2 * decrement_m  # for dt calc
    dt_fixed = _get_fixed_dt(porosity, max_thickness=max_thickness, dt_safety_factor=dt_safety_factor)
    current_thickness = start_thickness_m
    iteration = 0
    print(f"      Starting from {start_thickness_m*1000:.2f} mm (lowest sloughing case), decrement: {decrement_m*1000:.2f} mm")
    while current_thickness >= min_thickness_m:
        iteration += 1
        print(f"      Iteration {iteration}: thickness = {current_thickness*1000:.2f} mm")
        result = run_simulation(
            current_thickness, porosity, time, temperature, T_ambient,
            method=method, dt_safety_factor=dt_safety_factor, dt_fixed=dt_fixed
        )
        if result['completely_melted']:
            # All layers have become water (no sloughing)
            print(f"      ✓ All layers became water at {current_thickness*1000:.2f} mm → critical sloughing thickness")
            return current_thickness
        if result['sloughing']:
            print(f"      → Sloughing at this thickness, reducing...")
        else:
            print(f"      → No sloughing, reducing...")
        current_thickness -= decrement_m
    print(f"      ✗ Reached min thickness {min_thickness_m*1000:.2f} mm without all-water")
    return None


def find_critical_thickness_nosloughing_case(
    porosity, time, temperature, start_thickness_m,
    T_ambient=12.0, increment_m=0.0001, max_thickness_m=0.01,
    method='explicit', dt_safety_factor=0.9,
):
    """
    For a porosity that has only no-sloughing (e.g. Drain) cases: use one case's
    temperature profile, start at start_thickness_m, and increase thickness by
    0.1 mm until sloughing occurs. That thickness is the critical sloughing thickness.

    Parameters
    ----------
    porosity : float
    time : array [s]
    temperature : array [°C]
    start_thickness_m : float
        Starting thickness [m] (e.g. max thickness among no-sloughing cases).
    T_ambient : float [°C]
    increment_m : float
        Thickness step [m], default 0.0001 (0.1 mm).
    max_thickness_m : float
        Stop if thickness exceeds this [m].
    method, dt_safety_factor : passed to run_simulation.

    Returns
    -------
    float or None
        Critical thickness [m] (first thickness at which sloughing occurs), or None.
    """
    dt_fixed = _get_fixed_dt(porosity, max_thickness=max_thickness_m, dt_safety_factor=dt_safety_factor)
    current_thickness = start_thickness_m
    iteration = 0
    print(f"      Starting from {start_thickness_m*1000:.2f} mm (no-sloughing case), increment: {increment_m*1000:.2f} mm")
    while current_thickness <= max_thickness_m:
        iteration += 1
        print(f"      Iteration {iteration}: thickness = {current_thickness*1000:.2f} mm")
        result = run_simulation(
            current_thickness, porosity, time, temperature, T_ambient,
            method=method, dt_safety_factor=dt_safety_factor, dt_fixed=dt_fixed
        )
        if result['sloughing']:
            print(f"      ✓ Sloughing at {current_thickness*1000:.2f} mm → critical sloughing thickness")
            return current_thickness
        print(f"      → No sloughing (all water or incomplete), increasing...")
        current_thickness += increment_m
    print(f"      ✗ No sloughing up to {max_thickness_m*1000:.1f} mm")
    return None


def find_threshold_thickness(porosity, time, temperature, T_ambient=12.0,
                             start_thickness=0.0019, increment=0.0002, max_thickness=0.01):
    """
    Legacy: Find the threshold frost thickness by increasing thickness until sloughing.
    Prefer find_critical_thickness_sloughing_case / find_critical_thickness_nosloughing_case
    with experiment-grouped data.
    """
    method = 'explicit'
    dt_safety_factor = 0.9
    layer_density = 20000
    n_layers_max = max(10, int(np.round(max_thickness * layer_density)))
    props = get_initial_frost_properties(porosity=porosity)
    dt_max = calculate_max_stable_dt(
        n_layers=n_layers_max, frost_thickness=max_thickness,
        k_eff=props['k_eff'], rho_eff=props['rho_eff'], cp_eff=props['cp_eff'],
        safety_factor=0.4
    )
    dt_fixed = dt_max * dt_safety_factor
    current_thickness = start_thickness
    iteration = 0
    print(f"      Starting from {start_thickness*1000:.2f} mm, increment: {increment*1000:.2f} mm")
    while current_thickness <= max_thickness:
        iteration += 1
        print(f"      Iteration {iteration}: Testing thickness = {current_thickness*1000:.2f} mm")
        result = run_simulation(current_thickness, porosity, time, temperature, T_ambient,
                               method=method, dt_safety_factor=dt_safety_factor, dt_fixed=dt_fixed)
        if result['sloughing']:
            threshold = None
            if 'h_total_at_sloughing' in result.get('results', {}):
                threshold = result['results']['h_total_at_sloughing']
            elif result.get('results', {}).get('h_total') is not None and len(result['results']['h_total']) > 1:
                threshold = result['results']['h_total'][-2]
            elif result.get('results', {}).get('h_total') is not None and len(result['results']['h_total']) > 0:
                threshold = result['results']['h_total'][-1]
            if threshold is None:
                threshold = current_thickness
            print(f"      ✓ Sloughing at thickness {threshold*1000:.3f} mm")
            return threshold
        current_thickness += increment
    print(f"      ✗ No sloughing up to {max_thickness*1000:.1f} mm")
    return None




def plot_threshold_curve(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                         exp_data_dir='exp_data',
                         output_dir='figure', figsize=(10, 8),
                         thickness_step_mm=0.1,
                         surface_type='Hydrophilic'):
    """
    Find and plot the critical sloughing thickness for each porosity using experiment cases.

    For each frost porosity (Hydrophilic only, with a matching temperature data file):
    - If there is at least one "Slough" case: use the case with the *lowest* sloughing
      thickness and its temperature profile; reduce thickness by thickness_step_mm (0.1 mm)
      until "all layers have become water". That thickness is the critical sloughing thickness.
    - If all cases for that porosity are no-sloughing (e.g. Drain): use one case's
      temperature profile and increase thickness by thickness_step_mm until sloughing
      occurs. That thickness is the critical sloughing thickness.
    """
    print("=" * 60)
    print("Finding Critical Sloughing Thickness for Different Porosities")
    print("=" * 60)
    
    # Load experiment data grouped by porosity (only rows with existing temperature data file)
    print("\nLoading experiment data grouped by porosity (Hydrophilic, with data files)...")
    groups = load_experiment_data_grouped_by_porosity(
        data_file=data_file,
        surface_type=surface_type,
        exp_data_dir=exp_data_dir,
    )
    porosities = sorted(groups.keys())
    if not porosities:
        print("  No porosity groups with matching temperature data files found.")
        return None, None
    print(f"  Found {len(porosities)} porosity values with at least one case with data file")
    for p in porosities:
        cases = groups[p]
        slough = [c for c in cases if c['behavior'] == 'Slough']
        other = [c for c in cases if c['behavior'] != 'Slough']
        print(f"    Porosity {p:.3f}: {len(slough)} Slough, {len(other)} other")
    
    thickness_step_m = thickness_step_mm * 0.001  # 0.1 mm in m
    threshold_thicknesses = []
    threshold_porosities = []
    
    print(f"\nThickness step: {thickness_step_mm} mm")
    for i, porosity in enumerate(porosities):
        cases = groups[porosity]
        sloughing_cases = [c for c in cases if c['behavior'] == 'Slough']
        nosloughing_cases = [c for c in cases if c['behavior'] != 'Slough']
        
        # Load temperature profile from the chosen case
        if sloughing_cases:
            # Pick lowest-thickness sloughing case; use its temperature profile
            case = min(sloughing_cases, key=lambda c: c['thickness'])
            data_path = Path(exp_data_dir) / case['data_file']
            print(f"\n  Porosity {i+1}/{len(porosities)}: {porosity:.3f} (Slough case: {case['data_file']}, thickness {case['thickness_mm']:.2f} mm)")
            loader, time, temperature = load_defrost_data(str(data_path))
            T_ambient = float(case['row']['Air Dry Bulb [C]'])
            # For porosity < 0.9, start thickness from 2.5 mm; otherwise use lowest sloughing case thickness
            start_thickness = 0.0025 if porosity < 0.9 else case['thickness']
            critical = find_critical_thickness_sloughing_case(
                porosity, time, temperature, start_thickness,
                T_ambient=T_ambient,
                decrement_m=thickness_step_m,
                method='explicit', dt_safety_factor=0.9,
            )
        elif nosloughing_cases:
            # Pick a no-sloughing case (e.g. highest thickness) for temperature profile; increase thickness until sloughing
            case = max(nosloughing_cases, key=lambda c: c['thickness'])
            data_path = Path(exp_data_dir) / case['data_file']
            print(f"\n  Porosity {i+1}/{len(porosities)}: {porosity:.3f} (No-slough case: {case['data_file']}, thickness {case['thickness_mm']:.2f} mm)")
            loader, time, temperature = load_defrost_data(str(data_path))
            T_ambient = float(case['row']['Air Dry Bulb [C]'])
            start_thickness = case['thickness']
            critical = find_critical_thickness_nosloughing_case(
                porosity, time, temperature, start_thickness,
                T_ambient=T_ambient,
                increment_m=thickness_step_m,
                max_thickness_m=0.01,
                method='explicit', dt_safety_factor=0.9,
            )
        else:
            print(f"\n  Porosity {porosity:.3f}: no Slough or no-slough cases with data file, skipping")
            continue
        
        if critical is not None:
            threshold_thicknesses.append(critical * 1000)  # mm
            threshold_porosities.append(porosity)
            print(f"    Critical sloughing thickness: {critical*1000:.3f} mm")
        else:
            print(f"    Warning: Could not find critical thickness for this porosity")
    
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
