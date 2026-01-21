"""
Convergence Test for dt_safety_factor

This script tests different values of dt_safety_factor to determine
the optimal value for accurate and stable simulations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_defrost_data, get_frost_properties, parse_case_filename
from model_init import initialize_model
from solver import DefrostSolver
from temperature_interpolation import interpolate_temperature
from stability_criterion import calculate_max_stable_dt, get_initial_frost_properties

# Create figure directory if it doesn't exist
figure_dir = Path("figure")
figure_dir.mkdir(exist_ok=True)


def run_simulation(dt_safety_factor, data_file, n_layers, method='explicit'):
    """
    Run a single simulation with a given dt_safety_factor.
    
    Parameters
    ----------
    dt_safety_factor : float
        Safety factor for time step (0 < dt_safety_factor <= 1.0)
    data_file : str
        Name of the data file
    n_layers : int
        Number of layers
    method : str
        Solver method ('explicit' or 'implicit')
    
    Returns
    -------
    results : dict
        Simulation results
    dt : float
        Actual time step used
    """
    # Load data
    loader, time_raw, temperature_raw = load_defrost_data(data_file)
    
    # Apply temperature offset
    temperature_offset = -1  # [°C]
    temperature_raw = temperature_raw + temperature_offset
    
    frost_props = get_frost_properties(data_file)
    case_params = parse_case_filename(data_file)
    
    # Extract surface type from filename
    name = Path(data_file).stem
    parts = name.split('_')
    contact_angle_str = parts[1] if len(parts) > 1 else None
    
    # Extract ambient air temperature
    T_ambient = case_params.get('air_temp', None)
    
    # Get frost properties
    if not isinstance(frost_props, dict):
        raise ValueError(f"ERROR: No matching experiment data found for {data_file}")
    
    frost_thickness = frost_props['thickness']
    porosity = frost_props['porosity']
    
    if frost_thickness is None or frost_thickness <= 0:
        raise ValueError(f"ERROR: Invalid thickness: {frost_thickness}")
    
    # Calculate dt
    if method == 'explicit':
        props = get_initial_frost_properties(porosity=porosity)
        dt_max = calculate_max_stable_dt(
            n_layers=n_layers,
            frost_thickness=frost_thickness,
            k_eff=props['k_eff'],
            rho_eff=props['rho_eff'],
            cp_eff=props['cp_eff'],
            safety_factor=0.5  # CFL condition uses 0.5
        )
        dt = dt_max * dt_safety_factor
    else:
        dt = 0.1  # Default for implicit
    
    # Interpolate temperature data
    time, temperature = interpolate_temperature(time_raw, temperature_raw, dt=dt, kind='linear')
    
    # Initialize model
    model = initialize_model(
        n_layers=n_layers,
        frost_thickness=frost_thickness,
        porosity=porosity,
        T_initial=temperature[0],
        surface_type=contact_angle_str
    )
    
    # Create solver
    h_conv = 4.0
    solver = DefrostSolver(model, dt=dt, method=method, h_conv=h_conv, T_ambient=T_ambient)
    
    # Solve
    results = solver.solve(time, temperature, save_history=True)
    
    return results, dt


def interpolate_to_reference_time(time_ref, time_data, data_array):
    """
    Interpolate data to reference time points.
    
    Parameters
    ----------
    time_ref : array
        Reference time points
    time_data : array
        Time points for data
    data_array : array
        Data array (can be 1D or 2D)
    
    Returns
    -------
    interpolated_data : array
        Data interpolated to reference time points
    """
    if data_array is None or len(data_array) == 0:
        return None
    
    data_array = np.array(data_array)
    
    # Handle 1D arrays
    if data_array.ndim == 1:
        return np.interp(time_ref, time_data, data_array)
    
    # Handle 2D arrays (time, layers)
    elif data_array.ndim == 2:
        n_layers = data_array.shape[1]
        interpolated = np.zeros((len(time_ref), n_layers))
        for i in range(n_layers):
            interpolated[:, i] = np.interp(time_ref, time_data, data_array[:, i])
        return interpolated
    
    else:
        raise ValueError(f"Unsupported data array dimension: {data_array.ndim}")


def calculate_errors(results_ref, results_test, metrics=['h_total', 'temperature', 'alpha_ice']):
    """
    Calculate errors between reference and test solutions.
    
    Parameters
    ----------
    results_ref : dict
        Reference solution results
    results_test : dict
        Test solution results
    metrics : list
        List of metrics to compare
    
    Returns
    -------
    errors : dict
        Dictionary of error metrics for each quantity
    """
    errors = {}
    
    # Use reference time as the common time base
    time_ref = results_ref['time']
    
    for metric in metrics:
        if metric not in results_ref or results_ref[metric] is None:
            continue
        if metric not in results_test or results_test[metric] is None:
            continue
        
        # Get reference data
        data_ref = results_ref[metric]
        time_ref_data = results_ref['time']
        
        # Get test data and interpolate to reference time
        data_test = results_test[metric]
        time_test_data = results_test['time']
        
        # Interpolate test data to reference time points
        data_test_interp = interpolate_to_reference_time(time_ref_data, time_test_data, data_test)
        
        if data_test_interp is None:
            continue
        
        # Convert to numpy arrays
        data_ref = np.array(data_ref)
        data_test_interp = np.array(data_test_interp)
        
        # Calculate errors
        if data_ref.ndim == 1:
            # 1D arrays (e.g., h_total)
            error_abs = np.abs(data_ref - data_test_interp)
            error_rel = error_abs / (np.abs(data_ref) + 1e-10)  # Avoid division by zero
            
            errors[metric] = {
                'L2_error': np.sqrt(np.mean(error_abs**2)),
                'Linf_error': np.max(error_abs),
                'mean_abs_error': np.mean(error_abs),
                'max_rel_error': np.max(error_rel),
                'mean_rel_error': np.mean(error_rel),
            }
        elif data_ref.ndim == 2:
            # 2D arrays (e.g., temperature, alpha_ice)
            error_abs = np.abs(data_ref - data_test_interp)
            error_rel = error_abs / (np.abs(data_ref) + 1e-10)
            
            errors[metric] = {
                'L2_error': np.sqrt(np.mean(error_abs**2)),
                'Linf_error': np.max(error_abs),
                'mean_abs_error': np.mean(error_abs),
                'max_rel_error': np.max(error_rel),
                'mean_rel_error': np.mean(error_rel),
            }
    
    return errors


def test_dt_convergence(data_file="55min_60deg_83%_12C.txt", n_layers=40, method='explicit'):
    """
    Test convergence with different dt_safety_factor values.
    
    Parameters
    ----------
    data_file : str
        Name of the data file
    n_layers : int
        Number of layers
    method : str
        Solver method ('explicit' or 'implicit')
    """
    print("="*70)
    print("dt_safety_factor Convergence Test")
    print("="*70)
    print(f"Data file: {data_file}")
    print(f"Number of layers: {n_layers}")
    print(f"Method: {method}")
    print()
    
    # Test different safety factors (from finest to coarsest)
    # Start with a very fine time step as reference
    safety_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"Testing {len(safety_factors)} different dt_safety_factor values...")
    print(f"Safety factors: {safety_factors}")
    print()
    
    # Run reference solution (finest time step)
    print("Running reference solution (dt_safety_factor = 0.1)...")
    try:
        results_ref, dt_ref = run_simulation(0.1, data_file, n_layers, method)
        print(f"  Reference dt: {dt_ref:.6f} s")
        print(f"  Reference time steps: {len(results_ref['time'])}")
    except Exception as e:
        print(f"ERROR: Failed to run reference solution: {e}")
        return
    
    # Store results for all safety factors
    all_results = {}
    all_errors = {}
    all_dt = {}
    
    # Run simulations for each safety factor
    for sf in safety_factors:
        print(f"\nRunning simulation with dt_safety_factor = {sf:.2f}...")
        try:
            results, dt = run_simulation(sf, data_file, n_layers, method)
            all_results[sf] = results
            all_dt[sf] = dt
            
            print(f"  dt: {dt:.6f} s")
            print(f"  Time steps: {len(results['time'])}")
            
            # Calculate errors relative to reference
            errors = calculate_errors(results_ref, results, 
                                     metrics=['h_total', 'temperature', 'alpha_ice', 'alpha_water'])
            all_errors[sf] = errors
            
            # Print key error metrics
            if 'h_total' in errors:
                print(f"  h_total L2 error: {errors['h_total']['L2_error']:.6e} m")
                print(f"  h_total Linf error: {errors['h_total']['Linf_error']:.6e} m")
            if 'temperature' in errors:
                print(f"  Temperature L2 error: {errors['temperature']['L2_error']:.6e} °C")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create convergence plots
    print("\n" + "="*70)
    print("Creating convergence plots...")
    
    # Extract data for plotting
    safety_factors_plot = sorted([sf for sf in safety_factors if sf in all_errors])
    dt_values = [all_dt[sf] for sf in safety_factors_plot]
    
    # Plot 1: L2 error vs dt_safety_factor
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # h_total errors
    ax = axes[0, 0]
    if safety_factors_plot:
        h_total_l2 = [all_errors[sf]['h_total']['L2_error'] for sf in safety_factors_plot if 'h_total' in all_errors[sf]]
        h_total_linf = [all_errors[sf]['h_total']['Linf_error'] for sf in safety_factors_plot if 'h_total' in all_errors[sf]]
        sf_plot = [sf for sf in safety_factors_plot if 'h_total' in all_errors[sf]]
        
        if h_total_l2:
            ax.loglog(sf_plot, h_total_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(sf_plot, h_total_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('dt_safety_factor', fontsize=12)
            ax.set_ylabel('Error [m]', fontsize=12)
            ax.set_title('Total Thickness Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Temperature errors
    ax = axes[0, 1]
    if safety_factors_plot:
        temp_l2 = [all_errors[sf]['temperature']['L2_error'] for sf in safety_factors_plot if 'temperature' in all_errors[sf]]
        temp_linf = [all_errors[sf]['temperature']['Linf_error'] for sf in safety_factors_plot if 'temperature' in all_errors[sf]]
        sf_plot = [sf for sf in safety_factors_plot if 'temperature' in all_errors[sf]]
        
        if temp_l2:
            ax.loglog(sf_plot, temp_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(sf_plot, temp_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('dt_safety_factor', fontsize=12)
            ax.set_ylabel('Error [°C]', fontsize=12)
            ax.set_title('Temperature Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Alpha ice errors
    ax = axes[1, 0]
    if safety_factors_plot:
        alpha_ice_l2 = [all_errors[sf]['alpha_ice']['L2_error'] for sf in safety_factors_plot if 'alpha_ice' in all_errors[sf]]
        alpha_ice_linf = [all_errors[sf]['alpha_ice']['Linf_error'] for sf in safety_factors_plot if 'alpha_ice' in all_errors[sf]]
        sf_plot = [sf for sf in safety_factors_plot if 'alpha_ice' in all_errors[sf]]
        
        if alpha_ice_l2:
            ax.loglog(sf_plot, alpha_ice_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(sf_plot, alpha_ice_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('dt_safety_factor', fontsize=12)
            ax.set_ylabel('Error [-]', fontsize=12)
            ax.set_title('Ice Volume Fraction Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Alpha water errors
    ax = axes[1, 1]
    if safety_factors_plot:
        alpha_water_l2 = [all_errors[sf]['alpha_water']['L2_error'] for sf in safety_factors_plot if 'alpha_water' in all_errors[sf]]
        alpha_water_linf = [all_errors[sf]['alpha_water']['Linf_error'] for sf in safety_factors_plot if 'alpha_water' in all_errors[sf]]
        sf_plot = [sf for sf in safety_factors_plot if 'alpha_water' in all_errors[sf]]
        
        if alpha_water_l2:
            ax.loglog(sf_plot, alpha_water_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(sf_plot, alpha_water_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('dt_safety_factor', fontsize=12)
            ax.set_ylabel('Error [-]', fontsize=12)
            ax.set_title('Water Volume Fraction Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    plt.tight_layout()
    figure_path = figure_dir / 'dt_safety_factor_convergence.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{figure_path}'")
    plt.close()
    
    # Plot 2: Error vs actual dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # h_total errors vs dt
    ax = axes[0, 0]
    if safety_factors_plot:
        h_total_l2 = [all_errors[sf]['h_total']['L2_error'] for sf in safety_factors_plot if 'h_total' in all_errors[sf]]
        dt_plot = [all_dt[sf] for sf in safety_factors_plot if 'h_total' in all_errors[sf]]
        
        if h_total_l2:
            ax.loglog(dt_plot, h_total_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Time Step dt [s]', fontsize=12)
            ax.set_ylabel('L2 Error [m]', fontsize=12)
            ax.set_title('Total Thickness Error vs dt', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Temperature errors vs dt
    ax = axes[0, 1]
    if safety_factors_plot:
        temp_l2 = [all_errors[sf]['temperature']['L2_error'] for sf in safety_factors_plot if 'temperature' in all_errors[sf]]
        dt_plot = [all_dt[sf] for sf in safety_factors_plot if 'temperature' in all_errors[sf]]
        
        if temp_l2:
            ax.loglog(dt_plot, temp_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Time Step dt [s]', fontsize=12)
            ax.set_ylabel('L2 Error [°C]', fontsize=12)
            ax.set_title('Temperature Error vs dt', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Alpha ice errors vs dt
    ax = axes[1, 0]
    if safety_factors_plot:
        alpha_ice_l2 = [all_errors[sf]['alpha_ice']['L2_error'] for sf in safety_factors_plot if 'alpha_ice' in all_errors[sf]]
        dt_plot = [all_dt[sf] for sf in safety_factors_plot if 'alpha_ice' in all_errors[sf]]
        
        if alpha_ice_l2:
            ax.loglog(dt_plot, alpha_ice_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Time Step dt [s]', fontsize=12)
            ax.set_ylabel('L2 Error [-]', fontsize=12)
            ax.set_title('Ice Volume Fraction Error vs dt', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Alpha water errors vs dt
    ax = axes[1, 1]
    if safety_factors_plot:
        alpha_water_l2 = [all_errors[sf]['alpha_water']['L2_error'] for sf in safety_factors_plot if 'alpha_water' in all_errors[sf]]
        dt_plot = [all_dt[sf] for sf in safety_factors_plot if 'alpha_water' in all_errors[sf]]
        
        if alpha_water_l2:
            ax.loglog(dt_plot, alpha_water_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Time Step dt [s]', fontsize=12)
            ax.set_ylabel('L2 Error [-]', fontsize=12)
            ax.set_title('Water Volume Fraction Error vs dt', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = figure_dir / 'dt_safety_factor_convergence_vs_dt.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{figure_path}'")
    plt.close()
    
    # Print summary table
    print("\n" + "="*70)
    print("Convergence Summary")
    print("="*70)
    print(f"{'Safety Factor':<15} {'dt [s]':<12} {'h_total L2':<15} {'Temp L2':<15} {'Alpha_ice L2':<15}")
    print("-"*70)
    
    for sf in sorted(safety_factors_plot):
        if sf in all_errors:
            dt_val = all_dt[sf]
            h_l2 = all_errors[sf].get('h_total', {}).get('L2_error', np.nan)
            temp_l2 = all_errors[sf].get('temperature', {}).get('L2_error', np.nan)
            alpha_ice_l2 = all_errors[sf].get('alpha_ice', {}).get('L2_error', np.nan)
            
            print(f"{sf:<15.2f} {dt_val:<12.6f} {h_l2:<15.6e} {temp_l2:<15.6e} {alpha_ice_l2:<15.6e}")
    
    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)
    
    # Find the safety factor where errors start to increase significantly
    # (convergence threshold: error increases by more than 10% from previous value)
    if len(safety_factors_plot) >= 2:
        h_total_errors = [all_errors[sf]['h_total']['L2_error'] for sf in sorted(safety_factors_plot) if 'h_total' in all_errors[sf]]
        sf_sorted = sorted([sf for sf in safety_factors_plot if 'h_total' in all_errors[sf]])
        
        if len(h_total_errors) >= 2:
            # Find where error increases significantly
            recommended_sf = None
            for i in range(1, len(h_total_errors)):
                error_increase = (h_total_errors[i] - h_total_errors[i-1]) / h_total_errors[i-1]
                if error_increase > 0.1:  # 10% increase threshold
                    recommended_sf = sf_sorted[i-1]
                    break
            
            if recommended_sf is None:
                # If no significant increase, use the coarsest that's still accurate
                # Use the one with error < 1% of reference
                for i in range(len(h_total_errors)-1, -1, -1):
                    if h_total_errors[i] < 0.01 * h_total_errors[0]:
                        recommended_sf = sf_sorted[i]
                        break
            
            if recommended_sf is None:
                recommended_sf = 0.5  # Default fallback
            
            print(f"Recommended dt_safety_factor: {recommended_sf:.2f}")
            print(f"  Corresponding dt: {all_dt[recommended_sf]:.6f} s")
            if 'h_total' in all_errors[recommended_sf]:
                print(f"  h_total L2 error: {all_errors[recommended_sf]['h_total']['L2_error']:.6e} m")
            if 'temperature' in all_errors[recommended_sf]:
                print(f"  Temperature L2 error: {all_errors[recommended_sf]['temperature']['L2_error']:.6e} °C")
        else:
            print("Insufficient data for recommendation")
    else:
        print("Insufficient data for recommendation")
    
    print("="*70)
    
    return all_results, all_errors, all_dt


if __name__ == "__main__":
    # Test with default parameters
    results, errors, dt_values = test_dt_convergence(
        data_file="55min_60deg_83%_12C.txt",
        n_layers=40,
        method='explicit'
    )
