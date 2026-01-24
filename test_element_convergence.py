"""
Convergence Test for Element Size (Number of Layers)

This script tests different numbers of layers to determine
the optimal spatial resolution for accurate simulations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
from data_loader import load_defrost_data, get_frost_properties, parse_case_filename
from model_init import initialize_model
from solver import DefrostSolver
from temperature_interpolation import interpolate_temperature
from stability_criterion import calculate_max_stable_dt, get_initial_frost_properties

# Create figure directory if it doesn't exist
figure_dir = Path("figure")
figure_dir.mkdir(exist_ok=True)


def run_simulation(n_layers, data_file, dt_safety_factor=0.35, method='explicit'):
    """
    Run a single simulation with a given number of layers.
    
    Parameters
    ----------
    n_layers : int
        Number of layers (element size)
    data_file : str
        Name of the data file
    dt_safety_factor : float
        Safety factor for time step (0 < dt_safety_factor <= 1.0)
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


def interpolate_to_reference_layers(z_ref, z_data, data_array):
    """
    Interpolate spatial data to reference layer positions.
    
    Parameters
    ----------
    z_ref : array
        Reference layer positions (depth coordinates)
    z_data : array
        Data layer positions (depth coordinates)
    data_array : array
        Data array (time, layers) - 2D array
    
    Returns
    -------
    interpolated_data : array
        Data interpolated to reference layer positions (time, n_ref_layers)
    """
    if data_array is None or len(data_array) == 0:
        return None
    
    data_array = np.array(data_array)
    
    if data_array.ndim != 2:
        raise ValueError(f"Expected 2D array (time, layers), got {data_array.ndim}D")
    
    n_time = data_array.shape[0]
    n_ref = len(z_ref)
    interpolated = np.zeros((n_time, n_ref))
    
    # Interpolate each time step
    for t in range(n_time):
        # Use scipy interp1d for better handling of edge cases
        f = interp1d(z_data, data_array[t, :], kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        interpolated[t, :] = f(z_ref)
    
    return interpolated


def interpolate_spatial_data(results_test, n_layers_ref, frost_thickness):
    """
    Interpolate spatial data from test solution to reference grid.
    
    Parameters
    ----------
    results_test : dict
        Test solution results
    n_layers_ref : int
        Number of layers in reference solution
    frost_thickness : float
        Total frost thickness
    
    Returns
    -------
    results_interp : dict
        Results with spatial data interpolated to reference grid
    """
    results_interp = results_test.copy()
    
    # Create reference layer positions (centers of layers)
    dz_ref = frost_thickness / n_layers_ref
    z_ref = np.linspace(dz_ref/2, frost_thickness - dz_ref/2, n_layers_ref)
    
    # Get test layer positions
    n_layers_test = None
    if 'temperature' in results_test and results_test['temperature'] is not None:
        n_layers_test = results_test['temperature'].shape[1] if results_test['temperature'].ndim == 2 else None
    
    if n_layers_test is None:
        # Can't interpolate without knowing number of layers
        return results_interp
    
    dz_test = frost_thickness / n_layers_test
    z_test = np.linspace(dz_test/2, frost_thickness - dz_test/2, n_layers_test)
    
    # Interpolate 2D spatial arrays
    spatial_metrics = ['temperature', 'alpha_ice', 'alpha_water', 'alpha_air']
    
    for metric in spatial_metrics:
        if metric in results_test and results_test[metric] is not None:
            data = np.array(results_test[metric])
            if data.ndim == 2:
                results_interp[metric] = interpolate_to_reference_layers(z_ref, z_test, data)
    
    return results_interp


def calculate_errors(results_ref, results_test, metrics=['h_total', 'temperature', 'alpha_ice'], 
                     n_layers_ref=None, frost_thickness=None):
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
    n_layers_ref : int, optional
        Number of layers in reference solution (for spatial interpolation)
    frost_thickness : float, optional
        Total frost thickness (for spatial interpolation)
    
    Returns
    -------
    errors : dict
        Dictionary of error metrics for each quantity
    """
    errors = {}
    
    # Use reference time as the common time base
    time_ref = results_ref['time']
    
    # Interpolate test spatial data to reference grid if needed
    results_test_interp = results_test.copy()
    if n_layers_ref is not None and frost_thickness is not None:
        results_test_interp = interpolate_spatial_data(results_test, n_layers_ref, frost_thickness)
    
    for metric in metrics:
        if metric not in results_ref or results_ref[metric] is None:
            continue
        if metric not in results_test_interp or results_test_interp[metric] is None:
            continue
        
        # Get reference data
        data_ref = results_ref[metric]
        time_ref_data = results_ref['time']
        
        # Get test data and interpolate to reference time
        data_test = results_test_interp[metric]
        time_test_data = results_test_interp['time']
        
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


def run_simulation_wrapper(args):
    """
    Wrapper function for parallel execution of run_simulation.
    
    Parameters
    ----------
    args : tuple
        (n_layers, data_file, dt_safety_factor, method)
    
    Returns
    -------
    tuple
        (n_layers, results, dt) or (n_layers, None, None) if error
    """
    n_layers, data_file, dt_safety_factor, method = args
    try:
        results, dt = run_simulation(n_layers, data_file, dt_safety_factor, method)
        return (n_layers, results, dt)
    except Exception as e:
        print(f"  ERROR (n_layers={n_layers}): {e}")
        return (n_layers, None, None)


def test_element_convergence(data_file="55min_60deg_83%_12C.txt", dt_safety_factor=0.35, 
                             method='explicit', max_workers=None, use_parallel=True):
    """
    Test convergence with different numbers of layers (element sizes).
    
    Parameters
    ----------
    data_file : str
        Name of the data file
    dt_safety_factor : float
        Safety factor for time step (default: 0.35 from dt convergence test)
    method : str
        Solver method ('explicit' or 'implicit')
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    use_parallel : bool, optional
        Whether to use parallel execution. Default: True
    """
    print("="*70)
    print("Element Size (Number of Layers) Convergence Test")
    print("="*70)
    print(f"Data file: {data_file}")
    print(f"dt_safety_factor: {dt_safety_factor}")
    print(f"Method: {method}")
    print(f"Parallel execution: {use_parallel}")
    if use_parallel:
        import os
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        print(f"Max workers: {max_workers}")
    print()
    
    # Get frost thickness for spatial interpolation
    frost_props = get_frost_properties(data_file)
    if not isinstance(frost_props, dict):
        raise ValueError(f"ERROR: No matching experiment data found for {data_file}")
    frost_thickness = frost_props['thickness']
    
    # Test different numbers of layers (from finest to coarsest)
    # Reference should be the finest (most accurate) solution
    # Note: More layers = finer spatial resolution = longer computation time
    #       but better accuracy. Use 60-80 for thorough verification,
    #       or 40-50 if computation time is a concern.
    n_layers_list = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    reference_n_layers = 120  # Reference number of layers (finest, most accurate)
    
    print(f"Testing {len(n_layers_list)} different numbers of layers...")
    print(f"Number of layers: {n_layers_list}")
    print(f"Reference solution: n_layers = {reference_n_layers}")
    print()
    
    # Run reference solution - must be done first
    print(f"Running reference solution (n_layers = {reference_n_layers})...")
    print("  NOTE: This runs sequentially. Parallel execution will start after this completes.")
    try:
        results_ref, dt_ref = run_simulation(reference_n_layers, data_file, dt_safety_factor, method)
        print(f"  Reference dt: {dt_ref:.6f} s")
        print(f"  Reference time steps: {len(results_ref['time'])}")
        print(f"  Reference solution completed!")
    except Exception as e:
        print(f"ERROR: Failed to run reference solution: {e}")
        return
    
    # Store results for all layer counts
    all_results = {}
    all_errors = {}
    all_dt = {}
    
    # Prepare layer counts to run (exclude reference which is already done)
    n_layers_to_run = [n for n in n_layers_list if n != reference_n_layers]
    
    if use_parallel and len(n_layers_to_run) > 1:
        # Run simulations in parallel
        print(f"\nRunning {len(n_layers_to_run)} simulations in parallel...")
        
        # Prepare arguments for parallel execution
        args_list = [(n, data_file, dt_safety_factor, method) for n in n_layers_to_run]
        
        # Try parallel execution, fallback to sequential if it fails
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_n = {executor.submit(run_simulation_wrapper, args): args[0] 
                              for args in args_list}
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_n):
                    n = future_to_n[future]
                    completed += 1
                    try:
                        n_result, results, dt = future.result()
                        if results is not None:
                            all_results[n_result] = results
                            all_dt[n_result] = dt
                            print(f"  [{completed}/{len(n_layers_to_run)}] Completed n_layers={n_result}, dt={dt:.6f} s")
                        else:
                            print(f"  [{completed}/{len(n_layers_to_run)}] Failed n_layers={n_result}")
                    except Exception as e:
                        print(f"  [{completed}/{len(n_layers_to_run)}] Exception for n_layers={n}: {e}")
        except Exception as e:
            print(f"\nWARNING: Parallel execution failed: {e}")
            print("Falling back to sequential execution...")
            use_parallel = False  # Fall through to sequential execution
    
    if not use_parallel or len(n_layers_to_run) <= 1:
        # Run simulations sequentially
        print(f"\nRunning {len(n_layers_to_run)} simulations sequentially...")
        for n in n_layers_to_run:
            print(f"\nRunning simulation with n_layers = {n}...")
            try:
                results, dt = run_simulation(n, data_file, dt_safety_factor, method)
                all_results[n] = results
                all_dt[n] = dt
                print(f"  dt: {dt:.6f} s")
                print(f"  Time steps: {len(results['time'])}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    # Add reference solution to results
    all_results[reference_n_layers] = results_ref
    all_dt[reference_n_layers] = dt_ref
    
    # Calculate errors for all solutions (except reference)
    print("\nCalculating errors relative to reference solution...")
    for n in n_layers_list:
        if n == reference_n_layers:
            # Reference solution has no error
            all_errors[n] = {}
            continue
        
        if n not in all_results:
            continue
        
        print(f"  Calculating errors for n_layers={n}...")
        try:
            errors = calculate_errors(results_ref, all_results[n], 
                                     metrics=['h_total', 'temperature', 'alpha_ice', 'alpha_water'],
                                     n_layers_ref=reference_n_layers,
                                     frost_thickness=frost_thickness)
            all_errors[n] = errors
            
            # Print key error metrics
            if 'h_total' in errors:
                print(f"    h_total L2 error: {errors['h_total']['L2_error']:.6e} m")
            if 'temperature' in errors:
                print(f"    Temperature L2 error: {errors['temperature']['L2_error']:.6e} °C")
        except Exception as e:
            print(f"    ERROR calculating errors: {e}")
            continue
    
    # Create convergence plots
    print("\n" + "="*70)
    print("Creating convergence plots...")
    
    # Extract data for plotting
    n_layers_plot = sorted([n for n in n_layers_list if n in all_errors])
    element_sizes = [frost_thickness / n for n in n_layers_plot]  # Element size (layer thickness)
    
    # Plot 1: L2 error vs number of layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # h_total errors
    ax = axes[0, 0]
    if n_layers_plot:
        h_total_l2 = [all_errors[n]['h_total']['L2_error'] for n in n_layers_plot if 'h_total' in all_errors[n]]
        h_total_linf = [all_errors[n]['h_total']['Linf_error'] for n in n_layers_plot if 'h_total' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'h_total' in all_errors[n]]
        
        if h_total_l2:
            ax.loglog(n_plot, h_total_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(n_plot, h_total_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Error [m]', fontsize=12)
            ax.set_title('Total Thickness Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Temperature errors
    ax = axes[0, 1]
    if n_layers_plot:
        temp_l2 = [all_errors[n]['temperature']['L2_error'] for n in n_layers_plot if 'temperature' in all_errors[n]]
        temp_linf = [all_errors[n]['temperature']['Linf_error'] for n in n_layers_plot if 'temperature' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'temperature' in all_errors[n]]
        
        if temp_l2:
            ax.loglog(n_plot, temp_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(n_plot, temp_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Error [°C]', fontsize=12)
            ax.set_title('Temperature Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Alpha ice errors
    ax = axes[1, 0]
    if n_layers_plot:
        alpha_ice_l2 = [all_errors[n]['alpha_ice']['L2_error'] for n in n_layers_plot if 'alpha_ice' in all_errors[n]]
        alpha_ice_linf = [all_errors[n]['alpha_ice']['Linf_error'] for n in n_layers_plot if 'alpha_ice' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'alpha_ice' in all_errors[n]]
        
        if alpha_ice_l2:
            ax.loglog(n_plot, alpha_ice_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(n_plot, alpha_ice_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Error [-]', fontsize=12)
            ax.set_title('Ice Volume Fraction Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    # Alpha water errors
    ax = axes[1, 1]
    if n_layers_plot:
        alpha_water_l2 = [all_errors[n]['alpha_water']['L2_error'] for n in n_layers_plot if 'alpha_water' in all_errors[n]]
        alpha_water_linf = [all_errors[n]['alpha_water']['Linf_error'] for n in n_layers_plot if 'alpha_water' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'alpha_water' in all_errors[n]]
        
        if alpha_water_l2:
            ax.loglog(n_plot, alpha_water_l2, 'o-', label='L2 error', linewidth=2, markersize=8)
            ax.loglog(n_plot, alpha_water_linf, 's-', label='L∞ error', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Error [-]', fontsize=12)
            ax.set_title('Water Volume Fraction Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
    
    plt.tight_layout()
    figure_path = figure_dir / 'element_size_convergence.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{figure_path}'")
    plt.close()
    
    # Plot 2: Error vs element size (layer thickness)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # h_total errors vs element size
    ax = axes[0, 0]
    if n_layers_plot:
        h_total_l2 = [all_errors[n]['h_total']['L2_error'] for n in n_layers_plot if 'h_total' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'h_total' in all_errors[n]]
        element_sizes_plot = [frost_thickness / n for n in n_plot]
        
        if h_total_l2:
            ax.loglog(element_sizes_plot, h_total_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Element Size (Layer Thickness) [m]', fontsize=12)
            ax.set_ylabel('L2 Error [m]', fontsize=12)
            ax.set_title('Total Thickness Error vs Element Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Temperature errors vs element size
    ax = axes[0, 1]
    if n_layers_plot:
        temp_l2 = [all_errors[n]['temperature']['L2_error'] for n in n_layers_plot if 'temperature' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'temperature' in all_errors[n]]
        element_sizes_plot = [frost_thickness / n for n in n_plot]
        
        if temp_l2:
            ax.loglog(element_sizes_plot, temp_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Element Size (Layer Thickness) [m]', fontsize=12)
            ax.set_ylabel('L2 Error [°C]', fontsize=12)
            ax.set_title('Temperature Error vs Element Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Alpha ice errors vs element size
    ax = axes[1, 0]
    if n_layers_plot:
        alpha_ice_l2 = [all_errors[n]['alpha_ice']['L2_error'] for n in n_layers_plot if 'alpha_ice' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'alpha_ice' in all_errors[n]]
        element_sizes_plot = [frost_thickness / n for n in n_plot]
        
        if alpha_ice_l2:
            ax.loglog(element_sizes_plot, alpha_ice_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Element Size (Layer Thickness) [m]', fontsize=12)
            ax.set_ylabel('L2 Error [-]', fontsize=12)
            ax.set_title('Ice Volume Fraction Error vs Element Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Alpha water errors vs element size
    ax = axes[1, 1]
    if n_layers_plot:
        alpha_water_l2 = [all_errors[n]['alpha_water']['L2_error'] for n in n_layers_plot if 'alpha_water' in all_errors[n]]
        n_plot = [n for n in n_layers_plot if 'alpha_water' in all_errors[n]]
        element_sizes_plot = [frost_thickness / n for n in n_plot]
        
        if alpha_water_l2:
            ax.loglog(element_sizes_plot, alpha_water_l2, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Element Size (Layer Thickness) [m]', fontsize=12)
            ax.set_ylabel('L2 Error [-]', fontsize=12)
            ax.set_title('Water Volume Fraction Error vs Element Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = figure_dir / 'element_size_convergence_vs_size.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{figure_path}'")
    plt.close()
    
    # Print summary table
    print("\n" + "="*70)
    print("Convergence Summary")
    print("="*70)
    print(f"{'N Layers':<12} {'Element Size [m]':<18} {'h_total L2':<15} {'Temp L2':<15} {'Alpha_ice L2':<15}")
    print("-"*70)
    
    for n in sorted(n_layers_plot):
        if n in all_errors:
            element_size = frost_thickness / n
            h_l2 = all_errors[n].get('h_total', {}).get('L2_error', np.nan)
            temp_l2 = all_errors[n].get('temperature', {}).get('L2_error', np.nan)
            alpha_ice_l2 = all_errors[n].get('alpha_ice', {}).get('L2_error', np.nan)
            
            print(f"{n:<12} {element_size:<18.6e} {h_l2:<15.6e} {temp_l2:<15.6e} {alpha_ice_l2:<15.6e}")
    
    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)
    
    # Find the number of layers where errors start to increase significantly
    # (convergence threshold: error increases by more than 10% from previous value)
    if len(n_layers_plot) >= 2:
        h_total_errors = [all_errors[n]['h_total']['L2_error'] for n in sorted(n_layers_plot) if 'h_total' in all_errors[n]]
        n_sorted = sorted([n for n in n_layers_plot if 'h_total' in all_errors[n]])
        
        if len(h_total_errors) >= 2:
            # Find where error increases significantly (going from fine to coarse)
            recommended_n = None
            for i in range(len(h_total_errors)-1, 0, -1):  # Go backwards (coarse to fine)
                error_increase = (h_total_errors[i-1] - h_total_errors[i]) / (h_total_errors[i] + 1e-15)
                if error_increase > 0.1:  # 10% increase threshold
                    recommended_n = n_sorted[i]
                    break
            
            if recommended_n is None:
                # If no significant increase, use the coarsest that's still accurate
                # Use the one with error < 1% of finest
                for i in range(len(h_total_errors)-1, -1, -1):
                    if h_total_errors[i] < 0.01 * h_total_errors[0]:
                        recommended_n = n_sorted[i]
                        break
            
            if recommended_n is None:
                recommended_n = 40  # Default fallback
            
            print(f"Recommended number of layers: {recommended_n}")
            print(f"  Corresponding element size: {frost_thickness/recommended_n:.6e} m")
            print(f"  Corresponding dt: {all_dt[recommended_n]:.6f} s")
            if 'h_total' in all_errors[recommended_n]:
                print(f"  h_total L2 error: {all_errors[recommended_n]['h_total']['L2_error']:.6e} m")
            if 'temperature' in all_errors[recommended_n]:
                print(f"  Temperature L2 error: {all_errors[recommended_n]['temperature']['L2_error']:.6e} °C")
        else:
            print("Insufficient data for recommendation")
    else:
        print("Insufficient data for recommendation")
    
    print("="*70)
    
    return all_results, all_errors, all_dt


if __name__ == "__main__":
    # Test with default parameters
    # Set use_parallel=True to run simulations in parallel (faster)
    # Set max_workers to control number of parallel processes (None = use all CPUs)
    results, errors, dt_values = test_element_convergence(
        data_file="55min_60deg_83%_12C.txt",
        dt_safety_factor=0.35,  # Use recommended value from dt convergence test
        method='explicit',
        use_parallel=True,  # Enable parallel execution
        max_workers=4    # Use all available CPUs
    )
