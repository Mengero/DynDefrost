"""
Dynamic Defrost Model - Main Script

1-D Dynamic Defrost Model for simulating frost layer behavior during defrost cycles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_defrost_data, get_frost_properties, parse_case_filename
from model_init import initialize_model
from solver import DefrostSolver
from temperature_interpolation import interpolate_temperature, print_interpolation_info
from stability_criterion import calculate_max_stable_dt, get_initial_frost_properties
from surface_retention import calculate_surface_retention

# Create figure directory if it doesn't exist
figure_dir = Path("figure")
figure_dir.mkdir(exist_ok=True)


def main():
    """Main entry point for the dynamic defrost model."""
    # ===== User Parameters =====
    data_file = "90min_60deg_45%_22C.txt"
    # n_layers will be automatically calculated based on initial thickness and retention thickness
    # dt will be automatically calculated based on n_layers for explicit methods
    # For implicit methods, you can set dt manually if needed
    method = 'explicit'  # Solver method: 'explicit' or 'implicit'
    dt_safety_factor = 0.9
    # ===========================
    
    # Load data
    loader, time_raw, temperature_raw = load_defrost_data(data_file)
    
    # Apply -1.2°C offset to surface temperature readings
    # The real temperature is 1.2°C lower than the reading
    temperature_offset = -1  # [°C]
    temperature_raw = temperature_raw + temperature_offset
    print(f"\nApplied temperature offset: {temperature_offset:.1f}°C")
    print(f"  Original temperature range: {np.min(temperature_raw - temperature_offset):.1f}°C to {np.max(temperature_raw - temperature_offset):.1f}°C")
    print(f"  Adjusted temperature range: {np.min(temperature_raw):.1f}°C to {np.max(temperature_raw):.1f}°C")
    
    frost_props = get_frost_properties(data_file)
    
    # Extract surface type from filename (e.g., "60deg" from "55min_60deg_83%_12C.txt")
    case_params = parse_case_filename(data_file)
    contact_angle_str = None
    # Extract contact angle from filename parts
    name = Path(data_file).stem
    parts = name.split('_')
    if len(parts) > 1:
        contact_angle_str = parts[1]  # e.g., "60deg" or "160deg"
    
    # Create figure filename prefix based on data file
    # e.g., "sim_result_180min_60deg_55%_22C"
    figure_prefix = f"sim_result_{name}"
    
    # Extract ambient air temperature from case parameters
    T_ambient = case_params.get('air_temp', None)  # [°C]
    
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
    
    # Calculate surface retention thickness to determine number of layers
    # Determine contact angles from surface type
    theta_receding = None
    theta_advancing = None
    if contact_angle_str is not None:
        surface_type_clean = contact_angle_str.lower().strip()
        if '60' in surface_type_clean or surface_type_clean == '60deg':
            theta_receding = 60.0
            theta_advancing = 70.0
        elif '160' in surface_type_clean or surface_type_clean == '160deg':
            theta_receding = 155.0
            theta_advancing = 160.0
    
    # Calculate surface retention thickness if contact angles are available
    if theta_receding is not None and theta_advancing is not None:
        retention_result = calculate_surface_retention(theta_receding, theta_advancing)
        surface_retention_thickness = retention_result['thickness']
        
        # Calculate number of layers: n_layers = initial_thickness / (retention_thickness / 2)
        # This ensures each layer is approximately half the retention thickness
        n_layers = int(np.round(frost_thickness / 5e-5))
        
        # Ensure minimum of 1 layer
        n_layers = max(1, n_layers)
        
        print(f"\nSurface retention calculation:")
        print(f"  Contact angles: θ_R={theta_receding:.1f}°, θ_A={theta_advancing:.1f}°")
        print(f"  Maximum retention thickness: {surface_retention_thickness*1000:.4f} mm")
        print(f"  Calculated number of layers: {n_layers} (based on thickness / (retention_thickness / 2))")
    else:
        # Fallback: use a default number of layers if contact angles are not available
        n_layers = 40
        print(f"\nWarning: Contact angles not available. Using default n_layers = {n_layers}")
    
    # Automatically calculate optimal dt based on n_layers for explicit methods
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
        
        print(f"\nStability Analysis (Explicit Method):")
        print(f"  Number of layers: {n_layers}")
        print(f"  Spatial step (dx): {frost_thickness/n_layers*1000:.4f} mm")
        print(f"  Thermal diffusivity: {props['k_eff']/(props['rho_eff']*props['cp_eff']):.2e} m²/s")
        print(f"  Maximum stable dt: {dt_max:.4f} s")
        print(f"  Auto-selected dt: {dt:.4f} s ({dt_safety_factor*100:.0f}% of maximum)")
        stability_ratio = dt / dt_max
        print(f"  ✓ Stability ratio (dt/dt_max): {stability_ratio:.3f} (safe)")
    else:
        # For implicit methods, use a reasonable default or let user specify
        # Implicit methods are unconditionally stable, so dt can be larger
        dt = 0.1  # Default for implicit (can be larger if needed)
        print(f"\nUsing implicit method (unconditionally stable)")
        print(f"  Time step: {dt:.4f} s")
    
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
    
    # Create solver with convective heat transfer coefficient and ambient temperature
    h_conv = 1.5  # Natural convection heat transfer coefficient [W/(m²·K)]
    solver = DefrostSolver(model, dt=dt, method=method, h_conv=h_conv, T_ambient=T_ambient)
    
    if T_ambient is not None:
        print(f"  Ambient air temperature: {T_ambient:.1f}°C")
        print(f"  Convective heat transfer coefficient: {h_conv:.1f} W/(m²·K)")
    
    # Solve
    print(f"\nSolving defrost problem...")
    print(f"  Time steps: {len(time)}")
    print(f"  Duration: {time[-1]:.1f} s")
    print(f"  Method: {method}")
    print(f"  Time step: {dt} s")
    print(f"  History save interval: 1.0 s (to reduce memory usage)")
    
    results = solver.solve(time, temperature, save_history=True, history_save_interval=1.0)
    
    print(f"\nSolution completed!")
    print(f"  Final time: {results['time'][-1]:.1f} s")
    print(f"  Final temperature range: {np.min(results['temperature'][:, -1]):.1f}°C to {np.max(results['temperature'][:, -1]):.1f}°C")
    
    # Check for sloughing time point
    if results['sloughing'] is not None and len(results['sloughing']) > 0:
        sloughing_indices = np.where(results['sloughing'] == True)[0]
        if len(sloughing_indices) > 0:
            first_sloughing_idx = sloughing_indices[0]
            sloughing_time = results['time'][first_sloughing_idx]
            print(f"\n{'='*70}")
            print(f"SLOUGHING DETECTED!")
            print(f"{'='*70}")
            print(f"  Sloughing time: {sloughing_time:.2f} s")
            if results['h_total'] is not None and len(results['h_total']) > first_sloughing_idx:
                print(f"  Frost thickness at sloughing: {results['h_total'][first_sloughing_idx]*1000:.4f} mm")
            if results['h_crit'] is not None and len(results['h_crit']) > first_sloughing_idx:
                print(f"  Critical thickness at sloughing: {results['h_crit'][first_sloughing_idx]*1000:.4f} mm")
            print(f"{'='*70}")
        else:
            print(f"\n  No sloughing detected during simulation")
    else:
        print(f"\n  Sloughing status not available")
    
    # Plot total thickness vs time
    if results['h_total'] is not None and len(results['h_total']) > 0:
        plt.figure(figsize=(10, 6))
        
        # Ensure arrays have the same length (handle case where simulation stops early)
        time_plot = results['time']
        h_total_plot = results['h_total']
        min_len = min(len(time_plot), len(h_total_plot))
        time_plot = time_plot[:min_len]
        h_total_plot = h_total_plot[:min_len]
        
        plt.plot(time_plot, h_total_plot * 1000, 'b-', linewidth=2, label='Total frost thickness')
        
        # Add critical thickness if available
        if results['h_crit'] is not None and len(results['h_crit']) > 0:
            h_crit_plot = results['h_crit']
            min_len_crit = min(len(time_plot), len(h_crit_plot))
            time_plot_crit = time_plot[:min_len_crit]
            h_crit_plot = h_crit_plot[:min_len_crit]
            plt.plot(time_plot_crit, h_crit_plot * 1000, 'r--', linewidth=2, label='Critical detachment thickness')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Thickness [mm]', fontsize=12)
        plt.title('Total Frost Layer Thickness vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_total_thickness_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No thickness data available for plotting")
    
    # Plot individual layer thicknesses vs time
    if results['dx'] is not None and len(results['dx']) > 0:
        dx_array = np.array(results['dx'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), dx_array.shape[0])
        time_plot = time_plot[:min_len]
        dx_plot = dx_array[:min_len, :]  # Shape: (min_len, n_layers)
        
        n_layers = dx_plot.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Use colormap to distinguish layers
        colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
        
        # Plot each layer
        for layer_idx in range(n_layers):
            layer_thickness = dx_plot[:, layer_idx] * 1000  # Convert to mm
            # Only plot if layer has non-zero thickness at some point
            if np.max(layer_thickness) > 1e-6:  # Threshold to avoid plotting empty layers
                plt.plot(time_plot, layer_thickness, 
                        color=colors[layer_idx], 
                        linewidth=1.5, 
                        alpha=0.7,
                        label=f'Layer {layer_idx+1}')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Layer Thickness [mm]', fontsize=12)
        plt.title('Individual Layer Thickness vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add legend, but limit to avoid clutter if too many layers
        if n_layers <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        else:
            # For many layers, just show a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                      norm=plt.Normalize(vmin=0, vmax=n_layers-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Layer Index', fontsize=10)
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_layer_thickness_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No layer thickness data available for plotting")
    
    # Plot temperature of each layer vs time
    if results['temperature'] is not None and len(results['temperature']) > 0:
        temp_array = np.array(results['temperature'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), temp_array.shape[0])
        time_plot = time_plot[:min_len]
        temp_plot = temp_array[:min_len, :]  # Shape: (min_len, n_layers)
        
        n_layers = temp_plot.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Use colormap to distinguish layers
        colors = plt.cm.plasma(np.linspace(0, 1, n_layers))
        
        # Plot each layer
        for layer_idx in range(n_layers):
            layer_temp = temp_plot[:, layer_idx]
            plt.plot(time_plot, layer_temp, 
                    color=colors[layer_idx], 
                    linewidth=1.5, 
                    alpha=0.7,
                    label=f'Layer {layer_idx+1}')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Temperature [°C]', fontsize=12)
        plt.title('Layer Temperature vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add legend, but limit to avoid clutter if too many layers
        if n_layers <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        else:
            # For many layers, just show a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                      norm=plt.Normalize(vmin=0, vmax=n_layers-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Layer Index', fontsize=10)
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_layer_temperature_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No temperature data available for plotting")
    
    # Plot volume fraction of ice for each layer vs time
    if results['alpha_ice'] is not None and len(results['alpha_ice']) > 0:
        alpha_ice_array = np.array(results['alpha_ice'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), alpha_ice_array.shape[0])
        time_plot = time_plot[:min_len]
        alpha_ice_plot = alpha_ice_array[:min_len, :]  # Shape: (min_len, n_layers)
        
        n_layers = alpha_ice_plot.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Use colormap to distinguish layers
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_layers))
        
        # Plot each layer
        for layer_idx in range(n_layers):
            layer_alpha_ice = alpha_ice_plot[:, layer_idx]
            # Only plot if layer has non-zero ice fraction at some point
            if np.max(layer_alpha_ice) > 1e-6:
                plt.plot(time_plot, layer_alpha_ice, 
                        color=colors[layer_idx], 
                        linewidth=1.5, 
                        alpha=0.7,
                        label=f'Layer {layer_idx+1}')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Ice Volume Fraction [-]', fontsize=12)
        plt.title('Ice Volume Fraction per Layer vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add legend, but limit to avoid clutter if too many layers
        if n_layers <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        else:
            # For many layers, just show a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                      norm=plt.Normalize(vmin=0, vmax=n_layers-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Layer Index', fontsize=10)
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_ice_volume_fraction_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No ice volume fraction data available for plotting")
    
    # Plot volume fraction of water for each layer vs time
    if results['alpha_water'] is not None and len(results['alpha_water']) > 0:
        alpha_water_array = np.array(results['alpha_water'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), alpha_water_array.shape[0])
        time_plot = time_plot[:min_len]
        alpha_water_plot = alpha_water_array[:min_len, :]  # Shape: (min_len, n_layers)
        
        n_layers = alpha_water_plot.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Use colormap to distinguish layers
        colors = plt.cm.Reds(np.linspace(0.3, 1, n_layers))
        
        # Plot each layer
        for layer_idx in range(n_layers):
            layer_alpha_water = alpha_water_plot[:, layer_idx]
            # Only plot if layer has non-zero water fraction at some point
            if np.max(layer_alpha_water) > 1e-6:
                plt.plot(time_plot, layer_alpha_water, 
                        color=colors[layer_idx], 
                        linewidth=1.5, 
                        alpha=0.7,
                        label=f'Layer {layer_idx+1}')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Water Volume Fraction [-]', fontsize=12)
        plt.title('Water Volume Fraction per Layer vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add legend, but limit to avoid clutter if too many layers
        if n_layers <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        else:
            # For many layers, just show a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                      norm=plt.Normalize(vmin=0, vmax=n_layers-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Layer Index', fontsize=10)
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_water_volume_fraction_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No water volume fraction data available for plotting")
    
    # Plot shrinkage rate for each layer vs time
    if results['shrinkage_rate'] is not None and len(results['shrinkage_rate']) > 0:
        shrinkage_rate_array = np.array(results['shrinkage_rate'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), shrinkage_rate_array.shape[0])
        time_plot = time_plot[:min_len]
        shrinkage_rate_plot = shrinkage_rate_array[:min_len, :]  # Shape: (min_len, n_layers)
        
        n_layers = shrinkage_rate_plot.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Use colormap to distinguish layers
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_layers))
        
        # Plot each layer
        for layer_idx in range(n_layers):
            layer_shrinkage_rate = shrinkage_rate_plot[:, layer_idx] * 1e6  # Convert to μm/s for readability
            # Only plot if layer has non-zero shrinkage rate at some point
            if np.max(np.abs(layer_shrinkage_rate)) > 1e-6:
                plt.plot(time_plot, layer_shrinkage_rate, 
                        color=colors[layer_idx], 
                        linewidth=1.5, 
                        alpha=0.7,
                        label=f'Layer {layer_idx+1}')
        
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Shrinkage Rate [μm/s]', fontsize=12)
        plt.title('Layer Shrinkage Rate vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)  # Zero line
        
        # Add legend, but limit to avoid clutter if too many layers
        if n_layers <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        else:
            # For many layers, just show a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                                      norm=plt.Normalize(vmin=0, vmax=n_layers-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Layer Index', fontsize=10)
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_shrinkage_rate_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No shrinkage rate data available for plotting")
    
    # Plot effective density vs time
    if (results['alpha_ice'] is not None and results['alpha_water'] is not None and 
        results['dx'] is not None and len(results['alpha_ice']) > 0):
        alpha_ice_array = np.array(results['alpha_ice'])  # Shape: (n_time_steps, n_layers)
        alpha_water_array = np.array(results['alpha_water'])  # Shape: (n_time_steps, n_layers)
        dx_array = np.array(results['dx'])  # Shape: (n_time_steps, n_layers)
        time_plot = results['time']
        
        # Ensure arrays have the same length
        min_len = min(len(time_plot), alpha_ice_array.shape[0], 
                     alpha_water_array.shape[0], dx_array.shape[0])
        time_plot = time_plot[:min_len]
        alpha_ice_plot = alpha_ice_array[:min_len, :]
        alpha_water_plot = alpha_water_array[:min_len, :]
        dx_plot = dx_array[:min_len, :]
        
        # Calculate effective density for each time step
        # Use the same method as solver._calculate_effective_density()
        rho_ice = 917.0  # kg/m³
        rho_water = 1000.0  # kg/m³
        rho_air = 1.2  # kg/m³
        
        rho_eff_history = []
        n_layers = alpha_ice_plot.shape[1]
        
        # Get begin_idx and end_idx from model (if available)
        # For now, assume all layers are active (begin_idx=0, end_idx=n_layers-1)
        begin_idx = 0
        end_idx = n_layers - 1
        
        for t_idx in range(min_len):
            alpha_ice_t = alpha_ice_plot[t_idx, :]
            alpha_water_t = alpha_water_plot[t_idx, :]
            dx_t = dx_plot[t_idx, :]
            
            # Find the first layer with frost (alpha_ice > 0) starting from wall side (end_idx)
            first_frost_idx = None
            for i in range(end_idx, begin_idx - 1, -1):
                if alpha_ice_t[i] > 1e-10:  # Layer contains ice
                    first_frost_idx = i
                    break
            
            if first_frost_idx is None:
                rho_eff_history.append(0.0)
                continue
            
            # Calculate masses and thickness from first_frost_idx to end_idx (inclusive)
            layer_range = range(first_frost_idx, end_idx + 1)
            
            # Calculate masses per unit area
            m_ice_layers = alpha_ice_t[layer_range] * rho_ice * dx_t[layer_range]
            m_water_layers = alpha_water_t[layer_range] * rho_water * dx_t[layer_range]
            
            m_ice_total = np.sum(m_ice_layers)
            m_water_total = np.sum(m_water_layers)
            
            # Total thickness
            h_total = np.sum(dx_t[layer_range])
            
            if h_total > 0:
                # Calculate volumes
                V_total = h_total  # For unit area A = 1 m²
                V_ice = m_ice_total / rho_ice
                V_water = m_water_total / rho_water
                V_air = V_total - V_ice - V_water
                V_air = np.maximum(V_air, 0.0)  # Ensure non-negative
                
                # Calculate air mass
                m_air_total = V_air * rho_air
                
                # Total mass
                # m_total = m_ice_total + m_water_total + m_air_total
                m_total = m_ice_total + m_air_total
                
                # Effective density
                rho_eff = m_total / h_total
            else:
                rho_eff = 0.0
            
            rho_eff_history.append(rho_eff)
        
        rho_eff_array = np.array(rho_eff_history)
        
        # Plot effective density vs time
        plt.figure(figsize=(10, 6))
        plt.plot(time_plot, rho_eff_array, 'g-', linewidth=2, label='Effective density')
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Effective Density [kg/m³]', fontsize=12)
        plt.title('Effective Density vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add horizontal line at 215 kg/m³ (threshold for too much water)
        plt.axhline(y=215, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                   label='Water threshold (215 kg/m³)')
        
        plt.tight_layout()
        figure_path = figure_dir / f'{figure_prefix}_effective_density_vs_time.png'
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{figure_path}'")
        plt.close()
    else:
        print("Warning: No data available for effective density plot")
    
    return time, temperature, frost_props, model, results


if __name__ == "__main__":
    time, temperature, frost_props, model, results = main()
 