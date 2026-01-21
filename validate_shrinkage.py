"""
Shrinkage Model Validation Script

This script validates the shrinkage model by:
1. Setting initial conditions: alpha_ice = 0.2, thickness = 4mm
2. Calculating initial mass of ice (kg/m²)
3. Decreasing ice mass by 5% of initial mass every second
4. At each time step:
   - Update volume fraction of ice based on new mass
   - Calculate shrinkage to get updated thickness
   - Re-update volume fraction of ice based on new thickness
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
from model_init import DefrostModel
from solver import DefrostSolver


def validate_shrinkage_model(use_shrinkage=True):
    """
    Validate the shrinkage model with specified test conditions.
    
    Parameters
    ----------
    use_shrinkage : bool, optional
        Whether to apply shrinkage model. If False, thickness remains constant.
        Default: True
    """
    # Set up directories for logs and figures
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)  # Create log directory if it doesn't exist
    
    figure_dir = Path("figure")
    figure_dir.mkdir(exist_ok=True)  # Create figure directory if it doesn't exist
    
    # Only save log file for shrinkage case (without_shrinkage produces empty file)
    # Simplified filename without _with_shrinkage_ or _without_shrinkage_
    log_filepath = None
    if use_shrinkage:
        log_filename = f"shrinkage_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = log_dir / log_filename
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_filepath, mode='w'),
                logging.StreamHandler()  # Also print to console
            ]
        )
    else:
        # For without_shrinkage, only print to console
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.StreamHandler()  # Only print to console
            ]
        )
    
    logger = logging.getLogger()
    
    # Helper function to log and print
    def log_print(message):
        logger.info(message)
    
    log_print("=" * 70)
    if use_shrinkage:
        log_print("Shrinkage Model Validation (WITH shrinkage)")
    else:
        log_print("Shrinkage Model Validation (WITHOUT shrinkage)")
    log_print("=" * 70)
    if log_filepath:
        log_print(f"Log file: {log_filepath}")
    
    # Initial conditions
    alpha_ice_initial = 0.2
    thickness_initial = 4e-3  # 4 mm in meters
    n_layers = 1  # Single layer for validation
    
    # Create model
    model = DefrostModel(
        n_layers=n_layers,
        frost_thickness=thickness_initial,
        porosity=1.0 - alpha_ice_initial  # porosity = 1 - alpha_ice (assuming no water initially)
    )
    
    # Set initial volume fractions
    model.alpha_ice = np.array([alpha_ice_initial])
    model.alpha_water = np.array([0.0])
    model.alpha_air = np.array([1.0 - alpha_ice_initial])
    model.alpha_ice_initial = model.alpha_ice.copy()
    
    # Set initial temperature
    model.set_initial_temperature(-20.0)
    
    # Calculate initial mass of ice per unit area (kg/m²)
    # m''_ice = α_ice * ρ_ice * δ_A
    m_ice_initial = alpha_ice_initial * model.rho_ice * thickness_initial
    log_print(f"\nInitial Conditions:")
    log_print(f"  alpha_ice: {alpha_ice_initial}")
    log_print(f"  alpha_air: {model.alpha_air[0]}")
    log_print(f"  thickness: {thickness_initial*1000:.2f} mm")
    log_print(f"  Initial mass of ice: {m_ice_initial:.6f} kg/m²")
    
    # Mass decrease rate: 5% of initial mass per second
    mass_decrease_rate = 0.001 * m_ice_initial  # kg/(m²·s)
    log_print(f"  Mass decrease rate: {mass_decrease_rate:.6f} kg/(m²·s) ({mass_decrease_rate/m_ice_initial*100:.1f}% per second)")
    
    # Note: Minimum thickness is time-dependent and calculated at each time step
    # thickness_min(t) = m_ice(t) / rho_ice + m_water(t) / rho_water
    # This represents the minimum volume required by the current ice and water masses
    log_print(f"  Minimum thickness is calculated dynamically at each time step")
    log_print(f"    Formula: thickness_min(t) = m_ice(t) / ρ_ice + m_water(t) / ρ_water")
    
    # Time parameters
    dt = 1  # 1 second time step
    n_steps = 20000  # 20 seconds total
    
    # Create solver (we'll use it for shrinkage calculation)
    solver = DefrostSolver(model, dt=dt, method='explicit')
    
    # Initialize mass per unit area
    solver.m_double_prime_ice = np.array([m_ice_initial])
    solver.m_double_prime_water = np.array([0.0])
    
    # Storage for results
    time_history = []
    thickness_history = []
    thickness_min_history = []  # Minimum thickness based on mass
    alpha_ice_history = []
    alpha_H2O_history = []
    alpha_air_before_history = []
    m_ice_history = []
    alpha_ice_after_shrinkage_history = []
    alpha_H2O_after_shrinkage_history = []
    alpha_air_after_history = []
    shrink_rate_history = []
    r_pore_history = []
    eta_eff_history = []
    delta_over_eta_r_history = []
    stage2_flag_history = []  # Track when we're in stage 2 (thickness at minimum)
    
    # Calculate initial r_pore for display
    alpha_ice_init = alpha_ice_initial
    d_ice_init = solver.d_ice_initial
    if alpha_ice_init > 0:
        r_pore_init = (2.0 * (1.0 - alpha_ice_init)) / (3.0 * alpha_ice_init) * d_ice_init * 1e6  # Convert to μm
    else:
        r_pore_init = 0.0
    
    # Calculate initial eta_eff for display
    rho_frost_init = (alpha_ice_init * model.rho_ice + 
                     model.alpha_water[0] * model.rho_water + 
                     model.alpha_air[0] * model.rho_air)
    eta_dry_init = solver.eta_0 * np.exp(solver.b * (rho_frost_init / model.rho_ice))
    f_wet_init = 1.0 / (1.0 + solver.C_wet * model.alpha_water[0])
    eta_eff_init = eta_dry_init * f_wet_init
    
    # Calculate initial δ_A / (η_eff * r_pore)
    r_pore_init_m = r_pore_init * 1e-6  # Convert to meters
    if eta_eff_init > 0 and r_pore_init_m > 0:
        delta_over_eta_r_init = thickness_initial / (eta_eff_init * r_pore_init_m)
    else:
        delta_over_eta_r_init = 0.0
    
    # Initial state
    alpha_air_initial = model.alpha_air[0]
    # Calculate initial minimum thickness
    m_water_initial = solver.m_double_prime_water[0]
    thickness_min_initial = m_ice_initial / model.rho_ice + m_water_initial / model.rho_water
    
    time_history.append(0.0)
    thickness_history.append(model.dx[0])
    thickness_min_history.append(thickness_min_initial)
    alpha_ice_history.append(model.alpha_ice[0])
    alpha_H2O_history.append(model.alpha_water[0])
    alpha_air_before_history.append(alpha_air_initial)
    m_ice_history.append(solver.m_double_prime_ice[0])
    alpha_ice_after_shrinkage_history.append(model.alpha_ice[0])
    alpha_H2O_after_shrinkage_history.append(model.alpha_water[0])
    alpha_air_after_history.append(alpha_air_initial)
    shrink_rate_history.append(0.0)  # No shrinkage at initial time
    r_pore_history.append(r_pore_init)
    eta_eff_history.append(eta_eff_init)
    delta_over_eta_r_history.append(delta_over_eta_r_init)
    stage2_flag_history.append(False)  # Not in stage 2 initially
    
    log_print(f"\nStarting simulation for {n_steps} seconds...")
    log_print(f"{'Time (s)':<10} {'Mass (kg/m²)':<15} {'α_ice (b)':<12} {'α_H2O (b)':<12} {'α_air (b)':<12} {'Thick (mm)':<12} {'α_ice (a)':<12} {'α_H2O (a)':<12} {'α_air (a)':<12} {'r_pore (μm)':<12} {'η_eff (Pa·s)':<15} {'δ/(η·r)':<12} {'Shrink (mm/s)':<15} {'Stage2':<8}")
    log_print("-" * 220)
    log_print(f"{0.0:<10.1f} {m_ice_initial:<15.6f} {alpha_ice_initial:<12.6f} {model.alpha_water[0]:<12.6f} {alpha_air_initial:<12.6f} {thickness_initial*1000:<12.4f} {alpha_ice_initial:<12.6f} {model.alpha_water[0]:<12.6f} {alpha_air_initial:<12.6f} {r_pore_init:<12.4f} {eta_eff_init:<15.2e} {delta_over_eta_r_init:<12.2e} {'-':<15} {'No':<8}")
    
    # Time stepping
    in_stage2 = False  # Track when we've entered stage 2 (thickness constrained to minimum)
    ice_zero_time = None  # Track when ice volume fraction goes to 0
    
    for step in range(1, n_steps + 1):
        t = step * dt
        
        # Step 1: Decrease ice mass by 5% of initial mass
        # The melted ice converts to water, maintaining mass conservation
        m_ice_old = solver.m_double_prime_ice[0]
        m_ice_new = m_ice_old - mass_decrease_rate * dt
        m_ice_new = np.maximum(m_ice_new, 0.0)  # Don't go negative
        
        # Calculate change in ice mass
        delta_m_ice = m_ice_old - m_ice_new  # Mass of ice that melted [kg/m²]
        
        # Mass conservation: melted ice mass = water mass produced
        # delta_m_ice = delta_m_water
        m_water_old = solver.m_double_prime_water[0]
        m_water_new = m_water_old + delta_m_ice  # Melted ice becomes water
        
        # Update mass per unit area
        solver.m_double_prime_ice[0] = m_ice_new
        solver.m_double_prime_water[0] = m_water_new
        
        # Step 2: Update volume fraction of ice and water based on new mass and current thickness
        # α_ice = m''_ice / (ρ_ice * δ_A)
        # α_water = m''_water / (ρ_water * δ_A)
        current_thickness = model.dx[0]
        if current_thickness > 0:
            alpha_ice_before = m_ice_new / (model.rho_ice * current_thickness)
            alpha_H2O_before = m_water_new / (model.rho_water * current_thickness)
        else:
            alpha_ice_before = 0.0
            alpha_H2O_before = 0.0
        
        alpha_ice_before = np.clip(alpha_ice_before, 0.0, 1.0)
        alpha_H2O_before = np.clip(alpha_H2O_before, 0.0, 1.0)
        
        model.alpha_ice[0] = alpha_ice_before
        model.alpha_water[0] = alpha_H2O_before
        
        # Update air volume fraction (closure)
        model.alpha_air[0] = 1.0 - alpha_ice_before - alpha_H2O_before
        model.alpha_air[0] = np.maximum(model.alpha_air[0], 0.0)
        
        # Calculate pore radius before shrinkage (for display)
        # r_pore = (2(1 - α_ice)) / (3 * α_ice) * d_ice
        alpha_ice_i = model.alpha_ice_initial[0]
        if alpha_ice_i > 0 and alpha_ice_before > 0:
            d_ice = solver.d_ice_i[0] * (alpha_ice_before / alpha_ice_i) ** (1.0 / 3.0)
        else:
            d_ice = solver.d_ice_i[0]
        
        if alpha_ice_before > 0:
            r_pore = (2.0 * (1.0 - alpha_ice_before)) / (3.0 * alpha_ice_before) * d_ice
        else:
            r_pore = 1e-6  # Small value if no ice
        
        # Calculate effective viscosity (eta_eff) for display
        # Calculate frost density
        rho_frost = (alpha_ice_before * model.rho_ice + 
                    alpha_H2O_before * model.rho_water + 
                    model.alpha_air[0] * model.rho_air)
        
        # Calculate dry viscosity: η_dry = η_0 * exp(b * (ρ_frost / ρ_ice))
        eta_dry = solver.eta_0 * np.exp(solver.b * (rho_frost / model.rho_ice))
        
        # Calculate wetness reduction factor: f_wet = 1 / (1 + C_wet * α_H2O)
        f_wet = 1.0 / (1.0 + solver.C_wet * alpha_H2O_before)
        
        # Calculate effective viscosity: η_eff = η_dry * f_wet
        eta_eff = eta_dry * f_wet
        
        # Calculate δ_A / (η_eff * r_pore) for display
        if eta_eff > 0 and r_pore > 0:
            delta_over_eta_r = current_thickness / (eta_eff * r_pore)
        else:
            delta_over_eta_r = 0.0
        
        # Calculate time-dependent minimum thickness based on current masses
        # thickness_min(t) = m_ice(t) / rho_ice + m_water(t) / rho_water
        # This is the minimum volume required by current ice and water masses
        thickness_min = m_ice_new / model.rho_ice + m_water_new / model.rho_water
        
        # Step 3: Calculate shrinkage model (if enabled)
        thickness_before = current_thickness
        if use_shrinkage:
            # Update properties needed for shrinkage
            model.calculate_specific_heat()
            model.calculate_thermal_conductivity()
            
            # Calculate shrinkage (manually call the shrinkage function)
            solver._calculate_layer_shrinkage()
            
            # Get updated thickness after shrinkage
            thickness_after = model.dx[0]
            
            # Check if we've entered stage 2: thickness constrained to minimum
            # In stage 2, ice continues melting, water increases, but thickness = thickness_min
            # This happens because water density > ice density, so total volume decreases
            if thickness_after <= thickness_min:
                # Enter stage 2: enforce thickness = minimum required by current masses
                thickness_after = thickness_min
                model.dx[0] = thickness_min
                if not in_stage2:
                    in_stage2 = True
                    log_print(f"\n→ Entered Stage 2 at t = {t:.1f} s: Thickness constrained to minimum ({thickness_min*1000:.4f} mm)")
                    log_print(f"   Ice continues melting, water mass increases, but thickness stays at minimum")
                    log_print(f"   This occurs because ρ_water ({model.rho_water:.1f} kg/m³) > ρ_ice ({model.rho_ice:.1f} kg/m³)")
            
            # Calculate shrinkage rate (mm/s)
            shrink_rate = (thickness_before - thickness_after) / dt * 1000  # Convert to mm/s
        else:
            # No shrinkage: thickness remains constant
            thickness_after = current_thickness
            shrink_rate = 0.0
            
            # Still check if we would enter stage 2 (for consistency)
            if thickness_after <= thickness_min:
                thickness_after = thickness_min
                model.dx[0] = thickness_min
                if not in_stage2:
                    in_stage2 = True
                    log_print(f"\n→ Entered Stage 2 at t = {t:.1f} s: Thickness at minimum ({thickness_min*1000:.4f} mm)")
        
        # Step 4: Re-update volume fraction of ice and water based on new thickness
        # α_ice = m''_ice / (ρ_ice * δ_A^(t+Δt))
        # α_water = m''_water / (ρ_water * δ_A^(t+Δt))
        if thickness_after > 0:
            alpha_ice_after = m_ice_new / (model.rho_ice * thickness_after)
            alpha_H2O_after = m_water_new / (model.rho_water * thickness_after)
        else:
            alpha_ice_after = 0.0
            alpha_H2O_after = 0.0
        
        alpha_ice_after = np.clip(alpha_ice_after, 0.0, 1.0)
        alpha_H2O_after = np.clip(alpha_H2O_after, 0.0, 1.0)
        
        model.alpha_ice[0] = alpha_ice_after
        model.alpha_water[0] = alpha_H2O_after
        
        # Update air volume fraction (closure)
        model.alpha_air[0] = 1.0 - alpha_ice_after - alpha_H2O_after
        model.alpha_air[0] = np.maximum(model.alpha_air[0], 0.0)
        
        # Check if ice volume fraction goes to 0 (or very close to 0)
        if ice_zero_time is None and alpha_ice_after <= 1e-6:
            ice_zero_time = t
            log_print(f"\n→ Ice volume fraction reached 0 at t = {t:.2f} s")
        
        # Get alpha_air values
        alpha_air_before = model.alpha_air[0]  # After first update, before shrinkage
        alpha_air_after = model.alpha_air[0]  # After second update (after shrinkage)
        
        # Store results
        time_history.append(t)
        thickness_history.append(thickness_after)
        thickness_min_history.append(thickness_min)  # Store minimum thickness
        alpha_ice_history.append(alpha_ice_before)
        alpha_H2O_history.append(alpha_H2O_before)
        alpha_air_before_history.append(alpha_air_before)
        m_ice_history.append(m_ice_new)
        alpha_ice_after_shrinkage_history.append(alpha_ice_after)
        alpha_H2O_after_shrinkage_history.append(alpha_H2O_after)
        alpha_air_after_history.append(alpha_air_after)
        shrink_rate_history.append(shrink_rate)
        r_pore_history.append(r_pore * 1e6)  # Convert to μm for storage
        eta_eff_history.append(eta_eff)
        delta_over_eta_r_history.append(delta_over_eta_r)
        stage2_flag_history.append(in_stage2)
        
        # Print progress
        stage2_str = "Yes" if in_stage2 else "No"
        log_print(f"{t:<10.1f} {m_ice_new:<15.6f} {alpha_ice_before:<12.6f} {alpha_H2O_before:<12.6f} {alpha_air_before:<12.6f} {thickness_after*1000:<12.4f} {alpha_ice_after:<12.6f} {alpha_H2O_after:<12.6f} {alpha_air_after:<12.6f} {r_pore*1e6:<12.4f} {eta_eff:<15.2e} {delta_over_eta_r:<12.2e} {shrink_rate:<15.6f} {stage2_str:<8}")
    
    # Convert to numpy arrays for plotting
    time_history = np.array(time_history)
    thickness_history = np.array(thickness_history)
    thickness_min_history = np.array(thickness_min_history)
    alpha_ice_history = np.array(alpha_ice_history)
    alpha_H2O_history = np.array(alpha_H2O_history)
    alpha_air_before_history = np.array(alpha_air_before_history)
    m_ice_history = np.array(m_ice_history)
    alpha_ice_after_shrinkage_history = np.array(alpha_ice_after_shrinkage_history)
    alpha_H2O_after_shrinkage_history = np.array(alpha_H2O_after_shrinkage_history)
    alpha_air_after_history = np.array(alpha_air_after_history)
    shrink_rate_history = np.array(shrink_rate_history)
    r_pore_history = np.array(r_pore_history)
    eta_eff_history = np.array(eta_eff_history)
    delta_over_eta_r_history = np.array(delta_over_eta_r_history)
    
    # Plot results
    shrinkage_label = " (with shrinkage)" if use_shrinkage else " (no shrinkage)"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 to add new plot
    
    # Plot 1: Thickness vs time (exclude last time step)
    axes[0, 0].plot(time_history[:-1], thickness_history[:-1] * 1000, 'b-', linewidth=2, label=f'Layer thickness{shrinkage_label}')
    axes[0, 0].plot(time_history[:-1], thickness_min_history[:-1] * 1000, 'r--', linewidth=2, label='Minimum thickness (m_ice/ρ_ice + m_water/ρ_water)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Thickness (mm)')
    axes[0, 0].set_title('Layer Thickness vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Mass of ice vs time (exclude last time step)
    axes[0, 1].plot(time_history[:-1], m_ice_history[:-1], 'r-', linewidth=2, label='Ice mass')
    axes[0, 1].axhline(y=m_ice_initial, color='r', linestyle='--', alpha=0.5, label='Initial mass')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mass of ice (kg/m²)')
    axes[0, 1].set_title('Ice Mass vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Volume fractions of ice and water vs time (exclude last time step)
    axes[1, 0].plot(time_history[:-1], alpha_ice_history[:-1], 'g--', linewidth=2, label='α_ice (before shrinkage)')
    axes[1, 0].plot(time_history[:-1], alpha_ice_after_shrinkage_history[:-1], 'g-', linewidth=2, label='α_ice (after shrinkage)')
    axes[1, 0].plot(time_history[:-1], alpha_H2O_history[:-1], 'b--', linewidth=2, label='α_H2O (before shrinkage)')
    axes[1, 0].plot(time_history[:-1], alpha_H2O_after_shrinkage_history[:-1], 'b-', linewidth=2, label='α_H2O (after shrinkage)')
    axes[1, 0].axhline(y=alpha_ice_initial, color='g', linestyle=':', alpha=0.5, label='Initial α_ice')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Volume fraction')
    axes[1, 0].set_title('Volume Fractions of Ice and Water vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Thickness reduction rate vs time (exclude last time step)
    if len(thickness_history) > 2:
        thickness_rate = np.diff(thickness_history[:-1]) / dt * 1000  # mm/s
        time_rate = time_history[1:-1]
        axes[1, 1].plot(time_rate, thickness_rate, 'm-', linewidth=2, label='Shrinkage rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Thickness change rate (mm/s)')
        axes[1, 1].set_title('Shrinkage Rate vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].axis('off')
    
    # Plot 5: Shrinkage rate vs alpha_water (exclude last time step)
    # Only plot this for the WITH shrinkage case
    if use_shrinkage and len(shrink_rate_history) > 1:
        # Use alpha_H2O_after_shrinkage_history (after shrinkage) for consistency
        alpha_water_plot = alpha_H2O_after_shrinkage_history[:-1]
        shrink_rate_plot = shrink_rate_history[:-1]
        
        # Ensure arrays have the same length
        min_len = min(len(alpha_water_plot), len(shrink_rate_plot))
        if min_len > 0:
            alpha_water_plot = alpha_water_plot[:min_len]
            shrink_rate_plot = shrink_rate_plot[:min_len]
        
        # Only plot points where we have valid data
        # Note: shrink_rate should be >= 0 (shrinkage) or zero (no shrinkage)
        valid_mask = (alpha_water_plot >= 0) & (alpha_water_plot <= 1) & np.isfinite(shrink_rate_plot) & (shrink_rate_plot >= 0)
        
        if len(alpha_water_plot) == 0 or len(shrink_rate_plot) == 0:
            log_print(f"\nWarning: Empty arrays for shrinkage rate plot")
            axes[1, 2].axis('off')
        elif np.any(valid_mask):
            # Plot all valid points - use line plot for better visibility
            axes[1, 2].plot(alpha_water_plot[valid_mask], shrink_rate_plot[valid_mask], 
                           'm-', linewidth=2, alpha=0.7, label='Shrinkage rate')
            # Also add markers for better visibility
            axes[1, 2].plot(alpha_water_plot[valid_mask], shrink_rate_plot[valid_mask], 
                           'mo', markersize=3, alpha=0.5)
            axes[1, 2].set_xlabel('Water Volume Fraction (α_H2O)')
            axes[1, 2].set_ylabel('Shrinkage Rate (mm/s)')
            axes[1, 2].set_title('Shrinkage Rate vs Water Volume Fraction')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
            axes[1, 2].set_xlim([0, 1])
            # Set y-axis to show from 0 or minimum value
            if np.any(shrink_rate_plot[valid_mask] > 1e-6):  # Use small threshold to avoid zero-only plots
                y_min = 0
                y_max = np.max(shrink_rate_plot[valid_mask]) * 1.1
                axes[1, 2].set_ylim([y_min, y_max])
            else:
                # All shrinkage rates are essentially zero, set a small range
                axes[1, 2].set_ylim([0, 0.01])
                log_print(f"\nNote: Shrinkage rates are all near zero in shrinkage rate vs alpha_water plot")
        else:
            # Debug: print why no valid data
            log_print(f"\nDebug: No valid data for shrinkage rate vs alpha_water plot")
            log_print(f"  use_shrinkage: {use_shrinkage}")
            log_print(f"  len(shrink_rate_history): {len(shrink_rate_history)}")
            log_print(f"  len(alpha_water_plot): {len(alpha_water_plot)}")
            if len(alpha_water_plot) > 0:
                log_print(f"  alpha_water_plot range: [{np.min(alpha_water_plot):.6f}, {np.max(alpha_water_plot):.6f}]")
            if len(shrink_rate_plot) > 0:
                log_print(f"  shrink_rate_plot range: [{np.min(shrink_rate_plot):.6f}, {np.max(shrink_rate_plot):.6f}]")
            axes[1, 2].axis('off')
    else:
        # Hide the subplot if no shrinkage data
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    # Use different filenames for with/without shrinkage cases
    if use_shrinkage:
        figure_filename = 'shrinkage_validation_with_shrinkage.png'
    else:
        figure_filename = 'shrinkage_validation_without_shrinkage.png'
    figure_path = figure_dir / figure_filename
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    log_print(f"\nResults saved to '{figure_path}'")
    
    # Summary statistics
    log_print(f"\n" + "=" * 70)
    log_print("Summary Statistics:")
    log_print("=" * 70)
    log_print(f"Initial thickness: {thickness_initial*1000:.4f} mm")
    log_print(f"Final thickness: {thickness_history[-1]*1000:.4f} mm")
    log_print(f"Thickness reduction: {(thickness_initial - thickness_history[-1])*1000:.4f} mm")
    log_print(f"Thickness reduction percentage: {(1 - thickness_history[-1]/thickness_initial)*100:.2f}%")
    
    # Display lower limit information (calculate final minimum thickness)
    m_ice_final = m_ice_history[-1]
    m_water_final = solver.m_double_prime_water[0]
    thickness_min_final = m_ice_final / model.rho_ice + m_water_final / model.rho_water
    log_print(f"\nFinal minimum thickness limit: {thickness_min_final*1000:.4f} mm")
    log_print(f"  Based on final masses: m_ice={m_ice_final:.6f} kg/m², m_water={m_water_final:.6f} kg/m²")
    log_print(f"  Calculation: m_ice / ρ_ice + m_water / ρ_water = {m_ice_final:.6f}/{model.rho_ice:.1f} + {m_water_final:.6f}/{model.rho_water:.1f} = {thickness_min_final*1000:.4f} mm")
    log_print(f"Final thickness / Final lower limit ratio: {thickness_history[-1] / thickness_min_final:.4f}")
    
    # Also show initial minimum thickness for reference
    thickness_min_initial = m_ice_initial / model.rho_ice + 0.0 / model.rho_water
    log_print(f"\nInitial minimum thickness limit: {thickness_min_initial*1000:.4f} mm")
    log_print(f"  Based on initial masses: m_ice={m_ice_initial:.6f} kg/m², m_water=0.0 kg/m²")
    
    # Check if we entered stage 2
    if any(stage2_flag_history):
        stage2_start_time = time_history[np.argmax(stage2_flag_history)]
        log_print(f"\n→ Stage 2 entered at t = {stage2_start_time:.1f} s")
        log_print(f"   In Stage 2: Thickness constrained to minimum, ice continues melting")
        log_print(f"   Total time in Stage 2: {time_history[-1] - stage2_start_time:.1f} s")
    else:
        log_print(f"\n→ Simulation completed without entering Stage 2")
    
    # Check if ice volume fraction went to 0
    if ice_zero_time is not None:
        log_print(f"\n→ Ice volume fraction reached 0 at t = {ice_zero_time:.2f} s")
        log_print(f"   Final α_ice: {alpha_ice_after_shrinkage_history[-1]:.6e}")
    else:
        log_print(f"\n→ Ice volume fraction did not reach 0 during simulation")
        log_print(f"   Final α_ice: {alpha_ice_after_shrinkage_history[-1]:.6f}")
    log_print(f"\nInitial mass: {m_ice_initial:.6f} kg/m²")
    log_print(f"Final mass: {m_ice_history[-1]:.6f} kg/m²")
    log_print(f"Mass reduction: {m_ice_initial - m_ice_history[-1]:.6f} kg/m²")
    log_print(f"Mass reduction percentage: {(1 - m_ice_history[-1]/m_ice_initial)*100:.2f}%")
    log_print(f"\nInitial α_ice: {alpha_ice_initial:.6f}")
    log_print(f"Final α_ice (after shrinkage): {alpha_ice_after_shrinkage_history[-1]:.6f}")
    log_print(f"Initial α_H2O: {0.0:.6f}")
    log_print(f"Final α_H2O (after shrinkage): {alpha_H2O_after_shrinkage_history[-1]:.6f}")
    log_print(f"Initial α_air: {alpha_air_initial:.6f}")
    log_print(f"Final α_air (after shrinkage): {alpha_air_after_history[-1]:.6f}")
    
    return {
        'time': time_history,
        'thickness': thickness_history,
        'thickness_min': thickness_min_history,  # Minimum thickness based on mass
        'alpha_ice_before': alpha_ice_history,
        'alpha_H2O_before': alpha_H2O_history,
        'alpha_air_before': alpha_air_before_history,
        'alpha_ice_after': alpha_ice_after_shrinkage_history,
        'alpha_H2O_after': alpha_H2O_after_shrinkage_history,
        'alpha_air_after': alpha_air_after_history,
        'mass_ice': m_ice_history,
        'shrink_rate': shrink_rate_history,
        'r_pore': r_pore_history,
        'eta_eff': eta_eff_history,
        'delta_over_eta_r': delta_over_eta_r_history,
        'stage2': np.array(stage2_flag_history),
        'log_file': str(log_filepath) if log_filepath else None
    }


if __name__ == "__main__":
    # Create figure directory if it doesn't exist
    figure_dir = Path("figure")
    figure_dir.mkdir(exist_ok=True)
    
    # Run simulation WITH shrinkage
    print("\n" + "=" * 70)
    results_with_shrinkage = validate_shrinkage_model(use_shrinkage=True)
    
    # Run simulation WITHOUT shrinkage
    print("\n" + "=" * 70)
    results_without_shrinkage = validate_shrinkage_model(use_shrinkage=False)
    
    # Create comparison plots
    print("\n" + "=" * 70)
    print("Creating Comparison Plots")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Thickness comparison (exclude last time step)
    time_with = results_with_shrinkage['time'][:-1]
    time_without = results_without_shrinkage['time'][:-1]
    
    axes[0, 0].plot(time_with, results_with_shrinkage['thickness'][:-1] * 1000, 
                    'b-', linewidth=2, label='With shrinkage')
    axes[0, 0].plot(time_without, results_without_shrinkage['thickness'][:-1] * 1000, 
                    'r--', linewidth=2, label='Without shrinkage')
    # Add minimum thickness line (should be the same for both cases since it's based on mass)
    if 'thickness_min' in results_with_shrinkage:
        axes[0, 0].plot(time_with, results_with_shrinkage['thickness_min'][:-1] * 1000, 
                        'g:', linewidth=2, label='Minimum thickness')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Thickness (mm)')
    axes[0, 0].set_title('Layer Thickness Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Volume fraction of ice comparison (exclude last time step)
    axes[0, 1].plot(time_with, results_with_shrinkage['alpha_ice_after'][:-1], 
                    'g-', linewidth=2, label='α_ice (with shrinkage)')
    axes[0, 1].plot(time_without, results_without_shrinkage['alpha_ice_after'][:-1], 
                    'g--', linewidth=2, label='α_ice (without shrinkage)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Volume fraction of ice')
    axes[0, 1].set_title('Volume Fraction of Ice Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Volume fraction of water comparison (exclude last time step)
    axes[1, 0].plot(time_with, results_with_shrinkage['alpha_H2O_after'][:-1], 
                    'b-', linewidth=2, label='α_H2O (with shrinkage)')
    axes[1, 0].plot(time_without, results_without_shrinkage['alpha_H2O_after'][:-1], 
                    'b--', linewidth=2, label='α_H2O (without shrinkage)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Volume fraction of water')
    axes[1, 0].set_title('Volume Fraction of Water Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Thickness difference (exclude last time step)
    # Interpolate to common time points for comparison
    time_common = np.linspace(0, max(time_with[-1], time_without[-1]), 100)
    thickness_with_interp = np.interp(time_common, time_with, results_with_shrinkage['thickness'][:-1] * 1000)
    thickness_without_interp = np.interp(time_common, time_without, results_without_shrinkage['thickness'][:-1] * 1000)
    thickness_diff = thickness_without_interp - thickness_with_interp
    
    axes[1, 1].plot(time_common, thickness_diff, 'm-', linewidth=2, label='Thickness difference')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Thickness difference (mm)')
    axes[1, 1].set_title('Thickness Difference (No shrinkage - With shrinkage)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    comparison_figure_path = figure_dir / 'shrinkage_comparison.png'
    plt.savefig(comparison_figure_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plots saved to '{comparison_figure_path}'")
    
    # plt.show()  # Commented out - using non-interactive backend (Agg)
