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
import matplotlib.pyplot as plt
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
    print("=" * 70)
    if use_shrinkage:
        print("Shrinkage Model Validation (WITH shrinkage)")
    else:
        print("Shrinkage Model Validation (WITHOUT shrinkage)")
    print("=" * 70)
    
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
    print(f"\nInitial Conditions:")
    print(f"  alpha_ice: {alpha_ice_initial}")
    print(f"  thickness: {thickness_initial*1000:.2f} mm")
    print(f"  Initial mass of ice: {m_ice_initial:.6f} kg/m²")
    
    # Mass decrease rate: 5% of initial mass per second
    mass_decrease_rate = 0.0001 * m_ice_initial  # kg/(m²·s)
    print(f"  Mass decrease rate: {mass_decrease_rate:.6f} kg/(m²·s) ({mass_decrease_rate/m_ice_initial*100:.1f}% per second)")
    
    # Calculate fixed lower limit for thickness
    # Since all ice will eventually become water and there's no drainage,
    # the minimum thickness is determined by the initial ice mass
    thickness_min = m_ice_initial / model.rho_water
    print(f"  Lower thickness limit: {thickness_min*1000:.4f} mm (based on initial ice mass)")
    
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
    alpha_ice_history = []
    alpha_H2O_history = []
    m_ice_history = []
    alpha_ice_after_shrinkage_history = []
    alpha_H2O_after_shrinkage_history = []
    shrink_rate_history = []
    r_pore_history = []
    eta_eff_history = []
    delta_over_eta_r_history = []
    
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
    time_history.append(0.0)
    thickness_history.append(model.dx[0])
    alpha_ice_history.append(model.alpha_ice[0])
    alpha_H2O_history.append(model.alpha_water[0])
    m_ice_history.append(solver.m_double_prime_ice[0])
    alpha_ice_after_shrinkage_history.append(model.alpha_ice[0])
    alpha_H2O_after_shrinkage_history.append(model.alpha_water[0])
    shrink_rate_history.append(0.0)  # No shrinkage at initial time
    r_pore_history.append(r_pore_init)
    eta_eff_history.append(eta_eff_init)
    delta_over_eta_r_history.append(delta_over_eta_r_init)
    
    print(f"\nStarting simulation for {n_steps} seconds...")
    print(f"{'Time (s)':<10} {'Mass (kg/m²)':<15} {'α_ice (before)':<18} {'α_H2O (before)':<18} {'Thickness (mm)':<18} {'α_ice (after)':<18} {'α_H2O (after)':<18} {'r_pore (μm)':<15} {'η_eff (Pa·s)':<18} {'δ/(η·r) (1/Pa)':<18} {'Shrink rate (mm/s)':<20}")
    print("-" * 200)
    print(f"{0.0:<10.1f} {m_ice_initial:<15.6f} {alpha_ice_initial:<18.6f} {model.alpha_water[0]:<18.6f} {thickness_initial*1000:<18.4f} {alpha_ice_initial:<18.6f} {model.alpha_water[0]:<18.6f} {r_pore_init:<15.4f} {eta_eff_init:<18.2e} {delta_over_eta_r_init:<18.2e} {'-':<20}")
    
    # Time stepping
    simulation_stopped = False
    stop_reason = None
    
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
            
            # Enforce lower limit: thickness cannot be less than that required by water
            if thickness_after < thickness_min:
                thickness_after = thickness_min
                model.dx[0] = thickness_min
                simulation_stopped = True
                stop_reason = f"Thickness reached lower limit ({thickness_min*1000:.4f} mm) based on water mass"
            
            # Calculate shrinkage rate (mm/s)
            shrink_rate = (thickness_before - thickness_after) / dt * 1000  # Convert to mm/s
        else:
            # No shrinkage: thickness remains constant
            thickness_after = current_thickness
            shrink_rate = 0.0
            
            # Still check lower limit even without shrinkage
            if thickness_after < thickness_min:
                thickness_after = thickness_min
                model.dx[0] = thickness_min
                simulation_stopped = True
                stop_reason = f"Thickness reached lower limit ({thickness_min*1000:.4f} mm) based on water mass"
        
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
        
        # Store results
        time_history.append(t)
        thickness_history.append(thickness_after)
        alpha_ice_history.append(alpha_ice_before)
        alpha_H2O_history.append(alpha_H2O_before)
        m_ice_history.append(m_ice_new)
        alpha_ice_after_shrinkage_history.append(alpha_ice_after)
        alpha_H2O_after_shrinkage_history.append(alpha_H2O_after)
        shrink_rate_history.append(shrink_rate)
        r_pore_history.append(r_pore * 1e6)  # Convert to μm for storage
        eta_eff_history.append(eta_eff)
        delta_over_eta_r_history.append(delta_over_eta_r)
        
        # Print progress
        print(f"{t:<10.1f} {m_ice_new:<15.6f} {alpha_ice_before:<18.6f} {alpha_H2O_before:<18.6f} {thickness_after*1000:<18.4f} {alpha_ice_after:<18.6f} {alpha_H2O_after:<18.6f} {r_pore*1e6:<15.4f} {eta_eff:<18.2e} {delta_over_eta_r:<18.2e} {shrink_rate:<20.6f}")
        
        # Stop simulation if lower limit reached
        if simulation_stopped:
            print(f"\n⚠️  Simulation stopped at t = {t:.1f} s: {stop_reason}")
            break
    
    # Convert to numpy arrays for plotting
    time_history = np.array(time_history)
    thickness_history = np.array(thickness_history)
    alpha_ice_history = np.array(alpha_ice_history)
    alpha_H2O_history = np.array(alpha_H2O_history)
    m_ice_history = np.array(m_ice_history)
    alpha_ice_after_shrinkage_history = np.array(alpha_ice_after_shrinkage_history)
    alpha_H2O_after_shrinkage_history = np.array(alpha_H2O_after_shrinkage_history)
    shrink_rate_history = np.array(shrink_rate_history)
    r_pore_history = np.array(r_pore_history)
    eta_eff_history = np.array(eta_eff_history)
    delta_over_eta_r_history = np.array(delta_over_eta_r_history)
    
    # Plot results
    shrinkage_label = " (with shrinkage)" if use_shrinkage else " (no shrinkage)"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Thickness vs time (exclude last time step)
    axes[0, 0].plot(time_history[:-1], thickness_history[:-1] * 1000, 'b-', linewidth=2, label=f'Layer thickness{shrinkage_label}')
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
    
    # Plot 4: Thickness reduction rate (exclude last time step)
    if len(thickness_history) > 2:
        thickness_rate = np.diff(thickness_history[:-1]) / dt * 1000  # mm/s
        time_rate = time_history[1:-1]
        axes[1, 1].plot(time_rate, thickness_rate, 'm-', linewidth=2, label='Shrinkage rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Thickness change rate (mm/s)')
        axes[1, 1].set_title('Shrinkage Rate vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('shrinkage_validation.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to 'shrinkage_validation.png'")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)
    print(f"Initial thickness: {thickness_initial*1000:.4f} mm")
    print(f"Final thickness: {thickness_history[-1]*1000:.4f} mm")
    print(f"Thickness reduction: {(thickness_initial - thickness_history[-1])*1000:.4f} mm")
    print(f"Thickness reduction percentage: {(1 - thickness_history[-1]/thickness_initial)*100:.2f}%")
    
    # Display lower limit information
    print(f"\nLower thickness limit (fixed): {thickness_min*1000:.4f} mm")
    print(f"  Based on initial ice mass: {m_ice_initial:.6f} kg/m²")
    print(f"  Calculation: m_ice_initial / ρ_water = {m_ice_initial:.6f} / {model.rho_water:.1f} = {thickness_min*1000:.4f} mm")
    print(f"Final thickness / Lower limit ratio: {thickness_history[-1] / thickness_min:.4f}")
    
    if simulation_stopped:
        print(f"\n⚠️  Simulation stopped early: {stop_reason}")
    print(f"\nInitial mass: {m_ice_initial:.6f} kg/m²")
    print(f"Final mass: {m_ice_history[-1]:.6f} kg/m²")
    print(f"Mass reduction: {m_ice_initial - m_ice_history[-1]:.6f} kg/m²")
    print(f"Mass reduction percentage: {(1 - m_ice_history[-1]/m_ice_initial)*100:.2f}%")
    print(f"\nInitial α_ice: {alpha_ice_initial:.6f}")
    print(f"Final α_ice (after shrinkage): {alpha_ice_after_shrinkage_history[-1]:.6f}")
    print(f"Initial α_H2O: {0.0:.6f}")
    print(f"Final α_H2O (after shrinkage): {alpha_H2O_after_shrinkage_history[-1]:.6f}")
    
    return {
        'time': time_history,
        'thickness': thickness_history,
        'alpha_ice_before': alpha_ice_history,
        'alpha_H2O_before': alpha_H2O_history,
        'alpha_ice_after': alpha_ice_after_shrinkage_history,
        'alpha_H2O_after': alpha_H2O_after_shrinkage_history,
        'mass_ice': m_ice_history,
        'shrink_rate': shrink_rate_history,
        'r_pore': r_pore_history,
        'eta_eff': eta_eff_history,
        'delta_over_eta_r': delta_over_eta_r_history
    }


if __name__ == "__main__":
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
    plt.savefig('shrinkage_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plots saved to 'shrinkage_comparison.png'")
    
    plt.show()
