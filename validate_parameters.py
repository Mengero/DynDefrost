"""
Parameter Sensitivity Analysis for Shrinkage Model

This script tests the impact of different parameters on the shrinkage rate:
- b: Structural constant (typically 2-4)
- C_wet: Lubricant constant
- eta_0: Base viscosity [Pa·s]

For each parameter, we vary it while keeping others constant and plot the shrinkage rate.
"""

import numpy as np
import matplotlib.pyplot as plt
from model_init import DefrostModel
from solver import DefrostSolver


def run_shrinkage_simulation(b=None, C_wet=None, eta_0=None, n_steps=20, dt=1.0):
    """
    Run a shrinkage simulation with specified parameters.
    
    Parameters
    ----------
    b : float, optional
        Structural constant. If None, uses default from solver.
    C_wet : float, optional
        Lubricant constant. If None, uses default from solver.
    eta_0 : float, optional
        Base viscosity. If None, uses default from solver.
    n_steps : int
        Number of time steps
    dt : float
        Time step size [s]
    
    Returns
    -------
    dict
        Dictionary with time, thickness, and shrinkage rate histories
    """
    # Initial conditions
    alpha_ice_initial = 0.2
    thickness_initial = 4e-3  # 4 mm
    porosity = 1.0 - alpha_ice_initial  # porosity = 0.8
    
    # Initialize model
    model = DefrostModel(
        n_layers=1,
        frost_thickness=thickness_initial,
        porosity=porosity
    )
    
    # Set initial temperature
    model.T = np.full(model.n_layers, -20.0)  # -20°C
    
    # Initialize solver
    solver = DefrostSolver(model, dt=dt, method='explicit')
    
    # Override parameters if specified
    if b is not None:
        solver.b = b
    if C_wet is not None:
        solver.C_wet = C_wet
    if eta_0 is not None:
        solver.eta_0 = eta_0
    
    # Initial mass of ice
    m_ice_initial = alpha_ice_initial * model.rho_ice * thickness_initial
    mass_decrease_rate = 0.05 * m_ice_initial  # 5% per second
    
    # Initialize mass per unit area
    solver.m_double_prime_ice = np.array([m_ice_initial])
    solver.m_double_prime_water = np.array([0.0])
    
    # History arrays
    time_history = []
    thickness_history = []
    shrink_rate_history = []
    
    # Initial state
    time_history.append(0.0)
    thickness_history.append(model.dx[0])
    shrink_rate_history.append(0.0)
    
    # Time stepping
    for step in range(1, n_steps + 1):
        t = step * dt
        
        # Step 1: Decrease ice mass
        m_ice_old = solver.m_double_prime_ice[0]
        delta_m_ice = -mass_decrease_rate * dt
        m_ice_new = m_ice_old + delta_m_ice
        m_ice_new = np.maximum(m_ice_new, 0.0)
        
        # Update water mass (melted ice becomes water)
        m_water_old = solver.m_double_prime_water[0]
        m_water_new = m_water_old - delta_m_ice  # delta_m_ice is negative
        
        solver.m_double_prime_ice[0] = m_ice_new
        solver.m_double_prime_water[0] = m_water_new
        
        # Step 2: Update volume fractions based on current thickness
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
        model.alpha_air[0] = 1.0 - alpha_ice_before - alpha_H2O_before
        model.alpha_air[0] = np.maximum(model.alpha_air[0], 0.0)
        
        # Step 3: Calculate shrinkage
        thickness_before = current_thickness
        model.calculate_specific_heat()
        model.calculate_thermal_conductivity()
        solver._calculate_layer_shrinkage()
        thickness_after = model.dx[0]
        
        # Calculate shrinkage rate (mm/s)
        shrink_rate = (thickness_before - thickness_after) / dt * 1000
        
        # Step 4: Re-update volume fractions based on new thickness
        if thickness_after > 0:
            alpha_ice_after = m_ice_new / (model.rho_ice * thickness_after)
            alpha_H2O_after = m_water_new / (model.rho_water * thickness_after)
        else:
            alpha_ice_after = 0.0
            alpha_H2O_after = 0.0
        
        # Store results
        time_history.append(t)
        thickness_history.append(thickness_after)
        shrink_rate_history.append(shrink_rate)
    
    return {
        'time': np.array(time_history[:-1]),  # Exclude last time step
        'thickness': np.array(thickness_history[:-1]),
        'shrink_rate': np.array(shrink_rate_history[:-1])
    }


def test_parameter_b():
    """Test the impact of parameter b (structural constant)."""
    print("=" * 70)
    print("Testing Parameter: b (structural constant)")
    print("=" * 70)
    
    # Default values
    b_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    C_wet_default = 15.0
    eta_0_default = 1e7
    
    results = {}
    for b in b_values:
        print(f"Running simulation with b = {b}...")
        results[b] = run_shrinkage_simulation(b=b, C_wet=C_wet_default, eta_0=eta_0_default)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs time
    for b in b_values:
        axes[0].plot(results[b]['time'], results[b]['shrink_rate'], 
                   linewidth=2, label=f'b = {b}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter b on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Thickness vs time
    for b in b_values:
        axes[1].plot(results[b]['time'], results[b]['thickness'] * 1000, 
                   linewidth=2, label=f'b = {b}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter b on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_b.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'parameter_sensitivity_b.png'")
    
    return results


def test_parameter_C_wet():
    """Test the impact of parameter C_wet (lubricant constant)."""
    print("\n" + "=" * 70)
    print("Testing Parameter: C_wet (lubricant constant)")
    print("=" * 70)
    
    # Default values
    C_wet_values = [5.0, 10.0, 15.0, 20.0, 25.0]
    b_default = 3.0
    eta_0_default = 1e7
    
    results = {}
    for C_wet in C_wet_values:
        print(f"Running simulation with C_wet = {C_wet}...")
        results[C_wet] = run_shrinkage_simulation(b=b_default, C_wet=C_wet, eta_0=eta_0_default)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs time
    for C_wet in C_wet_values:
        axes[0].plot(results[C_wet]['time'], results[C_wet]['shrink_rate'], 
                   linewidth=2, label=f'C_wet = {C_wet}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter C_wet on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Thickness vs time
    for C_wet in C_wet_values:
        axes[1].plot(results[C_wet]['time'], results[C_wet]['thickness'] * 1000, 
                   linewidth=2, label=f'C_wet = {C_wet}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter C_wet on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_C_wet.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'parameter_sensitivity_C_wet.png'")
    
    return results


def test_parameter_eta_0():
    """Test the impact of parameter eta_0 (base viscosity)."""
    print("\n" + "=" * 70)
    print("Testing Parameter: eta_0 (base viscosity)")
    print("=" * 70)
    
    # Default values
    eta_0_values = [1e6, 5e6, 1e7, 5e7, 1e8]
    b_default = 3.0
    C_wet_default = 15.0
    
    results = {}
    for eta_0 in eta_0_values:
        print(f"Running simulation with eta_0 = {eta_0:.0e} Pa·s...")
        results[eta_0] = run_shrinkage_simulation(b=b_default, C_wet=C_wet_default, eta_0=eta_0)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs time
    for eta_0 in eta_0_values:
        axes[0].plot(results[eta_0]['time'], results[eta_0]['shrink_rate'], 
                   linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter η₀ on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Thickness vs time
    for eta_0 in eta_0_values:
        axes[1].plot(results[eta_0]['time'], results[eta_0]['thickness'] * 1000, 
                   linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter η₀ on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_eta_0.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'parameter_sensitivity_eta_0.png'")
    
    return results


def create_combined_comparison():
    """Create a combined comparison plot showing all parameters."""
    print("\n" + "=" * 70)
    print("Creating Combined Comparison Plot")
    print("=" * 70)
    
    # Run simulations with different parameter values
    # b variation
    b_results = {}
    for b in [2.0, 3.0, 4.0]:
        b_results[b] = run_shrinkage_simulation(b=b, C_wet=15.0, eta_0=1e7)
    
    # C_wet variation
    C_wet_results = {}
    for C_wet in [10.0, 15.0, 20.0]:
        C_wet_results[C_wet] = run_shrinkage_simulation(b=3.0, C_wet=C_wet, eta_0=1e7)
    
    # eta_0 variation
    eta_0_results = {}
    for eta_0 in [5e6, 1e7, 5e7]:
        eta_0_results[eta_0] = run_shrinkage_simulation(b=3.0, C_wet=15.0, eta_0=eta_0)
    
    # Create combined plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: b parameter
    for b in [2.0, 3.0, 4.0]:
        axes[0].plot(b_results[b]['time'], b_results[b]['shrink_rate'], 
                     linewidth=2, label=f'b = {b}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Parameter b Impact (C_wet=15.0, η₀=1e7 Pa·s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: C_wet parameter
    for C_wet in [10.0, 15.0, 20.0]:
        axes[1].plot(C_wet_results[C_wet]['time'], C_wet_results[C_wet]['shrink_rate'], 
                     linewidth=2, label=f'C_wet = {C_wet}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Shrinkage Rate (mm/s)')
    axes[1].set_title('Parameter C_wet Impact (b=3.0, η₀=1e7 Pa·s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: eta_0 parameter
    for eta_0 in [5e6, 1e7, 5e7]:
        axes[2].plot(eta_0_results[eta_0]['time'], eta_0_results[eta_0]['shrink_rate'], 
                     linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Shrinkage Rate (mm/s)')
    axes[2].set_title('Parameter η₀ Impact (b=3.0, C_wet=15.0)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_combined.png', dpi=150, bbox_inches='tight')
    print("Combined comparison saved to 'parameter_sensitivity_combined.png'")


if __name__ == "__main__":
    # Run all parameter sensitivity tests
    results_b = test_parameter_b()
    results_C_wet = test_parameter_C_wet()
    results_eta_0 = test_parameter_eta_0()
    
    # Create combined comparison
    create_combined_comparison()
    
    print("\n" + "=" * 70)
    print("Parameter Sensitivity Analysis Complete!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - parameter_sensitivity_b.png")
    print("  - parameter_sensitivity_C_wet.png")
    print("  - parameter_sensitivity_eta_0.png")
    print("  - parameter_sensitivity_combined.png")
    
    plt.show()
