"""
Parameter Sensitivity Analysis for Shrinkage Model

This script tests the impact of different parameters on the shrinkage rate:
- b: Structural constant (typically 2-4)
- C_wet: Lubricant constant
- eta_0: Base viscosity [Pa·s]

For each parameter, we vary it while keeping others constant and plot the shrinkage rate.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
from pathlib import Path
from model_init import DefrostModel
from solver import DefrostSolver

# Create figure directory if it doesn't exist
figure_dir = Path("figure")
figure_dir.mkdir(exist_ok=True)

# Global default parameters for sensitivity analysis
ETA_0_DEFAULT = 5e4  # Base viscosity [Pa·s] - change this value to update all tests

# Optimized parameters for comparison plot
# Modify these values to set your optimized case
OPTIMIZED_B = 3.0      # Structural constant
OPTIMIZED_C_WET = 700  # Lubricant constant
OPTIMIZED_ETA_0 = 2e4   # Base viscosity [Pa·s]
OPTIMIZED_D_ICE = 5e-4  # Initial ice grain diameter [m] (100 μm)


def run_shrinkage_simulation(b=None, C_wet=None, eta_0=None, d_ice_initial=None, n_steps=2000, dt=1.0):
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
    d_ice_initial : float, optional
        Initial ice grain diameter [m]. If None, uses default from solver.
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
    if d_ice_initial is not None:
        solver.d_ice_initial = d_ice_initial
        solver.d_ice_i = np.full(model.n_layers, d_ice_initial)
    
    # Initial mass of ice
    m_ice_initial = alpha_ice_initial * model.rho_ice * thickness_initial
    mass_decrease_rate = 0.005 * m_ice_initial  # 5% per second
    
    # Initialize mass per unit area
    solver.m_double_prime_ice = np.array([m_ice_initial])
    solver.m_double_prime_water = np.array([0.0])
    
    # History arrays
    time_history = []
    thickness_history = []
    shrink_rate_history = []
    alpha_water_history = []  # Track water volume fraction
    
    # Initial state
    time_history.append(0.0)
    thickness_history.append(model.dx[0])
    shrink_rate_history.append(0.0)
    alpha_water_history.append(0.0)  # Initial alpha_water is 0
    
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
        alpha_water_history.append(alpha_H2O_after)  # Store water volume fraction after shrinkage
    
    return {
        'time': np.array(time_history[:-1]),  # Exclude last time step
        'thickness': np.array(thickness_history[:-1]),
        'shrink_rate': np.array(shrink_rate_history[:-1]),
        'alpha_water': np.array(alpha_water_history[:-1])  # Water volume fraction
    }


def test_parameter_b():
    """Test the impact of parameter b (structural constant)."""
    print("=" * 70)
    print("Testing Parameter: b (structural constant)")
    print("=" * 70)
    
    # Default values
    b_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    C_wet_default = 15.0
    
    results = {}
    for b in b_values:
        print(f"Running simulation with b = {b}...")
        results[b] = run_shrinkage_simulation(b=b, C_wet=C_wet_default, eta_0=ETA_0_DEFAULT)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs alpha_water
    for b in b_values:
        axes[0].plot(results[b]['alpha_water'], results[b]['shrink_rate'], 
                   linewidth=2, label=f'b = {b}')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter b on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Thickness vs alpha_water
    for b in b_values:
        axes[1].plot(results[b]['alpha_water'], results[b]['thickness'] * 1000, 
                   linewidth=2, label=f'b = {b}')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter b on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_sensitivity_b.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{figure_path}'")
    
    return results


def test_parameter_C_wet():
    """Test the impact of parameter C_wet (lubricant constant)."""
    print("\n" + "=" * 70)
    print("Testing Parameter: C_wet (lubricant constant)")
    print("=" * 70)
    
    # Default values - extended range up to 100
    C_wet_values = [5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 75.0, 100.0]
    b_default = 3.0
    
    results = {}
    for C_wet in C_wet_values:
        print(f"Running simulation with C_wet = {C_wet}...")
        results[C_wet] = run_shrinkage_simulation(b=b_default, C_wet=C_wet, eta_0=ETA_0_DEFAULT)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs alpha_water
    for C_wet in C_wet_values:
        axes[0].plot(results[C_wet]['alpha_water'], results[C_wet]['shrink_rate'], 
                   linewidth=2, label=f'C_wet = {C_wet}')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter C_wet on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Thickness vs alpha_water
    for C_wet in C_wet_values:
        axes[1].plot(results[C_wet]['alpha_water'], results[C_wet]['thickness'] * 1000, 
                   linewidth=2, label=f'C_wet = {C_wet}')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter C_wet on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_sensitivity_C_wet.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{figure_path}'")
    
    return results


def test_parameter_eta_0():
    """Test the impact of parameter eta_0 (base viscosity)."""
    print("\n" + "=" * 70)
    print("Testing Parameter: eta_0 (base viscosity)")
    print("=" * 70)
    
    # Default values - range from 5e3 to 1e6
    eta_0_values = [5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
    b_default = 3.0
    C_wet_default = 15.0
    
    results = {}
    for eta_0 in eta_0_values:
        print(f"Running simulation with eta_0 = {eta_0:.0e} Pa·s...")
        results[eta_0] = run_shrinkage_simulation(b=b_default, C_wet=C_wet_default, eta_0=eta_0)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs alpha_water
    for eta_0 in eta_0_values:
        axes[0].plot(results[eta_0]['alpha_water'], results[eta_0]['shrink_rate'], 
                   linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter η₀ on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Thickness vs alpha_water
    for eta_0 in eta_0_values:
        axes[1].plot(results[eta_0]['alpha_water'], results[eta_0]['thickness'] * 1000, 
                   linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter η₀ on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_sensitivity_eta_0.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{figure_path}'")
    
    return results


def test_parameter_d_ice():
    """Test the impact of parameter d_ice_initial (initial ice grain diameter)."""
    print("\n" + "=" * 70)
    print("Testing Parameter: d_ice_initial (initial ice grain diameter)")
    print("=" * 70)
    
    # Default values
    d_ice_values = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]  # 50 μm, 100 μm, 200 μm, 500 μm, 1000 μm
    b_default = 3.0
    C_wet_default = 15.0
    
    results = {}
    for d_ice in d_ice_values:
        print(f"Running simulation with d_ice_initial = {d_ice*1e6:.0f} μm...")
        results[d_ice] = run_shrinkage_simulation(b=b_default, C_wet=C_wet_default, 
                                                   eta_0=ETA_0_DEFAULT, d_ice_initial=d_ice)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs alpha_water
    for d_ice in d_ice_values:
        axes[0].plot(results[d_ice]['alpha_water'], results[d_ice]['shrink_rate'], 
                   linewidth=2, label=f'd_ice = {d_ice*1e6:.0f} μm')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Impact of Parameter d_ice on Shrinkage Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Thickness vs alpha_water
    for d_ice in d_ice_values:
        axes[1].plot(results[d_ice]['alpha_water'], results[d_ice]['thickness'] * 1000, 
                   linewidth=2, label=f'd_ice = {d_ice*1e6:.0f} μm')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Impact of Parameter d_ice on Layer Thickness')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_sensitivity_d_ice.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{figure_path}'")
    
    return results


def create_optimized_comparison():
    """Create a comparison plot between default and optimized parameter settings."""
    print("\n" + "=" * 70)
    print("Creating Optimized vs Default Comparison Plot")
    print("=" * 70)
    
    # Default parameters (from solver defaults)
    print("Running simulation with DEFAULT parameters...")
    print(f"  b = {2.0} (solver default)")
    print(f"  C_wet = {15.0} (solver default)")
    print(f"  eta_0 = {1e5:.0e} Pa·s (solver default)")
    print(f"  d_ice_initial = {1e-4*1e6:.0f} μm (solver default)")
    results_default = run_shrinkage_simulation(
        b=2.0,  # Solver default
        C_wet=15.0,  # Solver default
        eta_0=1e5,  # Solver default
        d_ice_initial=1e-4  # Solver default
    )
    
    # Optimized parameters (user-defined)
    print("\nRunning simulation with OPTIMIZED parameters...")
    print(f"  b = {OPTIMIZED_B}")
    print(f"  C_wet = {OPTIMIZED_C_WET}")
    print(f"  eta_0 = {OPTIMIZED_ETA_0:.0e} Pa·s")
    print(f"  d_ice_initial = {OPTIMIZED_D_ICE*1e6:.0f} μm")
    results_optimized = run_shrinkage_simulation(
        b=OPTIMIZED_B,
        C_wet=OPTIMIZED_C_WET,
        eta_0=OPTIMIZED_ETA_0,
        d_ice_initial=OPTIMIZED_D_ICE
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Shrinkage rate vs alpha_water
    axes[0].plot(results_default['alpha_water'], results_default['shrink_rate'], 
                'b-', linewidth=2, label='Default parameters')
    axes[0].plot(results_optimized['alpha_water'], results_optimized['shrink_rate'], 
                'r--', linewidth=2, label='Optimized parameters')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Shrinkage Rate Comparison: Default vs Optimized')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Thickness vs alpha_water
    axes[1].plot(results_default['alpha_water'], results_default['thickness'] * 1000, 
                'b-', linewidth=2, label='Default parameters')
    axes[1].plot(results_optimized['alpha_water'], results_optimized['thickness'] * 1000, 
                'r--', linewidth=2, label='Optimized parameters')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Layer Thickness Comparison: Default vs Optimized')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_optimized_comparison.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"\nOptimized comparison plot saved to '{figure_path}'")
    
    return {
        'default': results_default,
        'optimized': results_optimized
    }


def create_combined_comparison():
    """Create a combined comparison plot showing all parameters."""
    print("\n" + "=" * 70)
    print("Creating Combined Comparison Plot")
    print("=" * 70)
    
    # Run simulations with different parameter values
    # b variation
    b_results = {}
    for b in [2.0, 3.0, 4.0]:
        b_results[b] = run_shrinkage_simulation(b=b, C_wet=15.0, eta_0=ETA_0_DEFAULT)
    
    # C_wet variation
    C_wet_results = {}
    for C_wet in [10.0, 25.0, 50.0, 100.0]:
        C_wet_results[C_wet] = run_shrinkage_simulation(b=3.0, C_wet=C_wet, eta_0=ETA_0_DEFAULT)
    
    # eta_0 variation
    eta_0_results = {}
    for eta_0 in [1e4, 5e4, 1e5, 5e5]:
        eta_0_results[eta_0] = run_shrinkage_simulation(b=3.0, C_wet=15.0, eta_0=eta_0)
    
    # d_ice variation
    d_ice_results = {}
    for d_ice in [5e-5, 1e-4, 2e-4]:
        d_ice_results[d_ice] = run_shrinkage_simulation(b=3.0, C_wet=15.0, eta_0=ETA_0_DEFAULT, d_ice_initial=d_ice)
    
    # Create combined plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Plot 1: b parameter
    for b in [2.0, 3.0, 4.0]:
        axes[0].plot(b_results[b]['alpha_water'], b_results[b]['shrink_rate'], 
                     linewidth=2, label=f'b = {b}')
    axes[0].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[0].set_ylabel('Shrinkage Rate (mm/s)')
    axes[0].set_title('Parameter b Impact (C_wet=15.0, η₀=1e7 Pa·s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    
    # Plot 2: C_wet parameter
    for C_wet in [10.0, 25.0, 50.0, 100.0]:
        axes[1].plot(C_wet_results[C_wet]['alpha_water'], C_wet_results[C_wet]['shrink_rate'], 
                     linewidth=2, label=f'C_wet = {C_wet}')
    axes[1].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[1].set_ylabel('Shrinkage Rate (mm/s)')
    axes[1].set_title(f'Parameter C_wet Impact (b=3.0, η₀={ETA_0_DEFAULT:.0e} Pa·s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    
    # Plot 3: eta_0 parameter
    for eta_0 in [1e4, 5e4, 1e5, 5e5]:
        axes[2].plot(eta_0_results[eta_0]['alpha_water'], eta_0_results[eta_0]['shrink_rate'], 
                     linewidth=2, label=f'η₀ = {eta_0:.0e} Pa·s')
    axes[2].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[2].set_ylabel('Shrinkage Rate (mm/s)')
    axes[2].set_title('Parameter η₀ Impact (b=3.0, C_wet=15.0)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim([0, 1])
    
    # Plot 4: d_ice parameter
    for d_ice in [5e-5, 1e-4, 2e-4]:
        axes[3].plot(d_ice_results[d_ice]['alpha_water'], d_ice_results[d_ice]['shrink_rate'], 
                     linewidth=2, label=f'd_ice = {d_ice*1e6:.0f} μm')
    axes[3].set_xlabel('Water Volume Fraction (α_H2O)')
    axes[3].set_ylabel('Shrinkage Rate (mm/s)')
    axes[3].set_title(f'Parameter d_ice Impact (b=3.0, C_wet=15.0, η₀={ETA_0_DEFAULT:.0e} Pa·s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].set_xlim([0, 1])
    
    plt.tight_layout()
    figure_path = figure_dir / 'parameter_sensitivity_combined.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"Combined comparison saved to '{figure_path}'")


if __name__ == "__main__":
    # Run all parameter sensitivity tests
    results_b = test_parameter_b()
    results_C_wet = test_parameter_C_wet()
    results_eta_0 = test_parameter_eta_0()
    results_d_ice = test_parameter_d_ice()
    
    # Create combined comparison
    create_combined_comparison()
    
    # Create optimized vs default comparison
    create_optimized_comparison()
    
    print("\n" + "=" * 70)
    print("Parameter Sensitivity Analysis Complete!")
    print("=" * 70)
    print("\nGenerated plots:")
    print(f"  - {figure_dir / 'parameter_sensitivity_b.png'}")
    print(f"  - {figure_dir / 'parameter_sensitivity_C_wet.png'}")
    print(f"  - {figure_dir / 'parameter_sensitivity_eta_0.png'}")
    print(f"  - {figure_dir / 'parameter_sensitivity_d_ice.png'}")
    print(f"  - {figure_dir / 'parameter_sensitivity_combined.png'}")
    print(f"  - {figure_dir / 'parameter_optimized_comparison.png'}")
    
    # plt.show()  # Commented out - using non-interactive backend (Agg)
