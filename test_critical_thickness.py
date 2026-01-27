"""
Test script for critical detachment frost thickness calculation.

Tests the critical thickness calculation for different contact angles
and water volume fractions before integrating into the solver.
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_base_adhesion(theta_deg):
    """
    Calculate base adhesion from wettability (contact angle).
    
    τ_base = 73 × 10⁻³ ⋅ (1 + cos θ) [N/m²]
    
    Parameters
    ----------
    theta_deg : float
        Contact angle [degrees]
    
    Returns
    -------
    float
        Base adhesion [N/m²]
    """
    theta_rad = np.deg2rad(theta_deg)
    tau_0 = 73e-3  # [N/m²]
    tau_base = tau_0 * (1 + np.cos(theta_rad))
    return tau_base


def calculate_f_water(alpha_water, A=-0.1, B=50, e=0.05, C=0.4):
    """
    Calculate f(alpha_water) using logarithmic form.
    
    f(alpha_water) = A * log(B * alpha_water + e) + C
    
    Parameters
    ----------
    alpha_water : float
        Water volume fraction (alpha_water) [-]
    A : float, optional
        Multiplier for log (should be negative). Default: -0.1 (will be calibrated)
    B : float, optional
        Multiplier for alpha_water. Default: 50 (will be calibrated)
    e : float, optional
        Offset to prevent singularity. Default: 0.05 (will be calibrated)
    C : float, optional
        Plateau/base value. Default: 0.4 (will be calibrated)
    
    Returns
    -------
    float
        f(alpha_water) value [-]
    """
    # Ensure alpha_water is in valid range [0, 1]
    alpha_water = np.clip(alpha_water, 0.0, 1.0)
    
    # Calculate argument of log
    arg = B * alpha_water + e
    
    # Ensure argument is positive and > 0
    if arg <= 0:
        arg = 1e-10  # Small positive value
    
    # Calculate f(alpha_water) = A * log(B * alpha_water + e) + C
    log_val = np.log(arg)
    
    f_value = A * log_val + C
    
    return f_value


def calculate_critical_thickness_base(tau_base, k=1000.0, rho_eff=None, g=9.81):
    """
    Calculate base critical detachment frost thickness (without f(alpha_water)).
    
    h_crit_base = k ⋅ τ_base / (ρ_eff ⋅ g)
    
    This is the base value before multiplying by f(alpha_water).
    
    Parameters
    ----------
    tau_base : float
        Base adhesion [N/m²]
    k : float, optional
        Retention/adhesion coefficient. Default: 1.0
    rho_eff : float, optional
        Effective density of frost [kg/m³]. If None, returns base value without density.
        Default: None
    g : float, optional
        Gravitational acceleration [m/s²]. Default: 9.81
    
    Returns
    -------
    float
        Base critical detachment thickness [m] or base value [N/m²] if rho_eff is None
    """
    if rho_eff is not None:
        if rho_eff > 0 and g > 0:
            h_crit_base = (k * tau_base) / (rho_eff * g)
        else:
            h_crit_base = np.inf  # Invalid case
    else:
        # Return just k * tau_base (without dividing by rho_eff * g)
        h_crit_base = k * tau_base
    
    return h_crit_base


def calculate_critical_thickness(tau_base, f_water, rho_eff, k=1.0, g=9.81):
    """
    Calculate critical detachment frost thickness.
    
    h_crit = (k ⋅ τ_base ⋅ f(alpha_water)) / (ρ_eff ⋅ g)
    
    Where τ_base = τ_0 ⋅ (1 + cos θ)
    
    Parameters
    ----------
    tau_base : float
        Base adhesion [N/m²]
    f_water : float
        f(alpha_water) function value [-]
    rho_eff : float
        Effective density of frost [kg/m³]
    k : float, optional
        Retention/adhesion coefficient. Default: 1.0
    g : float, optional
        Gravitational acceleration [m/s²]. Default: 9.81
    
    Returns
    -------
    float
        Critical detachment thickness [m]
    """
    if rho_eff > 0 and g > 0:
        h_crit = (k * tau_base * f_water) / (rho_eff * g)
    else:
        h_crit = np.inf  # Invalid case
    
    return h_crit


def calculate_effective_density(alpha_ice, alpha_water, dx, begin_idx, end_idx,
                                rho_ice=917.0, rho_water=1000.0, rho_air=1.2):
    """
    Calculate effective density from the first layer containing frost (alpha_ice > 0)
    counting from the wall side (end_idx) till end_idx.
    
    ρ_eff = m_total / h_total
    
    Where:
    - m_total = total mass (ice + water + air) per unit area [kg/m²]
    - h_total = total thickness from first frost layer to end_idx [m]
    
    The calculation starts from end_idx (wall side) and goes backwards to find
    the first layer with alpha_ice > 0, then calculates density from that layer
    to end_idx (inclusive).
    
    Parameters
    ----------
    alpha_ice : array-like
        Ice volume fraction for each layer [-]
    alpha_water : array-like
        Water volume fraction for each layer [-]
    dx : array-like
        Layer thickness for each layer [m]
    begin_idx : int
        Starting index of active layers (surface side)
    end_idx : int
        Ending index of active layers (wall side)
    rho_ice : float, optional
        Ice density [kg/m³]. Default: 917.0
    rho_water : float, optional
        Water density [kg/m³]. Default: 1000.0
    rho_air : float, optional
        Air density [kg/m³]. Default: 1.2
    
    Returns
    -------
    float
        Effective density [kg/m³]
    """
    # Convert to numpy arrays if needed
    alpha_ice = np.asarray(alpha_ice)
    alpha_water = np.asarray(alpha_water)
    dx = np.asarray(dx)
    
    # Find the first layer with frost (alpha_ice > 0) starting from wall side (end_idx)
    # going backwards towards begin_idx
    first_frost_idx = None
    for i in range(end_idx, begin_idx - 1, -1):
        if alpha_ice[i] > 1e-10:  # Layer contains ice
            first_frost_idx = i
            break
    
    # If no frost layer found, return 0.0
    if first_frost_idx is None:
        return 0.0
    
    # Calculate masses and thickness from first_frost_idx to end_idx (inclusive)
    # Range: [first_frost_idx, end_idx]
    layer_range = range(first_frost_idx, end_idx + 1)
    
    # Calculate masses per unit area for each layer in range
    m_ice_layers = alpha_ice[layer_range] * rho_ice * dx[layer_range]
    m_water_layers = alpha_water[layer_range] * rho_water * dx[layer_range]
    
    # Total masses
    m_ice_total = np.sum(m_ice_layers)
    m_water_total = np.sum(m_water_layers)
    
    # Total thickness
    h_total = np.sum(dx[layer_range])
    
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
        m_total = m_ice_total + m_water_total + m_air_total
        
        # Effective density
        rho_eff = m_total / h_total
    else:
        rho_eff = 0.0
    
    return rho_eff


if __name__ == "__main__":
    print("=" * 70)
    print("Critical Detachment Frost Thickness Calculation")
    print("=" * 70)
    
    # Test parameters
    contact_angles = [60, 160]  # degrees
    alpha_water_values = np.linspace(0.0, 1.0, 21)  # 0 to 1 in steps of 0.05
    g = 9.81  # Gravitational acceleration [m/s²]
    
    # Test cases: different frost densities
    test_cases = {
        'low_density': {
            'name': 'Low Density Frost',
            'alpha_ice': 0.1,  # ~0.1 (porosity ~0.9)
            'description': 'alpha_ice ≈ 0.1 (porosity ~0.9)'
        },
        'high_density': {
            'name': 'Dense Frost',
            'alpha_ice': 0.35,  # 0.35 (porosity ~0.65)
            'description': 'alpha_ice = 0.35 (porosity ~0.65)'
        }
    }
    
    # Create dummy layer arrays for testing
    h_total = 0.005  # 5 mm (total frost thickness)
    n_layers = 10
    dx_test = np.full(n_layers, h_total / n_layers)  # Uniform layer thickness
    
    begin_idx = 0
    end_idx = n_layers - 1
    
    # Calculate effective densities for each test case
    case_results = {}
    
    for case_key, case_info in test_cases.items():
        print(f"\n{'='*70}")
        print(f"Test Case: {case_info['name']} ({case_info['description']})")
        print(f"{'='*70}")
        
        # Create layer arrays for this case
        alpha_ice_test = np.zeros(n_layers)
        alpha_water_test = np.zeros(n_layers)
        
        # Set alpha_ice for all layers
        alpha_ice_test[:] = case_info['alpha_ice']
        alpha_water_test[:] = 0.0  # No water initially
        
        # Calculate effective density
        rho_eff = calculate_effective_density(alpha_ice_test, alpha_water_test, dx_test, 
                                              begin_idx, end_idx)
        
        # Calculate total masses for display
        m_ice_total = np.sum(alpha_ice_test * 917.0 * dx_test)
        m_water_total = np.sum(alpha_water_test * 1000.0 * dx_test)
        
        print(f"  Total ice mass: {m_ice_total:.3f} kg/m²")
        print(f"  Total water mass: {m_water_total:.3f} kg/m²")
        print(f"  Total frost thickness: {h_total*1000:.2f} mm")
        print(f"  Ice volume fraction: {case_info['alpha_ice']:.2f}")
        print(f"  Effective density: {rho_eff:.1f} kg/m³")
        
        case_results[case_key] = {
            'name': case_info['name'],
            'alpha_ice': case_info['alpha_ice'],
            'rho_eff': rho_eff,
            'results': {}
        }
    
    # Calibrated k values (from calibrate_critical_thickness.py)
    k_60 = 245.2220
    k_160 = 3340.7267
    
    # Calibrated f(alpha_water) parameters
    A_calibrated = -0.051472
    B_calibrated = 160.4478
    e_calibrated = 0.903741
    C_calibrated = 0.3466
    
    # Calculate results for each test case
    for case_key, case_data in case_results.items():
        rho_eff = case_data['rho_eff']
        print(f"\n{'='*70}")
        print(f"Calculating for {case_data['name']} (rho_eff = {rho_eff:.1f} kg/m³)")
        print(f"{'='*70}")
        
        for theta_deg in contact_angles:
            print(f"\n{'='*70}")
            print(f"Contact Angle: θ = {theta_deg}°")
            print(f"{'='*70}")
            
            # Select k based on contact angle
            if theta_deg == 60:
                k = k_60
            elif theta_deg == 160:
                k = k_160
            else:
                # Default to average if other angle
                k = (k_60 + k_160) / 2
            
            # Calculate base adhesion
            tau_base = calculate_base_adhesion(theta_deg)
            print(f"\nBase adhesion (τ_base): {tau_base*1000:.3f} mN/m²")
            print(f"Calibrated k value: {k:.4f}")
            
            # Calculate base critical thickness (without f(alpha_water))
            h_crit_base = calculate_critical_thickness_base(tau_base, k, rho_eff=rho_eff, g=g)
            print(f"Base critical thickness (without f(alpha_water)): {h_crit_base*1000:.4f} mm")
            print(f"  Formula: h_crit_base = k ⋅ τ_base / (ρ_eff ⋅ g)")
            print(f"  = {k:.4f} ⋅ {tau_base:.6f} / ({rho_eff:.1f} ⋅ {g:.2f}) = {h_crit_base*1000:.4f} mm")
            
            # Calculate for different alpha_water values
            h_crit_values = []
            f_water_values = []
            
            print(f"\n{'α_water':<12} {'f(alpha_water)':<18} {'h_crit (mm)':<15} {'h_crit/f (mm)':<15}")
            print("-" * 65)
            
            for alpha_water in alpha_water_values:
                f_water = calculate_f_water(alpha_water, A_calibrated, B_calibrated, e_calibrated, C_calibrated)
                h_crit = calculate_critical_thickness(tau_base, f_water, rho_eff, k, g)
                
                # Calculate h_crit/f (base value without f(alpha_water))
                h_crit_over_f = h_crit / f_water if f_water > 0 else np.inf
                
                h_crit_values.append(h_crit)
                f_water_values.append(f_water)
                
                # Print every 5th value to avoid clutter
                if len(alpha_water_values) <= 21 or np.isclose(alpha_water % 0.1, 0.0) or alpha_water == 0.0 or alpha_water == 1.0:
                    print(f"{alpha_water:<12.2f} {f_water:<18.4f} {h_crit*1000:<15.4f} {h_crit_over_f*1000:<15.4f}")
            
            # Store results after processing all alpha_water values
            case_data['results'][theta_deg] = {
                'alpha_water': alpha_water_values,
                'f_water': np.array(f_water_values),
                'h_crit': np.array(h_crit_values),
                'tau_base': tau_base
            }
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Colors and styles for different cases
    case_styles = {
        'low_density': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        'high_density': {'color': 'red', 'linestyle': '--', 'marker': 's'}
    }
    
    # Plot 1: f(alpha_water) vs α_water (same for all cases)
    for case_key, case_data in case_results.items():
        style = case_styles[case_key]
        results = case_data['results'][60]
        axes[0, 0].plot(alpha_water_values, results['f_water'], 
                       color=style['color'], linestyle=style['linestyle'], 
                       linewidth=2, label=f"θ=60° ({case_data['name']})")
    axes[0, 0].set_xlabel('Water Volume Fraction (α_water)')
    axes[0, 0].set_ylabel('f(alpha_water)')
    axes[0, 0].set_title('f(alpha_water) vs Water Volume Fraction')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xlim([0, 1])
    
    # Plot 2: h_crit vs α_water for θ=60° (compare densities)
    for case_key, case_data in case_results.items():
        style = case_styles[case_key]
        results = case_data['results'][60]
        axes[0, 1].plot(alpha_water_values, results['h_crit']*1000, 
                       color=style['color'], linestyle=style['linestyle'], 
                       linewidth=2, marker=style['marker'], markersize=4,
                       label=f"{case_data['name']} (ρ={case_data['rho_eff']:.0f} kg/m³)")
    axes[0, 1].set_xlabel('Water Volume Fraction (α_water)')
    axes[0, 1].set_ylabel('Critical Thickness (mm)')
    axes[0, 1].set_title('Critical Thickness vs α_water (θ = 60°)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim([0, 1])
    
    # Plot 3: h_crit vs α_water for θ=160° (compare densities)
    for case_key, case_data in case_results.items():
        style = case_styles[case_key]
        results = case_data['results'][160]
        axes[0, 2].plot(alpha_water_values, results['h_crit']*1000, 
                       color=style['color'], linestyle=style['linestyle'], 
                       linewidth=2, marker=style['marker'], markersize=4,
                       label=f"{case_data['name']} (ρ={case_data['rho_eff']:.0f} kg/m³)")
    axes[0, 2].set_xlabel('Water Volume Fraction (α_water)')
    axes[0, 2].set_ylabel('Critical Thickness (mm)')
    axes[0, 2].set_title('Critical Thickness vs α_water (θ = 160°)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_xlim([0, 1])
    
    # Plot 4: Base adhesion comparison
    tau_60 = case_results['low_density']['results'][60]['tau_base']
    tau_160 = case_results['low_density']['results'][160]['tau_base']
    axes[1, 0].bar(['θ = 60°', 'θ = 160°'], [tau_60*1000, tau_160*1000], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Base Adhesion (mN/m²)')
    axes[1, 0].set_title('Base Adhesion Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: h_crit ratio (160° / 60°) for different densities
    for case_key, case_data in case_results.items():
        style = case_styles[case_key]
        results_60 = case_data['results'][60]
        results_160 = case_data['results'][160]
        ratio = results_160['h_crit'] / results_60['h_crit']
        axes[1, 1].plot(alpha_water_values, ratio, 
                       color=style['color'], linestyle=style['linestyle'], 
                       linewidth=2, marker=style['marker'], markersize=4,
                       label=case_data['name'])
    axes[1, 1].set_xlabel('Water Volume Fraction (α_water)')
    axes[1, 1].set_ylabel('h_crit(160°) / h_crit(60°)')
    axes[1, 1].set_title('Critical Thickness Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_xlim([0, 1])
    
    # Plot 6: Density comparison (h_crit ratio between densities)
    for theta_deg in contact_angles:
        results_low = case_results['low_density']['results'][theta_deg]
        results_high = case_results['high_density']['results'][theta_deg]
        ratio = results_high['h_crit'] / results_low['h_crit']
        color = 'blue' if theta_deg == 60 else 'red'
        axes[1, 2].plot(alpha_water_values, ratio, 
                       color=color, linewidth=2, 
                       label=f'θ = {theta_deg}°')
    axes[1, 2].set_xlabel('Water Volume Fraction (α_water)')
    axes[1, 2].set_ylabel('h_crit(Dense) / h_crit(Low Density)')
    axes[1, 2].set_title('Density Effect on Critical Thickness')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('critical_thickness_test.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print("Plots saved to 'critical_thickness_test.png'")
    print(f"{'='*70}")
    
    plt.show()
