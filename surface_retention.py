"""
Surface Water Retention Calculator

This module calculates the maximum surface water retention based on contact angle
hysteresis (receding and advancing contact angles).

Based on the retention force model:
Γ = (k ⋅ γ_LV ⋅ (cos θ_R - cos θ_A)) / (δ ⋅ g)

Where:
- Γ = surface water retention [kg/m²]
- k = retention coefficient (depends on surface type)
- γ_LV = liquid-vapor surface tension [N/m]
- θ_R = receding contact angle [degrees]
- θ_A = advancing contact angle [degrees]
- δ = typical droplet spacing [m]
- g = gravitational acceleration [m/s²]
"""

import numpy as np


def calculate_surface_retention(theta_receding, theta_advancing, 
                                gamma_LV=72e-3, g=9.81, delta=None, k=None):
    """
    Calculate maximum surface water retention based on contact angle hysteresis.
    
    Parameters
    ----------
    theta_receding : float
        Receding contact angle [degrees]
    theta_advancing : float
        Advancing contact angle [degrees]
    gamma_LV : float, optional
        Liquid-vapor surface tension [N/m]. Default: 72e-3 N/m (water-air at 20°C)
    g : float, optional
        Gravitational acceleration [m/s²]. Default: 9.81 m/s²
    delta : float, optional
        Typical droplet spacing [m]. If None, will be determined based on surface type.
        Default: None
    k : float, optional
        Retention coefficient. If None, will be determined based on surface type.
        Default: None
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'retention': Surface water retention [kg/m²]
        - 'thickness': Maximum retention water layer thickness [m]
        - 'surface_type': Classification of surface type (str)
        - 'k': Retention coefficient used
        - 'delta': Droplet spacing used [m]
        - 'hysteresis': Contact angle hysteresis [degrees]
        - 'theta_eq': Estimated equilibrium contact angle [degrees]
    """
    # Calculate hysteresis
    hysteresis = theta_advancing - theta_receding
    
    # Estimate equilibrium contact angle (average of advancing and receding)
    theta_eq = (theta_advancing + theta_receding) / 2.0
    
    # Determine surface type and parameters if not provided
    if k is None or delta is None:
        surface_type, k_auto, delta_auto = _determine_surface_parameters(
            theta_advancing, theta_receding, hysteresis
        )
        if k is None:
            k = k_auto
        if delta is None:
            delta = delta_auto
    else:
        # If both k and delta are provided, classify surface type based on angles
        if theta_advancing > 120:
            surface_type = "superhydrophobic"
        elif theta_advancing < 80:
            surface_type = "hydrophilic"
        else:
            surface_type = "hydrophobic"
    
    # Convert angles to radians for cosine calculation
    theta_R_rad = np.deg2rad(theta_receding)
    theta_A_rad = np.deg2rad(theta_advancing)
    
    # Calculate retention: Γ = (k ⋅ γ_LV ⋅ (cos θ_R - cos θ_A)) / (δ ⋅ g)
    # Note: cos θ_R - cos θ_A will be positive since θ_R < θ_A typically
    cos_diff = np.cos(theta_R_rad) - np.cos(theta_A_rad)
    
    # Calculate retention in N/m² (force per unit area)
    retention_force = (k * gamma_LV * cos_diff) / (delta * g)
    
    # Convert to kg/m² (mass per unit area)
    # Since F = m*g, we have m = F/g, but F here is already per unit area
    # So retention_force is in N/m² = kg/(m·s²), dividing by g gives kg/m²
    retention = retention_force / g  # [kg/m²]
    
    # Calculate maximum retention water layer thickness
    # thickness = retention (kg/m²) / density_water (kg/m³) = meters
    rho_water = 1000.0  # Water density [kg/m³] at 20°C
    thickness = retention / rho_water  # [m]
    
    return {
        'retention': retention,
        'thickness': thickness,
        'surface_type': surface_type,
        'k': k,
        'delta': delta,
        'hysteresis': hysteresis,
        'theta_eq': theta_eq,
        'theta_advancing': theta_advancing,
        'theta_receding': theta_receding
    }


def _determine_surface_parameters(theta_advancing, theta_receding, hysteresis):
    """
    Determine surface type and corresponding parameters (k and δ).
    
    Parameters
    ----------
    theta_advancing : float
        Advancing contact angle [degrees]
    theta_receding : float
        Receding contact angle [degrees]
    hysteresis : float
        Contact angle hysteresis [degrees]
    
    Returns
    -------
    tuple
        (surface_type, k, delta)
        - surface_type: str, classification
        - k: float, retention coefficient
        - delta: float, droplet spacing [m]
    """
    # Superhydrophobic surface: θ > 120°
    if theta_advancing > 120:
        # Small hysteresis (Δθ ≈ 5-10°)
        # k ≈ 0.5-0.6 (θ_eq-dominated)
        # δ ≈ 5 mm
        if hysteresis < 15:
            k = 0.5  # Minimum retention case
        else:
            k = 0.6  # Slightly higher retention
        delta = 5e-3  # 5 mm
        surface_type = "superhydrophobic"
    
    # Hydrophilic surface: θ ≈ 40-80°
    elif theta_advancing < 80:
        # Large hysteresis (Δθ = 20-40°)
        # k ≈ 0.8-0.9 (θ_A, θ_R-dominated)
        # δ ≈ 3-5 mm
        if hysteresis > 30:
            k = 0.9  # Maximum retention case
        else:
            k = 0.85  # Hydrophobic retention
        delta = 4e-3  # 4 mm (average of 3-5 mm)
        surface_type = "hydrophilic"
    
    # Hydrophobic surface: 80° < θ < 150°
    else:
        # Intermediate case
        # Interpolate k between hydrophilic and superhydrophobic values
        # Based on contact angle: closer to 80° → more hydrophilic, closer to 150° → more superhydrophobic
        if theta_advancing < 115:
            # Closer to hydrophilic
            k = 0.75
            delta = 4e-3
        else:
            # Closer to superhydrophobic
            k = 0.65
            delta = 4.5e-3
        
        surface_type = "hydrophobic"
    
    return surface_type, k, delta


def calculate_retention_range(theta_receding, theta_advancing, 
                              gamma_LV=72e-3, g=9.81):
    """
    Calculate retention range for a given contact angle pair.
    
    This function calculates retention using both minimum and maximum
    parameter values to give a range.
    
    Parameters
    ----------
    theta_receding : float
        Receding contact angle [degrees]
    theta_advancing : float
        Advancing contact angle [degrees]
    gamma_LV : float, optional
        Liquid-vapor surface tension [N/m]. Default: 72e-3 N/m
    g : float, optional
        Gravitational acceleration [m/s²]. Default: 9.81 m/s²
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'min': Minimum retention [kg/m²]
        - 'max': Maximum retention [kg/m²]
        - 'typical': Typical retention [kg/m²]
        - 'surface_type': Surface type classification
    """
    # Get typical values
    result_typical = calculate_surface_retention(
        theta_receding, theta_advancing, gamma_LV, g
    )
    
    surface_type = result_typical['surface_type']
    
    # Calculate range based on surface type
    if surface_type == "superhydrophobic":
        # k: 0.5-0.6, delta: 5 mm
        result_min = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=5e-3, k=0.5
        )
        result_max = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=5e-3, k=0.6
        )
    elif surface_type == "hydrophilic":
        # k: 0.8-0.9, delta: 3-5 mm
        result_min = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=5e-3, k=0.8  # Larger delta, smaller k → lower retention
        )
        result_max = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=3e-3, k=0.9  # Smaller delta, larger k → higher retention
        )
    else:  # hydrophobic
        result_min = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=4.5e-3, k=0.65
        )
        result_max = calculate_surface_retention(
            theta_receding, theta_advancing, gamma_LV, g, 
            delta=4e-3, k=0.75
        )
    
    # Calculate thicknesses
    rho_water = 1000.0  # Water density [kg/m³] at 20°C
    thickness_min = result_min['retention'] / rho_water
    thickness_max = result_max['retention'] / rho_water
    thickness_typical = result_typical['retention'] / rho_water
    
    return {
        'min': result_min['retention'],
        'max': result_max['retention'],
        'typical': result_typical['retention'],
        'thickness_min': thickness_min,
        'thickness_max': thickness_max,
        'thickness_typical': thickness_typical,
        'surface_type': surface_type,
        'hysteresis': result_typical['hysteresis'],
        'theta_eq': result_typical['theta_eq']
    }


if __name__ == "__main__":
    """
    Example usage and validation.
    """
    print("=" * 70)
    print("Surface Water Retention Calculator")
    print("=" * 70)
    
    # Example 1: Superhydrophobic surface
    print("\n1. Superhydrophobic Surface (θ > 120°)")
    print("-" * 70)
    theta_R = 135
    theta_A = 140
    result = calculate_surface_retention(theta_R, theta_A)
    print(f"   Receding angle: {theta_R}°")
    print(f"   Advancing angle: {theta_A}°")
    print(f"   Hysteresis: {result['hysteresis']:.1f}°")
    print(f"   Surface type: {result['surface_type']}")
    print(f"   Retention coefficient (k): {result['k']:.2f}")
    print(f"   Droplet spacing (δ): {result['delta']*1000:.1f} mm")
    print(f"   Retention: {result['retention']*1000:.2f} g/m²")
    print(f"   Water layer thickness: {result['thickness']*1000:.4f} mm")
    print(f"   Expected: < 5 g/m²")
    
    # Example 2: Hydrophilic surface
    print("\n2. Hydrophilic Surface (θ ≈ 40-80°)")
    print("-" * 70)
    theta_R = 60
    theta_A = 70
    result = calculate_surface_retention(theta_R, theta_A)
    print(f"   Receding angle: {theta_R}°")
    print(f"   Advancing angle: {theta_A}°")
    print(f"   Hysteresis: {result['hysteresis']:.1f}°")
    print(f"   Surface type: {result['surface_type']}")
    print(f"   Retention coefficient (k): {result['k']:.2f}")
    print(f"   Droplet spacing (δ): {result['delta']*1000:.1f} mm")
    print(f"   Retention: {result['retention']*1000:.2f} g/m²")
    print(f"   Water layer thickness: {result['thickness']*1000:.4f} mm")
    print(f"   Expected: 50-200 g/m²")
    
    # Example 3: Hydrophobic surface
    print("\n3. Hydrophobic Surface (θ ≈ 80-150°)")
    print("-" * 70)
    theta_R = 90
    theta_A = 110
    result = calculate_surface_retention(theta_R, theta_A)
    print(f"   Receding angle: {theta_R}°")
    print(f"   Advancing angle: {theta_A}°")
    print(f"   Hysteresis: {result['hysteresis']:.1f}°")
    print(f"   Surface type: {result['surface_type']}")
    print(f"   Retention coefficient (k): {result['k']:.2f}")
    print(f"   Droplet spacing (δ): {result['delta']*1000:.1f} mm")
    print(f"   Retention: {result['retention']*1000:.2f} g/m²")
    print(f"   Water layer thickness: {result['thickness']*1000:.4f} mm")
    
    # Example 4: Retention range
    print("\n4. Retention Range for Hydrophilic Surface")
    print("-" * 70)
    theta_R = 60
    theta_A = 70
    range_result = calculate_retention_range(theta_R, theta_A)
    print(f"   Min retention: {range_result['min']*1000:.2f} g/m²")
    print(f"   Max retention: {range_result['max']*1000:.2f} g/m²")
    print(f"   Typical retention: {range_result['typical']*1000:.2f} g/m²")
    print(f"   Min thickness: {range_result['thickness_min']*1000:.4f} mm")
    print(f"   Max thickness: {range_result['thickness_max']*1000:.4f} mm")
    print(f"   Typical thickness: {range_result['thickness_typical']*1000:.4f} mm")
    
    print("\n" + "=" * 70)
