"""
Calibration script for critical detachment frost thickness.

Target values:
- alpha_water = 0.8:
  - θ = 60°: h_crit ≈ 2.5 mm
  - θ = 160°: h_crit ≈ 1.5 mm
- alpha_water = 0:
  - θ = 60°: h_crit ≈ 10 mm
  - θ = 160°: h_crit ≈ 5 mm

Note: Since tau_base(60°)/tau_base(160°) ≈ 24.87, but target h_crit ratio is 2,
we may need different k values for different contact angles, or adjust the formula.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def calculate_base_adhesion(theta_deg):
    """Calculate base adhesion from contact angle."""
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
        Water volume fraction [-]
    A : float
        Multiplier for log (should be negative). Default: -0.1
    B : float
        Multiplier for alpha_water. Default: 50
    e : float
        Offset to prevent singularity. Default: 0.05
    C : float
        Plateau/base value. Default: 0.4
    
    Returns
    -------
    float
        f(alpha_water) value [-]
    """
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


def calculate_critical_thickness(tau_base, f_water, rho_eff, k, g=9.81):
    """Calculate critical thickness."""
    if rho_eff > 0 and g > 0:
        h_crit = (k * tau_base * f_water) / (rho_eff * g)
    else:
        h_crit = np.inf
    return h_crit


def objective_function(params, rho_eff=101.1, g=9.81):
    """
    Objective function for calibration.
    
    Parameters: [k_60, k_160, A, B, e, C]
    - k_60: retention coefficient for 60°
    - k_160: retention coefficient for 160°
    - A, B, e, C: parameters for f(alpha_water) = A*log(B*alpha_water + e) + C
    - A should be negative
    """
    k_60, k_160, A, B, e, C = params
    
    # Ensure parameters are reasonable
    # k values must be positive, A must be negative, B and e must be positive
    if any(p <= 0 for p in [k_60, k_160, B, e]) or A >= 0 or C < 0:
        return 1e10
    
    # Target values
    targets = {
        (60, 0.0, k_60): 10e-3,    # 10 mm
        (60, 0.8, k_60): 2.5e-3,   # 2.5 mm
        (160, 0.0, k_160): 5e-3,    # 5 mm
        (160, 0.8, k_160): 1.5e-3,  # 1.5 mm
    }
    
    error_sum = 0.0
    
    for (theta, alpha_w, k), target in targets.items():
        tau_base = calculate_base_adhesion(theta)
        f_water = calculate_f_water(alpha_w, A, B, e, C)
        h_crit = calculate_critical_thickness(tau_base, f_water, rho_eff, k, g)
        
        # Use relative error
        if target > 0:
            relative_error = ((h_crit - target) / target) ** 2
        else:
            relative_error = (h_crit - target) ** 2
        error_sum += relative_error
    
    return error_sum


def calibrate_constants(rho_eff=101.1, g=9.81):
    """Calibrate constants with separate k values for each contact angle."""
    # Bounds: [k_60, k_160, A, B, e, C]
    # A should be negative
    bounds = [
        (10.0, 10000.0),   # k_60
        (10.0, 10000.0),   # k_160
        (-10.0, -0.001),  # A (negative)
        (1.0, 200.0),      # B
        (0.001, 1.0),      # e
        (0.0, 2.0),        # C
    ]
    
    # Optimize using differential evolution
    result = differential_evolution(
        objective_function,
        bounds,
        args=(rho_eff, g),
        seed=42,
        maxiter=2000,
        tol=1e-8,
        polish=True
    )
    
    k_60, k_160, A, B, e, C = result.x
    
    return {
        'k_60': k_60,
        'k_160': k_160,
        'A': A,
        'B': B,
        'e': e,
        'C': C,
        'error': result.fun,
        'success': result.success
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Critical Thickness Calibration")
    print("=" * 70)
    
    rho_eff = 101.1  # kg/m³
    g = 9.81  # m/s²
    
    print(f"\nTarget Values:")
    print(f"  alpha_water = 0.0:")
    print(f"    θ = 60°:  h_crit = 10.0 mm")
    print(f"    θ = 160°: h_crit = 5.0 mm")
    print(f"  alpha_water = 0.8:")
    print(f"    θ = 60°:  h_crit = 2.5 mm")
    print(f"    θ = 160°: h_crit = 1.5 mm")
    
    print(f"\nCalibrating constants (with separate k for each angle)...")
    calibrated = calibrate_constants(rho_eff, g)
    
    print(f"\n{'='*70}")
    print("Calibrated Parameters:")
    print(f"{'='*70}")
    print(f"  k_60 (retention coefficient for 60°): {calibrated['k_60']:.4f}")
    print(f"  k_160 (retention coefficient for 160°): {calibrated['k_160']:.4f}")
    print(f"  A (multiplier for log, negative): {calibrated['A']:.6f}")
    print(f"  B (multiplier for alpha_water): {calibrated['B']:.4f}")
    print(f"  e (offset in log): {calibrated['e']:.6f}")
    print(f"  C (plateau value): {calibrated['C']:.4f}")
    print(f"  Optimization error: {calibrated['error']:.2e}")
    print(f"  Success: {calibrated['success']}")
    
    # Verify
    print(f"\n{'='*70}")
    print("Verification with Calibrated Parameters:")
    print(f"{'='*70}")
    
    k_60 = calibrated['k_60']
    k_160 = calibrated['k_160']
    A = calibrated['A']
    B = calibrated['B']
    e = calibrated['e']
    C = calibrated['C']
    
    test_cases = [
        (60, 0.0, 10e-3, k_60),
        (60, 0.8, 2.5e-3, k_60),
        (160, 0.0, 5e-3, k_160),
        (160, 0.8, 1.5e-3, k_160),
    ]
    
    print(f"\n{'θ (deg)':<12} {'α_water':<12} {'Target (mm)':<15} {'Calculated (mm)':<18} {'Error (%)':<15}")
    print("-" * 75)
    
    for theta, alpha_w, target, k in test_cases:
        tau_base = calculate_base_adhesion(theta)
        f_water = calculate_f_water(alpha_w, A, B, e, C)
        h_crit = calculate_critical_thickness(tau_base, f_water, rho_eff, k, g)
        error_pct = ((h_crit - target) / target) * 100 if target > 0 else 0
        
        print(f"{theta:<12} {alpha_w:<12.1f} {target*1000:<15.2f} {h_crit*1000:<18.4f} {error_pct:<15.2f}")
    
    # Show f(alpha_water) values
    print(f"\n{'='*70}")
    print("f(alpha_water) Function:")
    print(f"{'='*70}")
    print(f"  f(alpha_water) = {A:.6f} * log({B:.4f} * alpha_water + {e:.6f}) + {C:.4f}")
    print(f"\n  f(0.0) = {calculate_f_water(0.0, A, B, e, C):.4f}")
    print(f"  f(0.8) = {calculate_f_water(0.8, A, B, e, C):.4f}")
    print(f"  f(1.0) = {calculate_f_water(1.0, A, B, e, C):.4f}")
    
    # Show a range of values
    print(f"\n  Range of f(alpha_water):")
    for alpha_w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        f_val = calculate_f_water(alpha_w, A, B, e, C)
        print(f"    f({alpha_w:.1f}) = {f_val:.4f}")
