"""
Stability Criterion Module for Dynamic Defrost Model

This module calculates the maximum stable time step for explicit methods
based on the Courant-Friedrichs-Lewy (CFL) condition for diffusion problems.
"""

import numpy as np


def calculate_max_stable_dt(n_layers, frost_thickness, k_eff, rho_eff, cp_eff, safety_factor=0.5):
    """
    Calculate maximum stable time step for explicit methods using CFL condition.
    
    For explicit diffusion schemes, the stability criterion is:
        dt <= C * (dx^2) / alpha
    
    where:
        dx = spatial step size = frost_thickness / n_layers
        alpha = thermal diffusivity = k / (rho * cp)
        C = safety factor (typically 0.5 for stability)
    
    Parameters
    ----------
    n_layers : int
        Number of layers in the 1-D grid
    frost_thickness : float
        Total frost thickness [m]
    k_eff : float
        Effective thermal conductivity [W/(m·K)]
    rho_eff : float
        Effective density [kg/m³]
    cp_eff : float
        Effective specific heat [J/(kg·K)]
    safety_factor : float, optional
        Safety factor for stability (0 < safety_factor <= 1). 
        Default: 0.5 (conservative)
        
    Returns
    -------
    float
        Maximum stable time step [s]
    """
    # Calculate spatial step size
    dx = frost_thickness / n_layers
    
    # Calculate thermal diffusivity
    alpha = k_eff / (rho_eff * cp_eff)
    
    # CFL condition: dt <= C * (dx^2) / alpha
    dt_max = safety_factor * (dx**2) / alpha
    
    return dt_max


def calculate_optimal_dt_and_layers(frost_thickness, k_eff, rho_eff, cp_eff, 
                                    target_dt=None, target_n_layers=None,
                                    safety_factor=0.5):
    """
    Calculate optimal time step and number of layers based on stability criterion.
    
    This function helps choose appropriate dt and n_layers for explicit methods.
    You can specify either target_dt or target_n_layers, and it will calculate
    the other parameter to satisfy the stability criterion.
    
    Parameters
    ----------
    frost_thickness : float
        Total frost thickness [m]
    k_eff : float
        Effective thermal conductivity [W/(m·K)]
    rho_eff : float
        Effective density [kg/m³]
    cp_eff : float
        Effective specific heat [J/(kg·K)]
    target_dt : float, optional
        Desired time step [s]. If provided, calculates required n_layers.
    target_n_layers : int, optional
        Desired number of layers. If provided, calculates max stable dt.
    safety_factor : float, optional
        Safety factor for stability. Default: 0.5
        
    Returns
    -------
    dict
        Dictionary with:
        - 'dt': Recommended time step [s]
        - 'n_layers': Recommended number of layers
        - 'dt_max': Maximum stable time step [s]
        - 'dx': Spatial step size [m]
        - 'alpha': Thermal diffusivity [m²/s]
    """
    # Calculate thermal diffusivity
    alpha = k_eff / (rho_eff * cp_eff)
    
    if target_dt is not None:
        # Calculate required n_layers for given dt
        # From: dt <= C * (dx^2) / alpha
        # dx^2 >= dt * alpha / C
        # (frost_thickness / n_layers)^2 >= dt * alpha / C
        # n_layers <= frost_thickness / sqrt(dt * alpha / C)
        dx_min = np.sqrt(target_dt * alpha / safety_factor)
        n_layers_max = int(frost_thickness / dx_min)
        n_layers = max(1, n_layers_max)  # At least 1 layer
        dt = target_dt
        dx = frost_thickness / n_layers
        dt_max = safety_factor * (dx**2) / alpha
    elif target_n_layers is not None:
        # Calculate max stable dt for given n_layers
        dx = frost_thickness / target_n_layers
        dt_max = safety_factor * (dx**2) / alpha
        dt = dt_max
        n_layers = target_n_layers
    else:
        raise ValueError("Either target_dt or target_n_layers must be provided")
    
    return {
        'dt': dt,
        'n_layers': n_layers,
        'dt_max': dt_max,
        'dx': dx,
        'alpha': alpha
    }


def get_initial_frost_properties(porosity=0.9):
    """
    Get initial frost properties for stability calculation.
    
    Parameters
    ----------
    porosity : float
        Initial frost porosity (0-1)
        
    Returns
    -------
    dict
        Dictionary with k_eff, rho_eff, cp_eff
    """
    # Pure material properties
    rho_ice = 917.0       # [kg/m³]
    rho_water = 1000.0    # [kg/m³]
    rho_air = 1.2         # [kg/m³]
    
    cp_ice = 2090.0       # [J/(kg·K)]
    cp_water = 4186.0     # [J/(kg·K)]
    cp_air = 1006.0       # [J/(kg·K)]
    
    k_ice = 2.22         # [W/(m·K)]
    k_water = 0.56       # [W/(m·K)]
    k_air = 0.024        # [W/(m·K)]
    
    # Initial volume fractions (no water initially)
    alpha_ice = 1 - porosity
    alpha_water = 0.0
    alpha_air = porosity
    
    # Effective properties (mass-weighted average)
    rho_eff = alpha_ice * rho_ice + alpha_water * rho_water + alpha_air * rho_air
    
    cp_eff = (alpha_ice * rho_ice * cp_ice + 
              alpha_water * rho_water * cp_water + 
              alpha_air * rho_air * cp_air) / rho_eff
    
    # Thermal conductivity (parallel model for initial estimate)
    k_eff = alpha_ice * k_ice + alpha_water * k_water + alpha_air * k_air
    
    return {
        'k_eff': k_eff,
        'rho_eff': rho_eff,
        'cp_eff': cp_eff
    }
