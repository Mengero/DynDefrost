"""
Model Initialization Module for Dynamic Defrost Model

This module handles the initialization of the 1-D defrost model,
including grid setup, material properties, and initial conditions.
"""

import numpy as np


class DefrostModel:
    """1-D Dynamic Defrost Model."""
    
    def __init__(self, n_layers=4, frost_thickness=0.005, porosity=0.9):
        """
        Initialize the defrost model.
        
        Parameters
        ----------
        n_layers : int
            Number of layers in the 1-D frost grid
        frost_thickness : float
            Initial frost layer thickness in meters
        porosity : float
            Initial frost porosity (0-1)
        """
        self.n_layers = n_layers
        self.frost_thickness = frost_thickness
        self.porosity = porosity
        
        # Material properties (pure substances)
        self.rho_ice = 917.0       # Ice density [kg/m³]
        self.rho_water = 1000.0    # Water density [kg/m³]
        self.rho_air = 1.2         # Air density [kg/m³]
        
        self.cp_ice = 2090.0       # Ice specific heat [J/(kg·K)]
        self.cp_water = 4186.0     # Water specific heat [J/(kg·K)]
        self.cp_air = 1006.0       # Air specific heat [J/(kg·K)]
        
        self.k_ice = 2.22          # Ice thermal conductivity [W/(m·K)]
        self.k_water = 0.6         # Water thermal conductivity [W/(m·K)]
        self.k_air = 0.026         # Air thermal conductivity [W/(m·K)]
        
        # Phase change
        self.T_melt = 0.0          # Melting temperature [°C]
        self.L_fusion = 334000.0   # Latent heat of fusion [J/kg]
        
        # Layer properties (arrays, one value per layer)
        self.dx = None             # Layer thickness [m]
        self.T = None              # Layer temperature [°C]
        self.H = None              # Layer enthalpy [J/m³]
        self.cp = None             # Layer effective specific heat [J/(kg·K)]
        self.k = None              # Layer effective thermal conductivity [W/(m·K)]
        self.rho = None            # Layer effective density [kg/m³]
        
        # Volume fractions per layer (sum to 1.0 for each layer)
        self.alpha_ice = None      # Volume fraction of ice [-]
        self.alpha_water = None    # Volume fraction of water [-]
        self.alpha_air = None      # Volume fraction of air [-]
        
        # Boundary conditions
        self.T_surface = None      # Surface temperature (heated side)
        self.T_ambient = None      # Ambient temperature
        
        # Initialize layers
        self._initialize_layers()
        
    def _initialize_layers(self):
        """Initialize layer properties."""
        n = self.n_layers
        
        # Layer thickness (uniform)
        self.dx = np.full(n, self.frost_thickness / n)
        
        # Initial volume fractions: ice + air = 1 (no water initially)
        # porosity = alpha_air, so alpha_ice = 1 - porosity
        self.alpha_ice = np.full(n, 1 - self.porosity)
        self.alpha_water = np.zeros(n)
        self.alpha_air = np.full(n, self.porosity)
        
        # Calculate effective properties
        self._calculate_effective_properties()
        
        print(f"Initialized {n} layers")
        print(f"  Layer thickness: {self.dx[0]*1000:.4f} mm each")
        print(f"  Initial volume fractions: alpha_ice={self.alpha_ice[0]:.3f}, alpha_water={self.alpha_water[0]:.3f}, alpha_air={self.alpha_air[0]:.3f}")
        
    def _calculate_effective_properties(self):
        """Calculate effective properties for each layer based on volume fractions."""
        # Effective density (volume-weighted average)
        self.rho = (self.alpha_ice * self.rho_ice + 
                    self.alpha_water * self.rho_water + 
                    self.alpha_air * self.rho_air)
        
        # Effective specific heat (mass-weighted average)
        mass_ice = self.alpha_ice * self.rho_ice
        mass_water = self.alpha_water * self.rho_water
        mass_air = self.alpha_air * self.rho_air
        total_mass = mass_ice + mass_water + mass_air
        
        self.cp = (mass_ice * self.cp_ice + 
                   mass_water * self.cp_water + 
                   mass_air * self.cp_air) / total_mass
        
        # Effective thermal conductivity (parallel model)
        self.k = (self.alpha_ice * self.k_ice + 
                  self.alpha_water * self.k_water + 
                  self.alpha_air * self.k_air)
        
    def set_initial_temperature(self, T_initial):
        """
        Set uniform initial temperature field and calculate enthalpy.
        
        Parameters
        ----------
        T_initial : float
            Initial temperature in °C
        """
        self.T = np.full(self.n_layers, T_initial)
        self._calculate_enthalpy()
        print(f"  Initial temperature: {T_initial}°C")
        print(f"  Initial enthalpy: {self.H[0]:.2e} J/m³")
        
    def _calculate_enthalpy(self):
        """Calculate enthalpy for each layer based on temperature and composition."""
        # Reference: H = 0 at T = 0°C for ice
        # H = rho * cp * T for sensible heat (simplified, no latent heat stored yet)
        self.H = self.rho * self.cp * self.T
        
    def set_boundary_conditions(self, T_surface, T_ambient=None):
        """
        Set boundary conditions.
        
        Parameters
        ----------
        T_surface : float
            Surface temperature (heated side) in °C
        T_ambient : float, optional
            Ambient temperature in °C
        """
        self.T_surface = T_surface
        self.T_ambient = T_ambient
        print(f"  Boundary: T_surface = {T_surface}°C")
        
    def get_layer_properties(self, layer_idx):
        """
        Get all properties for a specific layer.
        
        Parameters
        ----------
        layer_idx : int
            Layer index (0 to n_layers-1)
            
        Returns
        -------
        dict
            Layer properties
        """
        return {
            'dx': self.dx[layer_idx],
            'T': self.T[layer_idx],
            'H': self.H[layer_idx],
            'cp': self.cp[layer_idx],
            'k': self.k[layer_idx],
            'rho': self.rho[layer_idx],
            'alpha_ice': self.alpha_ice[layer_idx],
            'alpha_water': self.alpha_water[layer_idx],
            'alpha_air': self.alpha_air[layer_idx],
        }
    
    def print_layer_summary(self):
        """Print summary of all layer properties."""
        print("\nLayer Summary:")
        print("-" * 90)
        print(f"{'Layer':<6} {'dx(mm)':<8} {'T(°C)':<8} {'H(J/m³)':<12} {'cp':<8} {'k':<8} {'a_ice':<8} {'a_water':<8} {'a_air':<8}")
        print("-" * 90)
        for i in range(self.n_layers):
            print(f"{i:<6} {self.dx[i]*1000:<8.3f} {self.T[i]:<8.2f} {self.H[i]:<12.2e} "
                  f"{self.cp[i]:<8.1f} {self.k[i]:<8.4f} {self.alpha_ice[i]:<8.3f} "
                  f"{self.alpha_water[i]:<8.3f} {self.alpha_air[i]:<8.3f}")
        print("-" * 90)
        
    def get_summary(self):
        """
        Get model summary.
        
        Returns
        -------
        dict
            Model parameters and state summary
        """
        return {
            'n_layers': self.n_layers,
            'frost_thickness_mm': self.frost_thickness * 1000,
            'dx_mm': self.dx[0] * 1000 if self.dx is not None else None,
            'porosity': self.porosity,
            'T_initial': self.T[0] if self.T is not None else None,
        }
    
    def __repr__(self):
        summary = self.get_summary()
        return (
            f"DefrostModel:\n"
            f"  Layers: {summary['n_layers']}\n"
            f"  Frost thickness: {summary['frost_thickness_mm']:.2f} mm\n"
            f"  Layer thickness: {summary['dx_mm']:.4f} mm\n"
            f"  Initial porosity: {summary['porosity']:.3f}"
        )


def initialize_model(n_layers=4, frost_thickness=0.005, porosity=0.9, T_initial=-20.0):
    """
    Convenience function to initialize the defrost model.
    
    Parameters
    ----------
    n_layers : int
        Number of layers in the frost
    frost_thickness : float
        Initial frost layer thickness in meters
    porosity : float
        Frost porosity (0-1)
    T_initial : float
        Initial temperature in °C
        
    Returns
    -------
    DefrostModel
        Initialized model instance
    """
    model = DefrostModel(n_layers=n_layers, frost_thickness=frost_thickness, porosity=porosity)
    model.set_initial_temperature(T_initial)
    return model
