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
        self.alpha_ice_initial = None  # Initial volume fraction of ice (for thermal conductivity calculation)
        
        # Thermal conductivity model parameters
        self.f_ini = 1.0 / 3.0     # Initial weighting factor for parallel-series model
        
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
        
        # Store initial ice volume fraction for thermal conductivity calculation
        self.alpha_ice_initial = self.alpha_ice.copy()
        
        # Calculate effective properties
        self._calculate_effective_properties()
        
        print(f"Initialized {n} layers")
        print(f"  Layer thickness: {self.dx[0]*1000:.4f} mm each")
        print(f"  Initial volume fractions: alpha_ice={self.alpha_ice[0]:.3f}, alpha_water={self.alpha_water[0]:.3f}, alpha_air={self.alpha_air[0]:.3f}")
        
    def calculate_specific_heat(self):
        """
        Calculate the specific heat capacity for each layer.
        
        Based on the equation:
        C_p,A = α_ice,A C_p,ice,A + α_H2O,A C_p,H2O,A + α_air,A C_p,air,A
        
        where:
        C_p,ice,A = c_p,ice ρ_ice δ_A
        C_p,H2O,A = c_p,H2O ρ_H2O δ_A
        C_p,air,A = c_p,air ρ_air δ_A
        
        This function should be called at every time step as volume fractions
        change during the defrost process.
        
        Returns
        -------
        numpy.ndarray
            Specific heat capacity per unit mass for each layer [J/(kg·K)]
        """
        # Calculate volumetric heat capacity components for each layer
        # C_p,ice,A = c_p,ice * ρ_ice * δ_A
        C_p_ice_layer = self.cp_ice * self.rho_ice * self.dx
        # C_p,H2O,A = c_p,H2O * ρ_H2O * δ_A
        C_p_water_layer = self.cp_water * self.rho_water * self.dx
        # C_p,air,A = c_p,air * ρ_air * δ_A
        C_p_air_layer = self.cp_air * self.rho_air * self.dx
        
        # Calculate volumetric heat capacity for each layer
        # C_p,A = α_ice,A * C_p,ice,A + α_H2O,A * C_p,H2O,A + α_air,A * C_p,air,A
        C_p_volumetric = (self.alpha_ice * C_p_ice_layer + 
                         self.alpha_water * C_p_water_layer + 
                         self.alpha_air * C_p_air_layer)
        
        # Calculate effective density for each layer
        rho_eff = (self.alpha_ice * self.rho_ice + 
                  self.alpha_water * self.rho_water + 
                  self.alpha_air * self.rho_air)
        
        # Convert to specific heat per unit mass: c_p,A = C_p,A / (ρ_A * δ_A)
        # Since C_p,A already includes δ_A, we divide by (ρ_A * δ_A)
        self.cp = C_p_volumetric / (rho_eff * self.dx)
        
        # Store effective density
        self.rho = rho_eff
        
        return self.cp
    
    def calculate_thermal_conductivity(self):
        """
        Calculate the effective thermal conductivity for each layer.
        
        Based on the combined parallel-series model:
        k_eff = f k_p + (1 - f) k_s
        
        where:
        k_p = α_ice k_ice + α_H2O k_H2O + α_air k_air  (parallel model)
        k_s = 1 / (α_ice / k_ice + α_H2O / k_H2O + α_air / k_air)  (series model)
        f = f_ini * (α_ice / α_ice,initial),  f_ini = 1/3
        
        This function should be called at every time step as volume fractions
        change during the defrost process.
        
        Returns
        -------
        numpy.ndarray
            Effective thermal conductivity for each layer [W/(m·K)]
        """
        # Calculate parallel thermal conductivity for each layer
        # k_p = α_ice * k_ice + α_H2O * k_H2O + α_air * k_air
        k_parallel = (self.alpha_ice * self.k_ice + 
                     self.alpha_water * self.k_water + 
                     self.alpha_air * self.k_air)
        
        # Calculate series thermal conductivity for each layer
        # k_s = 1 / (α_ice / k_ice + α_H2O / k_H2O + α_air / k_air)
        # Avoid division by zero by using np.where or ensuring denominators are non-zero
        k_series_denom = (self.alpha_ice / self.k_ice + 
                          self.alpha_water / self.k_water + 
                          self.alpha_air / self.k_air)
        k_series = 1.0 / k_series_denom
        
        # Calculate weighting factor f for each layer
        # f = f_ini * (α_ice / α_ice,initial)
        # Avoid division by zero if initial ice fraction is zero
        with np.errstate(divide='ignore', invalid='ignore'):
            f = self.f_ini * (self.alpha_ice / self.alpha_ice_initial)
            # If initial ice fraction was zero, set f to 0
            f = np.where(self.alpha_ice_initial > 0, f, 0.0)
            # Ensure f is between 0 and 1
            f = np.clip(f, 0.0, 1.0)
        
        # Calculate effective thermal conductivity
        # k_eff = f * k_p + (1 - f) * k_s
        self.k = f * k_parallel + (1.0 - f) * k_series
        
        return self.k
    
    def calculate_thermal_resistance(self):
        """
        Calculate the effective thermal resistance between adjacent layers.
        
        For the interface between layer 0 and layer 1:
        R_0,1 = δ_0 / (2 * k_0,eff)
        
        For interfaces between layer A and A+1 (A >= 1):
        R_A,A+1 = (δ_A + δ_A+1) / (2 * k_A,eff)
        
        This function should be called at every time step as thermal conductivity
        and layer thicknesses may change during the defrost process.
        
        Returns
        -------
        numpy.ndarray
            Thermal resistance between layers [K·m²/W]
            Array has length (n_layers - 1), with R[i] being the resistance
            between layer i and layer i+1
        """
        if self.k is None or self.dx is None:
            raise ValueError("Thermal conductivity and layer thicknesses must be calculated first")
        
        n_interfaces = self.n_layers - 1
        R = np.zeros(n_interfaces)
        
        # For interface between layer 0 and layer 1: R = δ_0 / (2 * k_0,eff)
        R[0] = self.dx[0] / (2.0 * self.k[0])
        
        # For interfaces between layer A and A+1 (A >= 1): R = (δ_A + δ_A+1) / (2 * k_A,eff)
        for i in range(1, n_interfaces):
            R[i] = (self.dx[i] + self.dx[i+1]) / (2.0 * self.k[i])
        
        return R
    
    def calculate_heat_flux_between_layers(self):
        """
        Calculate the heat flux between adjacent layers.
        
        For interfaces between layer A and A+1 (A >= 0):
        q''_{A,A+1} = (T_A - T_{A+1}) / R_{A,A+1}
        
        This function should be called at every time step as temperatures
        and thermal resistances may change during the defrost process.
        
        Returns
        -------
        numpy.ndarray
            Heat flux between layers [W/m²]
            Array has length (n_layers - 1), with q[i] being the heat flux
            from layer i to layer i+1
        """
        if self.T is None:
            raise ValueError("Layer temperatures must be set first")
        
        # Calculate thermal resistances
        R = self.calculate_thermal_resistance()
        
        # Calculate heat flux for each interface
        n_interfaces = self.n_layers - 1
        q = np.zeros(n_interfaces)
        
        for i in range(n_interfaces):
            q[i] = (self.T[i] - self.T[i+1]) / R[i]
        
        return q
    
    def calculate_total_heat_input(self, T_wall=None):
        """
        Calculate the total heat input at the boundary (surface).
        
        q'' = (T_w - T_0) / R_0
        
        where:
        R_0 = δ_0 / (2 * k_0,eff)
        
        Parameters
        ----------
        T_wall : float, optional
            Wall/surface temperature [°C]. If None, uses self.T_surface
        
        Returns
        -------
        float
            Total heat input at the boundary [W/m²]
        """
        if self.T is None:
            raise ValueError("Layer temperatures must be set first")
        if self.k is None or self.dx is None:
            raise ValueError("Thermal conductivity and layer thicknesses must be calculated first")
        
        # Use provided T_wall or self.T_surface
        if T_wall is None:
            if self.T_surface is None:
                raise ValueError("Wall temperature must be provided or set via set_boundary_conditions()")
            T_wall = self.T_surface
        
        # Calculate R_0 = δ_0 / (2 * k_0,eff)
        R_0 = self.dx[0] / (2.0 * self.k[0])
        
        # Calculate heat flux: q'' = (T_w - T_0) / R_0
        q_total = (T_wall - self.T[0]) / R_0
        
        return q_total
    
    def _calculate_effective_properties(self):
        """Calculate effective properties for each layer based on volume fractions."""
        # Calculate effective density (volume-weighted average)
        self.rho = (self.alpha_ice * self.rho_ice + 
                    self.alpha_water * self.rho_water + 
                    self.alpha_air * self.rho_air)
        
        # Use the new specific heat calculation method
        self.calculate_specific_heat()
        
        # Use the new thermal conductivity calculation method
        self.calculate_thermal_conductivity()
        
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
