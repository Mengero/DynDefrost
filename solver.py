"""
Solver Module for Dynamic Defrost Model

This module implements the numerical solver for the 1-D defrost problem using
an enthalpy-based approach to handle phase change.

The solver updates enthalpy based on heat flux, then updates temperature
according to enthalpy zones (solid, mushy, liquid).
"""

import numpy as np
from model_init import DefrostModel


class DefrostSolver:
    """
    Implicit solver for the 1-D defrost problem.
    
    Uses fully implicit (backward Euler) method for stability with large gradients.
    """
    
    def __init__(self, model, dt=0.1):
        """
        Initialize the defrost solver.
        
        Parameters
        ----------
        model : DefrostModel
            Initialized defrost model
        dt : float, optional
            Time step [s]. Default: 0.1
        """
        self.model = model
        self.dt = dt
        
        # Enthalpy thresholds for phase change
        # L_f: enthalpy at start of melting (ice at 0°C) = 0 [J/kg]
        # L_H2O: enthalpy at end of melting (water at 0°C) = L_fusion [J/kg]
        # These are specific enthalpies [J/kg]
        self.L_f = 0.0  # Ice at 0°C (reference point)
        self.L_H2O = model.L_fusion  # Water at 0°C (after complete melting)
        
        # Initialize specific enthalpy for each layer [J/kg]
        # h = H / (ρ * dx) where H is volumetric enthalpy
        self._initialize_enthalpy()
        
        # History tracking
        self.time_history = []
        self.temperature_history = []
        self.enthalpy_history = []
        self.volume_fraction_history = []
    
    def _initialize_enthalpy(self):
        """
        Initialize specific enthalpy for each layer.
        
        Specific enthalpy h [J/kg] is related to volumetric enthalpy H [J/m³] by:
        h = H / ρ
        
        For the enthalpy update equation: h^(t+Δt) = h^t + (Δt / m'') * (q_in'' - q_out'')
        where m'' = ρ * dx is mass per unit area [kg/m²] used for the heat flux term.
        
        For ice below 0°C: h = cp_ice * T (relative to 0°C, so h = 0 at T = 0°C for ice)
        This means h < 0 for ice below freezing point.
        """
        n = self.model.n_layers
        self.h = np.zeros(n)  # Specific enthalpy [J/kg]
        
        # Calculate initial specific enthalpy
        # For ice below 0°C: h = cp_ice * T (relative to 0°C)
        # Since T < 0°C initially, h will be negative
        for i in range(n):
            if self.model.rho[i] > 0:
                # Convert volumetric enthalpy to specific enthalpy: h = H / ρ
                if self.model.H is not None:
                    self.h[i] = self.model.H[i] / self.model.rho[i]
                else:
                    # Calculate from temperature: h = cp * T (for ice below 0°C)
                    # This gives h < 0 for T < 0°C
                    self.h[i] = self.model.cp_ice * self.model.T[i]
            else:
                self.h[i] = 0.0
        
    def solve_time_step(self, T_surface):
        """
        Solve one time step using enthalpy-based approach.
        
        Steps:
        1. Calculate heat fluxes between layers
        2. Update enthalpy: h_A^(t+Δt) = h_A^t + (Δt / m'') * (q_in'' - q_out'')
        3. Update temperature based on enthalpy zones
        4. Update volume fractions based on enthalpy
        
        Parameters
        ----------
        T_surface : float
            Surface temperature at current time [°C]
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Update boundary condition
        self.model.T_surface = T_surface
        
        # Store old state
        h_old = self.h.copy()
        T_old = self.model.T.copy()
        
        # Update properties based on current state
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity()
        
        # Step 1: Calculate heat fluxes
        q_fluxes = self._calculate_heat_fluxes(T_surface)
        
        # Step 2: Update enthalpy for each layer
        # h_A^(t+Δt) = h_A^t + (Δt / m'') * (q_in'' - q_out'')
        # where m'' = ρ * dx is mass per unit area [kg/m²]
        for i in range(self.model.n_layers):
            m_double_prime = self.model.rho[i] * self.model.dx[i]  # Mass per unit area [kg/m²]
            
            if m_double_prime > 0:
                # Net heat flux into layer i [W/m²]
                q_net = q_fluxes['in'][i] - q_fluxes['out'][i]
                
                # Update specific enthalpy [J/kg]
                # Energy input = q_net * dt [J/m²]
                # Specific enthalpy change = (q_net * dt) / m'' [J/kg]
                self.h[i] = h_old[i] + (self.dt / m_double_prime) * q_net
            else:
                self.h[i] = h_old[i]
        
        # Step 3: Update temperature based on enthalpy
        self._update_temperature_from_enthalpy(h_old)
        
        # Step 4: Update volume fractions based on enthalpy
        self._update_volume_fractions_from_enthalpy()
        
        # Step 5: Recalculate properties with new volume fractions
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity()
        
        # Update volumetric enthalpy: H = h * ρ
        for i in range(self.model.n_layers):
            self.model.H[i] = self.h[i] * self.model.rho[i]
        
        return True
    
    def _calculate_heat_fluxes(self, T_surface):
        """
        Calculate heat fluxes into and out of each layer.
        
        Parameters
        ----------
        T_surface : float
            Surface temperature [°C]
        
        Returns
        -------
        dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²]
        """
        n = self.model.n_layers
        q_in = np.zeros(n)
        q_out = np.zeros(n)
        
        # Calculate thermal resistances
        R = self.model.calculate_thermal_resistance()
        
        # Layer 0: heat input from surface
        R_surface = self.model.dx[0] / (2.0 * self.model.k[0])
        q_surface = (T_surface - self.model.T[0]) / R_surface
        q_in[0] = q_surface
        
        # Heat flux from layer 0 to layer 1
        if n > 1:
            q_01 = (self.model.T[0] - self.model.T[1]) / R[0]
            q_out[0] = q_01
            q_in[1] = q_01
        
        # Interior layers
        for i in range(1, n - 1):
            # Heat flux from layer i-1 to i
            q_in[i] = (self.model.T[i-1] - self.model.T[i]) / R[i-1]
            
            # Heat flux from layer i to i+1
            q_out[i] = (self.model.T[i] - self.model.T[i+1]) / R[i]
        
        # Last layer: only heat input from previous layer
        if n > 1:
            i = n - 1
            q_in[i] = (self.model.T[i-1] - self.model.T[i]) / R[i-1]
            # Assume adiabatic boundary at the end (no heat output)
            q_out[i] = 0.0
        
        return {'in': q_in, 'out': q_out}
    
    def _update_temperature_from_enthalpy(self, h_old):
        """
        Update temperature based on enthalpy using the piecewise function.
        
        ΔT = {
            (h^(t+Δt) - h^t) / C_p,A,  if h < L_f or h > L_H2O
            0,                          if L_f ≤ h ≤ L_H2O
        }
        
        Also handles the "excessive melting" case where h^t < L_H2O but h^(t+Δt) > L_H2O.
        
        Parameters
        ----------
        h_old : numpy.ndarray
            Specific enthalpy at previous time step [J/kg]
        """
        n = self.model.n_layers
        T_melt = self.model.T_melt
        
        for i in range(n):
            h_new = self.h[i]
            h_prev = h_old[i]
            dh = h_new - h_prev
            
            # Get current specific heat capacity [J/(kg·K)]
            cp = self.model.cp[i]
            
            if cp <= 0:
                continue  # Skip if invalid cp
            
            # Determine enthalpy zone
            if h_new < self.L_f:
                # Solid phase (ice below 0°C)
                # ΔT = Δh / C_p
                dT = dh / cp
                self.model.T[i] += dT
                
            elif h_new > self.L_H2O:
                # Check for excessive melting case
                if h_prev < self.L_H2O:
                    # Excessive melting: h^t < L_H2O but h^(t+Δt) > L_H2O
                    # Part 1: Use energy to bring h to L_H2O (melting)
                    dh_melting = self.L_H2O - h_prev
                    # Part 2: Remaining energy increases temperature
                    dh_heating = h_new - self.L_H2O
                    dT = dh_heating / cp
                    self.model.T[i] = T_melt + dT
                else:
                    # Liquid phase (water above 0°C)
                    # ΔT = Δh / C_p
                    dT = dh / cp
                    self.model.T[i] += dT
                    
            else:
                # Mushy zone: L_f ≤ h ≤ L_H2O
                # Temperature stays at melting point
                # But check if we entered the mushy zone
                if h_prev < self.L_f:
                    # Transitioning from solid to mushy
                    # Part 1: Heat to L_f
                    dh_heating = self.L_f - h_prev
                    dT_part1 = dh_heating / cp
                    # Part 2: Remaining energy goes to melting (no temp change)
                    self.model.T[i] = T_melt
                elif h_prev > self.L_H2O:
                    # Transitioning from liquid to mushy (shouldn't happen, but handle it)
                    # Part 1: Cool to L_H2O
                    dh_cooling = h_prev - self.L_H2O
                    dT_part1 = -dh_cooling / cp
                    # Part 2: Remaining energy goes to freezing (no temp change)
                    self.model.T[i] = T_melt
                else:
                    # Already in mushy zone, temperature stays constant
                    self.model.T[i] = T_melt
    
    def _update_volume_fractions_from_enthalpy(self):
        """
        Update volume fractions based on specific enthalpy.
        
        In the mushy zone (L_f ≤ h ≤ L_H2O), the fraction of ice that has melted
        is proportional to (h - L_f) / (L_H2O - L_f).
        """
        n = self.model.n_layers
        
        for i in range(n):
            h = self.h[i]
            alpha_ice_init = self.model.alpha_ice_initial[i]
            
            if h < self.L_f:
                # Solid phase: all ice remains
                self.model.alpha_ice[i] = alpha_ice_init
                self.model.alpha_water[i] = 0.0
                # Air fraction remains constant
                
            elif h > self.L_H2O:
                # Liquid phase: all ice has melted
                self.model.alpha_ice[i] = 0.0
                self.model.alpha_water[i] = alpha_ice_init
                # Air fraction remains constant
                
            else:
                # Mushy zone: partial melting
                # Fraction melted = (h - L_f) / (L_H2O - L_f)
                if self.L_H2O > self.L_f:
                    fraction_melted = (h - self.L_f) / (self.L_H2O - self.L_f)
                else:
                    fraction_melted = 0.0
                
                fraction_melted = np.clip(fraction_melted, 0.0, 1.0)
                
                # Update volume fractions
                self.model.alpha_ice[i] = alpha_ice_init * (1.0 - fraction_melted)
                self.model.alpha_water[i] = alpha_ice_init * fraction_melted
                # Air fraction remains constant
            
            # Ensure volume fractions sum to 1
            total = self.model.alpha_ice[i] + self.model.alpha_water[i] + self.model.alpha_air[i]
            if total > 0:
                # Normalize if needed (shouldn't be necessary, but safety check)
                scale = 1.0 / total
                self.model.alpha_ice[i] *= scale
                self.model.alpha_water[i] *= scale
                self.model.alpha_air[i] *= scale
    
    def solve(self, time_array, T_surface_array, save_history=True):
        """
        Solve the defrost problem over the entire time domain.
        
        Parameters
        ----------
        time_array : numpy.ndarray
            Time points [s]
        T_surface_array : numpy.ndarray
            Surface temperature at each time point [°C]
        save_history : bool, optional
            Whether to save solution history. Default: True
        
        Returns
        -------
        dict
            Solution results with time, temperature, and other quantities
        """
        if len(time_array) != len(T_surface_array):
            raise ValueError("time_array and T_surface_array must have same length")
        
        n_steps = len(time_array)
        
        # Initialize history
        if save_history:
            self.time_history = []
            self.temperature_history = []
            self.enthalpy_history = []
            self.volume_fraction_history = []
        
        # Time stepping
        for i in range(1, n_steps):
            dt = time_array[i] - time_array[i-1]
            self.dt = dt
            
            # Interpolate surface temperature if needed
            T_surface = T_surface_array[i]
            
            # Solve one time step
            success = self.solve_time_step(T_surface)
            
            if not success:
                print(f"Warning: Solver failed at step {i}, time = {time_array[i]:.2f} s")
            
            # Save history
            if save_history:
                self.time_history.append(time_array[i])
                self.temperature_history.append(self.model.T.copy())
                self.enthalpy_history.append(self.h.copy())  # Save specific enthalpy
                self.volume_fraction_history.append({
                    'ice': self.model.alpha_ice.copy(),
                    'water': self.model.alpha_water.copy(),
                    'air': self.model.alpha_air.copy()
                })
        
        # Return results
        results = {
            'time': np.array(self.time_history) if save_history else time_array,
            'temperature': np.array(self.temperature_history) if save_history else None,
            'model': self.model
        }
        
        return results
