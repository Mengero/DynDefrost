"""
Solver Module for Dynamic Defrost Model

This module implements the numerical solver for the 1-D defrost problem using
an enthalpy-based approach to handle phase change.

The solver updates enthalpy based on heat flux, then updates temperature
according to enthalpy zones (solid, mushy, liquid).
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from model_init import DefrostModel


class DefrostSolver:
    """
    Implicit solver for the 1-D defrost problem.
    
    Uses fully implicit (backward Euler) method for stability with large gradients.
    """
    
    def __init__(self, model, dt=0.1, method='explicit'):
        """
        Initialize the defrost solver.
        
        Parameters
        ----------
        model : DefrostModel
            Initialized defrost model
        dt : float, optional
            Time step [s]. Default: 0.1
        method : str, optional
            Solver method: 'explicit' or 'implicit'. Default: 'explicit'
        """
        self.model = model
        self.dt = dt
        self.method = method.lower()
        
        if self.method not in ['explicit', 'implicit']:
            raise ValueError(f"Method must be 'explicit' or 'implicit', got '{method}'")
        
        # Implicit solver parameters
        self.max_iter = 20
        self.tolerance = 1e-6
        
        # Enthalpy thresholds for phase change (per layer, depends on alpha_ice_initial)
        # L_f: mixture enthalpy when ice reaches 0°C (before melting starts)
        # L_H2O: mixture enthalpy when all ice has melted (water at 0°C)
        # These are specific enthalpies [J/kg] for the MIXTURE, not pure ice
        self.L_f = None  # Will be calculated per layer
        self.L_H2O = None  # Will be calculated per layer
        
        # Calculate enthalpy thresholds for each layer
        self._calculate_enthalpy_thresholds()
        
        # Initialize specific enthalpy for each layer [J/kg]
        # h = H / (ρ * dx) where H is volumetric enthalpy
        self._initialize_enthalpy()
        
        # History tracking
        self.time_history = []
        self.temperature_history = []
        self.enthalpy_history = []
        self.volume_fraction_history = []
    
    def _calculate_mixture_enthalpy(self, T_K, T_ref_K, alpha_ice, alpha_water, alpha_air):
        """
        Calculate mixture specific enthalpy from component enthalpies.
        
        Based on:
        h_A,i = (α_ice,A * ρ_ice * h_ice + α_H2O * ρ_H2O * h_H2O + α_air * ρ_air * h_air) / ρ_A
        
        where:
        - h_ice = cp_ice * (T_K - T_ref_K)
        - h_H2O = cp_H2O * (T_K - T_ref_K)
        - h_air = cp_air * (T_K - T_ref_K)
        - ρ_A = α_ice,A * ρ_ice + α_H2O * ρ_H2O + α_air * ρ_air
        
        Parameters
        ----------
        T_K : float or numpy.ndarray
            Temperature in Kelvin
        T_ref_K : float
            Reference temperature in Kelvin (where h = 0)
        alpha_ice : float or numpy.ndarray
            Volume fraction of ice
        alpha_water : float or numpy.ndarray
            Volume fraction of water
        alpha_air : float or numpy.ndarray
            Volume fraction of air
        
        Returns
        -------
        float or numpy.ndarray
            Mixture specific enthalpy [J/kg]
        """
        # Calculate component enthalpies
        h_ice = self.model.cp_ice * (T_K - T_ref_K)
        h_H2O = self.model.cp_water * (T_K - T_ref_K)
        h_air = self.model.cp_air * (T_K - T_ref_K)
        
        # Calculate mixture density
        rho_A = (alpha_ice * self.model.rho_ice + 
                 alpha_water * self.model.rho_water + 
                 alpha_air * self.model.rho_air)
        
        # Calculate mixture enthalpy
        # h_A = (α_ice * ρ_ice * h_ice + α_H2O * ρ_H2O * h_H2O + α_air * ρ_air * h_air) / ρ_A
        numerator = (alpha_ice * self.model.rho_ice * h_ice + 
                    alpha_water * self.model.rho_water * h_H2O + 
                    alpha_air * self.model.rho_air * h_air)
        
        # Avoid division by zero
        if isinstance(rho_A, np.ndarray):
            h_A = np.where(rho_A > 0, numerator / rho_A, 0.0)
        else:
            h_A = numerator / rho_A if rho_A > 0 else 0.0
        
        return h_A
    
    def _calculate_enthalpy_thresholds(self):
        """
        Calculate enthalpy thresholds L_f and L_H2O for each layer.
        
        These thresholds represent enthalpy differences from the initial state:
        - L_f[i]: enthalpy gap (DeltaL_f) required to heat from initial temperature 
                  to melting point (0°C). This is the enthalpy when ice reaches 0°C.
        - L_H2O[i]: enthalpy when all ice has melted (water at 0°C)
        
        Uses component-based enthalpy calculation:
        h_A = (α_ice * ρ_ice * h_ice + α_H2O * ρ_H2O * h_H2O + α_air * ρ_air * h_air) / ρ_A
        where h_component = cp_component * (T_K - T_ref_K)
        
        L_H2O = L_f + alpha_ice_initial * L_fusion
                (L_f plus latent heat to melt all ice in the mixture)
        
        Note: cp values are in J/(kg·K), so we must use Kelvin temperature.
        """
        n = self.model.n_layers
        self.L_f = np.zeros(n)
        self.L_H2O = np.zeros(n)
        
        # Reference temperature: use initial temperature as reference (h = 0 at initial state)
        T_ref_K = self.model.T + 273.15  # Convert initial T from °C to K [K]
        
        # Melting temperature in Kelvin
        T_melt_K = self.model.T_melt + 273.15  # [K]
        
        for i in range(n):
            alpha_ice_init = self.model.alpha_ice_initial[i]
            alpha_air = self.model.alpha_air[i]
            
            # L_f: mixture enthalpy at melting point (0°C) relative to initial state
            # At initial state: ice + air, no water (h = 0 by definition)
            # At melting point: ice + air, no water (before melting starts)
            h_at_melt = self._calculate_mixture_enthalpy(
                T_K=T_melt_K,
                T_ref_K=T_ref_K[i],  # Reference is initial state
                alpha_ice=alpha_ice_init,
                alpha_water=0.0,
                alpha_air=alpha_air
            )
            
            # L_f is the enthalpy gap from initial to melting point
            # Since h_initial = 0 (by definition), L_f = h_at_melt
            self.L_f[i] = h_at_melt
            
            # L_H2O: mixture enthalpy when all ice has melted (water at 0°C)
            # At this state: no ice, water + air at T_melt
            # Mass conservation: when ice melts to water, mass is conserved
            # alpha_ice * rho_ice = alpha_water * rho_water
            # Therefore: alpha_water = alpha_ice * rho_ice / rho_water
            alpha_water_melted = alpha_ice_init * self.model.rho_ice / self.model.rho_water
            
            # The enthalpy includes:
            # 1. Sensible heat to raise from T_initial to T_melt (already in L_f for ice+air)
            # 2. Latent heat to melt all ice
            # Note: The component enthalpies only account for sensible heat, so we add latent heat separately
            h_all_melted = self._calculate_mixture_enthalpy(
                T_K=T_melt_K,
                T_ref_K=T_ref_K[i],  # Reference is initial state
                alpha_ice=0.0,
                alpha_water=alpha_water_melted,  # All ice converted to water (mass conserved)
                alpha_air=alpha_air
            )
            
            # L_H2O = sensible heat (water+air at T_melt) + latent heat
            # The latent heat L_fusion is per unit mass of ice
            # Convert to per unit mass of mixture using mass fraction
            # mass_fraction_ice = (alpha_ice * rho_ice) / rho_A
            rho_A = (alpha_ice_init * self.model.rho_ice + 
                     alpha_air * self.model.rho_air)
            if rho_A > 0:
                mass_fraction_ice = (alpha_ice_init * self.model.rho_ice) / rho_A
                latent_heat_per_mass_mixture = mass_fraction_ice * self.model.L_fusion
            else:
                latent_heat_per_mass_mixture = 0.0
            
            # L_H2O = enthalpy of water+air at T_melt + latent heat
            self.L_H2O[i] = h_all_melted + latent_heat_per_mass_mixture
        
        print(f"Enthalpy thresholds calculated:")
        print(f"  L_f range: [{np.min(self.L_f):.2f}, {np.max(self.L_f):.2f}] J/kg")
        print(f"  L_H2O range: [{np.min(self.L_H2O):.2f}, {np.max(self.L_H2O):.2f}] J/kg")
    
    def _initialize_enthalpy(self):
        """
        Initialize specific enthalpy for each layer.
        
        Specific enthalpy h [J/kg] is related to volumetric enthalpy H [J/m³] by:
        h = H / ρ
        
        For the enthalpy update equation: h^(t+Δt) = h^t + (Δt / m'') * (q_in'' - q_out'')
        where m'' = ρ * dx is mass per unit area [kg/m²] used for the heat flux term.
        
        Since L_f represents the enthalpy gap from initial state to melting point,
        we initialize h = 0 at the initial state (by definition).
        The enthalpy will increase as heat is added, reaching L_f when T reaches melting point.
        
        Note: If model.H is already set, we use that; otherwise initialize to 0.
        """
        n = self.model.n_layers
        self.h = np.zeros(n)  # Specific enthalpy [J/kg]
        
        # Initialize enthalpy from initial state
        # Since L_f is measured from initial state, h_initial = 0 by definition
        # We always initialize to 0, regardless of model.H, to ensure consistency
        # with the new enthalpy threshold calculation
        for i in range(n):
            self.h[i] = 0.0
        
    def solve_time_step(self, T_surface):
        """
        Solve one time step using enthalpy-based approach.
        
        Can use either explicit or implicit method based on self.method.
        
        Explicit: q evaluated at time t
        Implicit: q evaluated at time t+Δt (requires iteration)
        
        Parameters
        ----------
        T_surface : float
            Surface temperature at current time [°C]
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.method == 'explicit':
            return self._solve_time_step_explicit(T_surface)
        else:
            return self._solve_time_step_implicit(T_surface)
    
    def _solve_time_step_explicit(self, T_surface):
        """
        Explicit solver: heat fluxes evaluated at current time t.
        
        h^(t+Δt) = h^t + (Δt / m'') * q^t
        """
        # Update boundary condition
        self.model.T_surface = T_surface
        
        # Store old state
        h_old = self.h.copy()
        T_old = self.model.T.copy()
        
        # Update properties based on current state
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity()
        
        # Step 1: Calculate heat fluxes (EXPLICIT: using T^t)
        # Note: Uses current temperatures self.model.T (T^t)
        q_fluxes = self._calculate_heat_fluxes_general(T_surface, self.model.T)
        
        # Step 2: Update enthalpy for each layer (EXPLICIT)
        # h_A^(t+Δt) = h_A^t + (Δt / m'') * (q_in'' - q_out'')^t
        for i in range(self.model.n_layers):
            m_double_prime = self.model.rho[i] * self.model.dx[i]
            
            if m_double_prime > 0:
                q_net = q_fluxes['in'][i] - q_fluxes['out'][i]
                self.h[i] = h_old[i] + (self.dt / m_double_prime) * q_net
            else:
                self.h[i] = h_old[i]
        
        # Step 3-5: Update temperature, volume fractions, and properties
        self._update_temperature_from_enthalpy(h_old)
        self._update_volume_fractions_from_enthalpy_first()
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity()
        
        # Update volumetric enthalpy
        for i in range(self.model.n_layers):
            self.model.H[i] = self.h[i] * self.model.rho[i]
        
        return True
    
    def _solve_time_step_implicit(self, T_surface):
        """
        Implicit solver: heat fluxes evaluated at future time t+Δt.
        
        h^(t+Δt) = h^t + (Δt / m'') * q^(t+Δt)
        
        Since q^(t+Δt) depends on T^(t+Δt), and T^(t+Δt) depends on h^(t+Δt),
        we need to solve this iteratively.
        
        Uses Picard iteration (fixed-point iteration).
        """
        # Update boundary condition
        self.model.T_surface = T_surface
        
        # Store old state
        h_old = self.h.copy()
        T_old = self.model.T.copy()
        
        # Initial guess: use explicit step as starting point
        h_new = h_old.copy()
        T_new = T_old.copy()
        
        # Iterate until convergence
        for iteration in range(self.max_iter):
            # Store previous iteration values
            h_prev = h_new.copy()
            T_prev = T_new.copy()
            
            # Update model state for this iteration
            self.model.T = T_new.copy()
            self.h = h_new.copy()
            
            # Update properties based on current guess
            self._update_volume_fractions_from_enthalpy_first()
            self.model.calculate_specific_heat()
            self.model.calculate_thermal_conductivity()
            
            # Calculate heat fluxes (IMPLICIT: using T^(t+Δt))
            # Note: Uses future temperatures T_new (T^(t+Δt))
            q_fluxes = self._calculate_heat_fluxes_general(T_surface, T_new)
            
            # Update enthalpy: h^(t+Δt) = h^t + (Δt / m'') * q^(t+Δt)
            for i in range(self.model.n_layers):
                m_double_prime = self.model.rho[i] * self.model.dx[i]
                
                if m_double_prime > 0:
                    q_net = q_fluxes['in'][i] - q_fluxes['out'][i]
                    h_new[i] = h_old[i] + (self.dt / m_double_prime) * q_net
                else:
                    h_new[i] = h_old[i]
            
            # Update temperature from new enthalpy
            self.h = h_new.copy()
            self._update_temperature_from_enthalpy(h_old)
            T_new = self.model.T.copy()
            
            # Check convergence
            max_change_h = np.max(np.abs(h_new - h_prev))
            max_change_T = np.max(np.abs(T_new - T_prev))
            
            if max_change_h < self.tolerance and max_change_T < self.tolerance:
                # Converged!
                self.h = h_new
                self.model.T = T_new
                break
        
        # Final update of properties
        self._update_volume_fractions_from_enthalpy_first()
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity()
        
        # Update volumetric enthalpy
        for i in range(self.model.n_layers):
            self.model.H[i] = self.h[i] * self.model.rho[i]
        
        return True
    
    def _calculate_heat_fluxes(self, T_surface):
        """
        Calculate heat fluxes using CURRENT temperatures (for explicit method).
        
        This is a wrapper that calls the general function with self.model.T (T^t).
        
        Parameters
        ----------
        T_surface : float
            Surface temperature [°C]
        
        Returns
        -------
        dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²]
        """
        # EXPLICIT: Use current temperatures (T^t)
        return self._calculate_heat_fluxes_general(T_surface, self.model.T)
    
    def _calculate_heat_fluxes_implicit(self, T_surface, T):
        """
        Calculate heat fluxes using PROVIDED temperatures (for implicit method).
        
        This allows us to evaluate heat fluxes at t+Δt by passing T^(t+Δt).
        This is just an alias for clarity - calls the general function.
        
        Parameters
        ----------
        T_surface : float
            Surface temperature [°C]
        T : numpy.ndarray
            Layer temperatures to use for flux calculation [°C]
        
        Returns
        -------
        dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²]
        """
        # IMPLICIT: Use provided temperatures (T^(t+Δt))
        return self._calculate_heat_fluxes_general(T_surface, T)
    
    def _calculate_heat_fluxes_general(self, T_surface, T):
        """
        General function to calculate heat fluxes using any temperature array.
        
        This is the core implementation used by both explicit and implicit methods.
        The difference is which temperatures are passed:
        - Explicit: passes T^t (current temperatures)
        - Implicit: passes T^(t+Δt) (future temperatures)
        
        Parameters
        ----------
        T_surface : float
            Surface temperature [°C]
        T : numpy.ndarray
            Layer temperatures to use for flux calculation [°C]
            For explicit: T = self.model.T (T^t)
            For implicit: T = T_new (T^(t+Δt))
        
        Returns
        -------
        dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²]
        """
        n = self.model.n_layers
        q_in = np.zeros(n)
        q_out = np.zeros(n)
        
        # Calculate thermal resistances (using current properties)
        R = self.model.calculate_thermal_resistance()
        
        # Layer 0: heat input from surface
        R_surface = self.model.dx[0] / (2.0 * self.model.k[0])
        q_surface = (T_surface - T[0]) / R_surface
        q_in[0] = q_surface
        
        # Heat flux from layer 0 to layer 1
        if n > 1:
            q_01 = (T[0] - T[1]) / R[0]
            q_out[0] = q_01
            q_in[1] = q_01
        
        # Interior layers
        for i in range(1, n - 1):
            # Heat flux from layer i-1 to i
            q_in[i] = (T[i-1] - T[i]) / R[i-1]
            
            # Heat flux from layer i to i+1
            q_out[i] = (T[i] - T[i+1]) / R[i]
        
        # Last layer: only heat input from previous layer
        if n > 1:
            i = n - 1
            q_in[i] = (T[i-1] - T[i]) / R[i-1]
            # Assume adiabatic boundary at the end (no heat output)
            q_out[i] = 0.0
        
        return {'in': q_in, 'out': q_out}
    
    def _update_temperature_from_enthalpy(self, h_old):
        """
        Update temperature based on enthalpy using the piecewise function.
        
        IMPORTANT: L_f and L_H2O are per-layer values that depend on alpha_ice_initial.
        This is because h is the specific enthalpy of the MIXTURE, not pure ice.
        
        ΔT = {
            (h^(t+Δt) - h^t) / C_p,A,  if h < L_f[i] or h > L_H2O[i]
            0,                          if L_f[i] ≤ h ≤ L_H2O[i]
        }
        
        Also handles the "excessive melting" case where h^t < L_H2O[i] but h^(t+Δt) > L_H2O[i].
        
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
            
            # Get per-layer enthalpy thresholds (depend on alpha_ice_initial)
            L_f_i = self.L_f[i]
            L_H2O_i = self.L_H2O[i]
            
            # Get current specific heat capacity [J/(kg·K)]
            cp = self.model.cp[i]
            
            if cp <= 0:
                continue  # Skip if invalid cp
            
            # Determine enthalpy zone using per-layer thresholds
            if h_new < L_f_i:
                # Solid phase (ice below 0°C)
                # ΔT = Δh / C_p
                dT = dh / cp
                self.model.T[i] += dT
                
            elif h_new > L_H2O_i:
                # Check for excessive melting case
                if h_prev < L_H2O_i:
                    # Excessive melting: h^t < L_H2O[i] but h^(t+Δt) > L_H2O[i]
                    # Part 1: Use energy to bring h to L_H2O (melting)
                    dh_melting = L_H2O_i - h_prev
                    # Part 2: Remaining energy increases temperature
                    dh_heating = h_new - L_H2O_i
                    dT = dh_heating / cp
                    self.model.T[i] = T_melt + dT
                else:
                    # Liquid phase (water above 0°C)
                    # ΔT = Δh / C_p
                    dT = dh / cp
                    self.model.T[i] += dT
                    
            else:
                # Mushy zone: L_f[i] ≤ h ≤ L_H2O[i]
                # Temperature stays at melting point
                # But check if we entered the mushy zone
                if h_prev < L_f_i:
                    # Transitioning from solid to mushy
                    # Part 1: Heat to L_f[i]
                    dh_heating = L_f_i - h_prev
                    dT_part1 = dh_heating / cp
                    # Part 2: Remaining energy goes to melting (no temp change)
                    self.model.T[i] = T_melt
                elif h_prev > L_H2O_i:
                    # Transitioning from liquid to mushy (shouldn't happen, but handle it)
                    # Part 1: Cool to L_H2O[i]
                    dh_cooling = h_prev - L_H2O_i
                    dT_part1 = -dh_cooling / cp
                    # Part 2: Remaining energy goes to freezing (no temp change)
                    self.model.T[i] = T_melt
                else:
                    # Already in mushy zone, temperature stays constant
                    self.model.T[i] = T_melt
    
    def _update_volume_fractions_from_heat_flux_first(self, q_fluxes):
        """
        First volume fraction update based on heat flux (before thickness reduction).
        
        This is the FIRST update step that accounts for phase change (ice melting to water)
        based on the net heat flux. A second update will follow for thickness reduction.
        
        Uses the heat flux-based equations:
        
        α_ice,A^(t,1) = α_ice,A^t - (Δt * (q_in'' - q_out'')) / (h_sl * ρ_ice) * δ_A
        
        α_H2O,A^(t,1) = α_H2O,A^t + (Δt * (q_in'' - q_out'')) / (h_sl * ρ_H2O) * δ_A
        
        α_air,A^(t,1) = 1 - α_H2O,A^(t,1) - α_ice,A^(t,1)
        
        where:
        - h_sl = L_fusion (latent heat of fusion) [J/kg]
        - δ_A = dx[i] (layer thickness) [m]
        - q_in'' - q_out'' = net heat flux [W/m²]
        - Δt = time step [s]
        
        Parameters
        ----------
        q_fluxes : dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²] for each layer
        """
        n = self.model.n_layers
        h_sl = self.model.L_fusion  # Latent heat of fusion [J/kg]
        
        for i in range(n):
            # Calculate net heat flux for this layer
            q_net = q_fluxes['in'][i] - q_fluxes['out'][i]  # [W/m²]
            
            # Get layer thickness
            delta_A = self.model.dx[i]  # [m]
            
            # Store current volume fractions
            alpha_ice_t = self.model.alpha_ice[i]
            alpha_H2O_t = self.model.alpha_water[i]
            
            # Update ice volume fraction
            # α_ice,A^(t,1) = α_ice,A^t - (Δt * (q_in'' - q_out'')) / (h_sl * ρ_ice) * δ_A
            delta_alpha_ice = (self.dt * q_net) / (h_sl * self.model.rho_ice) * delta_A
            alpha_ice_new = alpha_ice_t - delta_alpha_ice
            
            # Ensure ice volume fraction doesn't go negative
            alpha_ice_new = np.maximum(alpha_ice_new, 0.0)
            
            # Update water volume fraction
            # α_H2O,A^(t,1) = α_H2O,A^t + (Δt * (q_in'' - q_out'')) / (h_sl * ρ_H2O) * δ_A
            delta_alpha_H2O = (self.dt * q_net) / (h_sl * self.model.rho_water) * delta_A
            alpha_H2O_new = alpha_H2O_t + delta_alpha_H2O
            
            # Ensure water volume fraction doesn't go negative
            alpha_H2O_new = np.maximum(alpha_H2O_new, 0.0)
            
            # Update air volume fraction (closure)
            # α_air,A^(t,1) = 1 - α_H2O,A^(t,1) - α_ice,A^(t,1)
            alpha_air_new = 1.0 - alpha_H2O_new - alpha_ice_new
            
            # Ensure air volume fraction doesn't go negative
            alpha_air_new = np.maximum(alpha_air_new, 0.0)
            
            # Normalize to ensure volume fractions sum to 1
            total = alpha_ice_new + alpha_H2O_new + alpha_air_new
            if total > 0:
                scale = 1.0 / total
                alpha_ice_new *= scale
                alpha_H2O_new *= scale
                alpha_air_new *= scale
            
            # Update model volume fractions
            self.model.alpha_ice[i] = alpha_ice_new
            self.model.alpha_water[i] = alpha_H2O_new
            self.model.alpha_air[i] = alpha_air_new
    
    def _update_volume_fractions_from_enthalpy_first(self):
        """
        First volume fraction update based on enthalpy (before thickness reduction).
        
        This is the FIRST update step that accounts for phase change (ice melting to water)
        based on the current enthalpy state. A second update will follow for thickness reduction.
        
        Uses the mixture enthalpy equation:
        h_A,i = (α_ice,A * ρ_ice * h_ice + α_H2O * ρ_H2O * h_H2O + α_air * ρ_air * h_air) / ρ_A
        
        where:
        - h_ice = cp_ice * (T_K - T_ref_K)
        - h_H2O = cp_H2O * (T_K - T_ref_K)
        - h_air = cp_air * (T_K - T_ref_K)
        - ρ_A = α_ice,A * ρ_ice + α_H2O * ρ_H2O + α_air * ρ_air
        
        IMPORTANT: Uses per-layer L_f and L_H2O thresholds that depend on alpha_ice_initial.
        
        In the mushy zone (L_f[i] ≤ h ≤ L_H2O[i]), the fraction of ice that has melted
        is proportional to (h - L_f[i]) / (L_H2O[i] - L_f[i]).
        
        Mass conservation is enforced: alpha_ice * rho_ice = alpha_water * rho_water
        """
        n = self.model.n_layers
        
        for i in range(n):
            h = self.h[i]
            alpha_ice_init = self.model.alpha_ice_initial[i]
            
            # Get per-layer enthalpy thresholds
            L_f_i = self.L_f[i]
            L_H2O_i = self.L_H2O[i]
            
            if h < L_f_i:
                # Solid phase: all ice remains
                self.model.alpha_ice[i] = alpha_ice_init
                self.model.alpha_water[i] = 0.0
                # Air fraction remains constant
                
            elif h > L_H2O_i:
                # Liquid phase: all ice has melted
                # Mass conservation: alpha_ice * rho_ice = alpha_water * rho_water
                # Therefore: alpha_water = alpha_ice_init * rho_ice / rho_water
                self.model.alpha_ice[i] = 0.0
                self.model.alpha_water[i] = alpha_ice_init * self.model.rho_ice / self.model.rho_water
                # Air fraction remains constant
                
            else:
                # Mushy zone: partial melting
                # Fraction melted = (h - L_f[i]) / (L_H2O[i] - L_f[i])
                if L_H2O_i > L_f_i:
                    fraction_melted = (h - L_f_i) / (L_H2O_i - L_f_i)
                else:
                    fraction_melted = 0.0
                
                fraction_melted = np.clip(fraction_melted, 0.0, 1.0)
                
                # Update volume fractions with mass conservation
                # Remaining ice: volume fraction decreases proportionally
                self.model.alpha_ice[i] = alpha_ice_init * (1.0 - fraction_melted)
                
                # Melted ice becomes water: mass is conserved
                # Mass of melted ice = alpha_ice_init * rho_ice * fraction_melted
                # This becomes water: alpha_water * rho_water = alpha_ice_init * rho_ice * fraction_melted
                # Therefore: alpha_water = alpha_ice_init * rho_ice * fraction_melted / rho_water
                self.model.alpha_water[i] = alpha_ice_init * self.model.rho_ice * fraction_melted / self.model.rho_water
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
