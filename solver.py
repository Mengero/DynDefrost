"""
Solver Module for Dynamic Defrost Model

This module implements the numerical solver for the 1-D defrost problem using
an enthalpy-based approach to handle phase change.

The solver updates enthalpy based on heat flux, then updates temperature
according to enthalpy zones (solid, mushy, liquid).
"""

from re import T
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from model_init import DefrostModel
from surface_retention import calculate_surface_retention


class DefrostSolver:
    """
    Implicit solver for the 1-D defrost problem.
    
    Uses fully implicit (backward Euler) method for stability with large gradients.
    """
    
    def __init__(self, model, dt=0.1, method='explicit', h_conv=10.0, T_ambient=None, verbose=True):
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
        h_conv : float, optional
            Convective heat transfer coefficient for natural convection [W/(m²·K)].
            Default: 5.0
        T_ambient : float, optional
            Ambient air temperature [°C]. If None, will use model.T_ambient if set.
            Default: None
        verbose : bool, optional
            If True, print progress messages during simulation. Default: True
        """
        self.model = model
        self.dt = dt
        self.method = method.lower()
        self.h_conv = h_conv if h_conv is not None else 5.0  # Convective heat transfer coefficient [W/(m²·K)]
        self.T_ambient = T_ambient if T_ambient is not None else model.T_ambient
        self.verbose = verbose
        
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
        # Reference temperature per layer [K] (initial state where h = 0); used when recalculating L_f, L_H2O
        self.T_ref_K = None
        
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
        self.dx_history = []  # Layer thickness history
        self.shrinkage_rate_history = []  # Shrinkage rate history [m/s] per layer
        
        # Store previous time step volume fractions for timewise updates
        self.alpha_ice_prev = None
        self.alpha_water_prev = None
        self.alpha_air_prev = None
        
        # Store mass per unit area for ice and water (for second update)
        self.m_double_prime_ice = None  # Mass per unit area of ice [kg/m²]
        self.m_double_prime_water = None  # Mass per unit area of water [kg/m²]
        
        # Shrinkage model parameters
        self.sigma = 72e-3  # Surface tension [N/m] (water-air at 0°C)
        self.eta_0 = 2e4  # Base viscosity [Pa·s]
        self.b = 3.0  # Structural constant (typically 2-4)
        self.C_wet = 700  # Lubricant constant
        self.d_ice_initial = 5e-4  # Initial ice grain diameter [m] (10 microns, typical)
        
        # Store initial ice grain diameter per layer
        self.d_ice_i = np.full(model.n_layers, self.d_ice_initial)
        
        # Critical detachment (sloughing) parameters
        # Calibrated k values for different contact angles
        self.k_60 = 245.2220   # Retention coefficient for 60° contact angle
        self.k_140 = 8000  # Retention coefficient for 140° contact angle
        
        # Calibrated f(alpha_water) function parameters
        # f(alpha_water) = A * log(B * alpha_water + e) + C
        self.A_crit = -0.051472  # Multiplier for log (negative)
        self.B_crit = 160.4478   # Multiplier for alpha_water
        self.e_crit = 0.903741   # Offset to prevent singularity
        self.C_crit = 0.3466     # Plateau/base value
        
        # Gravitational acceleration
        self.g = 9.81  # [m/s²]
        
        # Critical thickness and sloughing history
        self.h_crit_history = []
        self.h_total_history = []
        self.sloughing_status_history = []  # True if sloughing occurs (h_total < h_crit)
        
        # Temporary storage for shrinkage rates (updated each time step)
        self.current_shrinkage_rates = None
        
        # Flag to indicate if all layers have become water (simulation should end)
        self._all_layers_water = False
        
        # Track which layer is currently the beginning (facing air) and end (close to wall)
        # These may change after diffusion when outer layers melt
        self.begin_idx = 0  # Initially layer 0 faces air
        self.end_idx = model.n_layers - 1 if model.n_layers > 0 else None  # Initially last layer is close to wall
    
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
        # Store per layer for later recalculation when mass enters a layer (e.g. water diffusion)
        self.T_ref_K = (self.model.T + 273.15).copy()  # [K]
        T_ref_K = self.T_ref_K
        
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

        if self.verbose:
            print(f"Enthalpy thresholds calculated:")
            print(f"  L_f range: [{np.min(self.L_f):.2f}, {np.max(self.L_f):.2f}] J/kg")
            print(f"  L_H2O range: [{np.min(self.L_H2O):.2f}, {np.max(self.L_H2O):.2f}] J/kg")
    
    def _recalculate_layer_enthalpy_thresholds(self, layer_idx, alpha_ice_eff, alpha_air_eff):
        """
        Recalculate L_f and L_H2O for a single layer given effective initial composition.
        Used when mass (e.g. melted water) has entered the control volume, so the effective
        "initial" ice content (if we think of the added water as having been ice) is larger.
        
        Parameters
        ----------
        layer_idx : int
            Layer index
        alpha_ice_eff : float
            Effective initial ice volume fraction (ice + water-as-ice equivalent)
        alpha_air_eff : float
            Effective initial air volume fraction (1 - alpha_ice_eff for no water)
        """
        T_melt_K = self.model.T_melt + 273.15
        T_ref_K_i = self.T_ref_K[layer_idx]
        
        # L_f: mixture enthalpy at melting point (ice + air, no water)
        h_at_melt = self._calculate_mixture_enthalpy(
            T_K=T_melt_K,
            T_ref_K=T_ref_K_i,
            alpha_ice=alpha_ice_eff,
            alpha_water=0.0,
            alpha_air=alpha_air_eff
        )
        self.L_f[layer_idx] = h_at_melt
        
        # L_H2O: mixture enthalpy when all ice has melted
        alpha_water_melted = alpha_ice_eff * self.model.rho_ice / self.model.rho_water
        h_all_melted = self._calculate_mixture_enthalpy(
            T_K=T_melt_K,
            T_ref_K=T_ref_K_i,
            alpha_ice=0.0,
            alpha_water=alpha_water_melted,
            alpha_air=alpha_air_eff
        )
        rho_A = (alpha_ice_eff * self.model.rho_ice + alpha_air_eff * self.model.rho_air)
        if rho_A > 0:
            mass_fraction_ice = (alpha_ice_eff * self.model.rho_ice) / rho_A
            latent_heat_per_mass_mixture = mass_fraction_ice * self.model.L_fusion
        else:
            latent_heat_per_mass_mixture = 0.0
        self.L_H2O[layer_idx] = h_all_melted + latent_heat_per_mass_mixture
    
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
        
        # Initialize previous time step volume fractions
        self.alpha_ice_prev = self.model.alpha_ice.copy()
        self.alpha_water_prev = self.model.alpha_water.copy()
        self.alpha_air_prev = self.model.alpha_air.copy()
        
        # Initialize mass per unit area for ice and water
        # m''_ice = α_ice * ρ_ice * δ_A
        # m''_water = α_water * ρ_water * δ_A
        self.m_double_prime_ice = self.model.alpha_ice * self.model.rho_ice * self.model.dx
        self.m_double_prime_water = self.model.alpha_water * self.model.rho_water * self.model.dx
        
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
        
        # Step 0: Check for diffusion/merging of melted layers
        self._process_water_diffusion()
        
        # Update boundary condition
        self.model.T_surface = T_surface
        
        # Store old state
        h_old = self.h.copy()
        T_old = self.model.T.copy()
        
        # Update properties based on current state
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
        
        # Step 1: Calculate heat fluxes (EXPLICIT: using T^t)
        # Note: Uses current temperatures self.model.T (T^t)
        q_fluxes = self._calculate_heat_fluxes_general(T_surface, self.model.T)
        
        # Step 2: Update enthalpy for each layer (EXPLICIT)
        # h_A^(t+Δt) = h_A^t + (Δt / m'') * (q_in'' - q_out'')^t
        # Only update layers from begin_idx to end_idx (active layers)       
        for i in range(self.begin_idx, self.end_idx + 1):
            
            m_double_prime = self.model.rho[i] * self.model.dx[i]
            
            if m_double_prime > 0:
                q_net = q_fluxes['in'][i] - q_fluxes['out'][i]
                self.h[i] = h_old[i] + (self.dt / m_double_prime) * q_net
                
                # if abs((self.dt / m_double_prime) * q_net) > abs(self.h[i]):
                #     print(f"Warning: Layer {i} enthalpy change is too large: {abs((self.dt / m_double_prime) * q_net)} J/kg")
            else:
                self.h[i] = h_old[i]
        
        # Step 3: Update temperature based on enthalpy
        self._update_temperature_from_enthalpy(h_old)
        
        
        # Step 4: Update volume fractions based on heat flux (using previous time step)
        # This uses the equations: α^(t,1) = α^t - (Δt * q_net) / (h_sl * ρ) * δ
        self._update_volume_fractions_from_heat_flux_first(q_fluxes)
        

        # Step 5: Calculate layer shrinkage (thickness reduction)
        # This uses: 1/δ_A * d(δ_A)/dt = -σ / (η_eff * r_pore)
        shrinkage_rates = self._calculate_layer_shrinkage()
        self.current_shrinkage_rates = shrinkage_rates.copy()  # Store for history
        
        # Step 6: Second volume fraction update based on mass per unit area and new thickness
        # This recalculates volume fractions using mass conservation after thickness change
        self._update_volume_fractions_second(q_fluxes)
        
        # Step 7: Recalculate properties with new volume fractions and thicknesses
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
        
        # Step 8: Calculate critical sloughing thickness and check if frost can survive
        sloughing_info = self._check_sloughing()
        # Store sloughing info temporarily (will be saved to history in solve() if needed)
        self._latest_sloughing_info = sloughing_info
        
        # Check if sloughing occurs
        if sloughing_info['sloughing']:
            # Calculate water retention from the layer closest to the wall (first layer)
            if self.model.n_layers > 0:
                # Calculate effective density BEFORE modifying layers (at moment of sloughing)
                rho_eff = self._calculate_effective_density()
                # Store rho_eff in sloughing info for access by other modules
                self._latest_sloughing_info['rho_eff'] = rho_eff
                
                # Water mass per unit area in the first layer
                water_retention = self.model.alpha_water[0] * self.model.rho_water * self.model.dx[0]
                self.model.surface_retention = water_retention  # [kg/m²]
                self.model.surface_retention_thickness = water_retention / self.model.rho_water  # [m]
                
                # Set everything to 0 except the layer closest to the wall
                # Set all layers except first to zero thickness
                for i in range(1, self.model.n_layers):
                    self.model.dx[i] = 0.0
                    self.model.alpha_ice[i] = 0.0
                    self.model.alpha_water[i] = 0.0
                    self.model.alpha_air[i] = 0.0
                    self.model.T[i] = 0.0
                    self.model.H[i] = 0.0
                    self.m_double_prime_ice[i] = 0.0
                    self.m_double_prime_water[i] = 0.0
                
                # Set first layer: alpha_water = 1, everything else = 0
                self.model.dx[0] = self.model.surface_retention_thickness
                self.model.alpha_ice[0] = 0.0
                self.model.alpha_water[0] = 1.0
                self.model.alpha_air[0] = 0.0
                # Update masses for first layer
                self.m_double_prime_ice[0] = 0.0
                self.m_double_prime_water[0] = self.model.alpha_water[0] * self.model.rho_water * self.model.dx[0]

                if self.verbose:
                    print(f"\n{'='*70}")
                    print("SLOUGHING DETECTED!")
                    print(f"{'='*70}")
                    print(f"  h_total = {sloughing_info['h_total']*1000:.4f} mm")
                    print(f"  h_crit = {sloughing_info['h_crit']*1000:.4f} mm")
                    print(f"  Safety factor = {sloughing_info['safety_factor']:.4f}")
                    print(f"  Effective density = {rho_eff:.2f} kg/m³")
                    print(f"  Surface water retention = {water_retention*1000:.2f} g/m²")
                    print(f"  Surface water retention thickness = {self.model.surface_retention_thickness*1000:.4f} mm")
                    print(f"  Layer 0: alpha_water = {self.model.alpha_water[0]:.3f}, thickness = {self.model.dx[0]*1000:.4f} mm")
                    print(f"  All other layers set to zero")
                    print(f"{'='*70}\n")
            
            # Return False to indicate simulation should stop
            return False
        
        # Update volumetric enthalpy
        for i in range(self.model.n_layers):
            self.model.H[i] = self.h[i] * self.model.rho[i]
        
        # Store current volume fractions as previous for next time step
        self.alpha_ice_prev = self.model.alpha_ice.copy()
        self.alpha_water_prev = self.model.alpha_water.copy()
        self.alpha_air_prev = self.model.alpha_air.copy()
        
        return True
    
    def _solve_time_step_implicit(self, T_surface):
        """
        Implicit solver: heat fluxes evaluated at future time t+Δt.
        
        TODO: Implement implicit solver properly.
        For now, this is a placeholder that raises an error.
        """
        raise NotImplementedError("Implicit solver is not yet implemented. Please use explicit solver.")
    
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
        
        Uses tracked begin_idx and end_idx to handle cases where
        outer layers have melted and diffused away.
        
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
        
        # Skip layers that have been diffused away (dx = 0)
        # Only calculate fluxes for layers with non-zero thickness
        active_layers = np.where(self.model.dx > 1e-10)[0]
        
        if len(active_layers) == 0:
            # No active layers
            return {'in': q_in, 'out': q_out}
        
        # Check if active layers are adjacent (no gaps)
        # Active layers should form a continuous range from begin_idx to end_idx
        active_sorted = np.sort(active_layers)
        if len(active_sorted) > 1:
            # Check for gaps between consecutive active layers
            for i in range(len(active_sorted) - 1):
                if active_sorted[i + 1] - active_sorted[i] > 1:
                    # Non-adjacent layers detected - this is not realistic
                    raise ValueError(
                        f"Non-adjacent active layers detected in heat flux calculation. "
                        f"Active layers: {active_sorted}, gap between {active_sorted[i]} and {active_sorted[i+1]}. "
                        f"This indicates layers have been incorrectly diffused away, creating unrealistic gaps."
                    )
        
        # Get current begin and end layer indices
        begin_idx = self.begin_idx
        end_idx = self.end_idx
        
        if begin_idx is None or end_idx is None:
            # Fallback to original behavior if not set
            begin_idx = 0
            end_idx = n - 1
        
        # Calculate thermal resistances (using current properties)
        R = self.model.calculate_thermal_resistance()
        
        # Begin layer: heat input from surface (only if it's an active layer)
        if begin_idx in active_layers:
            R_surface = self.model.dx[begin_idx] / (2.0 * self.model.k[begin_idx])
            q_surface = (T_surface - T[begin_idx]) / R_surface
            q_in[begin_idx] = q_surface
        
        # Calculate heat fluxes between active layers
        # Active layers are guaranteed to be adjacent (checked above)
        active_sorted = np.sort(active_layers)
        
        # Interior heat fluxes between consecutive active layers
        for i in range(len(active_sorted) - 1):
            layer_i = active_sorted[i]
            layer_next = active_sorted[i + 1]
            
            # Since active layers are adjacent, layer_next == layer_i + 1
            # Use direct resistance R[layer_i]
            if layer_i < len(R):
                R_effective = R[layer_i]
                if R_effective > 0:
                    q_flux = (T[layer_i] - T[layer_next]) / R_effective
                    q_out[layer_i] = q_flux
                    q_in[layer_next] = q_flux
        
        # End layer: convective heat loss to ambient air (only if it's an active layer)
        if end_idx in active_layers:
            # Heat input from previous layer (if exists)
            # Since active layers are adjacent, the previous active layer is end_idx - 1
            # and the flux has already been calculated above
            
            # Convective heat transfer to ambient air with conductive resistance through half layer
            if self.T_ambient is not None:
                dx_end = self.model.dx[end_idx]  # End layer thickness [m]
                k_end = self.model.k[end_idx]    # End layer thermal conductivity [W/(m·K)]
                # Thermal resistance: convective + conductive through half layer thickness
                R_conv = 1.0 / self.h_conv  # Convective resistance [m²·K/W]
                R_cond = (dx_end / 2.0) / k_end  # Conductive resistance [m²·K/W]
                R_total = R_conv + R_cond
                q_out[end_idx] = (T[end_idx] - self.T_ambient) / R_total  # [W/m²]
            else:
                # Fallback to adiabatic if ambient temperature not set
                q_out[end_idx] = 0.0
            
            # if abs(q_in[end_idx]) > 200 or abs(q_out[end_idx]) > 200:
            #     print(f"Warning: End layer {end_idx} heat flux is too large: {abs(q_in[end_idx])} W/m² or {abs(q_out[end_idx])} W/m²")
        
        # Check for large fluxes in begin layer
        # if begin_idx in active_layers:
        #     if abs(q_in[begin_idx]) > 200 or abs(q_out[begin_idx]) > 200:
        #         print(f"Warning: Begin layer {begin_idx} heat flux is too large: {abs(q_in[begin_idx])} W/m² or {abs(q_out[begin_idx])} W/m²")
        
        return {'in': q_in, 'out': q_out}

    
    def _update_layer_below_mushy_zone(self, layer_idx, total_mass, h_new, h_old):
        """
        Update layer when enthalpy is below mushy zone (all water becomes ice).
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer
        total_mass : float
            Total mass after merging [kg/m²]
        h_new : float
            New enthalpy [J/kg]
        h_old : float
            Old enthalpy [J/kg]
        """
        dx = self.model.dx[layer_idx]
        
        if dx > 0:
            alpha_ice_new = total_mass / (self.model.rho_ice * dx)
            self.model.alpha_ice[layer_idx] = alpha_ice_new
            self.model.alpha_water[layer_idx] = 0.0
            
            # Update mass per unit area
            if self.m_double_prime_ice is not None:
                self.m_double_prime_ice[layer_idx] = alpha_ice_new * self.model.rho_ice * dx
            if self.m_double_prime_water is not None:
                self.m_double_prime_water[layer_idx] = 0.0
            
            # Recalculate properties
            self.model.calculate_specific_heat()
            self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
            
            # Update temperature
            cp = self.model.cp[layer_idx]
            if cp > 0:
                T_ref = self.model.T[layer_idx]
                dh = h_new - h_old
                dT = dh / cp
                self.model.T[layer_idx] = T_ref + dT
    
    def _update_layer_above_mushy_zone(self, layer_idx, total_water_mass, ice_mass, h_new, h_old, ice_layers):
        """
        Update layer when enthalpy is above mushy zone (ice becomes water, diffuses to next layer).
        
        Parameters
        ----------
        layer_idx : int
            Index of the receiving layer
        total_water_mass : float
            Total water mass after merging [kg/m²]
        ice_mass : float
            Ice mass in receiving layer [kg/m²]
        h_new : float
            New enthalpy [J/kg]
        h_old : float
            Old enthalpy [J/kg]
        ice_layers : numpy.ndarray
            Array of ice layer indices
        
        Returns
        -------
        bool
            True if all layers became water, False otherwise
        """
        # Convert all ice to water
        total_water_after_conversion = total_water_mass + ice_mass
        
        # When all ice melts, the resulting water has enthalpy h_new
        # (h_new already accounts for the latent heat of melting)
        water_enthalpy = h_new
        
        # Find next nearby ice layer (excluding current one)
        remaining_ice_layers = ice_layers[ice_layers != layer_idx]
        
        if len(remaining_ice_layers) > 0:
            # Find nearest ice layer
            distances = np.abs(remaining_ice_layers - layer_idx)
            next_ice_idx = remaining_ice_layers[np.argmin(distances)]
            
            # Set current layer to all water first (alpha_air will be calculated automatically)
            self._set_layer_to_all_water(layer_idx, total_water_after_conversion)
            
            # Recursively merge all water into next ice layer (will check mushy zone iteratively)
            all_water = self._merge_water_into_ice_layer(next_ice_idx, total_water_after_conversion, water_enthalpy)
            return all_water
        else:
            # No more ice layers - all layers become water
            self._set_layer_to_all_water(layer_idx, total_water_after_conversion)
            return True  # All layers are water
    
    def _update_layer_in_mushy_zone(self, layer_idx, water_mass_total, alpha_ice_keep):
        """
        Update layer when enthalpy is in mushy zone (keep alpha_ice same, increase alpha_water).
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer
        water_mass_total : float
            Total water mass after merging [kg/m²]
        alpha_ice_keep : float
            Alpha ice to keep (unchanged)
        """
        dx = self.model.dx[layer_idx]  # dx is constant
        
        if dx > 0:
            # Calculate alpha_water from water mass
            water_volume = water_mass_total / self.model.rho_water
            self.model.alpha_water[layer_idx] = water_volume / dx
            
            # Keep alpha_ice the same
            self.model.alpha_ice[layer_idx] = alpha_ice_keep
            
            # Calculate alpha_air from constraint: alpha_air = 1 - alpha_ice - alpha_water
            self.model.alpha_air[layer_idx] = 1.0 - alpha_ice_keep - self.model.alpha_water[layer_idx]
            
            # Update mass per unit area
            if self.m_double_prime_water is not None:
                self.m_double_prime_water[layer_idx] = water_mass_total
            
            # Recalculate properties
            self.model.calculate_specific_heat()
            self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
            
            # Temperature stays at melting point
            self.model.T[layer_idx] = self.model.T_melt
    
    def _set_layer_to_all_water(self, layer_idx, water_mass):
        """
        Set a layer to all water (no ice).
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer
        water_mass : float
            Water mass [kg/m²]
        """
        dx = self.model.dx[layer_idx]  # dx is constant
        
        self.model.alpha_ice[layer_idx] = 0.0
        
        # Calculate alpha_water from water mass
        if dx > 0:
            water_volume = water_mass / self.model.rho_water
            self.model.alpha_water[layer_idx] = water_volume / dx
        else:
            self.model.alpha_water[layer_idx] = 0.0
        
        # Calculate alpha_air from constraint: alpha_air = 1 - alpha_ice - alpha_water
        self.model.alpha_air[layer_idx] = 1.0 - 0.0 - self.model.alpha_water[layer_idx]
        if self.m_double_prime_ice is not None:
            self.m_double_prime_ice[layer_idx] = 0.0
        if self.m_double_prime_water is not None:
            self.m_double_prime_water[layer_idx] = water_mass
        
        # Recalculate properties
        self.model.calculate_specific_heat()
        self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
    
    def _reset_melted_layer(self, layer_idx):
        """
        Reset a melted layer to all zeros.
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer to reset
        """
        self.model.alpha_ice[layer_idx] = 0.0
        self.model.alpha_water[layer_idx] = 0.0
        self.model.alpha_air[layer_idx] = 0.0
        self.model.dx[layer_idx] = 0.0
        self.model.T[layer_idx] = 0.0
        self.h[layer_idx] = 0.0
        if self.m_double_prime_ice is not None:
            self.m_double_prime_ice[layer_idx] = 0.0
        if self.m_double_prime_water is not None:
            self.m_double_prime_water[layer_idx] = 0.0
    
    def _merge_water_into_ice_layer(self, layer_idx, water_mass_to_add, water_enthalpy):
        """
        Merge water mass into an ice layer iteratively.
        
        Process:
        1. Add water mass from diffusion layer(s) to receiver layer water mass
        2. Update volume fraction of water in receiver layer
        3. Keep volume fraction of ice the same
        4. Update volume fraction of air (from constraint: alpha_air = 1 - alpha_ice - alpha_water)
        5. Calculate receiver's new enthalpy
        6. Check if it's in/below/above mushy zone:
           - If below: update properties and temperature
           - If in: update properties and temperature
           - If above: all ice becomes water, add to diffusion water mass, bring to next receiving layer, repeat
        
        Parameters
        ----------
        layer_idx : int
            Index of the ice layer to merge water into
        water_mass_to_add : float
            Water mass to add to the layer [kg/m²] (from diffusion layer)
        water_enthalpy : float
            Specific enthalpy of the water being added [J/kg]
        
        Returns
        -------
        bool
            True if all layers became water, False otherwise
        """
        # Get current layer properties
        dx = self.model.dx[layer_idx]  # dx is constant during merging
        alpha_ice = self.model.alpha_ice[layer_idx]  # Will be kept the same initially
        
        # Step 1: Get current water mass in receiver layer
        if self.m_double_prime_water is not None:
            water_mass_receiver = self.m_double_prime_water[layer_idx]
        else:
            alpha_water_current = self.model.alpha_water[layer_idx]
            water_mass_receiver = alpha_water_current * self.model.rho_water * dx
        
        # Step 2: Add water mass from diffusion layer(s) to receiver layer water mass
        total_water_mass = water_mass_receiver + water_mass_to_add
        
        # Step 3: Update volume fraction of water in receiver layer
        if dx > 0:
            water_volume = total_water_mass / self.model.rho_water
            alpha_water_new = water_volume / dx
        else:
            alpha_water_new = 0.0
        
        # Step 4: Keep volume fraction of ice the same
        alpha_ice_keep = alpha_ice
        
        # Step 5: Update volume fraction of air (from constraint)
        alpha_air_new = 1.0 - alpha_ice_keep - alpha_water_new
        
        # Step 6: Calculate receiver's new enthalpy
        # Use mass-weighted average: h_new = (h_receiver * m_receiver + h_diffusion * m_diffusion) / m_total
        ice_mass = alpha_ice_keep * self.model.rho_ice * dx
        air_mass = alpha_air_new * self.model.rho_air * dx
        current_total_mass = ice_mass + water_mass_receiver + air_mass
        new_total_mass = current_total_mass + water_mass_to_add
        
        if new_total_mass <= 0:
            return False
        
        h_current = self.h[layer_idx]
        h_new = (h_current * current_total_mass + water_enthalpy * water_mass_to_add) / new_total_mass
        
        # Get mushy zone thresholds for this layer
        L_f_i = self.L_f[layer_idx]
        L_H2O_i = self.L_H2O[layer_idx]
        
        # Step 7: Check mushy zone and update accordingly
        if h_new > L_H2O_i:
            # Above mushy zone: all ice in receiving layer becomes water
            # Add this to the diffusion water mass, bring to next receiving layer, repeat
            total_water_after_conversion = total_water_mass + ice_mass
            
            # Calculate enthalpy of water being diffused (weighted average of all water)
            if total_water_after_conversion > 0:
                # Weighted average: (receiver_water * h_receiver + diffusion_water * h_diffusion + ice * h_after_melting) / total
                # For ice that melts, use h_new as its enthalpy (since it's all water now)
                water_enthalpy_for_next = (
                    water_mass_receiver * h_current +
                    water_mass_to_add * water_enthalpy +
                    ice_mass * h_new
                ) / total_water_after_conversion
            else:
                water_enthalpy_for_next = h_new
            
            # Set current layer to all water (alpha_air will be calculated automatically)
            self._set_layer_to_all_water(layer_idx, total_water_after_conversion)
            
            # Find next nearby ice layer
            ice_layers = np.where(self.model.alpha_ice > 1e-10)[0]
            remaining_ice_layers = ice_layers[ice_layers != layer_idx]
            
            if len(remaining_ice_layers) > 0:
                # Find nearest ice layer
                distances = np.abs(remaining_ice_layers - layer_idx)
                next_ice_idx = remaining_ice_layers[np.argmin(distances)]
                
                # Recursively merge into next layer (repeat the same process)
                return self._merge_water_into_ice_layer(
                    next_ice_idx, total_water_after_conversion, water_enthalpy_for_next
                )
            else:
                # No more ice layers - all layers become water
                self._all_layers_water = True
                return True
                
        elif h_new < L_f_i:
            # Below mushy zone: update properties and temperature
            self.h[layer_idx] = h_new
            
            # All water becomes ice
            if dx > 0:
                alpha_ice_new = new_total_mass / (self.model.rho_ice * dx)
                self.model.alpha_ice[layer_idx] = alpha_ice_new
                self.model.alpha_water[layer_idx] = 0.0
                # Calculate alpha_air from constraint
                self.model.alpha_air[layer_idx] = 1.0 - alpha_ice_new - 0.0
                
                # Update mass per unit area
                if self.m_double_prime_ice is not None:
                    self.m_double_prime_ice[layer_idx] = alpha_ice_new * self.model.rho_ice * dx
                if self.m_double_prime_water is not None:
                    self.m_double_prime_water[layer_idx] = 0.0
                
                # Recalculate properties after volume fraction changes
                self.model.calculate_specific_heat()
                self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
                
                # Update temperature
                cp = self.model.cp[layer_idx]
                if cp > 0:
                    T_ref = self.model.T[layer_idx]
                    dh = h_new - h_current
                    dT = dh / cp
                    self.model.T[layer_idx] = T_ref + dT
            
            return False
            
        else:
            # In mushy zone: update properties and temperature
            self.h[layer_idx] = h_new
            
            # Update volume fractions (already calculated above)
            self.model.alpha_water[layer_idx] = alpha_water_new
            self.model.alpha_ice[layer_idx] = alpha_ice_keep
            self.model.alpha_air[layer_idx] = alpha_air_new
            
            # Update mass per unit area
            if self.m_double_prime_water is not None:
                self.m_double_prime_water[layer_idx] = total_water_mass
            
            # Recalculate properties after volume fraction changes
            self.model.calculate_specific_heat()
            self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
            
            # Update temperature (stays at melting point in mushy zone, but update for consistency)
            self.model.T[layer_idx] = self.model.T_melt
            
            return False
    
    def _process_water_diffusion(self):
        """
        Process water diffusion from melted layers into receiving ice layers.
        
        This function implements the diffusion equations:
        1. Add water mass from diffusion layer(s) to receiver layer: 
           m^1_{R,H_2O} = m^0_{R,H_2O} + m_{D,H_2O}
        2. Update volume fraction of water in receiver layer:
           α^1_{R,H_2O} = m^1_{R,H_2O} / (ρ_{H_2O} * δ_R)
        3. Keep volume fraction of ice the same:
           α^1_{R,ice} = α^0_{R,ice}
        4. Update volume fraction of air from constraint:
           α^1_{R,air} = 1 - α^1_{R,H_2O} - α^1_{R,ice}
        5. Calculate receiver's new enthalpy
        6. Check if in/below/above mushy zone and update accordingly
        
        When layers completely melt (alpha_ice = 0), their water is merged into the nearest
        layer that still has ice. If multiple layers melt at the same time, they all merge
        into the same nearest ice layer. This prevents accumulation of pure water layers.
        
        This function should be called at the beginning of each time step,
        before updating boundary conditions and calculating heat fluxes.
        """
        n = self.model.n_layers
        if n == 0:
            return
        
        # Find all layers that are completely melted (alpha_ice = 0)
        melted_layers = np.where(self.model.alpha_ice < 1e-10)[0]
        
        if len(melted_layers) == 0:
            return  # No melted layers to merge
        
        # Find layers with ice (alpha_ice > 0)
        ice_layers = np.where(self.model.alpha_ice > 1e-10)[0]
        
        if len(ice_layers) == 0:
            # All layers are melted - nothing to merge into
            # Just reset all layers to 0
            for i in melted_layers:
                self._reset_melted_layer(i)
            self._all_layers_water = True
            return
        
        # Group melted layers by their nearest ice layer
        # This handles cases where melted layers are at both ends (i=0 and i=n-1)
        # Strategy: group melted layers by which ice layer is nearest to them
        if len(melted_layers) > 0:
            # Create a mapping: ice_layer_idx -> list of melted layers that are nearest to it
            ice_to_melted = {}
            
            for melted_idx in melted_layers:
                # Find nearest ice layer for this melted layer
                distances = np.abs(ice_layers - melted_idx)
                nearest_ice_idx = ice_layers[np.argmin(distances)]
                
                # Group this melted layer with others that have the same nearest ice layer
                if nearest_ice_idx not in ice_to_melted:
                    ice_to_melted[nearest_ice_idx] = []
                ice_to_melted[nearest_ice_idx].append(melted_idx)
            
            # Now start the diffusion process for each group
            for nearest_ice_idx, melted_group in ice_to_melted.items():
                # Step 1: Collect total water mass and total water enthalpy from diffusion layer(s)
                total_water_mass = 0.0
                total_water_enthalpy = 0.0
                
                for melted_idx in melted_group:
                    # Calculate water mass from melted layer
                    water_mass = self.m_double_prime_water[melted_idx]
                    total_water_mass += water_mass
                    
                    # Collect enthalpy contribution (weighted by mass)
                    if water_mass > 0:
                        total_water_enthalpy += self.h[melted_idx] * water_mass
                
                # Step 2: Start diffusion process (recursive function)
                # This function will handle all the equations and recursive calls if above L_H2O
                self._diffuse_water_into_layer(nearest_ice_idx, total_water_mass, total_water_enthalpy, ice_layers)
                
                # Step 3: Reset all melted layers in this group to 0
                for melted_idx in melted_group:
                    self._reset_melted_layer(melted_idx)
        
        # After diffusion, update which layers are facing air and close to wall
        # Outer layers (i=0, i=1, etc.) may have melted and diffused away
        self._update_surface_and_wall_layers()
        
        # Check if all layers are water - if so, simulation should end
        # This check happens after all merging is complete
        remaining_ice_layers_after = np.where(self.model.alpha_ice > 1e-10)[0]
        if len(remaining_ice_layers_after) == 0:
            # All layers are water - simulation ends
            # Set a flag that can be checked by the solver
            self._all_layers_water = True
    
    def _diffuse_water_into_layer(self, layer_idx, water_mass_to_add, water_enthalpy_to_add, ice_layers):
        """
        Recursive function to diffuse water into an ice layer.
        
        Implements the diffusion equations:
        1. Check capacity: Calculate maximum water that can fit based on air volume fraction
        2. Add water mass up to capacity: m^1_{R,H_2O} = m^0_{R,H_2O} + min(m_{D,H_2O}, capacity)
        3. α^1_{R,H_2O} = m^1_{R,H_2O} / (ρ_{H_2O} * δ_R)  (update alpha_water)
        4. α^1_{R,ice} = α^0_{R,ice}  (keep alpha_ice same)
        5. α^1_{R,air} = 1 - α^1_{R,H_2O} - α^1_{R,ice}  (update alpha_air)
        6. Calculate new enthalpy
        7. Check mushy zone:
           - If below/in: update properties and return
           - If above L_H2O: convert ice to water, update total_water_mass, recursively call self
        
        If water_mass_to_add exceeds capacity, excess water is distributed to the next nearest ice layer.
        
        Parameters
        ----------
        layer_idx : int
            Index of the receiving ice layer
        water_mass_to_add : float
            Water mass to add from diffusion [kg/m²]
        water_enthalpy_to_add : float
            Total enthalpy of the water being added [J] (mass-weighted)
        ice_layers : numpy.ndarray
            Array of ice layer indices (for finding next layer if needed)
        """
        # Get current layer properties
        dx = self.model.dx[layer_idx]  # dx is constant
        alpha_ice = self.model.alpha_ice[layer_idx]
        alpha_air_current = self.model.alpha_air[layer_idx]
        
        # Get current water mass in receiver layer
        if self.m_double_prime_water is not None:
            water_mass_receiver = self.m_double_prime_water[layer_idx]
        else:
            alpha_water_current = self.model.alpha_water[layer_idx]
            water_mass_receiver = alpha_water_current * self.model.rho_water * dx
        
        # Check capacity: maximum water that can fit in this layer
        # Capacity = alpha_air * rho_water * dx (available air space)
        max_water_capacity = alpha_air_current * self.model.rho_water * dx
        
        # Calculate how much water can actually be added to this layer
        water_mass_that_fits = min(water_mass_to_add, max_water_capacity)
        excess_water_mass = water_mass_to_add - water_mass_that_fits
        
        # Calculate enthalpy for the water that fits (proportional to mass)
        if water_mass_to_add > 0:
            water_enthalpy_that_fits = water_enthalpy_to_add * (water_mass_that_fits / water_mass_to_add)
            excess_water_enthalpy = water_enthalpy_to_add - water_enthalpy_that_fits
        else:
            water_enthalpy_that_fits = 0.0
            excess_water_enthalpy = 0.0
        
        # Equation 1: Add water mass from diffusion layer(s) to receiver layer (up to capacity)
        total_water_mass = water_mass_receiver + water_mass_that_fits
        
        # Equation 2: Update volume fraction of water in receiver layer
        if dx > 0:
            water_volume = total_water_mass / self.model.rho_water
            alpha_water_new = water_volume / dx
        else:
            alpha_water_new = 0.0
        
        # Equation 3: Keep volume fraction of ice the same
        alpha_ice_keep = alpha_ice
        
        # Equation 4: Update volume fraction of air from constraint
        alpha_air_new = 1.0 - alpha_ice_keep - alpha_water_new
        
        # Calculate masses for enthalpy calculation
        ice_mass = alpha_ice_keep * self.model.rho_ice * dx
        air_mass = alpha_air_new * self.model.rho_air * dx
        current_total_mass = ice_mass + water_mass_receiver + air_mass
        new_total_mass = current_total_mass + water_mass_that_fits
        
        if new_total_mass <= 0:
            return
        
        # Equation 5: Calculate receiver's new enthalpy
        # Use mass-weighted average: h_new = (h_receiver * m_receiver + h_diffusion * m_diffusion) / m_total
        h_current = self.h[layer_idx]
        h_new = (h_current * current_total_mass + water_enthalpy_that_fits) / new_total_mass
        
        # Recalculate mushy zone thresholds for this layer: more mass has entered (melted water),
        # so the effective "initial" ice content (ice + water-as-ice equivalent) is larger.
        # L_f and L_H2O depend on alpha_ice; both must be updated before the mushy zone check.
        if dx > 0 and self.model.rho_ice > 0:
            # Effective initial state: all (ice + water) in the control volume as ice + air
            ice_plus_water_mass = ice_mass + total_water_mass
            alpha_ice_eff = min(1.0, ice_plus_water_mass / (self.model.rho_ice * dx))
            alpha_air_eff = 1.0 - alpha_ice_eff
            if alpha_air_eff < 0:
                alpha_air_eff = 0.0
            self._recalculate_layer_enthalpy_thresholds(layer_idx, alpha_ice_eff, alpha_air_eff)
        
        # Get mushy zone thresholds (now updated for current layer composition)
        L_f_i = self.L_f[layer_idx]
        L_H2O_i = self.L_H2O[layer_idx]
        
        # Equation 6: Check mushy zone and update accordingly
        if h_new > L_H2O_i:
            # Above mushy zone: all ice becomes water, add to diffusion water mass, recursively call self
            total_water_after_conversion = total_water_mass + ice_mass
            
            # Calculate total enthalpy after conversion (ice melts to water)
            # The ice that melts contributes enthalpy h_new (since it's all water now)
            total_water_enthalpy_after_conversion = water_enthalpy_that_fits + h_new * ice_mass
            
            # Set current layer to all water
            self._set_layer_to_all_water(layer_idx, total_water_after_conversion)
            
            # Find next nearby ice layer
            remaining_ice_layers = ice_layers[ice_layers != layer_idx]
            
            if len(remaining_ice_layers) > 0:
                # Find nearest ice layer
                distances = np.abs(remaining_ice_layers - layer_idx)
                next_ice_idx = remaining_ice_layers[np.argmin(distances)]
                
                # Combine excess water (if any) with converted water
                total_water_to_next = total_water_after_conversion + excess_water_mass
                total_enthalpy_to_next = total_water_enthalpy_after_conversion + excess_water_enthalpy
                
                # Recursively call diffusion function with updated water mass and enthalpy
                self._diffuse_water_into_layer(next_ice_idx, total_water_to_next, total_enthalpy_to_next, remaining_ice_layers)
            else:
                # No more ice layers - all layers become water
                self._all_layers_water = True
                
        elif h_new < L_f_i:
            # Below mushy zone: update properties and temperature
            self.h[layer_idx] = h_new
            
            # All water becomes ice
            if dx > 0:
                alpha_ice_new = new_total_mass / (self.model.rho_ice * dx)
                self.model.alpha_ice[layer_idx] = alpha_ice_new
                self.model.alpha_water[layer_idx] = 0.0
                self.model.alpha_air[layer_idx] = 1.0 - alpha_ice_new - 0.0
                
                # Update mass per unit area
                if self.m_double_prime_ice is not None:
                    self.m_double_prime_ice[layer_idx] = alpha_ice_new * self.model.rho_ice * dx
                if self.m_double_prime_water is not None:
                    self.m_double_prime_water[layer_idx] = 0.0
                
                # Recalculate properties
                self.model.calculate_specific_heat()
                self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
                
                # Update temperature
                cp = self.model.cp[layer_idx]
                if cp > 0:
                    T_ref = self.model.T[layer_idx]
                    dh = h_new - h_current
                    dT = dh / cp
                    self.model.T[layer_idx] = T_ref + dT
            
            # If there's excess water that couldn't fit, distribute it to the next nearest ice layer
            if excess_water_mass > 1e-10:  # Small tolerance to avoid numerical issues
                remaining_ice_layers = ice_layers[ice_layers != layer_idx]
                
                if len(remaining_ice_layers) > 0:
                    # Find nearest ice layer
                    distances = np.abs(remaining_ice_layers - layer_idx)
                    next_ice_idx = remaining_ice_layers[np.argmin(distances)]
                    
                    # Recursively distribute excess water to next layer
                    self._diffuse_water_into_layer(next_ice_idx, excess_water_mass, excess_water_enthalpy, remaining_ice_layers)
                else:
                    # No more ice layers - excess water cannot be distributed
                    # This shouldn't happen in normal operation, but handle gracefully
                    pass
        else:
            # In mushy zone: update properties and temperature
            self.h[layer_idx] = h_new
            
            # Update volume fractions
            self.model.alpha_water[layer_idx] = alpha_water_new
            self.model.alpha_ice[layer_idx] = alpha_ice_keep
            self.model.alpha_air[layer_idx] = alpha_air_new
            
            # Update mass per unit area
            if self.m_double_prime_water is not None:
                self.m_double_prime_water[layer_idx] = total_water_mass
            
            # Recalculate properties
            self.model.calculate_specific_heat()
            self.model.calculate_thermal_conductivity(begin_idx=(self.begin_idx if self.begin_idx is not None else 0))
            
            # Update temperature (stays at melting point in mushy zone)
            self.model.T[layer_idx] = self.model.T_melt
            
            # If there's excess water that couldn't fit, distribute it to the next nearest ice layer
            if excess_water_mass > 1e-10:  # Small tolerance to avoid numerical issues
                remaining_ice_layers = ice_layers[ice_layers != layer_idx]
                
                if len(remaining_ice_layers) > 0:
                    # Find nearest ice layer
                    distances = np.abs(remaining_ice_layers - layer_idx)
                    next_ice_idx = remaining_ice_layers[np.argmin(distances)]
                    
                    # Recursively distribute excess water to next layer
                    self._diffuse_water_into_layer(next_ice_idx, excess_water_mass, excess_water_enthalpy, remaining_ice_layers)
                else:
                    # No more ice layers - excess water cannot be distributed
                    # This shouldn't happen in normal operation, but handle gracefully
                    pass
    
    def _update_surface_and_wall_layers(self):
        """
        Update which layers are the beginning (facing air) and end (close to wall) after diffusion.
        
        After diffusion, outer layers (i=0, i=1, etc.) may have melted and diffused away.
        This function finds:
        - The first non-zero layer (new begin layer facing air)
        - The last non-zero layer (new end layer closest to wall)
        
        Layers are considered "non-zero" if they have non-zero thickness (dx > 0).
        For layers close to wall, if they melt, water diffuses into nearby ice layers
        (no sloughing happens, so water stays in the system).
        """
        n = self.model.n_layers
        if n == 0:
            self.begin_idx = None
            self.end_idx = None
            return
        
        # Find all layers with non-zero thickness
        non_zero_layers = np.where(self.model.dx > 1e-10)[0]
        
        if len(non_zero_layers) == 0:
            # All layers are zero - no begin or end layer
            self.begin_idx = None
            self.end_idx = None
            return
        
        # Begin layer: first non-zero layer (facing air)
        # This is the layer that is now exposed to air after outer layers melted
        self.begin_idx = non_zero_layers[0]
        
        # End layer: last non-zero layer (closest to wall)
        # This is the layer that is now closest to the wall after outer layers melted
        self.end_idx = non_zero_layers[-1]
        
        # Note: If begin_idx > 0, it means layers 0, 1, ..., begin_idx-1 have melted
        # If end_idx < n-1, it means layers end_idx+1, ..., n-1 have melted
    
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
        T_melt = self.model.T_melt
        
        for i in range(self.begin_idx, self.end_idx + 1):
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
                    # if dT > 10:
                    #     print(f"Warning: Layer {i} temperature change is too large: {dT} K")
                    
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

            # if self.model.T[i] > 12 or self.model.T[i] < -30:
            #     print(f"Warning: Layer {i} temperature is too extreme: {self.model.T[i]} K")
    
    def _update_volume_fractions_from_heat_flux_first(self, q_fluxes):
        """
        First volume fraction update based on heat flux (before thickness reduction).
        
        This is the FIRST update step that accounts for phase change (ice melting to water)
        based on the net heat flux. A second update will follow for thickness reduction.
        
        IMPORTANT: Volume fractions are only updated when enthalpy is in the mushy zone
        (L_f[i] ≤ h ≤ L_H2O[i]). Outside the mushy zone, volume fractions remain unchanged.
        
        Uses the heat flux-based equations with PREVIOUS time step values:
        
        α_ice,A^(t,1) = α_ice,A^t - (Δt * (q_in'' - q_out'')) / (h_sl * ρ_ice) * δ_A
        
        α_H2O,A^(t,1) = α_H2O,A^t + (Δt * (q_in'' - q_out'')) / (h_sl * ρ_H2O) * δ_A
        
        α_air,A^(t,1) = 1 - α_H2O,A^(t,1) - α_ice,A^(t,1)
        
        where:
        - α_ice,A^t, α_H2O,A^t = volume fractions at previous time step
        - h_sl = L_fusion (latent heat of fusion) [J/kg]
        - δ_A = dx[i] (layer thickness) [m]
        - q_in'' - q_out'' = net heat flux [W/m²]
        - Δt = time step [s]
        
        Parameters
        ----------
        q_fluxes : dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²] for each layer
        """
        h_sl = self.model.L_fusion  # Latent heat of fusion [J/kg]
        
        for i in range(self.begin_idx, self.end_idx + 1):
            # Check if enthalpy is in the mushy zone
            h_current = self.h[i]
            L_f_i = self.L_f[i]
            L_H2O_i = self.L_H2O[i]
            
            # Only update volume fractions if in mushy zone
            if h_current < L_f_i or h_current > L_H2O_i:
                # Not in mushy zone - volume fractions remain unchanged
                continue
            
            # In mushy zone - proceed with volume fraction update
            # Calculate net heat flux for this layer
            q_net = q_fluxes['in'][i] - q_fluxes['out'][i]  # [W/m²]
            
            # Get layer thickness
            delta_A = self.model.dx[i]  # [m]
            
            # Use PREVIOUS time step volume fractions (α^t)
            # These are stored from the previous time step
            if self.alpha_ice_prev is not None:
                alpha_ice_t = self.alpha_ice_prev[i]
                alpha_H2O_t = self.alpha_water_prev[i]
            else:
                # Fallback to current values if previous not available (first step)
                alpha_ice_t = self.model.alpha_ice[i]
                alpha_H2O_t = self.model.alpha_water[i]
            
            # Update ice volume fraction using previous time step value
            # α_ice,A^(t,1) = α_ice,A^t - (Δt * (q_in'' - q_out'')) / (h_sl * ρ_ice) * δ_A
            # Rate of change: d(alpha_ice)/dt = -(q_net) / (h_sl * ρ_ice) * δ_A
            delta_alpha_ice = (self.dt * q_net) / (h_sl * self.model.rho_ice) * delta_A
            alpha_ice_new = alpha_ice_t - delta_alpha_ice
            
            # Ensure ice volume fraction doesn't go negative
            alpha_ice_new = np.maximum(alpha_ice_new, 0.0)
            
            # Update water volume fraction using previous time step value
            # α_H2O,A^(t,1) = α_H2O,A^t + (Δt * (q_in'' - q_out'')) / (h_sl * ρ_H2O) * δ_A
            # Rate of change: d(alpha_H2O)/dt = (q_net) / (h_sl * ρ_H2O) * δ_A
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

            # Special case: if alpha_ice reaches 0, force alpha_water = 1 and alpha_ice = 0
            if alpha_ice_new <= 1e-6:
                alpha_ice_new = 0.0
                alpha_H2O_new = 1.0
                alpha_air_new = 0.0
                # Update layer thickness to m_water / rho_water
                # Calculate water mass per unit area: includes existing water + melted ice
                # Use mass per unit area if available, otherwise calculate from volume fractions
                if self.m_double_prime_water is not None:
                    m_water = self.m_double_prime_water[i]
                else:
                    # Calculate total water mass: existing water + water from melted ice
                    # Water from melted ice: mass of ice that melted = (alpha_ice_t - alpha_ice_new) * rho_ice * delta_A
                    # Existing water: alpha_H2O_t * rho_water * delta_A
                    ice_melted = alpha_ice_t * self.model.rho_ice * delta_A  # All ice melted
                    water_from_ice = ice_melted  # Mass conserved: ice mass = water mass
                    existing_water = alpha_H2O_t * self.model.rho_water * delta_A
                    m_water = existing_water + water_from_ice
                # Set thickness to pure water layer thickness
                self.model.dx[i] = m_water / self.model.rho_water
                # Force enthalpy to L_H2O (water enthalpy at melting point)
                self.h[i] = self.L_H2O[i]

            # Update model volume fractions
            self.model.alpha_ice[i] = alpha_ice_new
            self.model.alpha_water[i] = alpha_H2O_new
            self.model.alpha_air[i] = alpha_air_new
    
    def _calculate_layer_shrinkage(self):
        """
        Calculate layer thickness shrinkage due to surface tension and viscosity.
        
        Based on the equation:
        1/δ_A * d(δ_A)/dt = -σ / (η_eff * r_pore)
        
        Which gives:
        d(δ_A)/dt = -σ * δ_A / (η_eff * r_pore)
        
        Therefore:
        δ_A^(t+Δt) = δ_A^t - (σ * δ_A^t * Δt) / (η_eff * r_pore)
        
        The effective viscosity is calculated using the Crocus model:
        - η_eff = η_dry * f_wet(α_H2O)
        - η_dry = η_0 * exp(b * (ρ_frost / ρ_ice))
        - f_wet = 1 / (1 + C_wet * α_H2O^(t,1))
        
        The pore radius is calculated using Kozeny-Carman relationship:
        - r_pore = (2(1 - α_ice^(t,1))) / (3 * α_ice^(t,1)) * d_ice
        
        The ice grain diameter evolves as:
        - d_ice = d_ice,i * (α_ice^(t,1) / α_ice,i^(t,1))^(1/3)
        
        Returns
        -------
        numpy.ndarray
            Shrinkage rate for each layer [m/s]
        """
        shrinkage_rates = np.zeros(self.model.n_layers)  # Store shrinkage rates for all layers
        
        for i in range(self.begin_idx, self.end_idx + 1):
            # Get current values (after first volume fraction update)
            alpha_ice = self.model.alpha_ice[i]
            alpha_H2O = self.model.alpha_water[i]
            alpha_air = self.model.alpha_air[i]
            delta_A_t = self.model.dx[i]  # Current layer thickness
            
            # If no ice or no air, no shrinkage
            # No air means no porosity, so no shrinkage can occur
            if alpha_ice <= 0 or alpha_air <= 0:
                shrinkage_rate = 0.0
                shrinkage_rates[i] = 0.0
                delta_A_new = delta_A_t  # No change in thickness
            else:
                # Calculate frost density
                rho_frost = (alpha_ice * self.model.rho_ice + 
                            alpha_H2O * self.model.rho_water + 
                            self.model.alpha_air[i] * self.model.rho_air)
                
                # Calculate dry viscosity: η_dry = η_0 * exp(b * (ρ_frost / ρ_ice))
                eta_dry = self.eta_0 * np.exp(self.b * (rho_frost / self.model.rho_ice))
                
                # Calculate wetness reduction factor: f_wet = 1 / (1 + C_wet * α_H2O^(t,1))
                f_wet = 1.0 / (1.0 + self.C_wet * alpha_H2O)
                
                # Calculate effective viscosity: η_eff = η_dry * f_wet
                eta_eff = eta_dry * f_wet
                
                # Calculate ice grain diameter: d_ice = d_ice,i * (α_ice^(t,1) / α_ice,i^(t,1))^(1/3)
                alpha_ice_i = self.model.alpha_ice_initial[i]
                if alpha_ice_i > 0 and alpha_ice > 0:
                    d_ice = self.d_ice_i[i] * (alpha_ice / alpha_ice_i) ** (1.0 / 3.0)
                else:
                    d_ice = self.d_ice_i[i]  # Keep initial if no ice
                
                # Calculate pore radius: r_pore = (2(1 - α_ice^(t,1))) / (3 * α_ice^(t,1)) * d_ice
                if alpha_ice > 0:
                    r_pore = (2.0 * (1.0 - alpha_ice)) / (3.0 * alpha_ice) * d_ice
                else:
                    r_pore = 1e-6  # Small value if no ice (avoid division by zero)
                
                # Calculate shrinkage rate: d(δ_A)/dt = -σ * δ_A / (η_eff * r_pore)
                if eta_eff > 0 and r_pore > 0:
                    shrinkage_rate = -self.sigma * delta_A_t / (eta_eff * r_pore)
                else:
                    shrinkage_rate = 0.0  # No shrinkage if invalid values
                
                # Store shrinkage rate for this layer
                shrinkage_rates[i] = shrinkage_rate
                
                # Update layer thickness: δ_A^(t+Δt) = δ_A^t + d(δ_A)/dt * Δt
                delta_A_new = delta_A_t + shrinkage_rate * self.dt
                
                # Ensure thickness doesn't go negative
                delta_A_new = np.maximum(delta_A_new, 1e-9)  # Minimum thickness
            
            # Update layer thickness
            self.model.dx[i] = delta_A_new
        
        return shrinkage_rates
    
    def _update_volume_fractions_second(self, q_fluxes):
        """
        Second volume fraction update based on mass per unit area and updated thickness.
        
        After shrinkage, the layer thickness has changed, so we need to recalculate
        volume fractions based on mass conservation.
        
        Steps:
        1. Update mass per unit area of ice: m''^(t+Δt)_ice = m''^t_ice + (dm''_{ice} / dt) Δt
           where: dm''_{ice} / dt = (q''_{in} - q''_{out}) / h_{sl}
        
        2. Update mass per unit area of water: m''^(t+Δt)_water = m''^t_water - (dm''_{ice} / dt) Δt
        
        3. Recalculate volume fractions using new thickness:
           α^(t+Δt)_ice = m''^(t+Δt)_ice / (ρ_ice * δ_A^(t+Δt))
           α^(t+Δt)_H2O = m''^(t+Δt)_H2O / (ρ_H2O * δ_A^(t+Δt))
           α^(t+Δt)_air = 1 - α^(t+Δt)_ice - α^(t+Δt)_H2O
        
        Parameters
        ----------
        q_fluxes : dict
            Dictionary with 'in' and 'out' arrays of heat fluxes [W/m²] for each layer
        """
        h_sl = self.model.L_fusion  # Latent heat of fusion [J/kg]
        
        # Initialize masses if not already done
        if self.m_double_prime_ice is None:
            self.m_double_prime_ice = self.model.alpha_ice * self.model.rho_ice * self.model.dx
        if self.m_double_prime_water is None:
            self.m_double_prime_water = self.model.alpha_water * self.model.rho_water * self.model.dx
        
        for i in range(self.begin_idx, self.end_idx + 1):
            # Get current (time t) mass per unit area
            m_double_prime_ice_t = self.m_double_prime_ice[i]
            m_double_prime_water_t = self.m_double_prime_water[i]
            
            # Check if enthalpy is in the mushy zone
            h_current = self.h[i]
            L_f_i = self.L_f[i]
            L_H2O_i = self.L_H2O[i]
            
            # Calculate net heat flux
            q_net = q_fluxes['in'][i] - q_fluxes['out'][i]  # [W/m²]
            
            # Only change mass if in mushy zone (L_f[i] ≤ h ≤ L_H2O[i])
            # Otherwise, dm_ice_dt = 0 (no phase change)
            if L_f_i <= h_current <= L_H2O_i:
                # In mushy zone - calculate rate of change of ice mass per unit area
                # dm''_{ice} / dt = (q''_{in} - q''_{out}) / h_{sl}
                dm_ice_dt = q_net / h_sl  # [kg/(m²·s)]
                # Ensure dm_ice_dt is non-negative (>= 0)
                if dm_ice_dt < 0:
                    dm_ice_dt = 0.0
            else:
                # Not in mushy zone - no phase change, dm_ice_dt = 0
                dm_ice_dt = 0.0  # [kg/(m²·s)]
            
            # Update mass per unit area of ice
            # m''^(t+Δt)_ice = m''^t_ice + (dm''_{ice} / dt) Δt
            m_double_prime_ice_new = m_double_prime_ice_t - dm_ice_dt * self.dt
            
            # Ensure ice mass doesn't go negative
            m_double_prime_ice_new = np.maximum(m_double_prime_ice_new, 0.0)
            
            # Update mass per unit area of water
            # m''^(t+Δt)_water = m''^t_water - (dm''_{ice} / dt) Δt
            # Note: When ice melts (dm_ice_dt < 0), water increases
            m_double_prime_water_new = m_double_prime_water_t + dm_ice_dt * self.dt
            
            # Ensure water mass doesn't go negative
            m_double_prime_water_new = np.maximum(m_double_prime_water_new, 0.0)
            
            # Get updated layer thickness (after shrinkage)
            delta_A_new = self.model.dx[i]  # [m]
            
            # Recalculate volume fractions using new thickness
            # α^(t+Δt)_ice = m''^(t+Δt)_ice / (ρ_ice * δ_A^(t+Δt))
            if delta_A_new > 0:
                alpha_ice_new = m_double_prime_ice_new / (self.model.rho_ice * delta_A_new)
            else:
                alpha_ice_new = 0.0
            
            # α^(t+Δt)_H2O = m''^(t+Δt)_H2O / (ρ_H2O * δ_A^(t+Δt))
            if delta_A_new > 0:
                alpha_H2O_new = m_double_prime_water_new / (self.model.rho_water * delta_A_new)
            else:
                alpha_H2O_new = 0.0
            
            # Ensure volume fractions are in valid range [0, 1]
            alpha_ice_new = np.clip(alpha_ice_new, 0.0, 1.0)
            alpha_H2O_new = np.clip(alpha_H2O_new, 0.0, 1.0)
            
            # Calculate air volume fraction (closure)
            # α^(t+Δt)_air = 1 - α^(t+Δt)_ice - α^(t+Δt)_H2O
            alpha_air_new = 1.0 - alpha_ice_new - alpha_H2O_new
            
            # Ensure air volume fraction is valid
            alpha_air_new = np.maximum(alpha_air_new, 0.0)
            
            # Normalize to ensure volume fractions sum to 1
            total = alpha_ice_new + alpha_H2O_new + alpha_air_new
            if not np.isclose(total, 1.0, atol=1e-3):
                print(f"Warning: Volume fractions do not sum to 1 (sum={total}) in layer {i}")
                scale = 1.0 / total
                alpha_ice_new *= scale
                alpha_H2O_new *= scale
                alpha_air_new *= scale
            
            # Special case: if alpha_ice reaches 0, force alpha_water = 1 and alpha_ice = 0
            if alpha_ice_new <= 1e-6:
                alpha_ice_new = 0.0
                alpha_H2O_new = 1.0
                alpha_air_new = 0.0
                # Also update masses to be consistent
                m_double_prime_ice_new = 0.0
                # Update layer thickness to m_water / rho_water
                self.model.dx[i] = m_double_prime_water_new / self.model.rho_water
                # Force enthalpy to L_H2O (water enthalpy at melting point)
                self.h[i] = self.L_H2O[i]
            
            # Update model volume fractions
            self.model.alpha_ice[i] = alpha_ice_new
            self.model.alpha_water[i] = alpha_H2O_new
            self.model.alpha_air[i] = alpha_air_new
            
            # Store updated mass per unit area for next time step
            self.m_double_prime_ice[i] = m_double_prime_ice_new
            self.m_double_prime_water[i] = m_double_prime_water_new
    
    def _calculate_base_adhesion(self, theta_deg):
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
    
    def _calculate_f_water(self, alpha_water):
        """
        Calculate f(alpha_water) using logarithmic form.
        
        f(alpha_water) = A * log(B * alpha_water + e) + C
        
        Parameters
        ----------
        alpha_water : float
            Water volume fraction [-]
        
        Returns
        -------
        float
            f(alpha_water) value [-]
        """
        alpha_water = np.clip(alpha_water, 0.0, 1.0)
        
        # Calculate argument of log
        arg = self.B_crit * alpha_water + self.e_crit
        
        # Ensure argument is positive and > 0
        if arg <= 0:
            arg = 1e-10  # Small positive value
        
        # Calculate f(alpha_water) = A * log(B * alpha_water + e) + C
        log_val = np.log(arg)
        f_value = self.A_crit * log_val + self.C_crit
        
        return f_value
    
    def _calculate_effective_density(self):
        """
        Calculate effective density from the first layer containing frost (alpha_ice > 0)
        counting from the wall side (end_idx) till end_idx.
        
        ρ_eff = m_total / h_total
        
        Where m_total includes ice, water, and air masses.
        
        The calculation starts from end_idx (wall side) and goes backwards to find
        the first layer with alpha_ice > 0, then calculates density from that layer
        to end_idx (inclusive).
        
        Returns
        -------
        float
            Effective density [kg/m³]
        """
        # Initialize masses if not already done
        if self.m_double_prime_ice is None:
            self.m_double_prime_ice = self.model.alpha_ice * self.model.rho_ice * self.model.dx
        if self.m_double_prime_water is None:
            self.m_double_prime_water = self.model.alpha_water * self.model.rho_water * self.model.dx
        
        # Find the first layer with frost (alpha_ice > 0) starting from wall side (end_idx)
        # going backwards towards begin_idx
        first_frost_idx = None
        for i in range(self.end_idx, self.begin_idx - 1, -1):
            if self.model.alpha_ice[i] > 1e-10:  # Layer contains ice
                first_frost_idx = i
                break
        
        # If no frost layer found, return 0.0
        if first_frost_idx is None:
            return 0.0
        
        # Calculate masses and thickness from first_frost_idx to end_idx (inclusive)
        # Range: [first_frost_idx, end_idx]
        layer_range = range(first_frost_idx, self.end_idx + 1)
        
        # Calculate total masses per unit area for layers in range
        m_ice_total = np.sum(self.m_double_prime_ice[layer_range])
        m_water_total = np.sum(self.m_double_prime_water[layer_range])
        
        # Calculate total thickness for layers in range
        h_total = np.sum(self.model.dx[layer_range])
        
        if h_total > 0:
            # Calculate volumes
            V_total = h_total  # For unit area A = 1 m²
            V_ice = m_ice_total / self.model.rho_ice
            V_water = m_water_total / self.model.rho_water
            V_air = V_total - V_ice - V_water
            V_air = np.maximum(V_air, 0.0)  # Ensure non-negative
            
            # Calculate air mass
            m_air_total = V_air * self.model.rho_air
            
            # Total mass
            m_total = m_ice_total + m_water_total + m_air_total
            
            # Effective density
            rho_eff = m_total / h_total
        else:
            rho_eff = 0.0
        
        return rho_eff
    
    def _calculate_critical_thickness(self):
        """
        Calculate critical detachment frost thickness.
        
        h_crit = (k ⋅ τ_base ⋅ f(alpha_water)) / (ρ_eff ⋅ g)
        
        Where:
        - τ_base = τ_0 ⋅ (1 + cos θ) (base adhesion)
        - f(alpha_water) = A * log(B * alpha_water + e) + C
        - ρ_eff = effective density of entire frost layer
        - k = retention coefficient (depends on contact angle)
        
        The water volume fraction (alpha_water) is taken from the first layer
        (surface layer, closest to the heated surface).
        
        Returns
        -------
        float
            Critical detachment thickness [m]
        """
        # Check if contact angle is available
        if self.model.theta_receding is None:
            return np.inf  # Cannot calculate without contact angle
        
        # Get contact angle (use receding angle for adhesion)
        theta_deg = self.model.theta_receding
        
        # Select k based on contact angle (interpolate if needed)
        if theta_deg <= 60:
            k = self.k_60
        elif theta_deg >= 140:
            k = self.k_140
        else:
            # Linear interpolation between 60° and 140°
            k = self.k_60 + (self.k_140 - self.k_60) * (theta_deg - 60) / (140 - 60)
        
        # Calculate base adhesion
        tau_base = self._calculate_base_adhesion(theta_deg)
        
        # Get water volume fraction from first layer (surface layer)
        alpha_water_surface = self.model.alpha_water[self.begin_idx] if self.model.n_layers > 0 else 0.0
        
        # Calculate f(alpha_water)
        f_water = self._calculate_f_water(alpha_water_surface)
        
        # Calculate effective density
        rho_eff = self._calculate_effective_density()
        
        # Calculate critical thickness (offset: hydrophilic +0.0005, superhydrophobic -0.0003)
        if rho_eff > 0 and self.g > 0:
            offset = 0.0005 if self.model.surface_type_classification == "hydrophilic" else (-0.0005 if self.model.surface_type_classification == "superhydrophobic" else 0.0)
            h_crit = (k * tau_base * f_water) / (rho_eff * self.g) + offset
        else:
            h_crit = np.inf  # Invalid case
        
        return h_crit
    
    def _check_sloughing(self):
        """
        Check if sloughing occurs by comparing total frost thickness with critical thickness.
        
        Sloughing occurs when: h_total > h_crit
        (When frost becomes too thick, gravitational force overcomes adhesion)
        
        Returns
        -------
        dict
            Dictionary with:
            - 'h_crit': Critical detachment thickness [m]
            - 'h_total': Total frost thickness [m]
            - 'sloughing': Boolean, True if sloughing occurs (h_total > h_crit)
            - 'safety_factor': h_crit / h_total (ratio, >1 means safe, <1 means sloughing)
        """
        h_crit = self._calculate_critical_thickness()
        h_total = np.sum(self.model.dx)
        
        sloughing = h_total > h_crit if h_crit < np.inf else False
        safety_factor = h_crit / h_total if h_total > 0 and h_crit < np.inf else np.inf
        
        return {
            'h_crit': h_crit,
            'h_total': h_total,
            'sloughing': sloughing,
            'safety_factor': safety_factor
        }
    
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
    
    def solve(self, time_array, T_surface_array, save_history=True, history_save_interval=1.0):
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
        history_save_interval : float, optional
            Interval in seconds for saving history. Only saves history at this interval
            to reduce memory usage. Default: 1.0 s. Always saves first and last time steps.
        
        Returns
        -------
        dict
            Solution results with time, temperature, and other quantities
        """
        if len(time_array) != len(T_surface_array):
            raise ValueError("time_array and T_surface_array must have same length")
        
        n_steps = len(time_array)
        
        # Calculate surface retention maximum before time marching
        if self.model.theta_receding is not None and self.model.theta_advancing is not None:
            retention_result = calculate_surface_retention(
                self.model.theta_receding,
                self.model.theta_advancing
            )
            # Store in model for later use
            self.model.surface_retention = retention_result['retention']
            self.model.surface_retention_thickness = retention_result['thickness']
            self.model.surface_type_classification = retention_result['surface_type']

            if self.verbose:
                print(f"\n{'='*70}")
                print("Surface Retention Maximum (before time marching):")
                print(f"{'='*70}")
                print(f"  Contact angles: θ_R={self.model.theta_receding:.1f}°, θ_A={self.model.theta_advancing:.1f}°")
                print(f"  Surface type: {retention_result['surface_type']}")
                print(f"  Maximum retention: {retention_result['retention']*1000:.2f} g/m²")
                print(f"  Maximum retention thickness: {retention_result['thickness']*1000:.4f} mm")
                print(f"  Retention coefficient (k): {retention_result['k']:.4f}")
                print(f"  Droplet spacing (δ): {retention_result['delta']*1e6:.2f} μm")
                print(f"{'='*70}\n")
        
        # Initialize history and sloughing info storage
        if save_history:
            self.time_history = []
            self.temperature_history = []
            self.enthalpy_history = []
            self.volume_fraction_history = []
            self.dx_history = []
            self.shrinkage_rate_history = []
            self.h_crit_history = []
            self.h_total_history = []
            self.sloughing_status_history = []
            
            # Track last saved time for interval-based saving
            last_saved_time = time_array[0] - history_save_interval  # Initialize to allow first save
            
            # Save initial state
            self.time_history.append(time_array[0])
            self.temperature_history.append(self.model.T.copy())
            self.enthalpy_history.append(self.h.copy())
            self.volume_fraction_history.append({
                'ice': self.model.alpha_ice.copy(),
                'water': self.model.alpha_water.copy(),
                'air': self.model.alpha_air.copy()
            })
            self.dx_history.append(self.model.dx.copy())
            if self.current_shrinkage_rates is not None:
                self.shrinkage_rate_history.append(self.current_shrinkage_rates.copy())
            else:
                self.shrinkage_rate_history.append(np.zeros(self.model.n_layers))
            # Initial sloughing check
            sloughing_info = self._check_sloughing()
            self.h_crit_history.append(sloughing_info['h_crit'])
            self.h_total_history.append(sloughing_info['h_total'])
            self.sloughing_status_history.append(sloughing_info['sloughing'])
            self._latest_sloughing_info = sloughing_info  # Store for consistency
        else:
            # Initialize sloughing info storage even if not saving history
            self._latest_sloughing_info = None
        
        # Time stepping
        total_duration = time_array[-1] - time_array[0]
        
        # Rotating spinner for progress indication
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0
        last_spinner_update = time_array[0]
        spinner_interval = 0.5  # Update spinner every 0.5 seconds of simulation time
        
        # Print initial message
        if self.verbose:
            print(f"\nSimulating... ", end='', flush=True)
        
        for i in range(1, n_steps):
            dt = time_array[i] - time_array[i-1]
            self.dt = dt
            
            # Interpolate surface temperature if needed
            T_surface = T_surface_array[i]
            current_time = time_array[i]
            
            # Solve one time step
            success = self.solve_time_step(T_surface)
            
            # Check if all layers have become water (simulation should end)
            if hasattr(self, '_all_layers_water') and self._all_layers_water:
                if self.verbose:
                    print(f"\r                                                              ", end='')
                    print(f"\rSimulation stopped at t = {time_array[i]:.2f} s - all layers have become water")
                break
            
            # Update rotating spinner
            time_since_last_spinner = current_time - last_spinner_update
            if time_since_last_spinner >= spinner_interval:
                # Calculate progress percentage
                elapsed_time = current_time - time_array[0]
                progress_pct = min(100.0, (elapsed_time / total_duration) * 100.0) if total_duration > 0 else 0.0

                # Print rotating spinner with progress
                if self.verbose:
                    print(f"\rSimulating... {spinner_chars[spinner_idx]} {progress_pct:5.1f}% (t = {current_time:.1f} s)", end='', flush=True)
                spinner_idx = (spinner_idx + 1) % len(spinner_chars)
                last_spinner_update = current_time
            
            # Determine if we should save history at this time step
            should_save_history = False
            if save_history:
                time_since_last_save = current_time - last_saved_time
                is_last_step = (i == n_steps - 1)
                
                # Save if interval has passed or if this is the last step
                if time_since_last_save >= history_save_interval or is_last_step:
                    should_save_history = True
                    last_saved_time = current_time
            
            # Save history (before checking for sloughing, so arrays stay in sync)
            if should_save_history:
                self.time_history.append(time_array[i])
                self.temperature_history.append(self.model.T.copy())
                self.enthalpy_history.append(self.h.copy())  # Save specific enthalpy
                self.volume_fraction_history.append({
                    'ice': self.model.alpha_ice.copy(),
                    'water': self.model.alpha_water.copy(),
                    'air': self.model.alpha_air.copy()
                })
                self.dx_history.append(self.model.dx.copy())  # Save layer thicknesses
                # Save shrinkage rates (use current or zeros if not calculated yet)
                if self.current_shrinkage_rates is not None:
                    self.shrinkage_rate_history.append(self.current_shrinkage_rates.copy())
                else:
                    self.shrinkage_rate_history.append(np.zeros(self.model.n_layers))
                
                # Save sloughing info (calculated in solve_time_step, stored in _latest_sloughing_info)
                if hasattr(self, '_latest_sloughing_info'):
                    sloughing_info = self._latest_sloughing_info
                    self.h_crit_history.append(sloughing_info['h_crit'])
                    self.h_total_history.append(sloughing_info['h_total'])
                    self.sloughing_status_history.append(sloughing_info['sloughing'])
            
            # Check for sloughing (check every step, even if not saving history)
            if not success:
                # Check if sloughing occurred (use latest sloughing info if available)
                sloughing_occurred = False
                if hasattr(self, '_latest_sloughing_info'):
                    sloughing_occurred = self._latest_sloughing_info.get('sloughing', False)
                elif self.sloughing_status_history and len(self.sloughing_status_history) > 0:
                    sloughing_occurred = self.sloughing_status_history[-1]
                
                if sloughing_occurred:
                    # Sloughing occurred - ensure last time step is saved to history
                    # Save even if it doesn't meet the interval condition
                    if save_history and not should_save_history:
                        # Last step wasn't saved yet, save it now
                        self.time_history.append(time_array[i])
                        self.temperature_history.append(self.model.T.copy())
                        self.enthalpy_history.append(self.h.copy())
                        self.volume_fraction_history.append({
                            'ice': self.model.alpha_ice.copy(),
                            'water': self.model.alpha_water.copy(),
                            'air': self.model.alpha_air.copy()
                        })
                        self.dx_history.append(self.model.dx.copy())
                        if self.current_shrinkage_rates is not None:
                            self.shrinkage_rate_history.append(self.current_shrinkage_rates.copy())
                        else:
                            self.shrinkage_rate_history.append(np.zeros(self.model.n_layers))
                        
                        # Save sloughing info for this final step
                        if hasattr(self, '_latest_sloughing_info'):
                            sloughing_info = self._latest_sloughing_info
                            self.h_crit_history.append(sloughing_info['h_crit'])
                            self.h_total_history.append(sloughing_info['h_total'])
                            self.sloughing_status_history.append(sloughing_info['sloughing'])
                    
                    # Sloughing occurred - stop simulation (clear spinner line first)
                    if self.verbose:
                        print(f"\r                                                              ", end='')
                        print(f"\rSimulation stopped at t = {time_array[i]:.2f} s due to sloughing")
                    break
                else:
                    if self.verbose:
                        print(f"\r                                                              ", end='')
                        print(f"\rWarning: Solver failed at step {i}, time = {time_array[i]:.2f} s")
        else:
            # Loop completed without break - ensure last time step is saved
            if save_history:
                # Check if last step was already saved
                last_saved_time = self.time_history[-1] if len(self.time_history) > 0 else None
                if last_saved_time is None or abs(last_saved_time - time_array[-1]) > 1e-6:
                    # Last step wasn't saved, save it now
                    self.time_history.append(time_array[-1])
                    self.temperature_history.append(self.model.T.copy())
                    self.enthalpy_history.append(self.h.copy())
                    self.volume_fraction_history.append({
                        'ice': self.model.alpha_ice.copy(),
                        'water': self.model.alpha_water.copy(),
                        'air': self.model.alpha_air.copy()
                    })
                    self.dx_history.append(self.model.dx.copy())
                    if self.current_shrinkage_rates is not None:
                        self.shrinkage_rate_history.append(self.current_shrinkage_rates.copy())
                    else:
                        self.shrinkage_rate_history.append(np.zeros(self.model.n_layers))
                    
                    # Save sloughing info for final step
                    if hasattr(self, '_latest_sloughing_info'):
                        sloughing_info = self._latest_sloughing_info
                        self.h_crit_history.append(sloughing_info['h_crit'])
                        self.h_total_history.append(sloughing_info['h_total'])
                        self.sloughing_status_history.append(sloughing_info['sloughing'])
            
            # Loop completed without break - clear spinner and show completion
            if self.verbose:
                print(f"\r                                                              ", end='')
                print(f"\rSimulation completed: t = {time_array[-1]:.2f} s")
        
        # Extract volume fractions from history
        alpha_ice_array = None
        alpha_water_array = None
        alpha_air_array = None
        if save_history and len(self.volume_fraction_history) > 0:
            alpha_ice_array = np.array([vf['ice'] for vf in self.volume_fraction_history])
            alpha_water_array = np.array([vf['water'] for vf in self.volume_fraction_history])
            alpha_air_array = np.array([vf['air'] for vf in self.volume_fraction_history])
        
        # Return results
        results = {
            'time': np.array(self.time_history) if save_history else time_array,
            'temperature': np.array(self.temperature_history) if save_history else None,
            'h_crit': np.array(self.h_crit_history) if save_history and len(self.h_crit_history) > 0 else None,
            'h_total': np.array(self.h_total_history) if save_history and len(self.h_total_history) > 0 else None,
            'sloughing': np.array(self.sloughing_status_history) if save_history and len(self.sloughing_status_history) > 0 else None,
            'dx': np.array(self.dx_history) if save_history and len(self.dx_history) > 0 else None,  # Layer thicknesses [m]
            'alpha_ice': alpha_ice_array,  # Volume fraction of ice per layer [n_time_steps, n_layers]
            'alpha_water': alpha_water_array,  # Volume fraction of water per layer [n_time_steps, n_layers]
            'alpha_air': alpha_air_array,  # Volume fraction of air per layer [n_time_steps, n_layers]
            'shrinkage_rate': np.array(self.shrinkage_rate_history) if save_history and len(self.shrinkage_rate_history) > 0 else None,  # Shrinkage rate [m/s] per layer [n_time_steps, n_layers]
            'model': self.model
        }
        
        return results
