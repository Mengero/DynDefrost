"""
Temperature Interpolation Module

This module handles interpolation of experimental temperature data to match
the simulation time step, smoothing out large gradients and preventing
temperature jumps.
"""

import numpy as np
from scipy.interpolate import interp1d


def interpolate_temperature(time_raw, temperature_raw, dt=0.1, kind='linear'):
    """
    Interpolate temperature data to match simulation time step.
    
    Experimental data typically has ~1 datapoint per second, but simulations
    require finer time resolution (e.g., dt = 0.1s) to avoid large temperature
    jumps and maintain stability.
    
    Parameters
    ----------
    time_raw : numpy.ndarray
        Original time array from experimental data [s]
    temperature_raw : numpy.ndarray
        Original temperature array from experimental data [°C]
    dt : float, optional
        Desired time step for interpolation [s]. Default: 0.1
    kind : str, optional
        Interpolation method. Options: 'linear', 'cubic', 'quadratic'.
        Default: 'linear' (recommended to avoid overshooting with large gradients)
    
    Returns
    -------
    tuple
        (time_interp, temperature_interp) - Interpolated time and temperature arrays
    """
    if len(time_raw) != len(temperature_raw):
        raise ValueError("time_raw and temperature_raw must have same length")
    
    if len(time_raw) < 2:
        raise ValueError("Need at least 2 data points for interpolation")
    
    # Create interpolation function
    # Use 'linear' interpolation to avoid overshooting with large gradients
    # bounds_error=False and fill_value='extrapolate' handle edge cases
    interp_func = interp1d(
        time_raw, 
        temperature_raw, 
        kind=kind,
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    # Create fine time array with specified dt
    time_interp = np.arange(time_raw[0], time_raw[-1] + dt, dt)
    
    # Interpolate temperature
    temperature_interp = interp_func(time_interp)
    
    return time_interp, temperature_interp


def print_interpolation_info(time_raw, temperature_raw, time_interp, temperature_interp, dt):
    """
    Print information about the interpolation process.
    
    Parameters
    ----------
    time_raw : numpy.ndarray
        Original time array [s]
    temperature_raw : numpy.ndarray
        Original temperature array [°C]
    time_interp : numpy.ndarray
        Interpolated time array [s]
    temperature_interp : numpy.ndarray
        Interpolated temperature array [°C]
    dt : float
        Time step used for interpolation [s]
    """
    print(f"\nInterpolating temperature data...")
    print(f"  Original data points: {len(time_raw)}")
    print(f"  Original time range: {time_raw[0]:.1f} s to {time_raw[-1]:.1f} s")
    if len(time_raw) > 1:
        print(f"  Original time step: ~{np.mean(np.diff(time_raw)):.2f} s")
    print(f"  Interpolated data points: {len(time_interp)}")
    print(f"  Interpolated time step: {dt} s")
    print(f"  Temperature range: {np.min(temperature_interp):.1f}°C to {np.max(temperature_interp):.1f}°C")
    
    # Calculate maximum temperature change rate
    if len(temperature_interp) > 1:
        dt_actual = np.diff(time_interp)
        dT_dt = np.diff(temperature_interp) / dt_actual
        max_rate = np.max(np.abs(dT_dt))
        print(f"  Maximum temperature change rate: {max_rate:.2f}°C/s")


if __name__ == "__main__":
    """Test the interpolation function."""
    # Create test data (simulating 1 datapoint per second)
    time_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    temperature_test = np.array([-20.0, -15.0, -10.0, -5.0, 0.0, 1.3])
    
    print("Test: Temperature Interpolation")
    print("=" * 70)
    print_interpolation_info(time_test, temperature_test, 
                           *interpolate_temperature(time_test, temperature_test, dt=0.1),
                           dt=0.1)
