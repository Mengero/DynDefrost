"""
Plot Representative Temperature Curve

This script excludes experiments with problematic temperature ranges and creates
a representative temperature vs time curve from the remaining good experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from scipy import stats


def parse_time_to_seconds(time_str, start_time=None):
    """Convert time string (HH:MM:SS) to seconds from start."""
    try:
        current_time = datetime.strptime(time_str.strip(), '%H:%M:%S')
        if start_time is None:
            return current_time, 0.0
        delta = current_time - start_time
        return current_time, delta.total_seconds()
    except (ValueError, AttributeError):
        return None, None


def filter_increasing_temperature(time, temperature, tolerance=0.01):
    """
    Filter data to keep only segments where temperature is constant or increasing.
    
    Parameters:
    -----------
    time : array
        Time array
    temperature : array
        Temperature array
    tolerance : float
        Tolerance for considering temperature as constant (in °C)
        
    Returns:
    --------
    tuple
        (filtered_time, filtered_temperature) arrays
    """
    if len(time) == 0 or len(temperature) == 0:
        return time, temperature
    
    # Find the starting point (lowest temperature)
    start_idx = np.argmin(temperature)
    
    # From the start, keep only increasing or constant segments
    filtered_indices = [start_idx]
    current_max_temp = temperature[start_idx]
    
    for i in range(start_idx + 1, len(temperature)):
        # Keep if temperature is greater than or equal to current max (within tolerance)
        if temperature[i] >= current_max_temp - tolerance:
            filtered_indices.append(i)
            current_max_temp = max(current_max_temp, temperature[i])
    
    filtered_indices = np.array(filtered_indices)
    return time[filtered_indices], temperature[filtered_indices]


def load_experimental_data(filepath, filter_decreasing=True):
    """
    Load experimental data from a .txt file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to data file
    filter_decreasing : bool
        If True, filter out decreasing temperature segments
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    data_lines = lines[1:] if len(lines) > 1 else []
    
    time_sensor1 = []
    temp_sensor1 = []
    start_time_sensor1 = None
    
    for line in data_lines:
        if not line.strip():
            continue
        
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        
        try:
            # Sensor 1 data (columns 2, 3)
            if parts[2].strip() and parts[3].strip():
                current_time, seconds = parse_time_to_seconds(parts[2], start_time_sensor1)
                if current_time is not None:
                    if start_time_sensor1 is None:
                        start_time_sensor1 = current_time
                        seconds = 0.0
                    time_sensor1.append(seconds)
                    temp_sensor1.append(float(parts[3]))
        except (ValueError, IndexError):
            continue
    
    time_array = np.array(time_sensor1)
    temp_array = np.array(temp_sensor1)
    
    # Filter out decreasing temperature segments
    if filter_decreasing and len(time_array) > 0:
        time_array, temp_array = filter_increasing_temperature(time_array, temp_array)
    
    return {
        'time': time_array,
        'temperature': temp_array,
        'filename': filepath.name
    }


def identify_problematic_experiments(data_dir='exp_data', min_data_points=100, 
                                     min_temp_range=15.0):
    """
    Identify experiments with problematic temperature ranges.
    
    Criteria:
    - Too few data points
    - Temperature range too small
    - Temperature doesn't reach reasonable values
    """
    data_path = Path(data_dir)
    data_files = sorted([f for f in data_path.glob('*.txt')])
    
    problematic = []
    good = []
    
    for filepath in data_files:
        try:
            data = load_experimental_data(filepath, filter_decreasing=True)
            
            if len(data['time']) < min_data_points:
                problematic.append(data['filename'])
                print(f"  {data['filename']}: Too few data points ({len(data['time'])})")
                continue
            
            temp_range = np.max(data['temperature']) - np.min(data['temperature'])
            if temp_range < min_temp_range:
                problematic.append(data['filename'])
                print(f"  {data['filename']}: Temperature range too small ({temp_range:.1f}°C)")
                continue
            
            # Check if temperature starts around -20°C and reaches above 0°C
            initial_temp = data['temperature'][0]
            final_temp = data['temperature'][-1]
            
            if initial_temp > -15 or final_temp < 0:
                problematic.append(data['filename'])
                print(f"  {data['filename']}: Unusual temperature range ({initial_temp:.1f}°C to {final_temp:.1f}°C)")
                continue
            
            good.append(data)
            
        except Exception as e:
            problematic.append(filepath.name)
            print(f"  {filepath.name}: Error loading - {e}")
    
    return good, problematic


def create_representative_curve(good_data, time_resolution=1.0, T_ambient=12.0):
    """
    Create a representative temperature curve by averaging/interpolating good experiments.
    Extends the curve to reach ambient temperature.
    
    Parameters:
    -----------
    good_data : list
        List of data dictionaries with 'time' and 'temperature' arrays
    time_resolution : float
        Time resolution in seconds for the representative curve
    T_ambient : float
        Ambient temperature to extend to [°C]
    """
    if not good_data:
        raise ValueError("No good data available")
    
    # Find the common time range
    min_time = max([np.min(d['time']) for d in good_data])
    max_time = min([np.max(d['time']) for d in good_data])
    
    # Create common time axis
    time_common = np.arange(min_time, max_time + time_resolution, time_resolution)
    
    # Interpolate each experiment to common time axis
    interpolated_temps = []
    
    for data in good_data:
        # Remove duplicate times (keep first occurrence)
        time_unique, indices = np.unique(data['time'], return_index=True)
        temp_unique = data['temperature'][indices]
        
        # Create interpolation function
        if len(time_unique) > 1:
            interp_func = interp1d(time_unique, temp_unique, 
                                  kind='linear', 
                                  bounds_error=False, 
                                  fill_value='extrapolate')
            temp_interp = interp_func(time_common)
            interpolated_temps.append(temp_interp)
    
    if not interpolated_temps:
        raise ValueError("Could not interpolate any data")
    
    # Calculate mean and standard deviation
    interpolated_temps = np.array(interpolated_temps)
    mean_temp = np.mean(interpolated_temps, axis=0)
    std_temp = np.std(interpolated_temps, axis=0)
    
    # Also calculate median for robustness
    median_temp = np.median(interpolated_temps, axis=0)
    
    # Extend the curve to reach ambient temperature
    # Calculate the gradient at the end (use last few points for stability)
    n_points_for_gradient = min(10, len(mean_temp))
    if n_points_for_gradient > 1:
        # Calculate average gradient from last n points
        time_gradient = time_common[-n_points_for_gradient:]
        temp_gradient = mean_temp[-n_points_for_gradient:]
        dT_dt = np.mean(np.diff(temp_gradient) / np.diff(time_gradient))
        
        # If gradient is positive and temperature hasn't reached ambient yet
        if dT_dt > 0 and mean_temp[-1] < T_ambient:
            # Extend time and temperature until reaching ambient
            current_time = time_common[-1]
            current_temp = mean_temp[-1]
            
            extended_time = [current_time]
            extended_temp = [current_temp]
            extended_std = [std_temp[-1]]
            extended_median = [median_temp[-1]]
            
            # Continue extending with the same gradient
            while extended_temp[-1] < T_ambient - 0.01:  # Stop when within 0.01°C of ambient
                next_time = extended_time[-1] + time_resolution
                next_temp = extended_temp[-1] + dT_dt * time_resolution
                
                # Don't exceed ambient temperature
                if next_temp >= T_ambient:
                    next_temp = T_ambient
                
                extended_time.append(next_time)
                extended_temp.append(next_temp)
                # Keep std dev constant at the last value (or gradually decrease)
                extended_std.append(std_temp[-1] * 0.5)  # Reduce uncertainty in extended region
                extended_median.append(next_temp)  # Median follows mean in extended region
            
            # Append extended data
            time_common = np.concatenate([time_common, extended_time[1:]])
            mean_temp = np.concatenate([mean_temp, extended_temp[1:]])
            std_temp = np.concatenate([std_temp, extended_std[1:]])
            median_temp = np.concatenate([median_temp, extended_median[1:]])
            
            print(f"  Extended curve from {current_temp:.2f}°C to {T_ambient:.1f}°C")
            print(f"  Extended time: {current_time:.1f} s to {time_common[-1]:.1f} s")
            print(f"  Gradient used: {dT_dt*60:.3f} °C/min")
        else:
            if mean_temp[-1] >= T_ambient:
                print(f"  Temperature already at or above ambient ({mean_temp[-1]:.2f}°C)")
            else:
                print(f"  Warning: Negative or zero gradient at end, cannot extend to ambient")
    
    return {
        'time': time_common,
        'mean_temperature': mean_temp,
        'median_temperature': median_temp,
        'std_temperature': std_temp,
        'n_experiments': len(good_data)
    }


def plot_representative_temperature(data_dir='exp_data', output_dir='figure', 
                                    figsize=(12, 8), T_ambient=12.0):
    """
    Plot representative temperature curve excluding problematic experiments.
    """
    print("=" * 60)
    print("Creating Representative Temperature Curve")
    print("=" * 60)
    
    # Identify good and problematic experiments
    print("\nIdentifying problematic experiments...")
    good_data, problematic = identify_problematic_experiments(data_dir)
    
    print(f"\nFound {len(problematic)} problematic experiments:")
    for p in problematic:
        print(f"  - {p}")
    
    print(f"\nUsing {len(good_data)} good experiments:")
    for d in good_data:
        print(f"  - {d['filename']}")
    
    if not good_data:
        print("ERROR: No good experiments found!")
        return
    
    # Create representative curve
    print("\nCreating representative curve...")
    representative = create_representative_curve(good_data, T_ambient=T_ambient)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Plot 1: Representative curve with individual experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(good_data)))
    
    for i, data in enumerate(good_data):
        label = data['filename'].replace('.txt', '')
        ax1.plot(data['time'] / 60, data['temperature'], 
                color=colors[i], linewidth=1, alpha=0.3, label=label)
    
    # Plot mean curve
    ax1.plot(representative['time'] / 60, representative['mean_temperature'], 
            'k-', linewidth=3, label='Mean (Representative)', zorder=10)
    
    # Plot median curve
    ax1.plot(representative['time'] / 60, representative['median_temperature'], 
            'b--', linewidth=2, alpha=0.7, label='Median', zorder=9)
    
    # Plot standard deviation band
    ax1.fill_between(representative['time'] / 60,
                     representative['mean_temperature'] - representative['std_temperature'],
                     representative['mean_temperature'] + representative['std_temperature'],
                     alpha=0.2, color='gray', label='±1 Std Dev')
    
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='0°C (melting point)')
    
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'Representative Temperature Curve (n={representative["n_experiments"]} experiments)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    
    # Plot 2: Standard deviation
    ax2.plot(representative['time'] / 60, representative['std_temperature'], 
            'r-', linewidth=2)
    ax2.fill_between(representative['time'] / 60, 0, representative['std_temperature'],
                    alpha=0.3, color='red')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Standard Deviation (°C)', fontsize=12)
    ax2.set_title('Temperature Variability Across Experiments', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'representative_temperature_curve.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Also create a simpler version with just the representative curve
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(representative['time'] / 60, representative['mean_temperature'], 
           'k-', linewidth=3, label='Representative Temperature')
    ax.fill_between(representative['time'] / 60,
                    representative['mean_temperature'] - representative['std_temperature'],
                    representative['mean_temperature'] + representative['std_temperature'],
                    alpha=0.3, color='gray', label='±1 Std Dev')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, 
              label='0°C (melting point)')
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Representative Temperature Curve (n={representative["n_experiments"]} experiments)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    output_file2 = output_path / 'representative_temperature_simple.png'
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Simple figure saved to: {output_file2}")
    
    # Save representative temperature data to file
    data_file = output_path / 'representative_temperature_data.txt'
    with open(data_file, 'w') as f:
        f.write("Time (s)\tTime (min)\tTemperature (°C)\tStd Dev (°C)\n")
        for i in range(len(representative['time'])):
            f.write(f"{representative['time'][i]:.2f}\t"
                   f"{representative['time'][i]/60:.4f}\t"
                   f"{representative['mean_temperature'][i]:.4f}\t"
                   f"{representative['std_temperature'][i]:.4f}\n")
    print(f"Representative temperature data saved to: {data_file}")
    
    # Also save as CSV
    data_file_csv = output_path / 'representative_temperature_data.csv'
    with open(data_file_csv, 'w') as f:
        f.write("Time_s,Time_min,Temperature_C,StdDev_C\n")
        for i in range(len(representative['time'])):
            f.write(f"{representative['time'][i]:.2f},"
                   f"{representative['time'][i]/60:.4f},"
                   f"{representative['mean_temperature'][i]:.4f},"
                   f"{representative['std_temperature'][i]:.4f}\n")
    print(f"Representative temperature data (CSV) saved to: {data_file_csv}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Representative Curve Statistics:")
    print("=" * 60)
    print(f"Time range: {representative['time'][0]/60:.1f} to {representative['time'][-1]/60:.1f} minutes")
    print(f"Temperature range: {np.min(representative['mean_temperature']):.2f} to {np.max(representative['mean_temperature']):.2f} °C")
    print(f"Average std dev: {np.mean(representative['std_temperature']):.2f} °C")
    print(f"Max std dev: {np.max(representative['std_temperature']):.2f} °C")
    
    return fig, fig2, representative


if __name__ == '__main__':
    fig1, fig2, representative = plot_representative_temperature()
    print("\nDone!")
