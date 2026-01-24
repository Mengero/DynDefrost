"""
Plot Temperature Variations for All Experimental Data

This script reads all experimental data files and plots temperature variations
during the defrost experiments, including Setpoint, Sensor 1, and Sensor 2 temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re


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


def load_experimental_data(filepath):
    """
    Load experimental data from a .txt file.
    
    Returns:
        dict with keys: 'time_setpoint', 'temp_setpoint', 'time_sensor1', 'temp_sensor1',
                       'time_sensor2', 'temp_sensor2', 'time_power', 'power', 'filename'
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Skip header line
    data_lines = lines[1:] if len(lines) > 1 else []
    
    # Initialize data storage
    time_setpoint = []
    temp_setpoint = []
    time_sensor1 = []
    temp_sensor1 = []
    time_sensor2 = []
    temp_sensor2 = []
    time_power = []
    power = []
    
    start_time_setpoint = None
    start_time_sensor1 = None
    start_time_sensor2 = None
    start_time_power = None
    
    for line in data_lines:
        if not line.strip():
            continue
        
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        
        try:
            # Setpoint data (columns 0, 1)
            if parts[0].strip() and parts[1].strip():
                current_time, seconds = parse_time_to_seconds(parts[0], start_time_setpoint)
                if current_time is not None:
                    if start_time_setpoint is None:
                        start_time_setpoint = current_time
                        seconds = 0.0
                    time_setpoint.append(seconds)
                    temp_setpoint.append(float(parts[1]))
            
            # Sensor 1 data (columns 2, 3)
            if parts[2].strip() and parts[3].strip():
                current_time, seconds = parse_time_to_seconds(parts[2], start_time_sensor1)
                if current_time is not None:
                    if start_time_sensor1 is None:
                        start_time_sensor1 = current_time
                        seconds = 0.0
                    time_sensor1.append(seconds)
                    temp_sensor1.append(float(parts[3]))
            
            # Sensor 2 data (columns 4, 5)
            if parts[4].strip() and parts[5].strip():
                current_time, seconds = parse_time_to_seconds(parts[4], start_time_sensor2)
                if current_time is not None:
                    if start_time_sensor2 is None:
                        start_time_sensor2 = current_time
                        seconds = 0.0
                    time_sensor2.append(seconds)
                    temp_sensor2.append(float(parts[5]))
            
            # Power data (columns 6, 7)
            if parts[6].strip() and parts[7].strip():
                current_time, seconds = parse_time_to_seconds(parts[6], start_time_power)
                if current_time is not None:
                    if start_time_power is None:
                        start_time_power = current_time
                        seconds = 0.0
                    time_power.append(seconds)
                    power.append(float(parts[7]))
                    
        except (ValueError, IndexError) as e:
            continue
    
    return {
        'time_setpoint': np.array(time_setpoint),
        'temp_setpoint': np.array(temp_setpoint),
        'time_sensor1': np.array(time_sensor1),
        'temp_sensor1': np.array(temp_sensor1),
        'time_sensor2': np.array(time_sensor2),
        'temp_sensor2': np.array(temp_sensor2),
        'time_power': np.array(time_power),
        'power': np.array(power),
        'filename': filepath.name
    }


def plot_all_temperatures(data_dir='exp_data', output_dir='figure', figsize=(16, 10)):
    """
    Plot temperature variations for all experimental data files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing experimental data files
    output_dir : str
        Directory to save the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all .txt files (excluding CSV)
    data_files = sorted([f for f in data_path.glob('*.txt')])
    
    if not data_files:
        print(f"No .txt files found in {data_dir}")
        return
    
    print(f"Found {len(data_files)} experimental data files")
    
    # Load all data
    all_data = []
    for filepath in data_files:
        print(f"Loading: {filepath.name}")
        try:
            data = load_experimental_data(filepath)
            if len(data['time_sensor1']) > 0:  # Only include files with valid data
                all_data.append(data)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            continue
    
    if not all_data:
        print("No valid data files found")
        return
    
    print(f"\nSuccessfully loaded {len(all_data)} data files")
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Create a grid of subplots - one for each experiment
    n_experiments = len(all_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    # Main plot: All Sensor 1 temperatures on one plot
    ax_main = plt.subplot2grid((n_rows + 1, n_cols), (0, 0), colspan=n_cols, rowspan=1)
    
    # Individual plots for each experiment
    axes = []
    for i in range(n_experiments):
        row = (i // n_cols) + 1
        col = i % n_cols
        ax = plt.subplot2grid((n_rows + 1, n_cols), (row, col))
        axes.append(ax)
    
    # Plot all Sensor 1 temperatures on main plot
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
    for i, data in enumerate(all_data):
        if len(data['time_sensor1']) > 0:
            label = data['filename'].replace('.txt', '')
            ax_main.plot(data['time_sensor1'] / 60, data['temp_sensor1'], 
                        color=colors[i], linewidth=1.5, alpha=0.7, label=label)
    
    ax_main.set_xlabel('Time (minutes)', fontsize=11)
    ax_main.set_ylabel('Temperature (°C)', fontsize=11)
    ax_main.set_title('All Experiments - Sensor 1 Temperature Variations', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1, label='0°C (melting)')
    ax_main.legend(loc='best', fontsize=7, ncol=2)
    
    # Plot individual experiments with all sensors
    for i, (data, ax) in enumerate(zip(all_data, axes)):
        filename = data['filename'].replace('.txt', '')
        
        # Plot Setpoint
        if len(data['time_setpoint']) > 0:
            ax.plot(data['time_setpoint'] / 60, data['temp_setpoint'], 
                   'k--', linewidth=1, alpha=0.5, label='Setpoint')
        
        # Plot Sensor 1
        if len(data['time_sensor1']) > 0:
            ax.plot(data['time_sensor1'] / 60, data['temp_sensor1'], 
                   'b-', linewidth=1.5, label='Sensor 1')
        
        # Plot Sensor 2 (if valid data - Sensor 2 seems to be around -97°C, might be ambient)
        if len(data['time_sensor2']) > 0:
            # Only plot if temperature is reasonable (not stuck at -97°C)
            temp2 = data['temp_sensor2']
            if np.std(temp2) > 0.1:  # Has variation
                ax.plot(data['time_sensor2'] / 60, temp2, 
                       'g-', linewidth=1, alpha=0.7, label='Sensor 2')
        
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel('Temp (°C)', fontsize=9)
        ax.set_title(filename, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.legend(loc='best', fontsize=7)
    
    # Hide unused subplots
    for i in range(len(all_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'all_temperature_variations.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Also create a summary plot with just Sensor 1 temperatures
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    for i, data in enumerate(all_data):
        if len(data['time_sensor1']) > 0:
            label = data['filename'].replace('.txt', '')
            ax2.plot(data['time_sensor1'] / 60, data['temp_sensor1'], 
                    color=colors[i], linewidth=2, alpha=0.8, label=label)
    
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title('All Experiments - Sensor 1 Temperature Variations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='0°C (melting point)')
    ax2.legend(loc='best', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    output_file2 = output_path / 'temperature_variations_summary.png'
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved to: {output_file2}")
    
    return fig, fig2


if __name__ == '__main__':
    print("=" * 60)
    print("Plotting Temperature Variations for All Experiments")
    print("=" * 60)
    
    fig1, fig2 = plot_all_temperatures()
    
    print("\nDone!")
    print("=" * 60)
