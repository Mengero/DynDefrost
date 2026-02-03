"""
Plot Temperature Repeatability

This script plots temperature curves from multiple experimental cases to demonstrate
that the experimental temperature control is repeatable across different tests.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv


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

    Parameters:
    -----------
    filepath : str or Path
        Path to data file

    Returns:
    --------
    dict with keys: 'time', 'temperature', 'filename'
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

    return {
        'time': np.array(time_sensor1),
        'temperature': np.array(temp_sensor1),
        'filename': filepath.name
    }


def plot_temperature_repeatability(data_dir='exp_data', output_dir='figure', figsize=(10, 7)):
    """
    Plot temperature curves from selected experimental cases to show repeatability.

    Parameters:
    -----------
    data_dir : str
        Directory containing experimental data files
    output_dir : str
        Directory to save the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    print("=" * 60)
    print("Plotting Temperature Repeatability")
    print("=" * 60)

    data_path = Path(data_dir)

    # Selected cases for repeatability demonstration
    # 2 cases from 60deg, 2 cases from 140deg (superhydrophobic)
    selected_cases = [
        '150min_60deg_45%_22C.txt',
        '180min_60deg_45%_22C.txt',
        '30min_140deg_63%_12C.txt',
        '90min_140deg_83%_12C.txt',
    ]

    # Colors for each case
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    print(f"\nLoading {len(selected_cases)} selected cases...")

    # Load data for each case
    loaded_data = []
    for case in selected_cases:
        filepath = data_path / case
        if filepath.exists():
            data = load_experimental_data(filepath)
            loaded_data.append(data)
            print(f"  Loaded: {case} ({len(data['time'])} points)")
        else:
            print(f"  WARNING: File not found: {case}")

    if not loaded_data:
        print("ERROR: No data files found!")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each case
    for i, data in enumerate(loaded_data):
        label = data['filename'].replace('.txt', '').replace('_', ' ')
        time_min = data['time'] / 60
        temp = data['temperature']

        # Plot line
        ax.plot(time_min, temp, color=colors[i], linewidth=2, label=label)

    # Add 0°C reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Customize plot
    ax.set_xlabel('Time (minutes)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=18, fontweight='bold')
    ax.set_title('Temperature Control Repeatability', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.grid(True, alpha=0.3)

    # Make axis edges thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Set plot region to 1:1 aspect ratio (square box)
    ax.set_box_aspect(1)

    # Add legend outside plotting region
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'temperature_repeatability.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    for data in loaded_data:
        t = data['time']
        temp = data['temperature']
        print(f"\n{data['filename']}:")
        print(f"  Duration: {t[-1]/60:.1f} minutes")
        print(f"  T_initial: {temp[0]:.1f}°C")
        print(f"  T_final: {temp[-1]:.1f}°C")
        print(f"  T_range: {np.max(temp) - np.min(temp):.1f}°C")

    return fig


def load_frost_growth_data(filepath='exp_data/defrost_sloughing_experiment_data.csv'):
    """
    Load frost growth data from CSV and group by experimental condition.

    Parameters:
    -----------
    filepath : str
        Path to the CSV data file

    Returns:
    --------
    dict : Keys are condition labels, values are dicts with 'time', 'thickness', 'porosity' arrays
    """
    filepath = Path(filepath)
    conditions = {}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('frosting time (min)', '').strip():
                continue

            try:
                surface = row['Surface Type'].strip()
                temp = int(float(row['Air Dry Bulb [C]']))
                rh = int(float(row['RH']) * 100)
                time_min = float(row['frosting time (min)'])
                thickness = float(row['t (mm)'])
                porosity = float(row['porosity (-)'])

                # Create condition label
                label = f"{surface} {temp}°C {rh}%RH"

                if label not in conditions:
                    conditions[label] = {
                        'time': [],
                        'thickness': [],
                        'porosity': [],
                        'surface': surface
                    }

                conditions[label]['time'].append(time_min)
                conditions[label]['thickness'].append(thickness)
                conditions[label]['porosity'].append(porosity)

            except (ValueError, KeyError) as e:
                continue

    # Convert lists to sorted numpy arrays
    for label in conditions:
        # Sort by time
        indices = np.argsort(conditions[label]['time'])
        conditions[label]['time'] = np.array(conditions[label]['time'])[indices]
        conditions[label]['thickness'] = np.array(conditions[label]['thickness'])[indices]
        conditions[label]['porosity'] = np.array(conditions[label]['porosity'])[indices]

    return conditions


def plot_frost_growth(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                      output_dir='figure', figsize=(10, 12)):
    """
    Plot frost thickness and porosity vs frosting time in two vertically stacked subplots.

    Parameters:
    -----------
    data_file : str
        Path to the CSV data file
    output_dir : str
        Directory to save the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    print("=" * 60)
    print("Plotting Frost Growth (Thickness & Porosity)")
    print("=" * 60)

    # Load data grouped by condition
    conditions = load_frost_growth_data(data_file)
    print(f"\nFound {len(conditions)} experimental conditions:")
    for label in conditions:
        print(f"  {label}: {len(conditions[label]['time'])} data points")

    # Colors for each condition (5 conditions)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Create figure with 2 vertically stacked subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Markers for each condition
    markers = ['o', 's', '^', 'D', 'v']

    # Plot each condition
    for i, (label, data) in enumerate(conditions.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Top subplot: Solid line with markers for thickness
        ax1.plot(data['time'], data['thickness'], color=color, linewidth=2,
                 linestyle='-', marker=marker, markersize=8, label=label)

        # Bottom subplot: Dashed line with markers for porosity
        ax2.plot(data['time'], data['porosity'], color=color, linewidth=2,
                 linestyle='--', marker=marker, markersize=8, label=label)

    # Customize top subplot (thickness)
    ax1.set_ylabel('Frost Thickness (mm)', fontsize=15, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='x', labelbottom=False)  # Remove x-tick labels from top plot
    ax1.set_title('Frost Growth vs Frosting Time', fontsize=17, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Make axis edges thicker for top subplot
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Set aspect ratio y:x = 1:2 (width is 2x height)
    ax1.set_box_aspect(0.5)

    # Customize bottom subplot (porosity)
    ax2.set_xlabel('Frosting Time (min)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Porosity (-)', fontsize=15, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=13, direction='in')
    ax2.grid(True, alpha=0.3)

    # Make axis edges thicker for bottom subplot
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    # Set aspect ratio y:x = 1:2 (width is 2x height)
    ax2.set_box_aspect(0.5)

    # Create legend (shared for both plots)
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, label in enumerate(conditions.keys()):
        color = colors[i % len(colors)]
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=label))
    # Add line style indicators
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=2,
                                  linestyle='-', label='— Thickness'))
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=2,
                                  linestyle='--', label='-- Porosity'))

    # Place legend outside on the right
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=11, framealpha=0.9)

    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.05)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'frost_growth.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    return fig


if __name__ == '__main__':
    fig1 = plot_temperature_repeatability()
    fig2 = plot_frost_growth()
    print("\nDone!")
