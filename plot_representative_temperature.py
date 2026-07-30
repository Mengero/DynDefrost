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


def plot_superhydrophobic_temperature_variation(data_dir='exp_data', output_dir='figure', figsize=(12, 8)):
    """
    Plot temperature vs time for each superhydrophobic (140deg) experiment only,
    so each experiment's temperature variation with time is visible.

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
    print("Plotting Superhydrophobic Temperature Variation")
    print("=" * 60)

    data_path = Path(data_dir)
    # Superhydrophobic experiments: filenames contain 140deg
    all_txt = sorted(data_path.glob('*.txt'))
    superhydrophobic_files = [f.name for f in all_txt if '140deg' in f.name]

    if not superhydrophobic_files:
        print("WARNING: No superhydrophobic (140deg) data files found.")
        return None

    print(f"\nFound {len(superhydrophobic_files)} superhydrophobic experiments:")
    loaded_data = []
    for case in superhydrophobic_files:
        filepath = data_path / case
        data = load_experimental_data(filepath)
        loaded_data.append(data)
        print(f"  {case} ({len(data['time'])} points)")

    if not loaded_data:
        print("ERROR: No data could be loaded!")
        return None

    # Colormap for many lines (one color per experiment)
    n_cases = len(loaded_data)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_cases, 10)))[:n_cases]
    if n_cases > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_cases))

    fig, ax = plt.subplots(figsize=figsize)

    for i, data in enumerate(loaded_data):
        label = data['filename'].replace('.txt', '').replace('_', ' ')
        time_min = data['time'] / 60
        temp = data['temperature']
        ax.plot(time_min, temp, color=colors[i], linewidth=2, label=label)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (minutes)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=18, fontweight='bold')
    ax.set_title('Superhydrophobic (140°) Experiments: Temperature vs Time', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_box_aspect(1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'temperature_superhydrophobic_variation.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Superhydrophobic temperature statistics:")
    print("=" * 60)
    for data in loaded_data:
        t = data['time']
        temp = data['temperature']
        print(f"  {data['filename']}: duration {t[-1]/60:.1f} min, T [{temp.min():.1f}, {temp.max():.1f}] °C")

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
                      output_dir='figure', figsize=(9, 7)):
    """
    Plot the defrost initial conditions as a porosity vs thickness map.

    Each experiment is one point in the (thickness, porosity) state space used
    to initialize the defrost model. Marker shape identifies the experimental
    condition and marker color encodes the frosting time, so separate
    experiments cannot be misread as one continuous frosting measurement.

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
    print("Plotting Defrost Initial Conditions (Thickness & Porosity)")
    print("=" * 60)

    # Load data grouped by condition
    conditions = load_frost_growth_data(data_file)
    print(f"\nFound {len(conditions)} experimental conditions:")
    for label in conditions:
        print(f"  {label}: {len(conditions[label]['time'])} data points")

    # Markers for each condition
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=figsize)

    # Shared color scale for frosting time across all conditions.
    # Single-hue light->dark blue ramp (darker = longer frosting); the palest
    # 25% of Blues is cut so short-time points don't wash out on white.
    from matplotlib.colors import LinearSegmentedColormap
    all_times = np.concatenate([conditions[label]['time'] for label in conditions])
    norm = plt.Normalize(all_times.min(), all_times.max())
    cmap = LinearSegmentedColormap.from_list(
        'Blues_trunc', plt.cm.Blues(np.linspace(0.25, 1.0, 256)))

    # Plot each condition: one marker shape per condition, color = frosting time
    for i, (label, data) in enumerate(conditions.items()):
        marker = markers[i % len(markers)]
        ax.scatter(data['thickness'], data['porosity'], c=data['time'],
                   cmap=cmap, norm=norm, marker=marker, s=110,
                   edgecolors='black', linewidths=0.8, zorder=3, label=label)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label('Frosting Time (min)', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel('Frost Thickness (mm)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Porosity (-)', fontsize=15, fontweight='bold')
    ax.set_title('Defrost Initial Conditions', fontsize=17, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax.grid(True, alpha=0.3)

    # Make axis edges thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_box_aspect(1)

    # Legend shows marker shapes only, unfilled: fill color encodes frosting
    # time, so legend markers must not carry any fill color
    legend = ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    for handle in legend.legend_handles:
        handle.set_facecolor('none')
        handle.set_edgecolor('black')

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
    fig2 = plot_superhydrophobic_temperature_variation()
    fig3 = plot_frost_growth()
    print("\nDone!")
