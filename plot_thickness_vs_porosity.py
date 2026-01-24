"""
Plot Frost Thickness vs Porosity

This script reads experimental data and plots frost thickness vs porosity,
with different markers for Slough behavior vs other behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def load_experimental_data(filepath):
    """
    Load experimental data from CSV file.
    
    Returns:
    --------
    dict with keys: 'thickness_mm', 'porosity', 'behavior', 'surface_type'
    """
    filepath = Path(filepath)
    
    thickness_mm = []
    porosity = []
    behavior = []
    surface_type = []
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows
            if not row.get('t (mm)', '').strip():
                continue
            
            try:
                thickness_mm.append(float(row['t (mm)']))
                porosity.append(float(row['porosity (-)']))
                behavior.append(row['Behavior'].strip())
                surface_type.append(row['Surface Type'].strip())
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    return {
        'thickness_mm': np.array(thickness_mm),
        'porosity': np.array(porosity),
        'behavior': behavior,
        'surface_type': surface_type
    }


def plot_single_surface(ax, surface_data, surface_name):
    """
    Plot data for a single surface type on the given axes.
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes to plot on
    surface_data : dict
        Data dictionary for the surface type
    surface_name : str
        Name of the surface type for title
    """
    # Separate data by behavior
    slough_indices = [i for i, b in enumerate(surface_data['behavior']) if b == 'Slough']
    other_indices = [i for i, b in enumerate(surface_data['behavior']) if b != 'Slough']
    
    # Plot Slough behavior with 'x' marker
    if slough_indices:
        slough_thickness = surface_data['thickness_mm'][slough_indices]
        slough_porosity = surface_data['porosity'][slough_indices]
        ax.scatter(slough_porosity, slough_thickness, 
                  marker='x', s=150, linewidths=3, 
                  color='red', label='Slough', zorder=3)
    
    # Plot other behaviors with 'o' marker
    if other_indices:
        other_thickness = surface_data['thickness_mm'][other_indices]
        other_porosity = surface_data['porosity'][other_indices]
        ax.scatter(other_porosity, other_thickness, 
                  marker='o', s=100, linewidths=2, 
                  edgecolors='blue', facecolors='none', 
                  label='Other behaviors', zorder=2)
    
    # Customize plot
    ax.set_xlabel('Frost Porosity (-)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frost Thickness (mm)', fontsize=12, fontweight='bold')
    ax.set_title(surface_name, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Set axis limits with some padding
    if len(surface_data['thickness_mm']) > 0:
        x_min, x_max = np.min(surface_data['porosity']), np.max(surface_data['porosity'])
        y_min, y_max = np.min(surface_data['thickness_mm']), np.max(surface_data['thickness_mm'])
        x_padding = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.01
        y_padding = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    return slough_indices, other_indices


def plot_thickness_vs_porosity(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                                output_dir='figure', figsize=(16, 6)):
    """
    Plot frost thickness vs porosity with different markers for behavior types.
    Creates separate plots for Hydrophilic and Superhydrophobic surfaces.
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV data file
    output_dir : str
        Directory to save the figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    print("=" * 60)
    print("Plotting Frost Thickness vs Porosity")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    data = load_experimental_data(data_file)
    
    n_points = len(data['thickness_mm'])
    print(f"Loaded {n_points} data points")
    
    # Separate data by surface type
    hydrophilic_indices = [i for i, s in enumerate(data['surface_type']) if s == 'Hydrophilic']
    superhydrophobic_indices = [i for i, s in enumerate(data['surface_type']) if s == 'Superhydrophobic']
    
    print(f"\nSurface type distribution:")
    print(f"  Hydrophilic: {len(hydrophilic_indices)} points")
    print(f"  Superhydrophobic: {len(superhydrophobic_indices)} points")
    
    # Create data dictionaries for each surface type
    hydrophilic_data = {
        'thickness_mm': data['thickness_mm'][hydrophilic_indices],
        'porosity': data['porosity'][hydrophilic_indices],
        'behavior': [data['behavior'][i] for i in hydrophilic_indices]
    }
    
    superhydrophobic_data = {
        'thickness_mm': data['thickness_mm'][superhydrophobic_indices],
        'porosity': data['porosity'][superhydrophobic_indices],
        'behavior': [data['behavior'][i] for i in superhydrophobic_indices]
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot Hydrophilic surface
    print("\nPlotting Hydrophilic surface...")
    slough_h, other_h = plot_single_surface(ax1, hydrophilic_data, 'Hydrophilic Surface')
    
    # Plot Superhydrophobic surface
    print("Plotting Superhydrophobic surface...")
    slough_s, other_s = plot_single_surface(ax2, superhydrophobic_data, 'Superhydrophobic Surface')
    
    # Add overall title
    fig.suptitle('Frost Thickness vs Porosity by Surface Type', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'thickness_vs_porosity.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    
    # Hydrophilic statistics
    print(f"\nHydrophilic Surface:")
    if slough_h:
        slough_thickness_h = hydrophilic_data['thickness_mm'][slough_h]
        slough_porosity_h = hydrophilic_data['porosity'][slough_h]
        print(f"  Slough behavior ({len(slough_h)} points):")
        print(f"    Thickness range: {np.min(slough_thickness_h):.2f} - {np.max(slough_thickness_h):.2f} mm")
        print(f"    Porosity range: {np.min(slough_porosity_h):.3f} - {np.max(slough_porosity_h):.3f}")
        print(f"    Mean thickness: {np.mean(slough_thickness_h):.2f} mm")
        print(f"    Mean porosity: {np.mean(slough_porosity_h):.3f}")
    
    if other_h:
        other_thickness_h = hydrophilic_data['thickness_mm'][other_h]
        other_porosity_h = hydrophilic_data['porosity'][other_h]
        print(f"  Other behaviors ({len(other_h)} points):")
        print(f"    Thickness range: {np.min(other_thickness_h):.2f} - {np.max(other_thickness_h):.2f} mm")
        print(f"    Porosity range: {np.min(other_porosity_h):.3f} - {np.max(other_porosity_h):.3f}")
        print(f"    Mean thickness: {np.mean(other_thickness_h):.2f} mm")
        print(f"    Mean porosity: {np.mean(other_porosity_h):.3f}")
    
    # Superhydrophobic statistics
    print(f"\nSuperhydrophobic Surface:")
    if slough_s:
        slough_thickness_s = superhydrophobic_data['thickness_mm'][slough_s]
        slough_porosity_s = superhydrophobic_data['porosity'][slough_s]
        print(f"  Slough behavior ({len(slough_s)} points):")
        print(f"    Thickness range: {np.min(slough_thickness_s):.2f} - {np.max(slough_thickness_s):.2f} mm")
        print(f"    Porosity range: {np.min(slough_porosity_s):.3f} - {np.max(slough_porosity_s):.3f}")
        print(f"    Mean thickness: {np.mean(slough_thickness_s):.2f} mm")
        print(f"    Mean porosity: {np.mean(slough_porosity_s):.3f}")
    
    if other_s:
        other_thickness_s = superhydrophobic_data['thickness_mm'][other_s]
        other_porosity_s = superhydrophobic_data['porosity'][other_s]
        print(f"  Other behaviors ({len(other_s)} points):")
        print(f"    Thickness range: {np.min(other_thickness_s):.2f} - {np.max(other_thickness_s):.2f} mm")
        print(f"    Porosity range: {np.min(other_porosity_s):.3f} - {np.max(other_porosity_s):.3f}")
        print(f"    Mean thickness: {np.mean(other_thickness_s):.2f} mm")
        print(f"    Mean porosity: {np.mean(other_porosity_s):.3f}")
    
    return fig, (ax1, ax2)


if __name__ == '__main__':
    fig, ax = plot_thickness_vs_porosity()
    print("\nDone!")
