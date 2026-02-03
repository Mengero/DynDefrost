"""
Plot Frost Thickness vs Porosity

This script reads experimental data and plots frost thickness vs porosity,
with different markers for Slough behavior vs other behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from scipy.optimize import curve_fit


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


def load_sloughing_threshold_data(filepath, keep_one_cap=True, cap_value=8.0):
    """
    Load sloughing threshold data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the sloughing threshold CSV file
    keep_one_cap : bool
        If True, keep one 8mm capped point for fitting (to capture rising trend)
    cap_value : float
        The cap value to identify capped points (default 8.0 mm)

    Returns:
    --------
    dict with keys: 'porosity', 'thickness_mm'
    """
    filepath = Path(filepath)

    porosity = []
    thickness_mm = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('Porosity', '').strip():
                continue
            try:
                porosity.append(float(row['Porosity']))
                thickness_mm.append(float(row['Threshold_Thickness_mm']))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

    porosity = np.array(porosity)
    thickness_mm = np.array(thickness_mm)

    # Handle capped values: keep only one 8mm point if requested
    if keep_one_cap:
        cap_indices = np.where(thickness_mm >= cap_value)[0]
        if len(cap_indices) > 1:
            # Keep only the first capped point, remove the rest
            remove_indices = cap_indices[1:]
            keep_mask = np.ones(len(porosity), dtype=bool)
            keep_mask[remove_indices] = False
            porosity = porosity[keep_mask]
            thickness_mm = thickness_mm[keep_mask]

    return {
        'porosity': porosity,
        'thickness_mm': thickness_mm
    }


def fit_sloughing_threshold(porosity, thickness_mm):
    """
    Fit a curve to the sloughing threshold data.

    Parameters:
    -----------
    porosity : array
        Porosity values
    thickness_mm : array
        Threshold thickness values in mm

    Returns:
    --------
    callable
        A function that takes porosity and returns fitted thickness
    """
    # Use exponential fit: t = a * exp(b * porosity) + c
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        # Initial guess
        popt, _ = curve_fit(exp_func, porosity, thickness_mm,
                           p0=[0.1, 5, 1], maxfev=5000)

        def fitted_curve(x):
            return exp_func(x, *popt)

        return fitted_curve
    except RuntimeError:
        # Fallback to polynomial fit if exponential fails
        coeffs = np.polyfit(porosity, thickness_mm, deg=3)

        def fitted_curve(x):
            return np.polyval(coeffs, x)

        return fitted_curve


def plot_single_surface(ax, surface_data, surface_name, fitted_curve=None, porosity_range=None):
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
    fitted_curve : callable, optional
        Fitted curve function for critical sloughing threshold
    porosity_range : tuple, optional
        (min, max) porosity range for plotting the fitted curve
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

    # Set axis limits FIRST based on experimental data only
    if len(surface_data['thickness_mm']) > 0:
        x_min, x_max = np.min(surface_data['porosity']), np.max(surface_data['porosity'])
        y_min, y_max = np.min(surface_data['thickness_mm']), np.max(surface_data['thickness_mm'])
        x_padding = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.01
        y_padding = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Plot fitted critical sloughing threshold line (AFTER setting axis limits)
    if fitted_curve is not None:
        if porosity_range is None:
            porosity_range = (x_min, x_max)
        x_fit = np.linspace(porosity_range[0], porosity_range[1], 200)
        y_fit = fitted_curve(x_fit)
        ax.plot(x_fit, y_fit, 'g-', linewidth=2, label='Critical threshold', zorder=1)

    # Customize plot
    ax.set_xlabel('Frost Porosity (-)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Frost Thickness (mm)', fontsize=18, fontweight='bold')
    ax.set_title(surface_name, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.grid(True, alpha=0.3)

    # Make axis edges thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Set plot region to 1:1 aspect ratio (square box)
    ax.set_box_aspect(1)

    # Add legend outside plotting region
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, framealpha=0.9)

    return slough_indices, other_indices


def plot_thickness_vs_porosity(data_file='exp_data/defrost_sloughing_experiment_data.csv',
                                output_dir='figure', figsize=(18, 8),
                                threshold_data_file='figure/sloughing_threshold_data.csv'):
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
    threshold_data_file : str
        Path to the sloughing threshold CSV file for fitted line (Hydrophilic only)
    """
    print("=" * 60)
    print("Plotting Frost Thickness vs Porosity")
    print("=" * 60)

    # Load experimental data
    print(f"\nLoading experimental data from: {data_file}")
    data = load_experimental_data(data_file)

    n_points = len(data['thickness_mm'])
    print(f"Loaded {n_points} data points")

    # Load sloughing threshold data for Hydrophilic surface
    fitted_curve_hydrophilic = None
    threshold_porosity_range = None
    if threshold_data_file and Path(threshold_data_file).exists():
        print(f"\nLoading sloughing threshold data from: {threshold_data_file}")
        threshold_data = load_sloughing_threshold_data(threshold_data_file, keep_one_cap=True)
        print(f"Loaded {len(threshold_data['porosity'])} threshold data points")

        # Fit the curve
        fitted_curve_hydrophilic = fit_sloughing_threshold(
            threshold_data['porosity'],
            threshold_data['thickness_mm']
        )
        threshold_porosity_range = (
            np.min(threshold_data['porosity']),
            np.max(threshold_data['porosity'])
        )
        print(f"Fitted curve for porosity range: {threshold_porosity_range[0]:.3f} - {threshold_porosity_range[1]:.3f}")
    else:
        print(f"\nWarning: Threshold data file not found: {threshold_data_file}")

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

    # Plot Hydrophilic surface with fitted threshold line
    print("\nPlotting Hydrophilic surface...")
    slough_h, other_h = plot_single_surface(
        ax1, hydrophilic_data, 'Hydrophilic Surface',
        fitted_curve=fitted_curve_hydrophilic,
        porosity_range=threshold_porosity_range
    )

    # Plot Superhydrophobic surface (no threshold data yet)
    print("Plotting Superhydrophobic surface...")
    slough_s, other_s = plot_single_surface(ax2, superhydrophobic_data, 'Superhydrophobic Surface')
    
    # Add overall title
    fig.suptitle('Frost Thickness vs Porosity by Surface Type',
                 fontsize=22, fontweight='bold', y=1.02)
    
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
