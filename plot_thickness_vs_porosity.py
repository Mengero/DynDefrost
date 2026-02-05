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
    Load sloughing threshold data from CSV file with section-based format.

    The file format has sections for each surface type:
        60deg
        Porosity,Threshold_Thickness_mm
        0.855,2.0
        ...

        140deg
        Porosity,Threshold_Thickness_mm
        0.896,2.16
        ...

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
    dict with keys 'Hydrophilic' and 'Superhydrophobic', each containing:
        'porosity': numpy array
        'thickness_mm': numpy array
    """
    filepath = Path(filepath)

    # Map section headers to surface types
    section_map = {
        '60deg': 'Hydrophilic',
        '140deg': 'Superhydrophobic',
    }

    result = {
        'Hydrophilic': {'porosity': [], 'thickness_mm': []},
        'Superhydrophobic': {'porosity': [], 'thickness_mm': []},
    }

    current_surface = None

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            if line in section_map:
                current_surface = section_map[line]
                continue

            # Skip the column header line
            if line.startswith('Porosity,'):
                continue

            # Parse data line
            if current_surface and ',' in line:
                try:
                    parts = line.split(',')
                    porosity = float(parts[0])
                    thickness = float(parts[1])
                    result[current_surface]['porosity'].append(porosity)
                    result[current_surface]['thickness_mm'].append(thickness)
                except (ValueError, IndexError):
                    continue

    # Convert to numpy arrays and apply capping logic
    for surface in result:
        porosity = np.array(result[surface]['porosity'])
        thickness_mm = np.array(result[surface]['thickness_mm'])

        # Handle capped values: keep only one 8mm point if requested
        if keep_one_cap and len(thickness_mm) > 0:
            cap_indices = np.where(thickness_mm >= cap_value)[0]
            if len(cap_indices) > 1:
                remove_indices = cap_indices[1:]
                keep_mask = np.ones(len(porosity), dtype=bool)
                keep_mask[remove_indices] = False
                porosity = porosity[keep_mask]
                thickness_mm = thickness_mm[keep_mask]

        result[surface]['porosity'] = porosity
        result[surface]['thickness_mm'] = thickness_mm

    return result


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


def plot_single_surface(ax, surface_data, surface_name, fitted_curve=None, porosity_range=None,
                        xlim=None, ylim=None):
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
    xlim : tuple, optional
        (min, max) fixed x-axis limits
    ylim : tuple, optional
        (min, max) fixed y-axis limits
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

    # Set axis limits - use fixed limits if provided, otherwise based on data
    if xlim is not None:
        ax.set_xlim(xlim)
        x_min, x_max = xlim
        # Set x-axis ticks at 0.05 intervals
        ax.set_xticks(np.arange(x_min, x_max + 0.01, 0.05))
    elif len(surface_data['thickness_mm']) > 0:
        x_min, x_max = np.min(surface_data['porosity']), np.max(surface_data['porosity'])
        x_padding = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.01
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
    else:
        x_min, x_max = 0.8, 1.0

    if ylim is not None:
        ax.set_ylim(ylim)
    elif len(surface_data['thickness_mm']) > 0:
        y_min, y_max = np.min(surface_data['thickness_mm']), np.max(surface_data['thickness_mm'])
        y_padding = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Plot fitted critical sloughing threshold line spanning full x range
    if fitted_curve is not None:
        # Use full axis range for the curve
        x_fit = np.linspace(x_min, x_max, 200)
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

    # Load sloughing threshold data for both surfaces
    fitted_curve_hydrophilic = None
    fitted_curve_superhydrophobic = None
    threshold_porosity_range_h = None
    threshold_porosity_range_s = None

    if threshold_data_file and Path(threshold_data_file).exists():
        print(f"\nLoading sloughing threshold data from: {threshold_data_file}")
        threshold_data = load_sloughing_threshold_data(threshold_data_file, keep_one_cap=True)

        # Fit curve for Hydrophilic
        if len(threshold_data['Hydrophilic']['porosity']) > 0:
            print(f"  Hydrophilic: {len(threshold_data['Hydrophilic']['porosity'])} threshold points")
            fitted_curve_hydrophilic = fit_sloughing_threshold(
                threshold_data['Hydrophilic']['porosity'],
                threshold_data['Hydrophilic']['thickness_mm']
            )
            threshold_porosity_range_h = (
                np.min(threshold_data['Hydrophilic']['porosity']),
                np.max(threshold_data['Hydrophilic']['porosity'])
            )
            print(f"    Porosity range: {threshold_porosity_range_h[0]:.3f} - {threshold_porosity_range_h[1]:.3f}")

        # Fit curve for Superhydrophobic
        if len(threshold_data['Superhydrophobic']['porosity']) > 0:
            print(f"  Superhydrophobic: {len(threshold_data['Superhydrophobic']['porosity'])} threshold points")
            fitted_curve_superhydrophobic = fit_sloughing_threshold(
                threshold_data['Superhydrophobic']['porosity'],
                threshold_data['Superhydrophobic']['thickness_mm']
            )
            threshold_porosity_range_s = (
                np.min(threshold_data['Superhydrophobic']['porosity']),
                np.max(threshold_data['Superhydrophobic']['porosity'])
            )
            print(f"    Porosity range: {threshold_porosity_range_s[0]:.3f} - {threshold_porosity_range_s[1]:.3f}")
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

    # Fixed axis limits for both plots
    xlim = (0.8, 1.0)
    ylim = (0, 5.0)

    # Plot Hydrophilic surface with fitted threshold line
    print("\nPlotting Hydrophilic surface...")
    slough_h, other_h = plot_single_surface(
        ax1, hydrophilic_data, 'Hydrophilic Surface',
        fitted_curve=fitted_curve_hydrophilic,
        porosity_range=threshold_porosity_range_h,
        xlim=xlim, ylim=ylim
    )

    # Plot Superhydrophobic surface with fitted threshold line
    print("Plotting Superhydrophobic surface...")
    slough_s, other_s = plot_single_surface(
        ax2, superhydrophobic_data, 'Superhydrophobic Surface',
        fitted_curve=fitted_curve_superhydrophobic,
        porosity_range=threshold_porosity_range_s,
        xlim=xlim, ylim=ylim
    )
    
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

    # Create third figure: Critical thresholds comparison
    print("\nPlotting critical thresholds comparison...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    x_fit = np.linspace(xlim[0], xlim[1], 200)

    # Plot Hydrophilic threshold
    if fitted_curve_hydrophilic is not None:
        y_fit_h = fitted_curve_hydrophilic(x_fit)
        ax3.plot(x_fit, y_fit_h, 'b-', linewidth=2.5, label='Hydrophilic')

    # Plot Superhydrophobic threshold
    if fitted_curve_superhydrophobic is not None:
        y_fit_s = fitted_curve_superhydrophobic(x_fit)
        ax3.plot(x_fit, y_fit_s, 'r-', linewidth=2.5, label='Superhydrophobic')

    # Customize plot
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xticks(np.arange(xlim[0], xlim[1] + 0.01, 0.05))
    ax3.set_xlabel('Frost Porosity (-)', fontsize=18, fontweight='bold')
    ax3.set_ylabel('Critical Sloughing Thickness (mm)', fontsize=18, fontweight='bold')
    ax3.set_title('Critical Sloughing Threshold Comparison', fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=16, direction='in')
    ax3.grid(True, alpha=0.3)

    # Make axis edges thicker
    for spine in ax3.spines.values():
        spine.set_linewidth(2)

    # Set plot region to 1:1 aspect ratio
    ax3.set_box_aspect(1)

    # Add legend
    ax3.legend(loc='upper left', fontsize=16, framealpha=0.9)

    plt.tight_layout()

    # Save third figure
    output_file3 = output_path / 'critical_threshold_comparison.png'
    fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_file3}")

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
