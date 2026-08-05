"""
Plot Frost Thickness Evolution During Defrost

This script runs the defrost simulation for several experimental cases of one
condition series and plots the total frost thickness vs defrost time together
with the critical sloughing thickness threshold. Cases that reach the
threshold slough (marked); cases that stay below it drain.

Simulation histories are cached under log/ so each case only runs once.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
from pathlib import Path
from main import run_simulation_with_params


# Cases to compare: one condition series (Hydrophilic, 22°C, 45%RH) covering
# both experimental outcomes (Drain and Slough)
DEFAULT_CASES = [
    '60min_60deg_45%_22C.txt',
    '90min_60deg_45%_22C.txt',
    '120min_60deg_45%_22C.txt',
    '180min_60deg_45%_22C.txt',
]


def get_case_history(data_file, results_dir='sim_results/defrost_histories'):
    """
    Run the defrost simulation for a case (or load previously saved results)
    and return the thickness/threshold histories.

    Results are saved as CSV files tracked in the repository, so each case is
    only ever simulated once (delete the CSV to force a re-run).

    Parameters
    ----------
    data_file : str
        Temperature data file name (e.g., "60min_60deg_45%_22C.txt")
    results_dir : str
        Directory for saved simulation histories

    Returns
    -------
    dict with keys: 'time' [s], 'h_total' [m], 'h_crit' [m],
                    'sloughing' (bool), 'sloughing_time' [s or nan]
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / (Path(data_file).stem + '.csv')

    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().lstrip('# ')
        meta = dict(item.split('=') for item in header.split(','))
        data = np.loadtxt(results_file, delimiter=',', skiprows=2)
        return {
            'time': data[:, 0],
            'h_total': data[:, 1],
            'h_crit': data[:, 2],
            'sloughing': meta['sloughing'] == 'True',
            'sloughing_time': float(meta['sloughing_time']),
        }

    print(f"  Simulating {data_file} (no saved results)...")
    sim = run_simulation_with_params(data_file, verbose=False)
    results = sim['results']

    history = {
        'time': np.asarray(results['time'], dtype=float),
        'h_total': np.asarray(results['h_total'], dtype=float),
        'h_crit': np.asarray(results['h_crit'], dtype=float),
        'sloughing': bool(sim['sloughing']),
        'sloughing_time': float(sim['sloughing_time']) if sim['sloughing_time'] is not None else np.nan,
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"# sloughing={history['sloughing']},"
                f"sloughing_time={history['sloughing_time']}\n")
        f.write("time_s,h_total_m,h_crit_m\n")
        np.savetxt(f, np.column_stack([history['time'], history['h_total'],
                                       history['h_crit']]), delimiter=',')
    print(f"  Results saved to {results_file}")
    return history


def plot_defrost_thickness_comparison(cases=None, output_dir='figure', figsize=(14, 6)):
    """
    Plot total frost thickness vs defrost time for several cases, with the
    critical sloughing thickness threshold and sloughing events marked.

    Two side-by-side panels:
    left  — full view: thickness together with the complete critical sloughing
            threshold curves, whose pre-melting values are an order of
            magnitude larger than the frost thickness;
    right — zoom on the frost thickness range, where the threshold curves
            cross the thickness curves at the sloughing events.

    Parameters
    ----------
    cases : list of str, optional
        Temperature data file names. Defaults to DEFAULT_CASES.
    output_dir : str
        Directory to save the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    print("=" * 60)
    print("Plotting Defrost Frost Thickness Comparison")
    print("=" * 60)

    if cases is None:
        cases = DEFAULT_CASES

    histories = {}
    for case in cases:
        histories[case] = get_case_history(case)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    max_thickness_mm = 0.0
    max_crit_mm = 0.0
    for i, case in enumerate(cases):
        hist = histories[case]
        color = colors[i % len(colors)]
        time_min = hist['time'] / 60
        h_total_mm = hist['h_total'] * 1000
        h_crit_mm = hist['h_crit'] * 1000
        max_thickness_mm = max(max_thickness_mm, np.nanmax(h_total_mm))
        max_crit_mm = max(max_crit_mm, np.nanmax(h_crit_mm[np.isfinite(h_crit_mm)]))

        frosting_time = Path(case).stem.split('_')[0].replace('min', '')
        outcome = 'slough' if hist['sloughing'] else 'drain'
        label = f"{frosting_time} min frosting ({outcome})"

        for ax in (ax1, ax2):
            # Frost thickness history (solid)
            ax.plot(time_min, h_total_mm, color=color, linewidth=2.5,
                    label=label if ax is ax1 else None)

            # Critical sloughing threshold history (dashed, same color)
            ax.plot(time_min, h_crit_mm, color=color, linewidth=1.5,
                    linestyle='--', alpha=0.7)

            # Mark the sloughing event (hollow circle) where thickness
            # reaches the threshold
            if hist['sloughing']:
                ax.plot(time_min[-1], h_total_mm[-1], marker='o', markersize=13,
                        markerfacecolor='none', markeredgecolor=color,
                        markeredgewidth=2.5, linestyle='None', zorder=5)

        print(f"  {case}: initial {h_total_mm[0]:.2f} mm, "
              f"{'sloughs at ' + format(time_min[-1], '.2f') + ' min' if hist['sloughing'] else 'drains'}")

    # Left panel: full view including the large pre-melting threshold values
    ax1.set_ylim(0, max_crit_mm * 1.05)
    # Right panel: zoom on the frost thickness range
    ax2.set_ylim(0, max_thickness_mm * 1.25)

    # Shade the zoomed region of the right panel in the left panel
    ax1.axhspan(0, max_thickness_mm * 1.25, color='gray', alpha=0.12, zorder=0)

    fig.suptitle('Frost Thickness During Defrost (Hydrophilic, 22°C 45%RH)',
                 fontsize=17, fontweight='bold')
    for ax in (ax1, ax2):
        ax.set_xlabel('Defrost Time (min)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Frost Thickness (mm)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13, direction='in')
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_box_aspect(1)

    # Legend: case lines plus line-style indicators (shared, outside right, no box)
    from matplotlib.lines import Line2D
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--'))
    labels.append('Critical sloughing threshold')
    handles.append(Line2D([0], [0], marker='o', markersize=11,
                          markerfacecolor='none', markeredgecolor='gray',
                          markeredgewidth=2.5, linestyle='None'))
    labels.append('Sloughing event')
    ax2.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=12, frameon=False)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'defrost_thickness_comparison.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    return fig


if __name__ == '__main__':
    plot_defrost_thickness_comparison()
    print("\nDone!")
