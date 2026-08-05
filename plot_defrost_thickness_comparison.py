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
            columns = f.readline().strip().split(',')
        meta = dict(item.split('=') for item in header.split(','))
        data = np.loadtxt(results_file, delimiter=',', skiprows=2)
        return {
            'time': data[:, 0],
            'h_total': data[:, 1],
            'h_crit': data[:, 2],
            # Older result files predate the wall-layer water fraction column
            'alpha_water_wall': data[:, 3] if len(columns) > 3 else None,
            'sloughing': meta['sloughing'] == 'True',
            'sloughing_time': float(meta['sloughing_time']),
        }

    print(f"  Simulating {data_file} (no saved results)...")
    sim = run_simulation_with_params(data_file, verbose=False)
    results = sim['results']

    time = np.asarray(results['time'], dtype=float)

    # Water volume fraction of the layer next to the wall (heated surface).
    # Layer 0 faces the air, so the wall side is the highest-index layer that
    # still exists (layers at the wall melt away first as defrost proceeds).
    alpha_water = np.asarray(results['alpha_water'], dtype=float)
    dx = np.asarray(results['dx'], dtype=float)
    n_steps = min(len(time), alpha_water.shape[0], dx.shape[0])
    alpha_water_wall = np.full(len(time), np.nan)
    for t in range(n_steps):
        existing = np.where(dx[t] > 1e-12)[0]
        if len(existing) > 0:
            alpha_water_wall[t] = alpha_water[t, existing[-1]]

    history = {
        'time': time,
        'h_total': np.asarray(results['h_total'], dtype=float),
        'h_crit': np.asarray(results['h_crit'], dtype=float),
        'alpha_water_wall': alpha_water_wall,
        'sloughing': bool(sim['sloughing']),
        'sloughing_time': float(sim['sloughing_time']) if sim['sloughing_time'] is not None else np.nan,
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"# sloughing={history['sloughing']},"
                f"sloughing_time={history['sloughing_time']}\n")
        f.write("time_s,h_total_m,h_crit_m,alpha_water_wall\n")
        np.savetxt(f, np.column_stack([history['time'], history['h_total'],
                                       history['h_crit'],
                                       history['alpha_water_wall']]),
                   delimiter=',')
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
                ax.plot(time_min[-1], h_total_mm[-1], marker='o', markersize=9,
                        markerfacecolor='none', markeredgecolor=color,
                        markeredgewidth=2, linestyle='None', zorder=5)

        print(f"  {case}: initial {h_total_mm[0]:.2f} mm, "
              f"{'sloughs at ' + format(time_min[-1], '.2f') + ' min' if hist['sloughing'] else 'drains'}")

    # Left panel: full view including the large pre-melting threshold values
    ax1.set_ylim(0, max_crit_mm * 1.05)
    # Right panel: zoom on the frost thickness range
    ax2.set_ylim(0, max_thickness_mm * 1.25)

    # Shade the zoomed region of the right panel in the left panel
    ax1.axhspan(0, max_thickness_mm * 1.25, color='gray', alpha=0.12, zorder=0)

    fig.suptitle('Frost Thickness During Defrost (Hydrophilic, 22°C 45%RH)',
                 fontsize=19, fontweight='bold')
    for ax in (ax1, ax2):
        ax.set_xlabel('Defrost Time (min)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Frost Thickness (mm)', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16, direction='in')
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_box_aspect(1)

    # Legend: case lines plus line-style indicators (shared, outside right, no box)
    from matplotlib.lines import Line2D
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--'))
    labels.append('Critical sloughing threshold')
    handles.append(Line2D([0], [0], marker='o', markersize=8,
                          markerfacecolor='none', markeredgecolor='gray',
                          markeredgewidth=2, linestyle='None'))
    labels.append('Sloughing event')
    ax2.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=14, frameon=False)

    plt.tight_layout()

    # Add horizontal space so the right panel's y-label clears the left panel
    fig.subplots_adjust(wspace=0.35)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'defrost_thickness_comparison.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    return fig


# One representative case pair per ambient condition / surface wettability:
# (condition label, case without dynamic defrosting, case with dynamic defrosting)
CONDITION_CASE_PAIRS = [
    ('Hydrophilic 22°C 45%RH', '90min_60deg_45%_22C.txt', '120min_60deg_45%_22C.txt'),
    ('Hydrophilic 22°C 55%RH', '60min_60deg_55%_22C.txt', '120min_60deg_55%_22C.txt'),
    ('Hydrophilic 12°C 83%RH', '30min_60deg_83%_12C.txt', '55min_60deg_83%_12C.txt'),
    ('Superhydrophobic 12°C 63%RH', '45min_140deg_63%_12C.txt', '90min_140deg_63%_12C.txt'),
    ('Superhydrophobic 12°C 83%RH', '35min_140deg_83%_12C.txt', '60min_140deg_83%_12C.txt'),
]


def _plot_condition_pairs(quantity, ylabel, output_file, output_dir='figure',
                          figsize=(10, 8)):
    """
    Shared plotting routine for the per-condition case pair figures.

    For each ambient condition / surface wettability, plot one case without
    dynamic defrosting (dashed) and one with dynamic defrosting (solid, with
    a hollow circle at the sloughing event). Color identifies the condition.

    Parameters
    ----------
    quantity : str
        History key to plot ('h_total' [m -> plotted in mm] or 'alpha_water_wall')
    ylabel : str
        Y-axis label
    output_file : str
        Output figure file name
    output_dir : str
        Directory to save the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, case_static, case_dynamic) in enumerate(CONDITION_CASE_PAIRS):
        color = colors[i % len(colors)]
        for case, style in ((case_static, '--'), (case_dynamic, '-')):
            try:
                hist = get_case_history(case)
            except Exception as exc:
                print(f"  WARNING: {case} could not be simulated ({exc}); skipped")
                continue
            values = hist[quantity]
            if values is None:
                print(f"  WARNING: {case} has no saved '{quantity}' data; "
                      f"delete its CSV to re-simulate")
                continue
            if quantity == 'h_total':
                values = values * 1000  # m -> mm
            time_min = hist['time'] / 60
            ax.plot(time_min, values, color=color, linewidth=2.5,
                    linestyle=style, label=label if style == '-' else None)
            if hist['sloughing']:
                # Mark the last finite sample (the final row can be NaN)
                finite = np.where(np.isfinite(values))[0]
                if len(finite) > 0:
                    ax.plot(time_min[finite[-1]], values[finite[-1]],
                            marker='o', markersize=9, markerfacecolor='none',
                            markeredgecolor=color, markeredgewidth=2,
                            linestyle='None', zorder=5)
            outcome = 'sloughs' if hist['sloughing'] else 'no sloughing'
            print(f"  {case}: {outcome}")

    ax.set_xlabel('Defrost Time (min)', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_box_aspect(1)

    # Legend: condition colors plus line-style indicators (no box)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-'))
    labels.append('Dynamic defrosting')
    handles.append(Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--'))
    labels.append('No dynamic defrosting')
    handles.append(Line2D([0], [0], marker='o', markersize=8,
                          markerfacecolor='none', markeredgecolor='gray',
                          markeredgewidth=2, linestyle='None'))
    labels.append('Sloughing event')
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=14, frameon=False)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / output_file
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    return fig


def plot_thickness_by_condition(output_dir='figure'):
    """
    Plot frost thickness during defrost for each ambient condition / surface
    wettability: one case without and one with dynamic defrosting.
    """
    print("=" * 60)
    print("Plotting Frost Thickness by Condition (dynamic vs no dynamic)")
    print("=" * 60)
    return _plot_condition_pairs(
        'h_total', 'Frost Thickness (mm)',
        'defrost_thickness_by_condition.png', output_dir)


def plot_wall_water_fraction(output_dir='figure'):
    """
    Plot the water volume fraction of the layer next to the wall (heated
    surface) during defrost for the same case pairs.
    """
    print("=" * 60)
    print("Plotting Wall-Layer Water Volume Fraction")
    print("=" * 60)
    return _plot_condition_pairs(
        'alpha_water_wall', 'Water Volume Fraction at Wall Layer (-)',
        'wall_water_fraction.png', output_dir)


def _condition_slug(label):
    """Turn a condition label into a file-name-friendly slug."""
    return (label.replace('°C', 'C').replace('%RH', 'RH')
            .replace(' ', '_').lower())


def _plot_single_condition(label, case_static, case_dynamic, quantity, ylabel,
                           color, output_dir='figure', figsize=(9, 7)):
    """
    Plot one quantity for one ambient condition: the case without dynamic
    defrosting (dashed) and the case with it (solid, hollow circle at the
    sloughing event).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for case, style, kind in ((case_static, '--', 'no dynamic defrosting'),
                              (case_dynamic, '-', 'dynamic defrosting')):
        try:
            hist = get_case_history(case)
        except Exception as exc:
            print(f"  WARNING: {case} could not be simulated ({exc}); skipped")
            continue
        values = hist[quantity]
        if values is None:
            print(f"  WARNING: {case} has no saved '{quantity}' data; "
                  f"delete its CSV to re-simulate")
            continue
        if quantity == 'h_total':
            values = values * 1000  # m -> mm
        time_min = hist['time'] / 60
        frosting_time = Path(case).stem.split('_')[0].replace('min', '')
        ax.plot(time_min, values, color=color, linewidth=2.5, linestyle=style,
                label=f"{frosting_time} min frosting — {kind}")
        if hist['sloughing']:
            # Mark the last finite sample (the final row can be NaN)
            finite = np.where(np.isfinite(values))[0]
            if len(finite) > 0:
                ax.plot(time_min[finite[-1]], values[finite[-1]], marker='o',
                        markersize=9, markerfacecolor='none',
                        markeredgecolor=color, markeredgewidth=2,
                        linestyle='None', zorder=5)
        outcome = 'sloughs' if hist['sloughing'] else 'no sloughing'
        print(f"  {case}: {outcome}")

    ax.set_xlabel('Defrost Time (min)', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_title(label, fontsize=19, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_box_aspect(1)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='o', markersize=8,
                          markerfacecolor='none', markeredgecolor='gray',
                          markeredgewidth=2, linestyle='None'))
    labels.append('Sloughing event')
    ax.legend(handles, labels, fontsize=14, frameon=False, loc='best')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    prefix = 'defrost_thickness' if quantity == 'h_total' else 'wall_water_fraction'
    output_file = output_path / f"{prefix}_{_condition_slug(label)}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {output_file}")
    plt.close(fig)

    return fig


def plot_per_condition_figures(output_dir='figure'):
    """
    Generate one figure per ambient condition / surface wettability for both
    the frost thickness and the wall-layer water volume fraction, each with
    the dynamic (solid) and no-dynamic (dashed) defrosting cases.
    """
    print("=" * 60)
    print("Plotting Per-Condition Defrost Figures")
    print("=" * 60)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (label, case_static, case_dynamic) in enumerate(CONDITION_CASE_PAIRS):
        color = colors[i % len(colors)]
        print(f"\n{label}:")
        _plot_single_condition(label, case_static, case_dynamic,
                               'h_total', 'Frost Thickness (mm)',
                               color, output_dir)
        _plot_single_condition(label, case_static, case_dynamic,
                               'alpha_water_wall',
                               'Water Volume Fraction at Wall Layer (-)',
                               color, output_dir)


if __name__ == '__main__':
    plot_defrost_thickness_comparison()
    plot_thickness_by_condition()
    plot_wall_water_fraction()
    plot_per_condition_figures()
    print("\nDone!")
