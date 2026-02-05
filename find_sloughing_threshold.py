"""
Find Sloughing Threshold for Different Porosities

This script finds the critical frost thickness for sloughing at each porosity:
- User selects surface type (Superhydrophobic or Hydrophilic)
- User selects which porosities to process
- For each porosity, finds critical thickness by:
  - If experiment shows "Slough": reduce thickness until no sloughing, take last sloughing thickness
  - If experiment shows no sloughing: increase thickness until sloughing occurs
- Results are saved to a log file
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Import simulation function from main.py
from main import run_simulation_with_params


def load_experiment_data(data_file='exp_data/defrost_sloughing_experiment_data.csv'):
    """
    Load experiment data from CSV file.

    Returns
    -------
    list of dict
        Each dict contains: surface_type, porosity, thickness_m, thickness_mm, behavior, data_file
    """
    filepath = Path(data_file)
    experiments = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                surface_type = row['Surface Type'].strip()
                porosity = float(row['porosity (-)'])
                thickness_m = float(row['t (m)'])
                thickness_mm = float(row['t (mm)'])
                behavior = row.get('Behavior', '').strip()
                frosting_time = int(float(row['frosting time (min)']))
                rh = int(round(float(row['RH']) * 100))
                air_temp = int(float(row['Air Dry Bulb [C]']))

                # Build data filename
                deg = '60deg' if surface_type == 'Hydrophilic' else '140deg'
                data_filename = f"{frosting_time}min_{deg}_{rh}%_{air_temp}C.txt"

                experiments.append({
                    'surface_type': surface_type,
                    'porosity': porosity,
                    'thickness_m': thickness_m,
                    'thickness_mm': thickness_mm,
                    'behavior': behavior,
                    'data_file': data_filename,
                    'T_ambient': air_temp,
                })
            except (ValueError, KeyError) as e:
                continue

    return experiments


def get_available_surfaces(experiments):
    """Get unique surface types from experiments."""
    return sorted(set(exp['surface_type'] for exp in experiments))


def get_porosities_for_surface(experiments, surface_type):
    """Get unique porosities for a given surface type, grouped with their experiment info."""
    surface_exps = [exp for exp in experiments if exp['surface_type'] == surface_type]

    # Group by porosity
    porosity_dict = {}
    for exp in surface_exps:
        p = exp['porosity']
        if p not in porosity_dict:
            porosity_dict[p] = []
        porosity_dict[p].append(exp)

    return porosity_dict


def prompt_surface_type(available_surfaces):
    """Prompt user to select surface type."""
    print("\n" + "=" * 60)
    print("SELECT SURFACE TYPE")
    print("=" * 60)

    for i, surface in enumerate(available_surfaces, 1):
        print(f"  {i}. {surface}")

    while True:
        try:
            choice = input("\nEnter number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(available_surfaces):
                return available_surfaces[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")


def prompt_porosity_selection(porosity_dict):
    """Prompt user to select which porosities to process."""
    print("\n" + "=" * 60)
    print("SELECT POROSITIES TO PROCESS")
    print("=" * 60)

    porosities = sorted(porosity_dict.keys())

    for i, p in enumerate(porosities, 1):
        exps = porosity_dict[p]
        behaviors = [exp['behavior'] for exp in exps]
        behavior_str = ', '.join(set(behaviors))
        print(f"  {i}. Porosity {p:.4f} - Behaviors: {behavior_str}")

    print(f"\n  a. All porosities")
    print(f"  q. Quit")

    while True:
        choice = input("\nEnter numbers separated by commas (e.g., 1,3,5), 'a' for all, or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            return None

        if choice.lower() == 'a':
            return porosities

        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = []
            for idx in indices:
                if 0 <= idx < len(porosities):
                    selected.append(porosities[idx])
                else:
                    print(f"Invalid index: {idx + 1}")
                    break
            else:
                if selected:
                    return selected
        except ValueError:
            print("Please enter valid numbers separated by commas.")


def find_critical_thickness(experiment, surface_type, step_mm=0.1, verbose=True):
    """
    Find critical sloughing thickness for a given experiment case.

    Parameters
    ----------
    experiment : dict
        Experiment data including data_file, thickness_m, behavior, porosity
    surface_type : str
        Surface type for simulation
    step_mm : float
        Thickness step in mm (default 0.1)
    verbose : bool
        Print progress messages

    Returns
    -------
    float or None
        Critical thickness in meters, or None if not found
    """
    step_m = step_mm / 1000.0  # Convert to meters
    data_file = experiment['data_file']
    porosity = experiment['porosity']
    behavior = experiment['behavior']
    exp_thickness = experiment['thickness_m']
    T_ambient = experiment.get('T_ambient', 12.0)

    # Check if data file exists
    data_path = Path('exp_data') / data_file
    if not data_path.exists():
        print(f"    WARNING: Data file not found: {data_file}")
        return None

    if verbose:
        print(f"\n  Processing porosity {porosity:.4f} ({behavior})")
        print(f"    Data file: {data_file}")
        print(f"    Experiment thickness: {exp_thickness*1000:.2f} mm")

    if behavior == 'Slough':
        # Start from experiment thickness, reduce until no sloughing
        # Take the LAST sloughing thickness as critical
        return _find_critical_slough_case(
            data_file, porosity, exp_thickness, T_ambient, surface_type, step_m, verbose
        )
    else:
        # Start from experiment thickness, increase until sloughing
        return _find_critical_noslough_case(
            data_file, porosity, exp_thickness, T_ambient, surface_type, step_m, verbose
        )


def _find_critical_slough_case(data_file, porosity, start_thickness, T_ambient,
                                surface_type, step_m, verbose):
    """
    For sloughing cases: reduce thickness until no sloughing.
    Return the last thickness that showed sloughing.
    """
    current_thickness = start_thickness
    last_sloughing_thickness = None
    min_thickness = 0.0005  # 0.5 mm minimum
    iteration = 0

    if verbose:
        print(f"    Strategy: Reduce thickness from {start_thickness*1000:.2f} mm until no sloughing")

    while current_thickness >= min_thickness:
        iteration += 1
        if verbose:
            print(f"    Iteration {iteration}: thickness = {current_thickness*1000:.2f} mm", end=" ")

        try:
            result = run_simulation_with_params(
                data_file=f"exp_data/{data_file}",
                frost_thickness=current_thickness,
                porosity=porosity,
                surface_type=surface_type,
                T_ambient=T_ambient,
                verbose=False
            )

            if result['sloughing']:
                if verbose:
                    print("-> Sloughing")
                last_sloughing_thickness = current_thickness
                current_thickness -= step_m
            else:
                if verbose:
                    print("-> No sloughing (all melted)")
                # Found the boundary - return last sloughing thickness
                if last_sloughing_thickness is not None:
                    if verbose:
                        print(f"    CRITICAL THICKNESS: {last_sloughing_thickness*1000:.2f} mm")
                    return last_sloughing_thickness
                else:
                    # First iteration had no sloughing - try increasing
                    if verbose:
                        print("    No sloughing at start, trying to increase...")
                    return _find_by_increasing(
                        data_file, porosity, start_thickness, T_ambient,
                        surface_type, step_m, verbose
                    )
        except Exception as e:
            if verbose:
                print(f"-> Error: {e}")
            current_thickness -= step_m

    # Reached minimum thickness
    if last_sloughing_thickness is not None:
        if verbose:
            print(f"    CRITICAL THICKNESS: {last_sloughing_thickness*1000:.2f} mm (reached min)")
        return last_sloughing_thickness

    if verbose:
        print(f"    No sloughing found down to {min_thickness*1000:.2f} mm")
    return None


def _find_critical_noslough_case(data_file, porosity, start_thickness, T_ambient,
                                  surface_type, step_m, verbose):
    """
    For no-sloughing cases: increase thickness until sloughing occurs.
    Return that thickness as critical.
    """
    current_thickness = start_thickness
    max_thickness = 0.0085  # 8.5 mm maximum
    iteration = 0

    if verbose:
        print(f"    Strategy: Increase thickness from {start_thickness*1000:.2f} mm until sloughing")

    while current_thickness <= max_thickness:
        iteration += 1
        if verbose:
            print(f"    Iteration {iteration}: thickness = {current_thickness*1000:.2f} mm", end=" ")

        try:
            result = run_simulation_with_params(
                data_file=f"exp_data/{data_file}",
                frost_thickness=current_thickness,
                porosity=porosity,
                surface_type=surface_type,
                T_ambient=T_ambient,
                verbose=False
            )

            if result['sloughing']:
                if verbose:
                    print("-> Sloughing")
                    print(f"    CRITICAL THICKNESS: {current_thickness*1000:.2f} mm")
                return current_thickness
            else:
                if verbose:
                    print("-> No sloughing")
                current_thickness += step_m
        except Exception as e:
            if verbose:
                print(f"-> Error: {e}")
            current_thickness += step_m

    # Reached maximum thickness without sloughing
    if verbose:
        print(f"    No sloughing up to {max_thickness*1000:.2f} mm, capping at {max_thickness*1000:.2f} mm")
    return max_thickness


def _find_by_increasing(data_file, porosity, start_thickness, T_ambient,
                        surface_type, step_m, verbose):
    """Helper to find critical by increasing when start has no sloughing."""
    current_thickness = start_thickness + step_m
    max_thickness = 0.0085

    while current_thickness <= max_thickness:
        if verbose:
            print(f"    Trying thickness = {current_thickness*1000:.2f} mm", end=" ")

        try:
            result = run_simulation_with_params(
                data_file=f"exp_data/{data_file}",
                frost_thickness=current_thickness,
                porosity=porosity,
                surface_type=surface_type,
                T_ambient=T_ambient,
                verbose=False
            )

            if result['sloughing']:
                if verbose:
                    print("-> Sloughing")
                    print(f"    CRITICAL THICKNESS: {current_thickness*1000:.2f} mm")
                return current_thickness
            else:
                if verbose:
                    print("-> No sloughing")
                current_thickness += step_m
        except Exception as e:
            if verbose:
                print(f"-> Error: {e}")
            current_thickness += step_m

    if verbose:
        print(f"    No sloughing found up to {max_thickness*1000:.2f} mm")
    return max_thickness


def save_results_to_log(results, log_file='figure/sloughing_threshold_log.csv'):
    """
    Save results to a log file (appends to existing file).

    Parameters
    ----------
    results : list of dict
        Each dict contains: surface_type, porosity, exp_thickness_mm,
        exp_behavior, critical_thickness_mm, timestamp
    log_file : str
        Path to log file
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Check if file exists to determine if we need header
    write_header = not log_path.exists()

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'surface_type', 'porosity', 'exp_thickness_mm',
                      'exp_behavior', 'critical_thickness_mm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for r in results:
            writer.writerow(r)

    print(f"\nResults appended to: {log_file}")


def main():
    """Main entry point for finding sloughing thresholds."""
    print("=" * 60)
    print("FIND CRITICAL SLOUGHING THRESHOLD")
    print("=" * 60)

    # Load experiment data
    print("\nLoading experiment data...")
    experiments = load_experiment_data()
    print(f"  Loaded {len(experiments)} experiment cases")

    # Get available surfaces
    available_surfaces = get_available_surfaces(experiments)
    if not available_surfaces:
        print("ERROR: No surface types found in experiment data")
        return

    # Prompt for surface type
    surface_type = prompt_surface_type(available_surfaces)
    if surface_type is None:
        print("\nExiting.")
        return

    print(f"\nSelected surface type: {surface_type}")

    # Get porosities for selected surface
    porosity_dict = get_porosities_for_surface(experiments, surface_type)
    if not porosity_dict:
        print(f"ERROR: No porosities found for {surface_type}")
        return

    # Prompt for porosity selection
    selected_porosities = prompt_porosity_selection(porosity_dict)
    if selected_porosities is None:
        print("\nExiting.")
        return

    print(f"\nSelected porosities: {[f'{p:.4f}' for p in selected_porosities]}")

    # Process each selected porosity
    print("\n" + "=" * 60)
    print("FINDING CRITICAL THICKNESSES")
    print("=" * 60)

    results = []
    timestamp = datetime.now().isoformat()

    for porosity in selected_porosities:
        # Get experiment cases for this porosity
        cases = porosity_dict[porosity]

        # Pick the case to use:
        # - If any sloughing case exists, use the one with lowest thickness
        # - Otherwise, use the case with highest thickness
        slough_cases = [c for c in cases if c['behavior'] == 'Slough']
        if slough_cases:
            case = min(slough_cases, key=lambda x: x['thickness_m'])
        else:
            case = max(cases, key=lambda x: x['thickness_m'])

        # Find critical thickness
        critical_thickness = find_critical_thickness(case, surface_type, verbose=True)

        # Store result
        result = {
            'timestamp': timestamp,
            'surface_type': surface_type,
            'porosity': porosity,
            'exp_thickness_mm': case['thickness_mm'],
            'exp_behavior': case['behavior'],
            'critical_thickness_mm': critical_thickness * 1000 if critical_thickness else None,
        }
        results.append(result)

        if critical_thickness:
            print(f"\n  >> Porosity {porosity:.4f}: Critical thickness = {critical_thickness*1000:.2f} mm")
        else:
            print(f"\n  >> Porosity {porosity:.4f}: Could not determine critical thickness")

    # Save results
    if results:
        save_results_to_log(results)

        # Also save to a simple CSV for plotting
        threshold_file = 'figure/sloughing_threshold_data.csv'
        with open(threshold_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Porosity', 'Threshold_Thickness_mm'])
            for r in results:
                if r['critical_thickness_mm'] is not None:
                    writer.writerow([r['porosity'], r['critical_thickness_mm']])
        print(f"Threshold data saved to: {threshold_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Surface type: {surface_type}")
    print(f"Porosities processed: {len(results)}")
    for r in results:
        crit = f"{r['critical_thickness_mm']:.2f} mm" if r['critical_thickness_mm'] else "N/A"
        print(f"  Porosity {r['porosity']:.4f}: {crit} (exp: {r['exp_behavior']})")

    print("\nDone!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
