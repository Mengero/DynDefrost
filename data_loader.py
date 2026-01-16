"""
Data Loader Module for Dynamic Defrost Model

This module handles loading and parsing experimental defrost data files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# Mapping from contact angle to surface type
SURFACE_TYPE_MAP = {
    '60deg': 'Hydrophilic',
    '160deg': 'Superhydrophobic',
}


class FrostPropertiesLoader:
    """Loads frost properties from the summary CSV file."""
    
    def __init__(self, filepath: str = None):
        """
        Initialize the frost properties loader.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the summary CSV file. Uses default if None.
        """
        if filepath is None:
            filepath = Path(__file__).parent / "exp_data" / "defrost_sloughing_experiment_data.csv"
        self.filepath = Path(filepath)
        self.data = None
        
    def load(self):
        """Load the summary CSV file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Summary file not found: {self.filepath}")
        
        # Read CSV data
        self.data = []
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split(',')
        
        # Parse data rows
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.strip().split(',')
            row = {}
            for i, h in enumerate(header):
                if i < len(values) and values[i]:
                    # Try to convert to float, otherwise keep as string
                    try:
                        row[h] = float(values[i])
                    except ValueError:
                        row[h] = values[i]
            self.data.append(row)
        
        print(f"Loaded {len(self.data)} frost property records")
        return self.data
    
    def get_properties(self, surface_type, air_temp, rh, frosting_time=None):
        """
        Get frost properties for a specific case.
        
        Parameters
        ----------
        surface_type : str
            'Hydrophilic' or 'Superhydrophobic'
        air_temp : float
            Air dry bulb temperature in °C
        rh : float
            Relative humidity (0-1)
        frosting_time : float, optional
            Frosting time in minutes. If None, returns all matching records.
            
        Returns
        -------
        dict or list
            Frost properties (porosity, thickness) or list of matching records
        """
        if self.data is None:
            self.load()
        
        # Filter by surface type, air temp, and RH
        matches = []
        for row in self.data:
            if (row.get('Surface Type') == surface_type and
                abs(row.get('Air Dry Bulb [C]', 0) - air_temp) < 0.1 and
                abs(row.get('RH', 0) - rh) < 0.01):
                matches.append(row)
        
        if frosting_time is not None:
            # Find closest frosting time
            for row in matches:
                if abs(row.get('frosting time (min)', 0) - frosting_time) < 0.1:
                    return {
                        'porosity': row.get('porosity (-)'),
                        'thickness': row.get('t (m)'),
                        'thickness_mm': row.get('t (mm)'),
                        'density': row.get('Density (kg/m^3)'),
                        'behavior': row.get('Behavior'),
                        'frosting_time': row.get('frosting time (min)'),
                    }
        
        return matches


class DefrostDataLoader:
    """Loads and processes defrost experimental data."""
    
    def __init__(self, filepath: str):
        """
        Initialize the data loader with a file path.
        
        Parameters
        ----------
        filepath : str
            Path to the experimental data file (.txt)
        """
        self.filepath = Path(filepath)
        self.time = None
        self.temperature = None
        
        # Frost properties from summary file
        self.porosity = None
        self.thickness = None
        
    def load(self):
        """
        Load and parse the experimental data file.
        
        Returns
        -------
        tuple
            (time, temperature) arrays in seconds and °C
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        # Read raw data
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Skip header line
        data_lines = lines[1:]
        
        # Parse data
        time_seconds = []
        temperature = []
        
        start_time = None
        
        for line in data_lines:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            try:
                # Parse time from Sensor 1 column (index 2)
                time_str = parts[2]
                current_time = datetime.strptime(time_str, '%H:%M:%S')
                
                if start_time is None:
                    start_time = current_time
                
                # Convert to seconds from start
                delta = current_time - start_time
                time_seconds.append(delta.total_seconds())
                
                # Parse temperature from Sensor 1 (index 3) - the -20°C to +1.3°C data
                temperature.append(float(parts[3]))
                    
            except (ValueError, IndexError):
                continue
        
        # Convert to numpy arrays
        self.time = np.array(time_seconds)
        self.temperature = np.array(temperature)
        
        print(f"Loaded {len(self.time)} data points from: {self.filepath.name}")
        print(f"  Duration: {self.time[-1]:.1f} s")
        print(f"  Temperature range: {self.temperature[0]:.1f}°C to {self.temperature[-1]:.1f}°C")
        
        return self.time, self.temperature
    
    def get_summary(self):
        """
        Get a summary of the loaded data.
        
        Returns
        -------
        dict
            Summary statistics of the data
        """
        if self.time is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return {
            'n_points': len(self.time),
            'duration_s': self.time[-1] - self.time[0],
            'temp_range': (np.min(self.temperature), np.max(self.temperature)),
            'temp_initial': self.temperature[0],
            'temp_final': self.temperature[-1],
        }
    
    def plot(self, figsize=(10, 6), save_path=None):
        """
        Plot Temperature vs. Time.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes objects
        """
        if self.time is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.time, self.temperature, 'b-', linewidth=1.5, label='Experimental Data')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Defrost Process - Temperature vs. Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add horizontal line at 0°C (melting point)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0°C (melting)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def calculate_heating_rate(self):
        """
        Calculate average heating rate.
        
        Returns
        -------
        float
            Heating rate in °C/min
        """
        if self.time is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        dt = self.time[-1] - self.time[0]
        dT = self.temperature[-1] - self.temperature[0]
        rate = dT / dt * 60
        print(f"  Average heating rate: {rate:.3f} °C/min")
        return rate
    
    def __repr__(self):
        if self.time is None:
            return f"DefrostDataLoader('{self.filepath}') - Data not loaded"
        summary = self.get_summary()
        return (
            f"DefrostDataLoader('{self.filepath}')\n"
            f"  Points: {summary['n_points']}\n"
            f"  Duration: {summary['duration_s']:.1f} s\n"
            f"  Temperature: {summary['temp_initial']:.1f}°C to {summary['temp_final']:.1f}°C"
        )


def parse_case_filename(filename):
    """
    Parse case parameters from filename.
    
    Parameters
    ----------
    filename : str
        Filename like '55min_60deg_83%_12C.txt'
        
    Returns
    -------
    dict
        Parsed parameters: frosting_time, surface_type, rh, air_temp
    """
    name = Path(filename).stem  # Remove extension
    parts = name.split('_')
    
    # Parse frosting time (e.g., '55min' -> 55.0)
    frosting_time_str = parts[0].replace('min', '')
    frosting_time = float(frosting_time_str)
    
    # Parse contact angle -> surface type
    contact_angle = parts[1]  # e.g., '60deg' or '160deg'
    surface_type = SURFACE_TYPE_MAP.get(contact_angle, contact_angle)
    
    # Parse RH (e.g., '83%' -> 0.83)
    rh_str = parts[2].replace('%', '')
    rh = float(rh_str) / 100
    
    # Parse air temp (e.g., '12C' -> 12.0)
    air_temp_str = parts[3].replace('C', '')
    air_temp = float(air_temp_str)
    
    return {
        'frosting_time': frosting_time,
        'surface_type': surface_type,
        'rh': rh,
        'air_temp': air_temp,
    }


def get_frost_properties(filename):
    """
    Get frost properties (porosity, thickness) for a specific case.
    
    Parameters
    ----------
    filename : str
        Case filename like '55min_60deg_83%_12C.txt'
        
    Returns
    -------
    dict
        Frost properties: porosity, thickness, etc.
    """
    params = parse_case_filename(filename)
    loader = FrostPropertiesLoader()
    
    result = loader.get_properties(
        surface_type=params['surface_type'],
        air_temp=params['air_temp'],
        rh=params['rh'],
        frosting_time=params['frosting_time']
    )
    
    if isinstance(result, dict):
        print(f"Frost properties for {filename}:")
        print(f"  Porosity: {result['porosity']:.3f}")
        print(f"  Thickness: {result['thickness']*1000:.2f} mm ({result['thickness']:.4e} m)")
    else:
        print(f"No matching record found for {filename}")
        
    return result


def get_default_data_path():
    """Get the default path to the experimental data file."""
    data_dir = Path(__file__).parent / "exp_data"
    return data_dir / "55min_60deg_83%_12C.txt"


def load_defrost_data(filepath: str = None):
    """
    Convenience function to load defrost data.
    
    Parameters
    ----------
    filepath : str, optional
        Path or filename of the experimental data file. Uses default if None.
        If just a filename is given, looks in exp_data folder.
        
    Returns
    -------
    tuple
        (loader, time, temperature)
    """
    if filepath is None:
        filepath = get_default_data_path()
    else:
        # If just a filename, prepend exp_data folder
        filepath = Path(filepath)
        if not filepath.exists():
            data_dir = Path(__file__).parent / "exp_data"
            filepath = data_dir / filepath.name
    loader = DefrostDataLoader(filepath)
    time, temperature = loader.load()
    return loader, time, temperature


def plot_defrost_data(filepath: str = None, figsize=(10, 6), save_path=None):
    """
    Convenience function to load and plot defrost data.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the experimental data file. Uses default if None.
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    tuple
        (fig, ax, time, temperature)
    """
    loader, time, temperature = load_defrost_data(filepath)
    fig, ax = loader.plot(figsize=figsize, save_path=save_path)
    return fig, ax, time, temperature
