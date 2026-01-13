"""
Data Loader Module for Dynamic Defrost Model

This module handles loading and parsing experimental defrost data files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


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


def get_default_data_path():
    """Get the default path to the experimental data file."""
    data_dir = Path(__file__).parent / "exp_data"
    return data_dir / "40deg_83%_12C.txt"


def load_defrost_data(filepath: str = None):
    """
    Convenience function to load defrost data.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the experimental data file. Uses default if None.
        
    Returns
    -------
    tuple
        (loader, time, temperature)
    """
    if filepath is None:
        filepath = get_default_data_path()
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
