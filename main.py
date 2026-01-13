"""
Dynamic Defrost Model - Main Script

1-D Dynamic Defrost Model for simulating frost layer behavior during defrost cycles.
"""

import matplotlib.pyplot as plt
from data_loader import load_defrost_data


def main():
    """Main entry point for the dynamic defrost model."""
    loader, time, temperature = load_defrost_data()
    loader.calculate_heating_rate()
    loader.plot()
    plt.show()
    return time, temperature


if __name__ == "__main__":
    time, temperature = main()
