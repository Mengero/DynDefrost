"""
Dynamic Defrost Model - Main Script

1-D Dynamic Defrost Model for simulating frost layer behavior during defrost cycles.
"""

import matplotlib.pyplot as plt
from data_loader import load_defrost_data, get_frost_properties
from model_init import initialize_model


def main():
    """Main entry point for the dynamic defrost model."""
    # ===== User Parameters =====
    data_file = "55min_40deg_83%_12C.txt"
    n_layers = 4
    # ===========================
    
    loader, time, temperature = load_defrost_data(data_file)
    frost_props = get_frost_properties(data_file)
    
    model = initialize_model(
        n_layers=n_layers,
        frost_thickness=frost_props['thickness'],
        porosity=frost_props['porosity'],
        T_initial=temperature[0]
    )
    
    loader.plot()
    plt.show()
    
    return time, temperature, frost_props, model


if __name__ == "__main__":
    time, temperature, frost_props, model = main()
