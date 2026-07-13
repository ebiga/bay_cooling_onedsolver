# Flight Conditions
altitude_ft=10000.
dISA_K=15.
Mach=0.25


# Inlet Information
inlet_position_m=6.


# Outlet Information
outlet_position_m=5.
Cp_exit=-0.1
outlet_type="OutletParallelRamp"


# PIPELINE ELEMENTS
# Piping types: pipe, bend [as many as needed in sequence]
# Consumer types: 
# If only width (no height) is provided, assumes circular section
    # {"type": "pipe", "length": _, "width": _, "height": _optional_}
    # {"type": "bend", "r_centerline": _, "width": _, "height": _optional_}
    # {"type": "VentingBay", "BAY_VOLUME_M3": _, "TARGET_ACPM": _, "KL": _}
    # {"type": "CoolingBay", "Q_BAY_LOAD_W": _, "T_SYSTEM_MAX_degC": _},
layout = [
    {"type": "pipe", "length": 0.45, "width": 0.45, "height": 0.45},
    {"type": "bend", "r_centerline": 0.45, "width": 0.45, "height": 0.45},
    {"type": "pipe", "length": 0.45, "width": 0.45},
    {"type": "VentingBay", "BAY_VOLUME_M3": 0.75, "TARGET_ACPM": 5.0, "KL": 1.5},
    {"type": "CoolingBay", "Q_BAY_LOAD_W": 3000., "T_SYSTEM_MAX_degC": 32.0},
]
