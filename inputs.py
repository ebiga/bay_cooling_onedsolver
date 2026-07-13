altitude_ft=10000.
dISA_K=15.
Mach=0.25
inlet_position_m=6.
outlet_position_m=5.
Cp_exit=-0.1
outlet_type="OutletParallelRamp"
BAY_VOLUME_M3=0.75
T_SYSTEM_MAX_degC=32.0
TARGET_ACPM=5.0
Q_BAY_LOAD_W=3000.

# PIPELINE ELEMENTS
# Types are pipe, bend, or plenum
# If only width (no height) is provided, assumes circular section
    # {"type": "pipe", "length": _, "width": _, "height": _optional_},
    # {"type": "bend", "r_centerline": _, "width": _, "height": _optional_},
    # {"type": "plenum", "KL": _},
layout = [
    {"type": "pipe", "length": 0.45, "width": 0.45, "height": 0.45},
    {"type": "bend", "r_centerline": 0.45, "width": 0.45, "height": 0.45},
    {"type": "pipe", "length": 0.45, "width": 0.45},
    {"type": "plenum", "KL": 1.5},
]
