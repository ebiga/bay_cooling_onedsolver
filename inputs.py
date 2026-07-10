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
K_SYS=1.5

# PIPELINE ELEMENTS
# Types are pipe or bend
# If only width is provided, assumes circular
layout = [
    {"type": "pipe", "length": 0.45, "width": 0.45, "height": 0.45},
    {"type": "bend", "r_centerline": 0.45, "width": 0.45, "height": 0.45},
    {"type": "pipe", "length": 0.45, "width": 0.45},
]
