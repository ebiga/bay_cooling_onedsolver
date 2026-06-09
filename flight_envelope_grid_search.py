import numpy as np

from oned_mdot_solver import *


# grid setup
altitudes = np.linspace(0., 40000., 5)
machs = np.linspace(0.10, 0.80, 5)
cps_exit = np.linspace(-0.2, 0.2, 5)


# empty the drawers
results = []
critical_condition = {}
max_area = -1.0


# loop the grid
for alt in altitudes:
    for mach in machs:
        for cp_exit in cps_exit:

            res = run_case(
                altitude_ft=alt,
                dISA_K=15.,
                Mach=mach,
                BAY_VOLUME_M3=2.4,
                TARGET_ACPM=5.0,
                Q_BAY_LOAD_W=1681.,
                T_SYSTEM_MAX_degC=32.0,
                Cp_exit=cp_exit,
                outlet_to_test="OutletParallelRamp",
                if_Solve_Ventilation=True,
                if_Solve_Cooling=False,
            )

            results.append({
                "Mach": mach,
                "Cp_exit": cp_exit,
                **res
            })
                        
            if res["status"] == "Success":
                area = res["inlet__area_cm2"]
                results.append((alt, mach, area))
                
                # We are looking for the MAXIMUM required area across the envelope
                if area > max_area:
                    max_area = area
                    critical_condition = {
                        "altitude_ft": alt,
                        "mach": mach,
                        "area_cm2": area,
                        "mfr": res["mfr"],
                    }



print(f" Worst-Case Altitude : {critical_condition['altitude_ft']:.0f} ft")
print(f" Worst-Case Mach     : {critical_condition['mach']:.3f}")
print(f" Max Req. Throat Area: {critical_condition['area_cm2']:.2f} cm²")
print(f" Operating MFR       : {critical_condition['mfr']:.3f}")
