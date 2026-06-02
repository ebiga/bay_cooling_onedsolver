import numpy as np

from oned_ventilation_solver import *

def map_flight_envelope_sizing(bay_volume, acpm, cp_exit, outlet_type, porosity=1.0):
    """
    Sweeps a 2D grid of the flight envelope to locate the absolute worst-case 
    sizing condition (maximum required throat area) for compliance verification.
    """
    # Define the boundaries of your operational flight envelope
    altitudes = np.linspace(0, 40000, 25)     # Sea level to 40k feet
    machs = np.linspace(0.15, 0.80, 25)       # Low-speed stall margin to Max Cruise
    
    max_area = -1.0
    critical_condition = {}
    
    # Track full results grid for optional contour plotting
    results_grid = []

    for alt in altitudes:
        for m in machs:
            # Freeze dISA at 0 (or a conservative hot day like +15 for sizing)
            p_inf, t_inf, _, _ = atmo(alt, dISA_degC=0.0)
            
            res = size_fire_zone_ventilation(
                bay_volume_m3=bay_volume,
                acpm=acpm,
                t_inf=t_inf,
                p_inf=p_inf,
                mach=m,
                cp_exit=cp_exit,
                outlet_type=outlet_type,
                grill_porosity=porosity
            )
            
            if res["status"] == "Success":
                area = res["a_throat_cm2"]
                results_grid.append((alt, m, area))
                
                # We are looking for the MAXIMUM required area across the envelope
                if area > max_area:
                    max_area = area
                    critical_condition = {
                        "altitude_ft": alt,
                        "mach": m,
                        "area_cm2": area,
                        "mfr": res["mfr"],
                        "cd_converged": res["Cd_used"]
                    }
                    
    return critical_condition, results_grid

# --- Execution Example ---
if __name__ == "__main__":
    # Using the same baseline architecture from your script
    VOLUME = 2.4
    TARGET_ACPM = 5.0
    CP_EXIT = -0.1
    OUTLET = "inverted_naca"

    crit_pt, grid_data = map_flight_envelope_sizing(VOLUME, TARGET_ACPM, CP_EXIT, OUTLET)

    print("=========================================================")
    print(f" CRITICAL DESIGN CONDITION LOCATED ({OUTLET.upper()})")
    print("=========================================================")
    print(f" Worst-Case Altitude : {crit_pt['altitude_ft']:.0f} ft")
    print(f" Worst-Case Mach     : {crit_pt['mach']:.3f}")
    print(f" Max Req. Throat Area: {crit_pt['area_cm2']:.2f} cm²")
    print(f" Operating MFR       : {crit_pt['mfr']:.3f}")
    print(f" Converged Outlet Cd : {crit_pt['cd_converged']:.3f}")
    print("=========================================================")