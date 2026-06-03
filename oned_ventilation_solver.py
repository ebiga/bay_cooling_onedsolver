import math

from scipy.optimize import minimize, Bounds

from auxfunctions import *
from InletOutletModels import *


def size_fire_zone_ventilation(bay_volume_m3, acpm, t_inf, p_inf, mach, cp_exit, outlet_type, grill_porosity=0.6, k_sys=1.5):
    """
    Sizes the required NACA inlet throat area to satisfy a minimum 
    Air Changes Per Minute (ACPM) requirement according to Part 25.1187.
    """
    gamm1 = gamma - 1.
    gamm2 = gamm1 / 2.
    
    # 1. Freestream Aero Conditions
    a_inf = math.sqrt(gamma * R_gas * t_inf)
    v_inf = mach * a_inf
    rho_inf = p_inf / (R_gas * t_inf)
    
    p_t_inf = p_inf * (1.0 + gamm2 * mach**2)**(gamma / gamm1)
    t_t_inf = t_inf * (1.0 + gamm2 * mach**2)
    qdin_inf = p_t_inf - p_inf
    
    # External static pressure at the exit dump location
    p_static_ext_exit = p_inf + cp_exit * qdin_inf
    
    # 2. Target Mass Flow Calculation based on Volumetric Changes
    vol_flow_rate_rps = (acpm / 60.0) * bay_volume_m3  # m^3/s
    mdot_target = rho_inf * vol_flow_rate_rps          # kg/s
    

    runtime_tracker = {"Cd": 0.6}
    
    def area_residual(x):
        mfr = x[0]
        
        # Calculate target throat area base on current MFR iteration
        a_throat_guess = mdot_target / (rho_inf * v_inf * mfr)
        
        # ASSUMPTION: Balanced layout footprint where Area_exit == Area_throat
        a_exit = a_throat_guess 
        
        # Dynamic Inlet Total Pressure Recovery
        eta_d = naca_pressure_recovery(mfr)
        p_t_1 = p_inf + eta_d * (p_t_inf - p_inf)
        
        # Node 1 (Throat State)
        m_1 = solve_throat_mach(mdot_target, a_throat_guess, p_t_1, t_t_inf)
        t_static_1 = t_t_inf / (1.0 + gamm2 * m_1**2)
        p_static_1 = p_t_1 / (1.0 + gamm2 * m_1**2)**(gamma / gamm1)
        rho_static_1 = p_static_1 / (R_gas * t_static_1)
        v_1 = m_1 * math.sqrt(gamma * R_gas * t_static_1)
        
        # System Internal Friction Losses
        dp_friction = k_sys * (0.5 * rho_static_1 * v_1**2)
        p_t_bay = p_t_1 - dp_friction
        
        # Isentropic expansion from degraded bay total pressure to external static pressure
        if p_t_bay > p_static_ext_exit:
            rho_t_bay = p_t_bay / (R_gas * t_t_inf)
            rho_exit = rho_t_bay * (p_static_ext_exit / p_t_bay)**(1.0 / gamma)
        else:
            # Fallback protection for unphysical intermediate solver steps
            rho_exit = p_static_ext_exit / (R_gas * t_t_inf)
            
        v_exit_nominal = mdot_target / (rho_exit * a_exit)
        R_vel = v_exit_nominal / v_inf
        
        Cd = get_outlet_cd(outlet_type, R_vel, grill_porosity)
        runtime_tracker["Cd"] = Cd  
        
        # Node 2 (Exit Drop Model via Discharge Coefficient)
        a_effective_exit = Cd * a_exit
        # Use the true corrected exit density for the dynamic backpressure delta P
        dp_outlet = (mdot_target**2) / (2.0 * rho_exit * (a_effective_exit)**2)        

        # The internal pressure must equal external dump pressure plus exit restriction losses
        calculated_p_t_bay = p_static_ext_exit + dp_outlet
        
        # Balance residual loop
        error = ((p_t_bay - calculated_p_t_bay) / p_t_inf)**2
        return error

    try:
        mfr_bounds = Bounds(0.01, 0.95)
        res = minimize(area_residual, x0=[0.4], method='Powell', bounds=mfr_bounds)
        final_mfr = res.x[0]
        final_area = mdot_target / (rho_inf * v_inf * final_mfr)
        
        return {
            "status": "Success",
            "mdot": mdot_target,
            "vol_flow": vol_flow_rate_rps * 60.0, # m3/min
            "a_throat_cm2": final_area * 10000.0,
            "mfr": final_mfr,
            "Cd_used": runtime_tracker["Cd"]
        }
    except Exception:
        return {
            "status": "Failed",
            "reason": "Solver execution failed. Ram air pressure cannot drive this mass flow through chosen outlet restriction."
        }




if __name__ == "__main__":
    # Compartment Parameters
    BAY_VOLUME_M3 = 2.4

    # Part 25.1187 target criteria
    TARGET_ACPM = 5.0
    
    # Environment & Flight parameters
    altitude_ft = 10000.
    dISA_K = 15.
    Mach = 0.25
    Cp_exit = -0.1
    
    p_inf, t_inf, _, _ = atmo(altitude_ft, dISA_K)
    
    outlets_to_test = [
        {"type": "OutletInvertedScoop", "porosity": 1.0},
        {"type": "OutletParallelRamp", "porosity": 1.0},
        {"type": "OutletDivergentRamp", "porosity": 1.0},
        {"type": "OutletGrill", "porosity": 0.65},
        {"type": "OutletGrill", "porosity": 0.45}
    ]
    
    print(f"--- Sizing Summary for Vol: {BAY_VOLUME_M3} m³ at {TARGET_ACPM} ACPM ---")
    for outlet in outlets_to_test:
        res = size_fire_zone_ventilation(
            bay_volume_m3=BAY_VOLUME_M3,
            acpm=TARGET_ACPM,
            t_inf=t_inf,
            p_inf=p_inf,
            mach=Mach,
            cp_exit=Cp_exit,
            outlet_type=outlet["type"],
            grill_porosity=outlet["porosity"]
        )
        
        if res["status"] == "Success":
            label = f"{outlet['type']} (Porosity: {outlet['porosity']})" if 'Grill' in outlet['type'] else outlet['type']
            print(f"\nOutlet Type        : {label}")
            print(f"  Effective Cd      : {res['Cd_used']:.3f}")
            print(f"  Target Mass Flow  : {res['mdot']:.4f} kg/s ({res['vol_flow']:.2f} m³/min)")
            print(f"  Req. Throat Area  : {res['a_throat_cm2']:.2f} cm²")
            print(f"  Operating MFR     : {res['mfr']:.3f}")
        else:
            print(f"\nOutlet Type {outlet['type']} Failed: {res['reason']}")
