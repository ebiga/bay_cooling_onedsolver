import math

from scipy.optimize import minimize, Bounds

from auxfunctions import *
from InletOutletModels import *


def size_ventilation(mdot_target_kg_s, T_inf_K, p_inf_Pa, Mach, Cp_exit, outlet_type, T_max_K=None, k_sys=1.5):
    """
    Sizes the required NACA inlet throat area to satisfy a target mfr and systems requirements: cooling or ventilation.
    """
    
    # Freestream Aerodynamics & Stagnation States
    a_inf = math.sqrt(gamma * R_gas * T_inf_K)
    v_inf = Mach * a_inf
    rho_inf = p_inf_Pa / (R_gas * T_inf_K)
    
    pt_inf = p_inf_Pa * (1.0 + gamm2 * Mach**2)**(gamma/gamm1)
    Tt_inf = T_inf_K  * (1.0 + gamm2 * Mach**2)

    qdin_inf = pt_inf - p_inf_Pa


    # External static pressure at the exit dump location
    p_static_ext_exit = p_inf_Pa + Cp_exit * qdin_inf


    # Fixed Compartment Exit Pressure Boundary
    p_static_2 = p_static_ext_exit
    Tt_2 = T_max_K  # ASSUMPTION: The air leaves at the maximum rated temperature limit

    
    # Define the Geometric Residual Function
    #_ This receives mfr instead of actual area cause it's more stable and more physically bound
    def area_residual(x):

        mfr = x[0]
        
        # Calculate target throat area base on current MFR iteration
        a_throat_guess = mdot_target_kg_s / (rho_inf * v_inf * mfr)
        
        # ASSUMPTION: Balanced layout footprint where Area_exit == Area_throat
        a_exit = a_throat_guess 
        
        # Dynamic Inlet Total Pressure Recovery
        eta_d = naca_pressure_recovery(mfr)
        pt_1 = p_inf_Pa + eta_d * qdin_inf
        Tt_1 = Tt_inf  # ASSUMPTION: no total temp losses at the inlet
        
        # Node 1 (Throat State)
        m_1 = solve_throat_mach(mdot_target_kg_s, a_throat_guess, pt_1, Tt_1)
        t_static_1 = Tt_1 / (1.0 + gamm2 * m_1**2)
        p_static_1 = pt_1 / (1.0 + gamm2 * m_1**2)**(gamma/gamm1)
        rho_static_1 = p_static_1 / (R_gas * t_static_1)
        v_1 = m_1 * math.sqrt(gamma * R_gas * t_static_1)


        # Losses: Friction
        dp_friction = k_sys * (0.5 * rho_static_1 * v_1**2)


        # Node 2-A (Exit, Thermal demand)
        #_ Static State via Quadratic Energy Equation
        dp_thermal = 0.

        if T_max_K:
            coeff_a = (R_gas * mdot_target_kg_s)**2 / (2.0 * cp_air * (p_static_2 * a_exit)**2)
            t_static_2 = (-1.0 + math.sqrt(1.0 + 4.0 * coeff_a * Tt_2)) / (2.0 * coeff_a)
            
            rho_static_2 = p_static_2 / (R_gas * t_static_2)
            v_2 = mdot_target_kg_s / (rho_static_2 * a_exit)
            m_2 = v_2 / math.sqrt(gamma * R_gas * t_static_2)
            pt_2 = p_static_2 * (1.0 + gamm2 * m_2**2)**(gamma/gamm1)
            
            # Losses: Thermal Expansion Rayleigh Penalty
            dp_thermal = (mdot_target_kg_s**2 / a_throat_guess**2) * ((1.0 / rho_static_2) - (1.0 / rho_static_1))


        # Bay total pressure with losses
        pt_bay = pt_1 - dp_friction - dp_thermal
       

        # Node 2-B (Exit, Drop Model via Discharge Coefficient)
        Tt_exit = T_max_K if T_max_K else Tt_inf

        # Node 2-B (Exit, Drop Model via Discharge Coefficient)
        #_ Isentropic expansion from degraded bay total pressure to external static pressure
        if pt_bay > p_static_ext_exit:
            rho_t_bay = pt_bay / (R_gas * Tt_exit)
            rho_exit = rho_t_bay * (p_static_ext_exit / pt_bay)**(1.0 / gamma)
        else:
            # Fallback protection for unphysical intermediate solver steps
            rho_exit = p_static_ext_exit / (R_gas * Tt_exit)
            
        v_exit_nominal = mdot_target_kg_s / (rho_exit * a_exit)
        R_vel = v_exit_nominal / v_inf
        
        # Nozzle effective discharge area
        Cd = get_outlet_cd(outlet_type, R_vel)
        a_effective_exit = Cd * a_exit

        # Use the true corrected exit density for the dynamic backpressure delta P
        dp_outlet = (mdot_target_kg_s**2) / (2.0 * rho_exit * (a_effective_exit)**2)


        # THE CONVERGENCE RESIDUAL:
        # Energy balance requires that available bay pressure minus outlet drop matches the target exit plane state
        # Hard Physical Constraint: Available pressure must drive the flow out to ambient
        error_pressure = ((pt_bay - dp_outlet - p_static_ext_exit) / p_static_ext_exit)**2.

        error_naca_eff = ((eta_d - 0.85)/0.85)**2.

        return error_pressure + error_naca_eff


    # Solve for required area
    try:
        mfr_bounds = Bounds( 0.0001, 0.9999 )

        res = minimize( area_residual, x0=[0.5], method='Powell', bounds=mfr_bounds)
        final_mfr = res.x[0]
        final_area = mdot_target_kg_s / (rho_inf * v_inf * final_mfr)
        
        return {
            "status": "Success",
            "mdot": mdot_target_kg_s,
            "a_throat_cm2": final_area * 10000.0,
            "mfr": final_mfr,
        }
    except Exception:
        return {
            "status": "Failed",
            "reason": "Solver execution failed. Ram air pressure cannot drive this mass flow through chosen outlet restriction."
        }




if __name__ == "__main__":

    if_Solve_Ventilation = True
    if_Solve_Cooling = False

    # Environment & Flight parameters
    altitude_ft = 10000.
    dISA_K = 15.
    Mach = 0.25
    
    p_inf, T_inf, rho_inf, _ = atmo(altitude_ft, dISA_K)


    # SYSTEM PARAMETERS
    # Compartment Parameters
    BAY_VOLUME_M3 = 2.4
    # Part 25.1187 target criteria
    TARGET_ACPM = 5.0

    # total heat rejected by systems into the bay
    Q_BAY_LOAD_W = 1681.
    # Systems are rated up to this temperature
    T_SYSTEM_MAX_degC = 32.0

    # Outlet exit pressure
    Cp_exit = -0.1
    outlet_to_test = "OutletParallelRamp"


    # massflow rate demands
    if if_Solve_Ventilation:
        vol_flow_rate_rps = (TARGET_ACPM / 60.0) * BAY_VOLUME_M3
        mdot_target = rho_inf * vol_flow_rate_rps

        res = size_ventilation(mdot_target, T_inf, p_inf, Mach, Cp_exit, outlet_to_test)

        if res["status"] == "Success":
            print(f"  Target Mass Flow : {res['mdot']:.4f} kg/s")
            print(f"  NACA Throat Area : {res['a_throat_cm2']:.2f} cm²")
            print(f"  Operating MFR    : {res['mfr']:.3f}")
        else:
            print(f"  Sizing Failed: {res.get('reason')}")


    if if_Solve_Cooling:
        T_max = T_SYSTEM_MAX_degC + 273.15
        Tt_inf = T_inf * (1.0 + gamm2 * Mach**2)

        dT_allowed = T_max - Tt_inf
        mdot_target = Q_BAY_LOAD_W / (cp_air * dT_allowed)

        res = size_ventilation(mdot_target, T_inf, p_inf, Mach, Cp_exit, outlet_to_test, T_max)
        
        if res["status"] == "Success":
            print(f"  Target Mass Flow : {res['mdot']:.4f} kg/s")
            print(f"  NACA Throat Area : {res['a_throat_cm2']:.2f} cm²")
            print(f"  Operating MFR    : {res['mfr']:.3f}")
        else:
            print(f"  Sizing Failed: {res.get('reason')}")
