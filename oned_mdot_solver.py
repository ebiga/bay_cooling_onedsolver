import math

from scipy.optimize import minimize, Bounds

from auxfunctions import *
from InletOutletModels import *


def size_ventilation(mdot_target_kg_s, T_max_K=None):
    """
    Sizes the required NACA inlet throat area to satisfy a target mfr and systems requirements.
    Calculates ram and spillage drag inline to support drag-targeted optimization routines.
    """

    # Get atmospheric conditions
    p_inf_Pa, T_inf_K, rho_inf, mu = atmo(inputs.altitude_ft, inputs.dISA_K)


    # Freestream Aerodynamics & Stagnation States
    a_inf = math.sqrt(gamma * R_gas * T_inf_K)
    v_inf = inputs.Mach * a_inf
    rho_inf = p_inf_Pa / (R_gas * T_inf_K)

    pt_inf = p_inf_Pa * (1.0 + gamm2 * inputs.Mach**2)**(gamma/gamm1)
    Tt_inf = T_inf_K  * (1.0 + gamm2 * inputs.Mach**2)

    qdin_inf = pt_inf - p_inf_Pa


    # External static pressure at the exit dump location
    p_static_ext_exit = p_inf_Pa + inputs.Cp_exit * qdin_inf

    # Local state tracking dictionary to capture the optimizer's current metrics
    state_tracker = {
        "drag_ram": 0.0,
        "drag_spillage": 0.0,
        "drag_total": 0.0
    }
    
    def area_residual(x):

        mfr = x[0]

        # Calculate target throat area base on current MFR iteration
        a_throat_guess = mdot_target_kg_s / (rho_inf * v_inf * mfr)
        
        a_exit = a_throat_guess * x[1]


        # Dynamic Inlet Total Pressure Recovery
        #_ Boundary layer thickness
        ReM = rho_inf * v_inf / mu
        delta_bl = BoundaryLayerThickness(ReM, inputs.inlet_position_m)
        
        # Geometrical throat depth step for scaling calculation
        # Assuming a rectangular aspect ratio width/height profile from your design rules
        eta_d, Cd_spill = naca_pressure_recovery(mfr, delta=delta_bl, area=a_throat_guess)

        pt_1 = p_inf_Pa + eta_d * qdin_inf
        Tt_1 = Tt_inf  # ASSUMPTION: no total temp losses at the inlet


        # Node 1 (Throat State)
        m_1 = solve_throat_mach(mdot_target_kg_s, a_throat_guess, pt_1, Tt_1)
        t_static_1 = Tt_1 / (1.0 + gamm2 * m_1**2)
        p_static_1 = pt_1 / (1.0 + gamm2 * m_1**2)**(gamma/gamm1)
        rho_static_1 = p_static_1 / (R_gas * t_static_1)
        v_1 = m_1 * math.sqrt(gamma * R_gas * t_static_1)


        # Losses: Friction
        dp_friction = inputs.K_SYS * (0.5 * rho_static_1 * v_1**2)


        # Node 2-A (Exit, Thermal demand)
        # Assume equipment heat raises total temperature but does not
        # directly impose a total-pressure penalty.
        Tt_bay = T_max_K if T_max_K else Tt_inf

        # Bay total pressure with losses
        pt_bay = pt_1 - dp_friction
       

        # Node 2-B (Exit, Drop Model via Discharge Coefficient)
        Tt_exit = Tt_bay

        #_ Isentropic expansion from degraded bay total pressure to external static pressure
        rho_t_bay = max(pt_bay, p_static_ext_exit) / (R_gas * Tt_exit)
        rho_exit = rho_t_bay * (p_static_ext_exit / pt_bay)**(1.0 / gamma)
            
        v_exit_nominal = mdot_target_kg_s / (rho_exit * a_exit)

        # MOMENTUM FLUX RATIO (J) & GEOMETRIC BOUNDARY LAYER SCALING
        J = (rho_exit * v_exit_nominal**2) / (rho_inf * v_inf**2 * (1.0 - inputs.Cp_exit))

        # Nozzle effective discharge area
        delta_bl = BoundaryLayerThickness(ReM, inputs.outlet_position_m)

        Cd = get_outlet_cd(inputs.outlet_type, J, delta=delta_bl, a_exit=a_exit)
        a_effective_exit = Cd * a_exit

        # Use the true corrected exit density for the dynamic backpressure delta P
        dp_outlet = (mdot_target_kg_s**2) / (2.0 * rho_exit * (a_effective_exit)**2)


        # DRAG
        # 1. Ram Drag
        drag_ram = mdot_target_kg_s * (v_inf - v_exit_nominal)
        
        # 2. Spillage Drag
        drag_spillage = Cd_spill * qdin_inf * a_throat_guess
        
        drag_total = drag_ram + drag_spillage

        # Push to outer scope state tracking dictionary
        state_tracker["drag_ram"] = drag_ram
        state_tracker["drag_spillage"] = drag_spillage
        state_tracker["drag_total"] = drag_total


        # THE CONVERGENCE RESIDUAL:
        # Energy balance requires that available bay pressure minus outlet drop matches the target exit plane state
        # Hard Physical Constraint: Available pressure must drive the flow out to ambient
        error_pressure = abs((pt_bay - dp_outlet - p_static_ext_exit) / p_static_ext_exit)

        print(f"mfr: {mfr:.3e}, pressure err: {error_pressure:.3e}, naca eff: {eta_d:.3e}, outlet eff: {Cd:.3f}, drag: {drag_total:.2f}")

        return error_pressure


    # Solve for required area
    try:
        mfr_bounds = ( 0.1, 1. )
        aexit_bounds = ( 0.5, 10. )

        res = minimize( area_residual, x0=[0.5, 1.], method='Powell', bounds=[mfr_bounds, aexit_bounds])

        # Force a final evaluation at the exact converged minimum to update state_tracker
        _ = area_residual(res.x)

        final_mfr = res.x[0]

        inlet__area = mdot_target_kg_s / (rho_inf * v_inf * final_mfr)
        outlet_area = res.x[1] * inlet__area
        
        return {
            "status": "Success",
            "mdot": mdot_target_kg_s,
            "inlet__area_cm2": inlet__area * 10000.0,
            "outlet_area_cm2": outlet_area * 10000.0,
            "mfr": final_mfr,
            "drag_ram_N": state_tracker["drag_ram"],
            "drag_spillage_N": state_tracker["drag_spillage"],
            "drag_total_N": state_tracker["drag_total"],
        }
    except Exception:
        return {
            "status": "Failed",
            "reason": "Solver execution failed. Ram air pressure cannot drive this mass flow through chosen outlet restriction."
        }




def run_case():

    # Get atmospheric data
    _, T_inf, rho_inf, _ = atmo(inputs.altitude_ft, inputs.dISA_K)


    # Compute the required mass flow rate
    mdot_target_acpm = mdot_target_thermal = False

    # Required MFR for: Ventilation
    if inputs.TARGET_ACPM:
        T_max_K = (inputs.T_SYSTEM_MAX_degC + 273.15) if inputs.T_SYSTEM_MAX_degC else None

        vol_flow_rate_rps = (inputs.TARGET_ACPM / 60.0) * inputs.BAY_VOLUME_M3
        mdot_target_acpm = rho_inf * vol_flow_rate_rps

        print(f"  Mass Flow to vent: {mdot_target_acpm:.4f} kg/s")

    # Required MFR for: Cooling
    if inputs.Q_BAY_LOAD_W:
        try:
            T_max_K = inputs.T_SYSTEM_MAX_degC + 273.15
        except Exception:
            return {
                "status": "Failed",
                "reason": "Q_BAY_LOAD_W also requires T_SYSTEM_MAX_degC to be defined."
            }

        Tt_inf = T_inf * (1.0 + gamm2 * inputs.Mach**2)
        dT_allowed = T_max_K - Tt_inf
        mdot_target_thermal = inputs.Q_BAY_LOAD_W / (cp_air * dT_allowed)

        print(f"  Mass Flow to cool: {mdot_target_thermal:.4f} kg/s")

    # The target massflow rate is the highest
    try:
        mdot_target = max(mdot_target_acpm, mdot_target_thermal)
    except Exception:
        return {
            "status": "Failed",
            "reason": "No TARGET_ACPM and/or Q_BAY_LOAD_W defined."
        }


    # Find the appropriate inlet and outlet areas.
    res = size_ventilation(mdot_target_kg_s=mdot_target, T_max_K=T_max_K)

    return res


if __name__ == "__main__":

    res = run_case()

    if res["status"] == "Success":
        print(f"  Target Mass Flow : {res['mdot']:.4f} kg/s")
        print(f"  Throat Area, Inlet  : {res['inlet__area_cm2']:.2f} cm²")
        print(f"  Throat Area, Outlet : {res['outlet_area_cm2']:.2f} cm²")
        print(f"  Operating MFR    : {res['mfr']:.3f}")
        print(f"  Ram Drag         : {res['drag_ram_N']:.2f} N")
        print(f"  Spillage Drag    : {res['drag_spillage_N']:.2f} N")
        print(f"  Total Drag       : {res['drag_total_N']:.2f} N")
    else:
        print(f"  Sizing Failed: {res.get('reason')}")
