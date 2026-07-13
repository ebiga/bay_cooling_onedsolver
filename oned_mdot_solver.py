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
    state_tracker = {}
    
    def evaluate_system_physics(x):
        """
        Executes the 1D aerothermal pipeline for a given geometry state.
        """

        mfr = x[0]
        area_ratio = x[1]

        # Calculate target throat area base on current MFR iteration
        a_throat_guess = mdot_target_kg_s / (rho_inf * v_inf * mfr)
        a_exit = a_throat_guess * area_ratio


        # Dynamic Inlet Total Pressure Recovery
        #_ Boundary layer thickness
        ReM = rho_inf * v_inf / mu
        delta_bl = BoundaryLayerThickness(ReM, inputs.inlet_position_m)
        
        # Geometrical throat depth step for scaling calculation
        # Assuming a rectangular aspect ratio width/height profile from your design rules
        eta_d, Cd_spill = naca_pressure_recovery(mfr, delta=delta_bl, area=a_throat_guess)

        pt_1 = p_inf_Pa + eta_d * qdin_inf
        Tt_1 = Tt_inf  # ASSUMPTION: no total temp losses at the inlet


        # =========================================================================
        # Node 1 (Throat State)
        # =========================================================================
        m_1 = solve_throat_mach(mdot_target_kg_s, a_throat_guess, pt_1, Tt_1)
        t_static_1 = Tt_1 / (1.0 + gamm2 * m_1**2)
        p_static_1 = pt_1 / (1.0 + gamm2 * m_1**2)**(gamma/gamm1)
        rho_static_1 = p_static_1 / (R_gas * t_static_1)
        v_1 = m_1 * math.sqrt(gamma * R_gas * t_static_1)

        mu_static_1 = ViscositySutherland(t_static_1)

        # INTERNAL DUCTING TOTAL PRESSURE DROP ACCUMULATOR
        dp_internal_total = 0.0
        Tt_bay = Tt_inf

        # Ensure your naming matches your input object attribute (e.g., inputs.layout)
        for element in inputs.layout:
            elem_type = element["type"]

            if elem_type == "pipe":
                # Call straight duct loss function (Returns: k, dp, area)
                _, dp_elem, _ = straight_duct_loss(
                    mdot=mdot_target_kg_s, 
                    rho=rho_static_1, 
                    mu=mu_static_1, 
                    length=element["length"], 
                    width=element["width"], 
                    height=element.get("height")
                )
                dp_internal_total += dp_elem


            elif elem_type == "bend":
                # Call curved bend loss function (Fixed: removed mu, fixed unpacking)
                _, dp_elem, _ = bend_loss(
                    mdot=mdot_target_kg_s,
                    rho=rho_static_1,
                    r_centerline=element["r_centerline"],
                    width=element["width"],
                    height=element.get("height")
                )
                dp_internal_total += dp_elem


            elif elem_type == "VentingBay":
                # Just a bulk K loss model
                dp_elem = element["KL"] * 0.5 * rho_static_1 * v_1**2
                dp_internal_total += dp_elem


            elif elem_type == "CoolingBay":
                # Assuming plenum losses came from other pimping elements
                dp_elem = 0.
                dp_internal_total += dp_elem

                # Assume equipment heat raises total temperature but does not
                # directly impose an additional total-pressure penalty.
                Tt_bay = T_max_K

            else:
                raise ValueError(f"Unknown ducting element type encountered: {elem_type}")


        # =========================================================================
        # Node 2-A (Exit, Thermal demand)
        # =========================================================================
        # Bay total pressure with losses
        pt_bay = pt_1 - dp_internal_total       

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

        # Cache results for optimizer evaluation reads
        state_tracker["drag_ram"] = drag_ram
        state_tracker["drag_spillage"] = drag_spillage
        state_tracker["drag_total"] = drag_total
        state_tracker["error_pressure"] = error_pressure
        state_tracker["inlet_area"] = a_throat_guess
        state_tracker["outlet_area"] = a_exit
        state_tracker["Cd"] = Cd
        state_tracker["eta_d"] = eta_d

        return

    # --- Optimizer Sub-Functions for SLSQP ---
    def obj_drag(x):
        evaluate_system_physics(x)
        return state_tracker["drag_total"]

    def const_pressure(x):
        evaluate_system_physics(x)
        return state_tracker["error_pressure"]



    # Solve for required area
    try:
        mfr_bounds = ( 0.1, 1. )
        aexit_bounds = ( 0.5, 10. )

        # Define equality constraint: pressure residual must equal zero
        constraints = {"type": "eq", "fun": const_pressure}

        res = minimize(
            obj_drag, 
            x0=[0.5, 1.0], 
            method="SLSQP", 
            bounds=[mfr_bounds, aexit_bounds],
            constraints=constraints,
            options={"ftol": 1e-8}
        )

        if not res.success:
            raise RuntimeError("SLSQP failed to converge constraints cleanly.")

        # Ensure track states match the final optimized vector exactly
        evaluate_system_physics(res.x)        

        return {
            "status": "Success",
            "mdot": mdot_target_kg_s,
            "inlet__area_cm2": state_tracker["inlet_area"]  * 10000.0,
            "outlet_area_cm2": state_tracker["outlet_area"] * 10000.0,
            "mfr": res.x[0],
            "outlet_cd": state_tracker["Cd"],
            "inlet_eta": state_tracker["eta_d"],
            "drag_ram_N": state_tracker["drag_ram"],
            "drag_spillage_N": state_tracker["drag_spillage"],
            "drag_total_N": state_tracker["drag_total"],        }

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


    # LOOP THE INPUT AND DETERMINE THE CASES
    for item in inputs.layout:

        # Required MFR for: Ventilation
        if item.get("type") == "VentingBay":
            BAY_VOLUME_M3 = item["BAY_VOLUME_M3"]
            TARGET_ACPM   = item["TARGET_ACPM"]
    
            T_max_K = None

            vol_flow_rate_rps = (TARGET_ACPM / 60.0) * BAY_VOLUME_M3
            mdot_target_acpm = rho_inf * vol_flow_rate_rps

            print(f"  Mass Flow to vent: {mdot_target_acpm:.4f} kg/s")

        # Required MFR for: Cooling
        if item.get("type") == "CoolingBay":
            T_SYSTEM_MAX_degC = item["T_SYSTEM_MAX_degC"]
            Q_BAY_LOAD_W      = item["Q_BAY_LOAD_W"]

            T_max_K = T_SYSTEM_MAX_degC + 273.15

            Tt_inf = T_inf * (1.0 + gamm2 * inputs.Mach**2)
            dT_allowed = T_max_K - Tt_inf
            mdot_target_thermal = Q_BAY_LOAD_W / (cp_air * dT_allowed)

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
