import math

from scipy.optimize import minimize, Bounds

from auxfunctions import *
from InletOutletModels import *


def size_ventilation(mdot_target_kg_s, T_max_K=None, dPtot_driven_Pa=None):
    """
    Sizes the required NACA inlet throat area to satisfy a target mfr and systems requirements.
    Calculates ram and spillage drag inline to support drag-targeted optimization routines.
    """

    # Get atmospheric conditions
    p_inf_Pa, T_inf_K, rho_inf, mu = atmo(inputs.altitude_ft, inputs.dISA_K)


    # Freestream Aerodynamics & Stagnation States
    a_inf = ComputeSpeedOfSoundFromTemperature(T_inf_K)
    v_inf = inputs.Mach * a_inf
    rho_inf = ComputeConstitutiveDensity(p_inf_Pa, T_inf_K)

    pt_inf = p_inf_Pa * isentM(inputs.Mach, 'pressure')
    Tt_inf = T_inf_K  * isentM(inputs.Mach, 'temperature')

    qdin_inf = 0.5 * rho_inf * v_inf**2


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
        #_ For now assuming the local Mach as freestream Mach
        eta_d, Cd_spill = naca_pressure_recovery(mfr=mfr, Machinf=inputs.Mach, delta=delta_bl, area=a_throat_guess)

        pt_1 = p_inf_Pa + eta_d * qdin_inf
        Tt_1 = Tt_inf  # ASSUMPTION: no total temp losses at the inlet


        # =========================================================================
        # CASCADING INTERNAL LOSS LOOP WITH DENSITY AND VELOCITY TRACKING
        # =========================================================================
        pt_current = pt_1
        Tt_current = Tt_1
        section_area = a_throat_guess

        for element in inputs.layout:
            elem_type = element["type"]

            # 1. Get the element section area
            area_elem, dh = ElementArea(element)

            if not area_elem:
                area_elem = section_area

            # 2. Solve static properties entering this specific element
            M_local, t_local, p_local, rho_local, v_local, mu_local = solve_local_states(
                mdot=mdot_target_kg_s,
                area=area_elem,
                pt=pt_current,
                Tt=Tt_current
            )

            # 3. Compute delta-P and delta-T across this element
            #_ First reset for each new elements
            dp_elem = 0.0
            dTt_elem = 0.0

            #_ Now deal with the element
            if elem_type == "pipe":
                # Frictional duct loss
                _, dp_elem = straight_duct_loss(
                    mdot=mdot_target_kg_s,
                    rho=rho_local,
                    mu=mu_local,
                    length=element["length"],
                    area=area_elem,
                    diam_hydro=dh
                )


            elif elem_type == "bend":
                # Centrifugal and wall friction bend loss
                _, dp_elem = bend_loss(
                    mdot=mdot_target_kg_s,
                    rho=rho_local,
                    mu=mu_local,
                    r_centerline=element["r_centerline"],
                    area=area_elem,
                    diam_hydro=dh
                )


            elif elem_type == "VentingBay":
                # Local dump loss (sudden expansion) based on entering dynamic pressure
                q_local = 0.5 * rho_local * v_local**2
                dp_elem = element["KL"] * q_local


            elif elem_type == "CoolingBay":
                # Thermal load updates total temperature
                if T_max_K is not None:
                    dTt_elem = T_max_K - Tt_current


            elif elem_type == "FanCooler":
                # Active pressure deficit/loss across the unit
                dp_elem = element["TotalPressureDrop_Pa"]


            # 4. Cascade the total states forward to the next element
            pt_current -= dp_elem
            Tt_current += dTt_elem
            section_area = area_elem


        # =========================================================================
        # EXIT: Nozzle Plane Hand-off
        # =========================================================================
        # The final total states leaving the very last element in your layout
        pt_bay = pt_current
        Tt_exit = Tt_current

        #_ Isentropic expansion from degraded bay total pressure to external static pressure
        rho_t_bay = ComputeConstitutiveDensity(pt_bay, Tt_exit)
        rho_exit = rho_t_bay * (p_static_ext_exit / pt_bay)**(1.0 / gamma)
            
        v_exit_nominal = mdot_target_kg_s / (rho_exit * a_exit)

        # MASS FLUX RATIO (J), corrected by simplified exit conditions
        # Local External Flow State at Exit
        Mach_local = math.sqrt(5.0 * ((1.0 + gamm2 * inputs.Mach**2) / (1.0 + 0.5 * gamma * inputs.Mach**2 * inputs.Cp_exit)**(gamm1/gamma) - 1.0))

        T_local = Tt_inf * (p_static_ext_exit / pt_inf)**(gamm1 / gamma)
        rho_local = ComputeConstitutiveDensity(p_static_ext_exit, T_local)

        v_local = Mach_local * ComputeSpeedOfSoundFromTemperature(T_local)

        # Finally the mass flow ratio
        J = (rho_exit * v_exit_nominal) / (rho_local * v_local)

        # Nozzle effective discharge area
        Cdischarge, CD_base = get_outlet_cd(outlet_type=inputs.outlet_type, Vrel=J, Mach_e=Mach_local)
        a_effective_exit = Cdischarge * a_exit

        # Use the true corrected exit density for the dynamic backpressure delta P
        dp_outlet = (mdot_target_kg_s**2) / (2.0 * rho_exit * (a_effective_exit)**2)


        # DRAG
        # 1. Ram Drag
        drag_ram = mdot_target_kg_s * (v_inf - v_exit_nominal)
        
        # 2. Spillage Drag
        drag_spillage = Cd_spill * qdin_inf * a_throat_guess

        # 3. Discharge drag
        drag_base = CD_base * qdin_inf * a_exit

        # Total        
        drag_total = drag_ram + drag_spillage + drag_base


        # THE CONVERGENCE RESIDUAL:
        # Energy balance requires that available bay pressure minus outlet drop matches the target exit plane state
        # Hard Physical Constraint: Available pressure must drive the flow out to ambient
        error_pressure = abs((pt_bay - dp_outlet - p_static_ext_exit) / p_static_ext_exit)

        print(f"mfr: {mfr:.3e}, pressure err: {error_pressure:.3e}, naca eff: {eta_d:.3e}, outlet eff: {Cdischarge:.3f}, drag: {drag_total:.2f}")

        # Cache results for optimizer evaluation reads
        state_tracker["drag_ram"] = drag_ram
        state_tracker["drag_spillage"] = drag_spillage
        state_tracker["drag_total"] = drag_total
        state_tracker["error_pressure"] = error_pressure
        state_tracker["inlet_area"] = a_throat_guess
        state_tracker["outlet_area"] = a_exit
        state_tracker["Cdischarge"] = Cdischarge
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
        aexit_bounds = ( 0.2, 4. )

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
            "outlet_cd": state_tracker["Cdischarge"],
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
    # The target massflow rate is the highest
    mdot_target = 0.


    # LOOP THE INPUT AND DETERMINE THE CASES
    for item in inputs.layout:

        # Required MFR for: Ventilation
        if item.get("type") == "VentingBay":
            BAY_VOLUME_M3 = item["BAY_VOLUME_M3"]
            TARGET_ACPM   = item["TARGET_ACPM"]
    
            T_max_K = None
            dPtot_driven_Pa = None

            vol_flow_rate_rps = (TARGET_ACPM / 60.0) * BAY_VOLUME_M3
            mdot_target_acpm = rho_inf * vol_flow_rate_rps

            mdot_target = max(mdot_target, mdot_target_acpm)

            print(f"  Mass Flow to vent: {mdot_target_acpm:.4f} kg/s")

        # Required MFR for: Cooling
        if item.get("type") == "CoolingBay":
            T_SYSTEM_MAX_degC = item["T_SYSTEM_MAX_degC"]
            Q_BAY_LOAD_W      = item["Q_BAY_LOAD_W"]

            T_max_K = T_SYSTEM_MAX_degC + 273.15
            dPtot_driven_Pa = None

            Tt_inf = T_inf * (1.0 + gamm2 * inputs.Mach**2)
            dT_allowed = T_max_K - Tt_inf
            mdot_target_thermal = Q_BAY_LOAD_W / (cp_air * dT_allowed)

            mdot_target = max(mdot_target, mdot_target_thermal)

            print(f"  Mass Flow to cool: {mdot_target_thermal:.4f} kg/s")

        # Required MFR for: Fan Blower
        if item.get("type") == "FanCooler":
            mdot_target_fan = item["MassFlowRate_kg_s"]
            dPtot_driven_Pa = item["TotalPressureDrop_Pa"]

            T_max_K = None

            mdot_target = max(mdot_target, mdot_target_fan)

            print(f"  Mass Flow to fan: {mdot_target_fan:.4f} kg/s")

    # Find the appropriate inlet and outlet areas.
    res = size_ventilation(mdot_target_kg_s=mdot_target, T_max_K=T_max_K, dPtot_driven_Pa=dPtot_driven_Pa)

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
