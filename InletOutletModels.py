import math


def naca_pressure_recovery(mfr):
    """
    Empirical fit for standard NACA submerged flush inlet pressure recovery.
    Based on the classics, NACA RM A7I30 / NACA ACR 5120
    eta is defined based on dynamic pressure recovery:
        Ptot_recovered = Ps + eta*(Ptot_inf - Ps_inf)
    """

    mfr = max(0., min(mfr, 1.))

    eta = -1.1 * (mfr - 0.65)**2 + 0.85
    eta = min(max(0.1, eta), 1.)

    K_spill = 0.4
    Cd_spill = K_spill * (1.0 - mfr)**2

    return eta, Cd_spill


def get_outlet_cd(outlet_type, R_vel, porosity=0.6):
    """
    Returns the dynamic discharge coefficient adjusted for external crossflow.
    The discharge coefficient is defined on exit area correction:
        Area_actual = Area_ideal * Cd
    dependent on the exit velocity ratio
        R = V_exit / V_inf
    """
    R_vel = max(0.001, R_vel)
    
    if outlet_type == "OutletInvertedScoop":
        # Reference C: Hoerner Aft-Facing Extractor Scoop
        return 0.80 - 0.05 * math.exp(-R_vel)
    elif outlet_type == "OutletParallelRamp":
        # Reference A: NACA TN 3924 Parallel Flush Slot
        return 0.60 * (1.0 - 0.60 * math.exp(-1.8 * R_vel))
    elif outlet_type == "OutletDivergentRamp":
        # Reference B: ESDU 86001 Flush Divergent Ramp Outlet
        return 0.70 * (1.0 - 0.45 * math.exp(-2.2 * R_vel))
    elif outlet_type == "OutletGrill":
        # Reference D: Idelchik Perforated Screen Interaction
        return 0.62 * porosity * (1.0 - 0.55 * math.exp(-1.5 * R_vel))
    else:
        raise ValueError(f"Unknown outlet type: {outlet_type}")
