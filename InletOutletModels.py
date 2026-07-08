import math



def naca_pressure_recovery(mfr, delta=None, area=None, C_vortex=2.0, aspect_r=4):
    """
    Empirical fit for standard NACA submerged flush inlet pressure recovery.
    Based on the classics, NACA RM A7I30 / NACA ACR 5120
    eta is defined based on dynamic pressure recovery:
        Ptot_recovered = Ps + eta*(Ptot_inf - Ps_inf)

    Extended NACA submerged flush inlet pressure recovery incorporating the 
    ESDU-based boundary layer thickness penalty model.
    
    Parameters:
    mfr      : Mass flow ratio (V_throat / V_inf)
    delta    : Incoming boundary layer thickness [m]
    area     : Inlet throat area [m2]
    C_vortex : Vortex scavenging efficiency (typically 1.8 to 2.2)
    aspect_r : Inlet throat aspect ration (in the range 3:1 to 5:1)

    h_i      : Inlet throat height [m]
    """

    # Clip for safety
    mfr = max(1e-4, min(mfr, 1.))

    # Clean baseline recovery profile
    eta = -1.1 * (mfr - 0.65)**2 + 0.85
    eta = min(max(0.1, eta), 1.)

    # Kinematic spillage
    K_spill = 0.4
    Cd_spill = K_spill * (1.0 - mfr)**2


    if delta:
        # Turbulent 1/7th power-law boundary layer momentum thickness estimation
        theta = 0.097 * delta

        # ESDU exponential degradation tracking
        h_i = math.sqrt(area / aspect_r)
        loss_exponent = -C_vortex * (theta / (h_i * math.sqrt(mfr)))
        eta = eta * math.exp(loss_exponent)


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
