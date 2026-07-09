import math
import numpy as np



def naca_pressure_recovery(mfr, delta=0, area=None, C_vortex=2.0, aspect_r=4):
    """
    Empirical fit for standard NACA submerged flush inlet pressure recovery.
    Based on the classics, NACA RM A7I30 / NACA ACR 5120
    eta is defined based on dynamic pressure recovery:
        Ptot_recovered = Ps + eta*(Ptot_inf - Ps_inf)

    Extended NACA submerged flush inlet pressure recovery incorporating the 
    ESDU 86002 boundary layer thickness penalty model.
    
    Parameters:
    mfr      : Mass flow ratio (V_throat / V_inf)
    delta    : Incoming boundary layer thickness [m]
    area     : Inlet throat area [m2]
    C_vortex : Vortex scavenging efficiency (typically 1.8 to 2.2)
    aspect_r : Inlet throat aspect ratio (in the range 3:1 to 5:1)

    h_i      : Inlet throat height [m]
    """

    # Clip for safety
    mfr = max(1e-4, min(mfr, 1.))

    # Kinematic spillage
    K_spill = 0.4
    Cd_spill = K_spill * (1.0 - mfr)**2


    if delta > 0:
        if area is None:
            raise ValueError("Area must be provided to calculate throat height (h_i) when delta > 0.")
        
        # Turbulent 1/7th power-law boundary layer momentum thickness estimation
        theta = 0.097 * delta
        h_i = math.sqrt(area / aspect_r)
        theta_over_hi = theta / max(1e-5, h_i)
    else:
        # Pure clean air baseline condition
        theta_over_hi = 0.0


    # Scaling factor for the entry lip (d_1_fl / d_t) for a standard 7-degree ramp
    lip_factor = 1.1602  


    # ESDU 86002 TABLE LOOKUPS
    
    # Figure 17: Maximum Ram Pressure Efficiency (eta_m) vs theta/d_t
    fig17_theta_dt = [0.00, 0.05, 0.10, 0.163, 0.25, 0.35, 0.50]
    fig17_eta_m    = [0.85, 0.78, 0.73, 0.688, 0.62, 0.56, 0.50]
    eta_m = np.interp(theta_over_hi, fig17_theta_dt, fig17_eta_m)

    # Figure 18: Optimal Modified Mass Flow Ratio (mu_m) vs theta/d_t
    fig18_theta_dt = [0.00, 0.05, 0.10, 0.163, 0.25, 0.35, 0.50]
    fig18_mu_m     = [0.45, 0.41, 0.38, 0.345, 0.31, 0.28, 0.25]
    mu_m = np.interp(theta_over_hi, fig18_theta_dt, fig18_mu_m)

    # 3. Calculate Off-Design Mass Flow Parameter (mu - mu_m)
    mu = mfr * lip_factor
    delta_mu = mu - mu_m

    # Figure 19a: Off-Design Mass Flow Correction Delta (Delta_eta_mf) vs (mu - mu_m)
    fig19a_delta_mu  = [-0.20, 0.00, 0.070, 0.232, 0.420, 0.589, 0.80]
    fig19a_delta_eta = [-0.01, 0.00, -0.006, -0.035, -0.050, 0.053, 0.02]
    delta_eta_mf = np.interp(delta_mu, fig19a_delta_mu, fig19a_delta_eta)

    # Optional tuning parameter hook: C_vortex can scale the BL sensitivity slightly 
    # if you want to shift the baseline curve up or down. Default is 2.0 (neutral).
    vortex_scaling = (2.0 / C_vortex) if C_vortex > 0 else 1.0


    # TOTAL RECOVERY BOOKKEEPING
    eta = eta_m + (delta_eta_mf * vortex_scaling)
    eta = min(max(0.05, eta), 1.0)


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
