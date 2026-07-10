import math
import numpy as np

import inputs



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



def get_outlet_cd(outlet_type, J, delta=0, a_exit=None, aspect_r=4, porosity=0.6):
    """
    The discharge coefficient is defined on exit area correction:
        Area_actual = Area_ideal * Cd
    Returns the dynamic discharge coefficient adjusted for external crossflow
    momentum flux ratio (J) and incoming boundary layer immersion (delta).

    The boundary layer physics are modeled via analytical integration of a 
    turbulent 1/7th power-law velocity profile over the nozzle height.
    
    Parameters:
    -----------
    outlet_type : str   -> "OutletInvertedScoop", "OutletParallelRamp", etc.
    J           : float -> Momentum flux ratio ((rho_exit * V_exit^2) / (rho_inf * V_inf^2))
    delta       : float -> Local boundary layer thickness at the exit plane [m]
    a_exit      : float -> Area of the exit nozzle opening [m2]
    aspect_r    : float -> Outlet throat aspect ratio (typically in the range 3:1 to 5:1)
    porosity    : float -> Open area ratio (only applied to the OutletGrill branch)
    """
        
    # Extract exit plane height to calculate non-dimensional boundary layer thickness
    h_exit = math.sqrt(a_exit / aspect_r)

    sqrt_J = math.sqrt(J)
    
    # Calculate non-dimensional boundary layer immersion parameter
    bar_delta = delta / max(1e-5, h_exit)
    
    # 1. Analytically integrate 1/7th power law profile over the nozzle opening height
    # Yields the effective crossflow velocity reduction factor (f_v = V_local,eff / V_local)
    if bar_delta >= 1.0:
        f_v = 0.875 * (bar_delta ** (-1.0 / 7.0))
    elif bar_delta > 0.0:
        f_v = 1.0 - 0.125 * bar_delta
    else:
        f_v = 1.0
        
    # 2. Scale the crossflow momentum flux to find the true effective J seen by the jet
    # Hiding inside a thick boundary layer reduces crossflow momentum, increasing J_eff
    J_eff = J / max(1e-5, f_v * f_v)
    sqrt_J_eff = math.sqrt(J_eff)
    
    if outlet_type == "OutletInvertedScoop":
        # Aft-Facing Extractor Scoop (Hoerner Drag, Ch. 12 & NACA ACR 5I20)
        cd_max = 0.80 - 0.08 * min(0.5, bar_delta)
        return cd_max - 0.05 * math.exp(-sqrt_J)
        
    elif outlet_type == "OutletParallelRamp":
        # Parallel Flush Slot (ESDU 86002 / NACA TN 3924)
        return 0.62 * (1.0 - 0.60 * math.exp(-2.0 * sqrt_J_eff))
        
    elif outlet_type == "OutletDivergentRamp":
        # Flush Divergent Ramp Outlet (ESDU 86002 / NACA TN 3924)
        return 0.70 * (1.0 - 0.40 * math.exp(-2.5 * sqrt_J_eff))
        
    elif outlet_type == "OutletGrill":
        # Flush Grill (Gritsch / Dittrich & Graves)
        cd_baseline = 0.62 * porosity
        vr_eff = 1.0 / max(1e-3, sqrt_J_eff)
        return cd_baseline / math.sqrt(1.0 + 1.1 * vr_eff**2)        

    else:
        raise ValueError(f"Unknown outlet type: {outlet_type}")
