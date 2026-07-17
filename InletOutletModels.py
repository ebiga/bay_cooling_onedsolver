import math
import numpy as np

import inputs



def naca_pressure_recovery(mfr, Machinf, delta=0, area=None, aspect_r=4):
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
    aspect_r : Inlet throat aspect ratio (in the range 3:1 to 5:1)

    h_i      : Inlet throat height [m]
    """

    # Clip for safety
    mfr = max(1e-4, min(mfr, 1.))


    # Kinematic spillage
    # Modelled according to AGARD-AG-264 data in Fig. 6.9, incl compressibility.
    m_squared_limit = min(0.95, Machinf**2)
    la_factor = 1.0 - m_squared_limit

    Cd0 = 0.175 / la_factor
    Cdcorr = 0.35 * (1 - (1-mfr)**2.)

    Cd_spill = Cd0 - Cdcorr


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


    # TOTAL RECOVERY BOOKKEEPING
    eta = eta_m + delta_eta_mf
    eta = min(max(0.01, eta), 1.0)


    return eta, Cd_spill



def get_outlet_cd(outlet_type, J, Machinf):
    """
    Determine the dynamic discharge coefficient (Cd) and static base drag properties
    for different outlet configurations under external crossflow.
    Data based on: NACA TN-3466.

    The discharge coefficient remains dynamic with respect to J.

    The base drag coefficient is evaluated at J=0 (mfr=0) to prevent double-counting 
    the thrust/ram-drag terms, corrected for Mach number using AGARD-AG-264.

    The base drag is modeled as:
        Drag_base = Cd_base * q_inf * a_exit

    Parameters:
    -----------
    outlet_type : str   -> "OutletInvertedScoop", "OutletParallelRamp", etc.
    J           : float -> Momentum flux ratio ((rho_exit * V_exit^2) / (rho_inf * V_inf^2))
    a_exit      : float -> Area of the exit nozzle opening [m2]
    aspect_r    : float -> Outlet throat aspect ratio (typically in the range 3:1 to 5:1)
    Machinf     : float -> Freestream Mach number for compressibility corrections [-]

    Returns:
    --------
    cd          : float -> Dynamic discharge coefficient [-]
    Cd_base     : float -> Static base drag coefficient at J=0 (referenced to q_inf) [-]
    """


    # Mass-balanced relative velocity. 
    Vrel = math.sqrt(J)


    # Compressibility correction terms
    m_squared_limit = min(0.95, Machinf**2)
    # Prandtl-Glauert
    pg_factor = math.sqrt(1.0 - m_squared_limit)



    # OUTLET TYPE EVALUATION
    if outlet_type == "OutletInvertedScoop":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: NACA TN-3466, Fig.18, M0.7, flush 3.
        mfr = [0.192, 0.220, 0.248, 0.280, 0.349, 0.434, 0.490, 0.600, 0.689, 0.799, 0.903]
        K__ = [1.398, 1.170, 0.988, 0.831, 0.761, 0.797, 0.812, 0.785, 0.776, 0.782, 0.808]
        cd = np.interp(Vrel, mfr, K__)

        # --- BASE DRAG ---
        # Ref: AGARD-AG-264, Section 6.3.2 ("Scoop Outlets"), Fig. 6.14 df=20 AF=1.
        Cd_base_0 = 0.25
        Cd_base = Cd_base_0 / pg_factor

    elif outlet_type == "OutletParallelRamp":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: NACA TN-3466, Fig.12, M0.7, flush 4.
        mfr = [0., 0.252, 0.359, 0.484, 0.577, 0.708, 0.857]
        K__ = [0., 0.560, 0.669, 0.776, 0.828, 0.878, 0.913]
        cd = np.interp(Vrel, mfr, K__)

        # --- BASE DRAG ---
        # Ref: AGARD-AG-264, Section 6.3.1 ("Flush Outlets"), Fig.6.21.
        Cd_base_0 = 0.12
        Cd_base = Cd_base_0 / pg_factor

    elif outlet_type == "OutletGrill":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: NACA TN-3466, Fig.6, M0.7, AR 6.
        mfr = [0., 0.177, 0.251, 0.396, 0.525, 0.655]
        K__ = [0., 0.346, 0.439, 0.554, 0.642, 0.680]
        cd = np.interp(Vrel, mfr, K__)

        # --- BASE DRAG ---
        # Ref: AGARD-AG-264, Section 6.3.1 ("Flush Outlets").
        Cd_base_0 = 0.01
        Cd_base = Cd_base_0 / pg_factor

    else:
        raise ValueError(f"Unknown outlet type: {outlet_type}")

    return cd, Cd_base



def straight_duct_loss(mdot, rho, mu, length, area, diam_hydro, roughness=0.0015e-3):
    """
    Computes total pressure loss for a straight duct section using the 
    explicit Churchill (1977) friction correlation across all flow regimes.
    Supports both rectangular and circular geometry profiles.
    """

    # Local velocity and dynamic pressure
    v = mdot / (rho * area)
    q = 0.5 * rho * v**2
    
    # Safeguard Reynolds number limits against low or zero flow bounds
    re = max(10.0, (rho * v * diam_hydro) / max(1e-7, mu))
    rel_roughness = roughness / diam_hydro

    # Churchill Correlation Sub-components (eliminates transcendental iteration loops)
    term_a_inner = (7.0 / re)**0.9 + 0.27 * rel_roughness
    log_arg = max(1e-12, term_a_inner)
    A = (-2.457 * math.log(log_arg))**16
    B = (37530.0 / re)**16
    
    f = 8.0 * ((8.0 / re)**12 + 1.0 / ((A + B)**1.5))**(1.0 / 12.0)
    
    k_duct = f * (length / diam_hydro)
    dp_duct = k_duct * q
    
    return k_duct, dp_duct



def bend_loss(mdot, rho, r_centerline, area, diam_hydro):
    """
    Computes total pressure loss across a 90-degree bend based on geometric sharpness.
    Uses a standard empirical curve-fit optimized for clean ducting networks.
    """

    # Local velocity and dynamic pressure
    v = mdot / (rho * area)
    q = 0.5 * rho * v**2
    
    # Enforce minimum curvature ratio to avoid math runtime issues
    r_ratio = max(0.1, r_centerline / diam_hydro)
    
    # Idelchik-aligned bend loss formulation
    k_bend = 0.131 + 0.163 * (1.0 / r_ratio)**3.5
    dp_bend = k_bend * q
    
    return k_bend, dp_bend



def ElementArea(element):
    '''
    Determine local cross-sectional flow area for different elements.

    Input: element from 'layout' list in inputs.py
    Output:     area, in m2
                hydraulic diameter, in m
    '''

    elem_type = element["type"]


    if elem_type in ["pipe", "bend"]:
        width = element["width"]
        height = element.get("height")

        if height:
            # Rectangular cross section
            area_elem = width * height

            perimeter = 2.0 * (width + height)
            diam_hydro = (4.0 * area_elem) / perimeter

        else:
            # Circular cross section
            area_elem = (math.pi * width**2) / 4.0

            diam_hydro = width


    elif elem_type == "FanCooler":
        area_elem = element["FanArea_m2"]

        diam_hydro = None


    elif elem_type in ["VentingBay", "CoolingBay"]:
        area_elem = None

        diam_hydro = None

    else:
        raise ValueError(f"Unknown ducting element: {elem_type}")
    

    return area_elem, diam_hydro
