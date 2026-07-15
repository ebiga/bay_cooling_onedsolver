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



def get_outlet_cd(outlet_type, J, delta=0., a_exit=None, aspect_r=4., porosity=0.6):
    """
    Determine the dynamic discharge coefficient (Cd) and dynamic base drag properties
    for different outlet configurations under external crossflow.

    The discharge coefficient is defined on exit area correction:
        Area_actual = Area_ideal * Cd

    The base drag is modeled as:
        Drag_base = Cd_base * q_inf * area_base

    Parameters:
    -----------
    outlet_type : str   -> "OutletInvertedScoop", "OutletParallelRamp", etc.
    J           : float -> Momentum flux ratio ((rho_exit * V_exit^2) / (rho_inf * V_inf^2))
    delta       : float -> Local boundary layer thickness at the exit plane [m]
    a_exit      : float -> Area of the exit nozzle opening [m2]
    aspect_r    : float -> Outlet throat aspect ratio (typically in the range 3:1 to 5:1)
    porosity    : float -> Open area ratio (only applied to the OutletGrill branch)

    Returns:
    --------
    cd          : float -> Dynamic discharge coefficient [-]
    Cd_base     : float -> Dynamic base drag coefficient (referenced to q_inf and area_base) [-]
    area_base   : float -> Statistically/geometrically derived solid base area [m2]
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
    
    # Base drag scales with local dynamic pressure ratio (shielding effect)
    shielding = f_v * f_v


    if outlet_type == "OutletInvertedScoop":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: Hoerner, S.F., "Fluid-Dynamic Drag," 1965. 
        #      Chapter XI ("Internal-Flow Systems"), Section 4 ("Outlets"), pp. 11-15.
        # Ref: Henry, J.R., "Design of Power-Plant Installations: Pressure-Loss and Drag 
        #      Estimates of Inlet and Outlet Ducts," NACA Wartime Report L-344 (originally 
        #      issued as ACR 5I20), pp. 23-28, 1945.
        cd_max = 0.80 - 0.08 * min(0.5, bar_delta)
        cd = cd_max - 0.05 * math.exp(-sqrt_J)
        
        # --- BASE DRAG ---
        # Physical model: A protruding cowl creates a high-pressure recovery region ahead, 
        # but a massive rearward separation wake. As jet outflow increases (sqrt_J), base 
        # pressure recovers due to jet entrainment (base bleed).
        # Base area: Estimated as the projected rearward solid lip/cowl frontal area (~15% of exit)
        area_base = a_exit * 0.15
        Cd_base_0 = 0.22      # Static base drag of raw cowl profile
        k_bleed = 1.5         # Moderate bleed efficacy due to protrusion
        Cd_base = Cd_base_0 * shielding * math.exp(-k_bleed * sqrt_J_eff)
        
    elif outlet_type == "OutletParallelRamp":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: ESDU 86002, "Drag and mass flow of internal flow systems: flush-mounted outlets," 
        #      Section 5 (Discharge Characteristics of Flush Rectangular Slots), pp. 8-12.
        # Ref: Wornom, D.E., "Discharge Coefficients of Various Outer-Skin Outlets for Aircraft," 
        #      NACA Technical Note 3924, pp. 11-14, 1957.
        cd = 0.62 * (1.0 - 0.60 * math.exp(-2.0 * sqrt_J_eff))
        
        # --- BASE DRAG ---
        # Physical model: Sits completely flush with the boundary layer. The only physical base 
        # is the blunt splitter-lip thickness. Parallel jet flow quickly re-energizes this wake.
        # Base area: Set as the trailing-edge splitter thickness area (~5% of exit)
        area_base = a_exit * 0.05
        Cd_base_0 = 0.08      # Very low static drag due to flush profile
        k_bleed = 2.5         # Highly effective base bleed because flow is parallel to lip
        Cd_base = Cd_base_0 * shielding * math.exp(-k_bleed * sqrt_J_eff)
        
    elif outlet_type == "OutletDivergentRamp":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: ESDU 86002, "Drag and mass flow of internal flow systems: flush-mounted outlets," 
        #      Section 6 (Divergent Ramp Outlets), pp. 14-17.
        # Ref: Wornom, D.E., "Discharge Coefficients of Various Outer-Skin Outlets for Aircraft," 
        #      NACA Technical Note 3924, pp. 15-18, 1957.
        cd = 0.70 * (1.0 - 0.40 * math.exp(-2.5 * sqrt_J_eff))
        
        # --- BASE DRAG ---
        # Physical model: Diverging sidewalls create localized boundary layer growth/separation 
        # prior to ejection, leading to slightly higher static wake drag than the parallel slot.
        # Base area: Set as the trailing edge/ramp termination step (~8% of exit)
        area_base = a_exit * 0.08
        Cd_base_0 = 0.12      # Moderate static drag from diffusion step
        k_bleed = 2.0         # Slightly degraded base bleed due to diffusion angles
        Cd_base = Cd_base_0 * shielding * math.exp(-k_bleed * sqrt_J_eff)
        
    elif outlet_type == "OutletGrill":
        # --- DISCHARGE COEFFICIENT ---
        # Ref: Dittrich, R.T., and Graves, C.C., "Discharge Coefficients for Combustor-Liner 
        #      Orifices. I - Circular Orifices with Parallel Flow," NACA Technical Note 3663, 
        #      pp. 12-16, 1956.
        # Ref: Gritsch, M. et al., "Discharge Coefficient Measurements of Waved-Edge and 
        #      Standard Film Cooling Holes," ASME Paper No. 98-GT-541, 1998.
        Cd_baseline = 0.62 * porosity
        vr_eff = 1.0 / max(1e-3, sqrt_J_eff)
        cd = Cd_baseline / math.sqrt(1.0 + 1.1 * vr_eff**2)        
        
        # --- BASE DRAG ---
        # Physical model: Grill bars act as localized bluff bodies transverse to the external 
        # crossflow. High static drag is generated by the solid blockages. Blowing bleed is 
        # highly effective at reducing separation, but turning skin losses maintain a small floor.
        # Base area: Calculated directly from the solid frontal area (1.0 - porosity)
        area_base = a_exit * (1.0 - porosity)
        Cd_base_0 = 0.32      # High blunt body static drag of the bar matrix
        k_bleed = 1.8         # Decent bleed effect through discrete holes
        Cd_base_dyn = Cd_base_0 * shielding * math.exp(-k_bleed * sqrt_J_eff)
        Cd_base = max(0.02, Cd_base_dyn) # Base drag floor to capture turning/skin friction losses

    else:
        raise ValueError(f"Unknown outlet type: {outlet_type}")

    return cd, Cd_base, area_base



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
