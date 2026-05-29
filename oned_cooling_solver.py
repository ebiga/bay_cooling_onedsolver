import math
from scipy.optimize import root_scalar
from scipy.optimize import minimize, LinearConstraint, Bounds


gamma=1.4
R_gas=287.05
cp_air = 1005.0


def atmo(altitude_ft, dISA_degC):
    """
    Plain vanilla ISA computer.
    """
    altitude_m = altitude_ft * 0.3048
    p_Pa = 101325. * ( 1. - 0.000022558*altitude_m )**5.2559
    T_K = 288.15 - 0.0065*altitude_m + dISA_degC
    rho_kg_m3 = p_Pa / (R_gas * T_K)
    mu_kg_ms = 0.00001716 * (T_K/273.15)**1.5 * (273.15+110.)/(T_K+110.)
    return p_Pa, T_K, rho_kg_m3, mu_kg_ms


def naca_pressure_recovery(mfr):
    """
    Empirical polynomial fit for a standard NACA submerged flush inlet 
    pressure recovery factor (eta_d) vs Mass Flow Ratio (MFR).
    # NACA RM A7130 / NACA ACR 5120
    """
    mfr_clamped = max(0.01, min(mfr, 0.99)) # Passive limit guardrail
    eta = -1.1 * (mfr_clamped - 0.65)**2 + 0.85
    return max(0.1, eta)


def solve_throat_mach(mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa):
    """
    Inverts the 1D compressible Mass Flow Parameter (MFP) equation 
    to find the true subsonic static Mach number at the throat.
    """

    def throat_Mach_for_target_mfr(x, mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa):
        gamm1 = gamma - 1.

        # define freestream total properties
        rhotot_kg_m3 = Ptot_Pa / (R_gas * Ttot_Pa)

        # define the local (inlet) conditions to be optimised
        isenM = 1. + 0.5*gamm1 * x[0]**2.

        p1_Pa = Ptot_Pa / (isenM**(gamma/gamm1))
        rho1_kg_m3 = rhotot_kg_m3 / (isenM**(1./gamm1))
        M1 = mfr_target_kg_m3/(S_throat_m2 * math.sqrt(gamma*p1_Pa*rho1_kg_m3))

        return (x[0] - M1)**2.

    choking_constraint_b = Bounds( 0., 1. )
    choking_constraint_c = LinearConstraint( 1., 0., 1. )
    
    res = minimize( throat_Mach_for_target_mfr, x0=[0.5], args=(mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa), method='Powell', bounds=choking_constraint_b, constraints=choking_constraint_c)
    MM = res.x[0] + math.sqrt(res.fun)

    return MM


def size_bay_ventilation(q_cooling, t_inf, p_inf, mach, cp_exit, t_max_celsius, k_sys=2.0):
    """
    Sizes the required NACA inlet throat area for an equipment bay based on 
    the maximum rated component temperature boundary condition.
    """
    gamm1 = gamma - 1.
    gamm2 = gamm1/2.
    
    # Convert system limit to Kelvin
    t_max = t_max_celsius + 273.15
    
    # 1. Freestream Aerodynamics & Stagnation States
    a_inf = math.sqrt(gamma * R_gas * t_inf)
    v_inf = mach * a_inf
    rho_inf = p_inf / (R_gas * t_inf)
    
    p_t_inf = p_inf * (1.0 + gamm2 * mach**2)**(gamma/gamm1)
    t_t_inf = t_inf * (1.0 + gamm2 * mach**2)

    qdin_inf = p_t_inf - p_inf
    
    # Thermal Feasibility Check
    if t_t_inf >= t_max:
        return {
            "status": "Infeasible", 
            "reason": f"Ram air total temp ({t_t_inf-273.15:.1f}°C) exceeds max system rating ({t_max_celsius}°C)."
        }
        
    # 2. Dynamic Mass Flow Requirement based on Ambient Environment
    dt_allowed = t_max - t_t_inf
    mdot = q_cooling / (cp_air * dt_allowed)
    
    # 3. Fixed Compartment Exit Pressure Boundary
    p_static_2 = p_inf + cp_exit * qdin_inf
    t_t_2 = t_max  # ASSUMPTION: The air leaves at the maximum rated temperature limit
    
    # 4. Define the Geometric Residual Function for Scipy
    def area_residual(x):
        a_throat_guess = x[0]
            
        a_exit = a_throat_guess  # ASSUMPTION: Balanced duct area assumption
        
        # Calculate local MFR
        mfr = mdot / (rho_inf * v_inf * a_throat_guess)
            
        eta_d = naca_pressure_recovery(mfr)
        p_t_1 = p_inf + eta_d * (p_t_inf - p_inf)
        t_t_1 = t_t_inf  # ASSUMPTION: no total temp losses at the inlet
        
        # Node 1 (Throat): Subsonic Static State
        m_1 = solve_throat_mach(mdot, a_throat_guess, p_t_1, t_t_1)
        t_static_1 = t_t_1 / (1.0 + gamm2 * m_1**2)
        p_static_1 = p_t_1 / (1.0 + gamm2 * m_1**2)**(gamma/gamm1)
        rho_static_1 = p_static_1 / (R_gas * t_static_1)
        v_1 = m_1 * math.sqrt(gamma * R_gas * t_static_1)
        
        # Node 2 (Exit): Static State via Quadratic Energy Equation
        coeff_a = (R_gas * mdot)**2 / (2.0 * cp_air * (p_static_2 * a_exit)**2)
        t_static_2 = (-1.0 + math.sqrt(1.0 + 4.0 * coeff_a * t_t_2)) / (2.0 * coeff_a)
        
        rho_static_2 = p_static_2 / (R_gas * t_static_2)
        v_2 = mdot / (rho_static_2 * a_exit)
        m_2 = v_2 / math.sqrt(gamma * R_gas * t_static_2)
        p_t_2 = p_static_2 * (1.0 + gamm2 * m_2**2)**(gamma/gamm1)
        
        # Losses (Friction + Thermal Expansion Rayleigh Penalty)
        dp_friction = k_sys * (0.5 * rho_static_1 * v_1**2)
        dp_thermal = (mdot**2 / a_throat_guess**2) * ((1.0 / rho_static_2) - (1.0 / rho_static_1))
        
        calculated_p_t_2 = p_t_1 - dp_friction - dp_thermal
        return calculated_p_t_2 - p_t_2

    # Solve for required area
    try:
        throat_area_bounds = Bounds( 0.001, 0.5 )
        
        a_throat_guess = mdot / (rho_inf * v_inf)

        res = minimize( area_residual, x0=[a_throat_guess], method='Powell', bounds=throat_area_bounds)
        final_area = res.x[0]

        final_mfr = mdot / (rho_inf * v_inf * final_area)
        
        return {
            "status": "Success",
            "mdot": mdot,
            "dt_air": dt_allowed,
            "a_throat_cm2": final_area * 10000.0,
            "mfr": final_mfr,
            "eta_d": naca_pressure_recovery(final_mfr)
        }
    except ValueError:
        return {
            "status": "Failed",
            "reason": "Available ram pressure cannot overcome duct losses for this mass flow requirement."
        }




if __name__ == "__main__":

    # Flight conditions
    altitude_ft = 0.
    dISA_K = 0.
    Mach = 0.5

    # total heat rejected by systems into the bay
    Q_BAY_LOAD_W = 4500.
    # Systems are rated up to this temperature
    T_SYSTEM_MAX_degC = 70.0

    # outlet pressure coefficient
    Cp_exit = 0.1
    
    p_inf, t_inf, _, _ = atmo(altitude_ft, dISA_K)

    sl_sim = size_bay_ventilation(
        Q_BAY_LOAD_W, t_inf, p_inf, Mach, Cp_exit, T_SYSTEM_MAX_degC
    )
    
    if sl_sim["status"] == "Success":
        print(f"  Calculated Target mdot   : {sl_sim['mdot']:.4f} kg/s")
        print(f"  Resulting Air ΔT         : {sl_sim['dt_air']:.1f} K")
        print(f"  REQUIRED NACA Throat Area: {sl_sim['a_throat_cm2']:.2f} cm²")
        print(f"  Operating MFR            : {sl_sim['mfr']:.3f}")
    else:
        print(f"  Sizing Failed: {sl_sim.get('reason')}")
