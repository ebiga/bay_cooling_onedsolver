import math

from scipy.optimize import minimize, Bounds


gamma  = 1.4
R_gas  = 287.05
cp_air = 1005.0

gamm1 = gamma - 1.
gamm2 = gamm1 / 2.



def atmo(altitude_ft, dISA_degC):
    """
    Plain vanilla ISA computer.
    """
    altitude_m = altitude_ft * 0.3048
    p_Pa = 101325. * ( 1. - 0.000022558*altitude_m )**5.2559
    T_K = 288.15 - 0.0065*altitude_m + dISA_degC
    rho_kg_m3 = p_Pa / (R_gas * T_K)
    mu_kg_ms = ViscositySutherland(T_K)

    return p_Pa, T_K, rho_kg_m3, mu_kg_ms



def solve_throat_mach(mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa):
    """
    Inverts the 1D compressible Mass Flow Parameter (MFP) equation 
    to find the true subsonic static Mach number at the throat.
    """

    def throat_Mach_for_target_mfr(x, mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa):

        # define freestream total properties
        rhotot_kg_m3 = Ptot_Pa / (R_gas * Ttot_Pa)

        # define the local (inlet) conditions to be optimised
        isenM = 1. + 0.5*gamm1 * x[0]**2.

        p1_Pa = Ptot_Pa / (isenM**(gamma/gamm1))
        rho1_kg_m3 = rhotot_kg_m3 / (isenM**(1./gamm1))
        M1 = mfr_target_kg_m3/(S_throat_m2 * math.sqrt(gamma*p1_Pa*rho1_kg_m3))

        return (x[0] - M1)**2.

    choking_constraint_b = Bounds( 0., 1. )
    
    res = minimize( throat_Mach_for_target_mfr, x0=[0.5], args=(mfr_target_kg_m3, S_throat_m2, Ptot_Pa, Ttot_Pa), method='Powell', bounds=choking_constraint_b)
    MM = res.x[0]

    return MM



def BoundaryLayerThickness(ReM, position):
    """
    Estimates the boundary layer thickess at a given position.
    
    Parameters:
    ReM         : Reynolds number per unit length [1/m]
    position    : Where to measure the boundary layer thickness [m]
    """

    return 0.37*position*((ReM*position)**-0.2)



def ViscositySutherland(T_K):

    return 0.00001716 * (T_K/273.15)**1.5 * (273.15+110.)/(T_K+110.)



def solve_local_states(mdot, area, pt, Tt):
    """
    Calls solve_throat_mach to extract physical static properties for a given area.
    """

    # Get the local Mach number
    M_local = solve_throat_mach(mdot, area, pt, Tt)
    
    # Compute the corresponding static thermodynamic state
    isenM = 1.0 + 0.5 * gamm1 * M_local**2
    
    t_local = Tt / isenM
    p_local = pt / (isenM**(gamma / gamm1))
    rho_local = p_local / (R_gas * t_local)
    v_local = M_local * math.sqrt(gamma * R_gas * t_local)
    mu_local = ViscositySutherland(t_local)
    
    return M_local, t_local, p_local, rho_local, v_local, mu_local
