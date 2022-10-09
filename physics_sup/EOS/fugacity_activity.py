import numpy as np
from select_para import *

# Composition-dependent fugacity coefficients

def fugacity(p, T, x, y, z, components_norm, phase, molality): #Computes fugacity/activity for all components
    if phase == 'Gas':
        phi_c, Vm, hg_dev = vapour_liquid_fa(p, T, x, y, z, components_norm)
    elif phase == "Aq":
        phi_c, Hcoeff = aqueous(p, T, components_norm, molality)

    return phi_c


def vapour_liquid_fa(p, T, x, y, z, components):
    single_HC = 1

    # Peng-Robinson EoS
    NC = np.size(z)
    R = 8.3145E-5

    # attraction parameter
    ai = np.zeros(NC)

    ai_forward = np.zeros(NC)  # Added for Enthalpy departure
    dt = 0.001 * T
    T_forward = T + dt  # Added for Enthalpy departure

    for i in range(0, np.size(ai)):
        C_a = 1
        Tc = props(components[i], "Tc")
        Pc = props(components[i], "Pc")
        ac = props(components[i], "ac")
        kappa = 0.37464 + 1.54226*ac - 0.26992*ac**2
        alpha = (1+C_a*kappa*(1-np.sqrt(T/Tc)))**2
        ai[i] = 0.45724*R**2*Tc**2*alpha/Pc

        alpha_forward = (1 + C_a * kappa * (1 - np.sqrt(T_forward / Tc))) ** 2  # Added for Enthalpy departure
        ai_forward[i] = 0.45724 * R ** 2 * Tc ** 2 * alpha_forward / Pc  # Added for Enthalpy departure

    dij = [["H2O", "CO2", "C1", "N2", "H2S"],  # from Aspen plus (DOI 10.1016/j.fluid.2016.06.012)
           [0, 0.19014, 0.47893, 0.32547, 0.105],
           [0.19014, 0, 0.100, -0.017, 0.0974],
           [0.47893, 0.100, 0, 0.0311, 0.0503],
           [0.32547, -0.017, 0.0311, 0, 0.1767],
           [-1.10329, 0.0974, 0.0503, 0.1767, 0]]  # binary interaction parameters for CH4, CO2, H2O. dij == dji??
    aij = np.zeros((NC, NC))
    a = aij

    aij_forward = np.zeros((NC, NC))  # Added for Enthalpy departure
    a_forward = aij_forward  # Added for Enthalpy departure

    for i in range(0, NC):
        indexi = dij[0][:].index(components[i]) + 1  #
        for j in range(0, NC):
            indexj = dij[0][:].index(components[j])
            aij[i, j] = np.sqrt(ai[i]*ai[j])*(1-dij[indexi][indexj])
            a[i, j] = aij[i, j]*y[i]*y[j]

            aij_forward[i, j] = np.sqrt(ai_forward[i] * ai_forward[j]) * (1 - dij[indexi][indexj])  # Added for Enthalpy departure
            a_forward[i, j] = aij_forward[i, j] * y[i] * y[j]  # Added for Enthalpy departure

    a = np.sum(a)
    a_forward = np.sum(a_forward)  # Added for Enthalpy departure
    dadT = (a_forward - a) / dt  # Added for Enthalpy departure

    A = a*p/(R**2*T**2)

    # repulsion parameter
    bi = np.zeros(NC)
    for i in range(0, NC):
        C_b = 1
        Tc = props(components[i], "Tc")
        Pc = props(components[i], "Pc")
        bi[i] = C_b*0.0778*R*Tc/Pc*y[i]
    b = np.sum(bi)
    B = b*p/(R*T)

    # solve for compressibility Z
    ZZ = [1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
    # print(ZZ)
    Z = np.roots(ZZ)  # 3 real roots: 2-phase region; 1 real root: single phase region
    # if in single phase region, Z close to 1 will probably be gas, Z close to 0 will probably be liquid
    # print("Z", Z)

    # interpret solution for Z and calculate fugacity coefficients
    phi_c = np.zeros(NC)
    if np.sum(np.imag(Z) == 0) == 1:  # single phase region
        index = np.nonzero(np.imag(Z) == 0)  # find real root
        Z = np.real(Z[index])   # Z reduces to only the real root
        for i in range(0, NC):
            phi_c[i] = np.exp(bi[i]/b*(Z-1)-np.log(Z-B)-A/(2.828*B)*(2*np.sum(y*aij[:, i])/a - bi[i]/b)*np.log((Z+2.414*B)/(Z-0.414*B)))
    else: #multiple roots -> Largest corresponds to vapour phase

        Z = np.amax(Z) #
        for i in range(0, NC):
            phi_c[i] = np.exp(bi[i] / b * (Z - 1) - np.log(Z - B) - A / (2.828 * B) * (
                        2 * np.sum(y * aij[:, i]) / a - bi[i] / b) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))

    hg_dev = R*T*(Z-1)+(T*dadT-a)/(2*np.sqrt(2)*b)*np.log((Z+2.414*B)/(Z-0.414*B))
    Vm = Z*R*T/p  # molar volume
    return phi_c, Vm, hg_dev*1e5 #Added for Enthalpy Departure


def aqueous(p, T, components, molality):
    # Activity-fugacity model following Ziabaksh
    NC = np.size(components)

    # parameters for partitioning between vapour and aqueous phase
    Tc = props("H2O", "Tc")  # water critical T [K]
    Pc = props("H2O", "Pc")  # water critical p [bar]
    tau = 1-T/Tc  # 1-T_r
    a = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502]  # constants in P_s
    P_s = Pc*np.exp(Tc/T*(a[0]*tau + a[1]*tau**1.5 + a[2]*tau**3 + a[3]*tau**3.5 + a[4]*tau**4 + a[5]*tau**7.5))
    tc = T-273.15  # temperature in Celsius
    V0 = (1 + 18.159725E-3*tc)/(0.9998396 + 18.224944E-3*tc - 7.922210E-6*tc**2 - 55.44846E-9*tc**3 + 149.7562E-12*tc**4 - 393.2952E-15*tc**5)
    B = 19654.320 + 147.037*tc - 2.21554*tc**2 + 1.0478E-2*tc**3 - 2.2789E-5*tc**4
    A1 = 3.2891 - 2.3910E-3*tc + 2.8446E-4*tc**2 - 2.8200E-6*tc**3 + 8.477E-9*tc**4
    A2 = 6.245E-5 - 3.913E-6*tc - 3.499E-8*tc**2 + 7.942E-10*tc**3 - 3.299E-12*tc**4
    V = V0 - V0*p/(B+A1*p+A2*p**2)  # volume of pure water at p [cm3/g]
    M = 18.0152  # molar mass of water
    R = 8.3145E1  # cm3 bar K-1 mol-1
    f0_H2O = P_s*np.exp((p-P_s)*M*V/(R*T))  # fugacity of pure water
    rho0_H2O = 1/V
    K0_H2O = np.exp(-2.209 + 3.097E-2*tc - 1.098E-4*tc**2 + 2.048E-7*tc**3)  # equilibrium constant for H2O at 1 bar
    lab = [["C1", "CO2", "N2", "SO2", "H2S", "C2H6"],                                   # coefficients for lambda
           [-5.7066455E-1, -0.0652869, -2.0939363, -5.096151E-2, 1.03658689, 0],         # c1
           [7.2997588E-4, 1.6790636E-4, 3.1445269E-3, 2.8865149E-4, -1.1784797E-3, 0],   # c2
           [1.5176903E2, 40.838951, 3.913916E2, 0, -1.7754826E2, 0],                     # c3
           [3.1927112E-5, 0, -2.9973977E-7, 0, -4.5313285E-4, 0],                        # c4
           [0, 0, 0, 1.1145002E-2, 0, 0],                                                # c5
           [-1.642651E-5, -3.9266518E-2, -1.5918098E-5, 0, 0, 0],                        # c6
           [0, 0, 0, -2.487817E-5, 0, 0],                                                # c7
           [0, 2.1157167E-2, 0, 0, 0, 0],                                                # c8
           [0, 6.5486487E-6, 0, 0, 0, 0],                                                # c9
           [0, 0, 0, 0, 0.4775165E2, 0]]                                                 # c10
    ksi = [["C1", "CO2", "N2", "SO2", "H2S", "C2H6"],                                   # coefficients for ksi
           [-2.9990084E-3, -1.144624E-2, -6.3981858E-3, -7.1462699E-3, 0.010274152, 0],  # c1
           [0, 2.8274958E-5, 0, 0, 0, 0],                                                # c2
           [0, 1.3980876E-2, 0, 0, 0, 0],                                                # c6
           [0, -1.4349005E-2, 0, 0, 0, 0]]                                               # c8
    par = [["C1", "CO2", "N2", "SO2", "H2S", "C2H6"],                                   # coefficients for Henry's cst
           [-0.092248, -0.114535, -0.008194, 0.198907, 0.77357854, 0],                   # eta
           [-5.779280, -5.279063, -5.175337, -1.552047, 0.27049433, 0],                  # tau
           [7.262730, 6.187967, 6.906469, 2.242564, 0.27543436, 0],                      # beta
           [0, 0, 0, -0.009847, 0, 0]]                                                   # Gamma

    # construct K-values
    phi_c = np.zeros(NC)
    for i in range(0, NC):
        if components[i] != "H2O":  # for non-H2O components: calculate Henry's and activity coefficients
            index = lab[0][:].index(components[i])
            dB = par[2][index] + par[4][index]*p + par[3][index]*np.sqrt(1E3/T)
            k_H = np.exp((1-par[1][index])*np.log(f0_H2O) + par[1][index]*np.log(R*T/M*rho0_H2O) + 2*rho0_H2O*dB)
            labda_c = lab[1][index] + lab[2][index]*T + lab[3][index]/T + lab[4][index]*p + lab[5][index]*p/T + \
                      lab[6][index]*p/(630-T) + lab[7][index]*T*np.log(p)
            ksi_c = ksi[1][index] + ksi[2][index]*T + ksi[3][index]*p/T + ksi[4][index]*p/(630-T)
            m_c = molality
            m_a = molality
            gamma = np.exp((2*m_c*labda_c) + (m_a*m_c*ksi_c))
            Hcoeff = k_H * gamma
            phi_c[i] = k_H*gamma/p # = phi*xna/Xaq from 2.33
        elif components[i] == "H2O":  # for H2O: follow "true equilibrium" by Spycher (described in Ziabaksh)
            p0 = 1  # reference pressure of 1 bar
            V_H2O = 18.1  # average partial molar volume over pressure interval p-p0
            phi_c[i] = K0_H2O*np.exp((p-p0)*V_H2O/(R*T))/p # = phi*xna/Xaq

    return phi_c, Hcoeff
