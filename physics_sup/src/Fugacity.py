import numpy as np
from src.Props import *



# Composition-dependent fugacity coefficients


def fugacity(p, T, x, components, phase, molality,props,EOS):
    if phase == "V" or phase == "L":
        phi_c, Vm = vapour_liquid(p, T, x, components, phase,props,EOS)
        #print("WHAT" , phi_c,Vm)
    elif phase == "Aq":
        phi_c = aqueous(p, T, components, molality,props)
    return phi_c,Vm


def vapour_liquid(p, T, x, components, phase,props,EOS):
    if EOS==1:
      # PREOS
      m1 = (1 + np.sqrt(2))
      m2 = (1 - np.sqrt(2))
      oma = 0.457235529
      omb = 0.077796074
      c1 = 0.37464
      c2 = 1.54226
      c3 = 0.2669

    elif EOS==2:
      # SRKEOS
      m1 = 0
      m2 = 2
      oma = 0.4274802
      omb = 0.086640350
      c1 = 0.48
      c2 = 1.57
      c3 = -0.17

    Peneloux = True  # perform Peneloux correction for molar volume

    # Peng-Robinson EoS #https://www.e-education.psu.edu/png520/m11_p2.html
    NC = np.size(x)
    R = 8.3145e-5


    ai = np.zeros(NC)
    bi = np.zeros(NC)
    for i in range(0, np.size(ai)):
        C_a =  C_b = 1
        Tc = props.Tc[i]
        Pc = props.Pc[i]
        w = props.w[i]
        kappa = c1 + c2 * w - c3 * w ** 2
        alpha = (1 + C_a * kappa * (1 - np.sqrt(T / Tc))) ** 2
        ai[i] = oma * R ** 2 * Tc ** 2 / Pc * alpha  # attraction parameter
        bi[i] = C_b * omb * R * Tc / Pc  # repulsion parameter



    '''
    dij = [["H2O",    "CO2",   "CH4",   "N2",    "H2S",  "H2", "C2H6","C3H8"],  # from Aspen plus (DOI 10.1016/j.fluid.2016.06.012)
           [0,        0.19014, 0.47893, 0.32547, 0.105,  "", 0,0],
           [0.19014,  0,       0.100,   -0.017,  0.0974, -0.1622, 0.129,0.131],
           [0.47893,  0.100,   0,       0.0311,  0.0503, 0.0156, 0.01,0.016],
           [0.32547,  -0.017,  0.0311,  0,       0.1767, 0.103, 0.039,0.083],
           [-1.10329, 0.0974,  0.0503,  0.1767,  0,      "", 0.084,0.082],
           ["",       -0.1622, 0.0156,  0.103,   "",     0, 0,0],
           [0,       0.1239,  0.01,   0.039,     0.084, 0 , 0,-0.006],
           [0, 0.131, 0.016, 0.083,0.082,0,-0.006,0]]  # binary interaction parameters. dij == dji

    


     dij = [["H2O",    "CO2",   "C1",   "N2",    "H2S",  "H2", "C2","C3"],  # from Aspen plus (DOI 10.1016/j.fluid.2016.06.012)
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]]  # binary interaction parameters for PREOS. dij == dji
           
    
    dij = [["CO2", "N2", "H2O", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "nC6", "nC7", "nC8", "nC9", "nC10"],
           [	0,     0,     0.2,    0.1,     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, .11, 0.096, .15, 0.1277, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0],
           [	       0,     0,     0.275,  0.1,     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.176, .12, 0.1002, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0],
           [	    0.2,   0.275, 0,      0.4907,  0.4911, 0.5469, 0.508, 0.508, 0.5, 0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.12, .48, 0, 0, 0, 0, 0, 0, 0.508, 0, 0, 0, 0],
           [	      0.1,   0.1,   0.4907, 0,       0, 0, 0, 0, 0, 0, 0.028739997, 0.033919997, 0.036999997, 0.039659997, 0.041619997, 0, 0.085, 0, 0.092810, 0.130663, 0.130663, 0.130663, 0.04848, 0.5268, 0, 0.0, 0.025,  0.046, 0.061],
           [	     0.1,   0.1,   0.4911, 0,       0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0.07, 0, 0, 0.006, 0.006, 0.006, 0.01, 0.01, 0, 0, 0, 0, 0],
           [	 0.1,   0.1,   0.5469, 0,       0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0.07, 0, 0, 0.006, 0.006, 0.006, 0.01, 0.01, 0, 0, 0, 0, 0],
           [	    0.1,   0.1,   0.508,  0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	    0.1,   0.1,   0.508,  0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	    0.1,   0.1,   0.5,    0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	     0.1,   0.1,   0.5,    0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	    0.1,   0.1,   0.48,   0.02874, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	     0.1,   0.1,   0.48,   0.03392, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	 0.1,   0.1,   0.48,   0.037,   0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	   0.1,   0.1,   0.48,   0.03966, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	 0.1,   0.1,   0.48,   0.04162, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	 0.11,  0.1,   0.45,   0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0],
           [	    0.096, 0.176, 0.12,   0.085,    0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.06, 0.125, 0.125, 0.125, 0.125]]
    '''


    aij = np.zeros((NC, NC))
    am = 0
    bm= 0
    #Quadratic Mixing Rule - qmr
    for i in range(0, NC):
        #indexi = dij[0][:].index(components[i]) + 1  #
        for j in range(0, NC):
            #indexj = dij[0][:].index(components[j])
            aij[i, j] = (ai[i])**(1/2)*(ai[j])**(1/2)#*(1-dij[indexi][indexj])
            am += aij[i, j]*x[i]*x[j]
        bm += bi[i] * x[i]
    #print("EOS", x)
    if Peneloux:
        # Peneloux volume shift parameter (1982)
        c = 0
        for i in range(0, NC):
            z_ra = 0.29056 - 0.08775 * props.w[i]
            c += x[i]*(0.50033 * R * props.Tc[i] / props.Pc[i] * (0.25969 - z_ra))


        bm = bm  # Peneloux correction for b and V

    A = am * p / (R ** 2 * T ** 2)
    B = bm*p/(R*T)
    print("EOS", A, B)

    # solve for compressibility Z
    E0 = -(A*B + m1*m2*B**2 * (B+1))
    E1 = A - (2 * (m1 + m2) - 1) * B ** 2 - (m1 + m2) * B
    E2 = (m1 + m2 - 1) * B - 1

    Z = np.roots([1, E2,E1,E0]) # 3 real roots: 2-phase region; 1 real root: supercritical??

    # interpret solution for Z and calculate fugacity coefficients
    if np.sum(np.imag(Z) == 0) == 1:  # supercritical??
        index = np.nonzero(np.imag(Z) == 0)  # find real root
        Z = np.real(Z[index])   # Z reduces to only the real root

    else:
        if phase == "V":
            Z = np.amax(Z)
        elif phase == "L":
            Z = np.amin(Z)
        else:
            Z = np.amax(Z)

    # fugacity coefficients for either V or L phase
    phi_c = np.zeros(NC)

    for k in range(0, NC):  # for 2-phase: fugacity coefficient for each phase, based on x and y input
        phi_c[k] = np.exp(bi[k] / bm * (Z - 1) - np.log(Z - B) - A / (2.828 * B) * (2 * np.sum(x*aij[:, k])
                                                    / am - bi[k] / bm) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))
    #print ("phi" , phi_c)

    Vm = Z*R*T/p  # molar volume

    Vm = Vm    # Peneloux correction
    print
    return phi_c, Vm


def aqueous(p, T, components, molality,props):
    # Activity-fugacity model following Ziabaksh
    NC = np.size(components)

    # parameters for partitioning between vapour and aqueous phase
    Tc = H2O.Tc  # water critical T [K]
    Pc = H2O.Pc  # water critical p [bar]
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
    lab = [["C1", "CO2", "N2", "SO2", "H2S", "C2"],                                   # coefficients for lambda
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
    ksi = [["C1", "CO2", "N2", "SO2", "H2S", "C2"],                                   # coefficients for ksi
           [-2.9990084E-3, -1.144624E-2, -6.3981858E-3, -7.1462699E-3, 0.010274152, 0],  # c1
           [0, 2.8274958E-5, 0, 0, 0, 0],                                                # c2
           [0, 1.3980876E-2, 0, 0, 0, 0],                                                # c6
           [0, -1.4349005E-2, 0, 0, 0, 0]]                                               # c8
    par = [["C1", "CO2", "N2", "SO2", "H2S", "C2"],                                   # coefficients for Henry's cst
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
            phi_c[i] = k_H*gamma/p
        elif components[i] == "H2O":  # for H2O: follow "true equilibrium" by Spycher (described in Ziabaksh)
            p0 = 1  # reference pressure of 1 bar
            V_H2O = 18.1  # average partial molar volume over pressure interval p-p0
            phi_c[i] = K0_H2O*np.exp((p-p0)*V_H2O/(R*T))/p


    return phi_c
