from src.Activity import *

def K_phase(p, T, components,phase):
    if phase == ['V', 'L']:
        K = vapour_liquid(p, T, components)
        K = 1/K
    elif phase == ["L", "V"]:
        K = vapour_liquid(p, T, components)
        K= 1/K
    elif phase == ['V', 'Aq']:
        KVL = vapour_liquid(p, T, components)
        KVAq = vapour_aqueous(p, T, components)
        K = 1/KVAq


    elif phase == ["Aq", "V"]:
        K = vapour_aqueous(p, T,components)
        K = K

    elif phase == ['L', 'Aq']:
        KVL = vapour_liquid(p, T, components)
        KVAq = vapour_aqueous(p, T, components)
        K = KVL/KVAq
    elif phase == ['Aq', 'L']:
        KVL = vapour_liquid(p, T,  components)
        KVAq = vapour_aqueous(p, T, components)
        K = KVL/KVAq
    elif phase == ["L"]:
        K = vapour_liquid(p, T,  components)
        K = 1 / K
    return K

def wilsonK(p,T,components):
    mix = Mix(components)
    NC = np.size(components)
    K=np.zeros(NC)
    for i in range(0,NC):
        K[i] = (mix.Pc[i] / p) * np.exp(5.373 * (1 + mix.w[i]) * (1 - mix.Tc[i] / T))

    return K

def vapour_liquid(p, T, components):
    NC = np.size(components)
    K_VL = np.zeros(NC)
    for i in range(0, NC):
        if components[i] == H2O:
            aii = [-133.67, 0.63288, 3.19211E-3]
            K_VL[i] = (aii[0] + aii[1] * T) / p + aii[2] * p  # K_i = x_iV/x_iAq
        else:
            K_VL[i] = wilsonK(p,T,components)[i]
    return K_VL

def vapour_aqueous(p, T,components):
    mix = Mix(components)
    NC = np.size(components)
    K_VAq = np.zeros(NC)
    if p < 10:
        p = 10
    if T < 300:
        T = 300

    for i in range(0, NC):
        if components[i] == H2O:
            aii = [12.048399, 4030.18245, -38.15]
            P_sat = np.exp(aii[0] - aii[1] / (T + aii[2]))
            j_inf = 1
            K_VAq[i] = P_sat / p * j_inf  # K_i = x_iV/x_iAq
        elif components[i] == CO2:
            if p < 200:
                K_VAq[i] = 20 # 5
            else:
                K_VAq[i] = 5
        elif components[i] == N2:
                K_VAq[i] = 100
        else:
            b = [0.688, 0.642, 0]                            # b1, b2, b3
            N = [[C1, C2, C3, iC4 , nC4 , iC5 , nC5, nC6, nC7, nC8, nC9, nC10 ,CO2, N2, H2S],
                 [ 1,     2,    3,    4,     4,       5,       5,    6,     7,     8,     9,      10,   1 ,    0,    0  ]]  # number of carbon atoms
            aii = [5.927140, -6.096480, 1.288620, 0.169347, 15.25180, -15.68750, -13.47210, 0.43577]  # a11-a24, Empirical parameters used in equations A.29 and A.30
            Tc = mix.Tc
            a1 = aii[0] + aii[1]*Tc[i]/T + aii[2]*np.log(T/Tc[i]) + aii[3]*T**6 / (Tc[i]**6)
            a2 = aii[4] + aii[5]*Tc[i]/T + aii[6]*np.log(T/Tc[i]) + aii[7]*T**6 / (Tc[i]**6)
            P_sat = mix.Pc[i] * np.exp(a1 + mix.w[i] * a2)
            index = N[0][:].index(components[i])
            j_inf = np.exp(b[0] + b[1] * N[1][index])  # + HC[3][index] / HC[4][index])# inifinite dilution activity coefficient
            K_VAq[i] = P_sat / p * j_inf  # K_i = x_iV/x_iAq
    return K_VAq



def Kval(p, T, components,phases):  # calculates K-values for specified phases
    NP = np.size(phases)
    NC=np.size(components)
    K = np.zeros((NP-1, NC))
    # calculation of initial or updated K-values
    #print('NP',NP)
    if (NP - 2) == 0: #two phase
        K = K_phase(p, T,components, phases)
        #print('here')
    else: # 3-phase
        for m in range(1, NP):
            K[m-1, :] =K_phase(p, T,components,[phases[0], phases[m]])
            #print(phases[0],phases[m], K[m-1, :])
    #print ('Kval',K)
    return K