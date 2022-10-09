import numpy as np
from src.Props import *

# Composition-dependent fugacity coefficients



def Activity(p, T, components, molality):
    # Activity-fugacity model following Ziabaksh
    mix = Mix(components)
    NC = np.size(components)
    Tc=  10E-12
    Pc = 10E-12
    # parameters for partitioning between vapour and aqueous phase
    for i in range(0,NC):
        if mix.names[i]=='H2O':
            Tc = mix.Tc[i]  # water critical T [K]
            Pc = mix.Pc[i]  # water critical p [bar]
    tau = 1-T/Tc  # 1-T_r
    tc = T - 273.15
    a = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502]
    #a = [-7.85951783, 0, 0, 0, 0, 0]  # constants in P_s
    P_s =Pc*np.exp(Tc/T*(a[0]*tau + a[1]*tau**1.5 + a[2]*tau**3 + a[3]*tau**3.5 + a[4]*tau**4 + a[5]*tau**6.5)) #saturation pressure (Shibue ,2003)
    P_s=(0.61121*np.exp((18.678-tc/234.5)*(tc/(257.14+tc))))/100
    #print('Ps',P_s,P_s1)

    tc = T-273.15  # temperature in Celsius
    V0 = (1 + 18.159725E-3*tc)/(0.9998396 + 18.224944E-3*tc - 7.922210E-6*tc**2 - 55.44846E-9*tc**3 + 149.7562E-12*tc**4 - 393.2952E-15*tc**5)
    B = 19654.320 + 147.037*tc - 2.21554*tc**2 + 1.0478E-2*tc**3 - 2.2789E-5*tc**4
    A1 = 3.2891 - 2.3910E-3*tc + 2.8446E-4*tc**2 - 2.8200E-6*tc**3 + 8.477E-9*tc**4
    A2 = 6.245E-5 - 3.913E-6*tc - 3.499E-8*tc**2 + 7.942E-10*tc**3 - 3.299E-12*tc**4
    V = V0 - V0*p/(B+A1*p+A2*p**2)  # volume of pure water at p [cm3/g], (Fine and Millero, XXXX)
    M = 18.0152  # molar mass of water
    R = 8.3145E1  # cm3 bar K-1 mol-1
    tf= (T - 273.15)*(9/5) + 32
    phi_w=0.9958+ 9.6833E-5*tf - 6.175E-7*tf**2 - 3.08333E-10*tf**3
    f0_H2O = phi_w*P_s*np.exp((p-P_s)*M*V/(R*T))  # fugacity of pure water, (King et al, xxx)

    #print('fo_H2O',f0_H2O,P_s*np.exp((p-P_s)*M*1.227568/(R*T)))
    rho0_H2O = 1/V #density of water
    K0_H2O = np.exp(-2.209 + 3.097E-2*tc - 1.098E-4*tc**2 + 2.048E-7*tc**3)  # equilibrium constant for H2O at 1 bar, (Spycher et al, XXXX)
    lab = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                     # coefficients for lambda
           [-5.7066455E-1,  -2.143686,      0.513068, 0.52862384   ,  0.52862384, 0, 0, 0, 0, 0, 0, 0, -0.0652869, 1.03658689,-2.0939363],                       # c1
           [7.2997588E-4,   2.598765E-3,   -0.000958, -1.0298104E-3, -1.0298104E-3, 0, 0, 0, 0, 0, 0, 0, 1.6790636E-4, -1.1784797E-3, 3.1445269E-3],                # c2*T
           [1.5176903E2,    4.6942351E2,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40.838951, -1.7754826E2,3.913916E2],                                                  # c3/T
           [3.1927112E-5,  -4.6849541E-5,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.5313285E-4,-2.9973977E-7],                                                          # c4*P
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],                                                                      # c5/P
           [-1.642651E-5,   0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.9266518E-2, 0, -1.5918098E-5],                                                          # c6*(P/T)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],                                                                     # c7*(T/P^2)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.1157167E-2, 0, 0 ],                                                          # c8*(P/(630-T))
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.5486487E-6, 0, 0 ],                                                          # c9*T*ln(P)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.775165E1, 0],                                                           # c10*(P/T^2)
           [0,              -8.4616602E-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],                                                                     # c11*P^2T
           [0,              1.095219E-6,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                                                                      # c12*PT

    ksi = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                   # coefficients for ksi, Calibrated second-order interaction parameters
           [-2.9990084E-3, -1.0165947E-2, -0.007485, 0.0206946 , 0.0206946,  0, 0, 0, 0, 0, 0, 0, -1.144624E-2, 0.010274152, -6.3981858E-3],  # c1
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              2.8274958E-5,  0, 0 ],   # c2*T
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              1.3980876E-2,  0, 0 ],  # c6*(P/T)
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              -1.4349005E-2, 0, 0 ]]  # c8*(P/(630-T))

    par = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                   # coefficients for Henry's cst
           [-0.1196, -0.6091, -1.1471, -1.6849 , -1.6849,  0, 0, 0, 0, 0, 0, 0, -0.114535, 0.77357854,-0.008194],                  # eta
           [-10.9926, -16.3482, -25.3879, -33.8492 , -33.8492,  0, 0, 0, 0, 0, 0, 0, -5.279063, 0.27049433,-5.175337],            # tau
           [14.4019, 20.0628, 28.2616, 36.1457, 36.1457, 0, 0, 0, 0, 0, 0, 0, 6.187967, 0.27543436, 6.906469],                    # beta
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                  # Gamma

    # construct fugacity-values
    phi_c = np.zeros(NC)
    for i in range(0, NC):
        if mix.names[i]== 'C20':
            phi_c[i]=1E-20
            #print(mix.names[i])
        else:
            if mix.names[i] != 'H2O':  # for non-H2O components: calculate Henry's and activity coefficients
                index = lab[0][:].index(mix.names[i])
                dB = par[2][index] + par[3][index]*np.sqrt(1E3/T)

            # H --> Henry's constant

                H= (np.exp((1-par[1][index])*np.log(f0_H2O) + par[1][index]*np.log(R*T/M*rho0_H2O) + 2*rho0_H2O*dB))/1


                #H=np.real(H)
                # Derivation of activity coefficient, gamma

                labda_c = lab[1][index] + lab[2][index]*T + lab[3][index]/T + lab[4][index]*p + lab[5][index]/p + \
                          lab[6][index]*p/T + lab[7][index]*(T/p**2)+lab[8][index]*(p/(630-T)) +\
                          lab[9][index]*T*np.log(p) + lab[10][index]*(p/T**2)+lab[11][index]*T*p**2+lab[12][index]*p*T

                ksi_c = ksi[1][index] + ksi[2][index]*T + ksi[3][index]*p/T + ksi[4][index]*p/(630-T)
                #molality=50
                m_c = molality
                m_a = molality
                gamma = np.exp((2*m_c*labda_c) + (m_a*m_c*ksi_c)) #activity coefficient
                #print(m_c,m_a,gamma,molality)
                if labda_c== 0.0:
                    #gamma=10000
                    H=10000000

                phi_c[i] = H*gamma/p #aqeous phase fugacity
               # print('lksaldklsakdsa', H,gamma,labda_c, phi_c[i])

                #print(mix.names[i], H)

            elif mix.names[i] == 'H2O':  # for H2O: follow "true equilibrium" by Spycher (described in Ziabaksh)
                p0 = 1  # reference pressure of 1 bar
                V_H2O = 18.1  # average partial molar volume over pressure interval p-p0
                #phi_c[i] = (K0_H2O*p) * np.exp((p - p0) * V_H2O / (R * T)) / p

                phi_c[i] =(f0_H2O/p)#*np.exp((p-p0)*V_H2O/(R*T)) #aqeous phase water fugacity
                #0.86
                #print('fw',f0_H2O,phi_c[i])



    return phi_c

def Henry(p, T, components, molality):
    # Activity-fugacity model following Ziabaksh
    mix = Mix(components)
    NC = np.size(components)
    Tc=  10E-12
    Pc = 10E-12
    # parameters for partitioning between vapour and aqueous phase
    for i in range(0,NC):
        if mix.names[i]=='H2O':
            Tc = mix.Tc[i]  # water critical T [K]
            Pc = mix.Pc[i]  # water critical p [bar]
    tau = 1-T/Tc  # 1-T_r
    a = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502]
    #a = [-7.85951783, 0, 0, 0, 0, 0]  # constants in P_s
    P_s =Pc*np.exp(Tc/T*(a[0]*tau + a[1]*tau**1.5 + a[2]*tau**3 + a[3]*tau**3.5 + a[4]*tau**4 + a[5]*tau**6.5)) #saturation pressure (Shibue ,2003)
    #print('Ps',P_s)

    tc = T-273.15  # temperature in Celsius
    V0 = (1 + 18.159725E-3*tc)/(0.9998396 + 18.224944E-3*tc - 7.922210E-6*tc**2 - 55.44846E-9*tc**3 + 149.7562E-12*tc**4 - 393.2952E-15*tc**5)
    B = 19654.320 + 147.037*tc - 2.21554*tc**2 + 1.0478E-2*tc**3 - 2.2789E-5*tc**4
    A1 = 3.2891 - 2.3910E-3*tc + 2.8446E-4*tc**2 - 2.8200E-6*tc**3 + 8.477E-9*tc**4
    A2 = 6.245E-5 - 3.913E-6*tc - 3.499E-8*tc**2 + 7.942E-10*tc**3 - 3.299E-12*tc**4
    V = V0 - V0*p/(B+A1*p+A2*p**2)  # volume of pure water at p [cm3/g], (Fine and Millero, XXXX)
    M = 18.0152  # molar mass of water
    R = 8.3145E1  # cm3 bar K-1 mol-1

    tf = (T - 273.15) * (9 / 5) + 32
    phi_w = 0.9958 + 9.6833E-5 * tf - 6.175E-7 * tf ** 2 - 3.08333E-10 * tf ** 3
    f0_H2O = phi_w * P_s * np.exp((p - P_s) * M * V / (R * T))  # fugacity of pure water, (King et al, xxx)

    #print('fo_H2O',f0_H2O,P_s*np.exp((p-P_s)*M*1.227568/(R*T)))
    rho0_H2O = 1/V #density of water
    K0_H2O = np.exp(-2.209 + 3.097E-2*tc - 1.098E-4*tc**2 + 2.048E-7*tc**3)  # equilibrium constant for H2O at 1 bar, (Spycher et al, XXXX)
    lab = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                     # coefficients for lambda
           [-5.7066455E-1,  -2.143686,      0.513068, 0.52862384   ,  0.52862384, 0, 0, 0, 0, 0, 0, 0, -0.0652869, 1.03658689,-2.0939363],                       # c1
           [7.2997588E-4,   2.598765E-3,   -0.000958, -1.0298104E-3, -1.0298104E-3, 0, 0, 0, 0, 0, 0, 0, 1.6790636E-4, -1.1784797E-3, 3.1445269E-3],                # c2*T
           [1.5176903E2,    4.6942351E2,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40.838951, -1.7754826E2,3.913916E2],                                                  # c3/T
           [3.1927112E-5,  -4.6849541E-5,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.5313285E-4,-2.9973977E-7],                                                          # c4*P
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],                                                                      # c5/P
           [-1.642651E-5,   0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.9266518E-2, 0, -1.5918098E-5],                                                          # c6*(P/T)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],                                                                     # c7*(T/P^2)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.1157167E-2, 0, 0 ],                                                          # c8*(P/(630-T))
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.5486487E-6, 0, 0 ],                                                          # c9*T*ln(P)
           [0,              0,              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.775165E1, 0],                                                           # c10*(P/T^2)
           [0,              -8.4616602E-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],                                                                     # c11*P^2T
           [0,              1.095219E-6,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                                                                      # c12*PT

    ksi = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                   # coefficients for ksi, Calibrated second-order interaction parameters
           [-2.9990084E-3, -1.0165947E-2, -0.007485, 0.0206946 , 0.0206946,  0, 0, 0, 0, 0, 0, 0, -1.144624E-2, 0.010274152, -6.3981858E-3],  # c1
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              2.8274958E-5,  0, 0 ],   # c2*T
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              1.3980876E-2,  0, 0 ],  # c6*(P/T)
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                              -1.4349005E-2, 0, 0 ]]  # c8*(P/(630-T))

    par = [['C1', 'C2', 'C3', 'iC4' , 'nC4','iC5','nC5','nC6','nC7','nC8','nC9','nC10','CO2', 'H2S','N2'],                                   # coefficients for Henry's cst
           [-0.1196, -0.6091, -1.1471, -1.6849 , -1.6849,  0, 0, 0, 0, 0, 0, 0, -0.114535, 0.77357854,-0.008194],                  # eta
           [-10.9926, -16.3482, -25.3879, -33.8492 , -33.8492,  0, 0, 0, 0, 0, 0, 0, -5.279063, 0.27049433,-5.175337],            # tau
           [14.4019, 20.0628, 28.2616, 36.1457, 36.1457, 0, 0, 0, 0, 0, 0, 0, 6.187967, 0.27543436, 6.906469],                    # beta
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                  # Gamma

    # construct fugacity-values
    phi_c = np.zeros(NC)
    K3 = np.zeros(NC)
    for i in range(0, NC):
        #print(mix.names[i])
        if mix.names[i]== 'C20':
            phi_c[i]=1E-20
        else:
            if mix.names[i] != 'H2O':  # for non-H2O components: calculate Henry's and activity coefficients
                index = lab[0][:].index(mix.names[i])
                dB = par[2][index] + par[3][index]*np.sqrt(1E3/T)

            # H --> Henry's constant

                H= np.exp((1-par[1][index])*np.log(f0_H2O) + par[1][index]*np.log(R*T/M*rho0_H2O) + 2*rho0_H2O*dB)



                # Derivation of activity coefficient, gamma

                labda_c = lab[1][index] + lab[2][index]*T + lab[3][index]/T + lab[4][index]*p + lab[5][index]/p + \
                          lab[6][index]*p/T + lab[7][index]*(T/p**2)+lab[8][index]*(p/(630-T)) +\
                          lab[9][index]*T*np.log(p) + lab[10][index]*(p/T**2)+lab[11][index]*T*p**2+lab[12][index]*p*T

                ksi_c = ksi[1][index] + ksi[2][index]*T + ksi[3][index]*p/T + ksi[4][index]*p/(630-T)

                m_c = molality
                m_a = molality
                gamma = np.exp((2*m_c*labda_c) + (m_a*m_c*ksi_c)) #activity coefficient
                #print(m_c,m_a,gamma,molality)
                if labda_c== 0.0:
                    #gamma=10000
                    H=1000


                phi_c[i] = H*gamma/p #aqeous phase fugacity
                #print('lksaldklsakdsa', H,gamma,labda_c, phi_c[i])

                #print(mix.names[i], H)
                K3[i] = ((mix.Pc[i] / p) * np.exp(5.373 * (1 + mix.w[i]) * (1 - mix.Tc[i] / T))) * (p / H)
            elif mix.names[i] == 'H2O':  # for H2O: follow "true equilibrium" by Spycher (described in Ziabaksh)
                K3[i] = 1000
                p0 = 1  # reference pressure of 1 bar
                V_H2O = 18.1  # average partial molar volume over pressure interval p-p0
                #phi_c[i] = (K0_H2O/p) * np.exp((p - p0) * V_H2O / (R * T)) / p

                phi_c[i] =1*(f0_H2O/(p))*np.exp((p-p0)*V_H2O/(R*T)) #aqeous phase water fugacity
                #0.86
                #print('fw',f0_H2O)





    return K3