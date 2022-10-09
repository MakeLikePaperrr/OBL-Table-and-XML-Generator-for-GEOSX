import numpy as np
from physics_sup.EOS import k_initial
from physics_sup.EOS import k_update

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def HC_normalize(V, x, components, phases):
    # only hydrocarbon componentss
    xi = np.zeros(np.size(components) - ("NaCl" in components) - ("Hydrate" in components))
    yi = np.zeros(np.size(components) - ("NaCl" in components) - ("Hydrate" in components))
    components = [None] * np.size(xi)
    m = 0
    for i in range(0, np.size(components)):
        if components[i] != "NaCl" and components[i] != "Hydrate":
            yi[m] = x[phases.index("Gas"), i]
            xi[m] = yi[m]
            components[m] = components[i]
            m += 1
    # normalize overall composition to only hydrocarbon phases
    zi = yi
    return xi, yi, zi, components

def ZiabakhshGanji_2P(p, T, zc, Cm, components, phases, min_z):
    # normalize to non-salt components
    #zc, components = normalize.normalize(z, components, "NaCl")

    NC = np.size(zc)  # number of components (normalized)
    NP = np.size(phases)

    def k(phases):  # calculates K-values for specified phases
        NP = np.size(phases)

        K = np.zeros((NP-1, NC))

        # calculation of initial or updated K-values
        if initial:  # K-values for initially specified phases
            for m in range(1, NP):
                K[m-1, :] = k_initial.ideal(p, T, zc, components, [phases[0], phases[m]])

        else:  # updated fugacity and K-values for phases present according to flash results with initial guess
            # calculate salt molality
            if "NaCl" in components:
                x_solute = z[components.index("NaCl")]
                x_water = V[phases.index("Aq")]*x[phases.index("Aq"), components.index("H2O")]*\
                          (1-z[components.index("NaCl")])  # overall molar fraction of aqueous phase - water
                molality = 55.509*x_solute/x_water  # [mol NaCl/kg H2O]
            else:
                molality = Cm #1.368 80 kg nacl per 1000kg water, corresponds to 1368mol/1000kg
            # print("molality", molality)

            xi, yi, zi, c = HC_normalize(V, x, components, phases)
            #('Outcome of normalization',xi,yi,zi,c)
            phi_c = np.zeros((NP, NC))

            # HC phase fugacity
            phi_c[phases.index("Gas"), :] = k_update.fugacity(p, T, xi, yi, zi, components, "Gas", molality)

            for m in range(0, NP):
                if phases[m] != "Gas":  # fugacity coefficients of non-HC phases
                    phi_c[m, :] = k_update.fugacity(p, T, xi, yi, zi, components, phases[m], molality)
                if m > 0:
                    if np.count_nonzero(phi_c[m, :]) != 0:  # if a row of K contains only zeros, then only one HC
                        K[m-1, :] = phi_c[0, :]/phi_c[m, :] #Divide out the different Hcs

        return K, phases

    def flash(zc, K, phases):
        # performs flash with calculated K-values.  Runs next iteration if any phase fraction is negative
        # ph0 is phases taken into account. Output will be ph1: positive phases in negative flash

        NP = np.size(phases)
        #print(zc,K)
        if NP == 2:  # perform 2 phase flash
            maxk = np.amax(K)
            mink = np.amin(K)
            eps = min_z
            V_low = 1 / (1 - maxk) + eps
            V_high = 1 / (1 - mink) - eps
            V_mid = (V_low + V_high) / 2

            def obj(zc, K, V):
                f = (zc * (1 - K)) / (1 + (K - 1) * V)
                f = f[0]
                f= np.sum(f)
                #f = f[0]+f[1]+f[2]
                return f

            max_iter = 200
            for i in range(1, max_iter):
                if obj(zc, K, V_mid) * obj(zc, K, V_low) < 0:
                    V_high = V_mid
                else:
                    V_low = V_mid
                V_mid = (V_high + V_low) / 2
                if np.absolute(obj(zc, K, V_mid)) < 1E-8:
                    #print(i,'iterations needed')
                    break
            if i >= max_iter:
                print("Flash warning!!!")

            V = np.array([1 - V_mid, V_mid])  # V_mid is molar fraction of non-reference phase
            x1 = zc / (V[1] * (K - 1) + 1)
            x2 = K * x1
            x = np.block([[x1], [x2]])


            if np.sum(V < 0) == 1:  # negative phase fractions
                x = np.zeros((NP, NC))
                for i in range(0, np.size(V)):

                    if V[i] < 0:
                        V[i] = 0
                    else:
                        V[i] = 1
                        x[i, :] = zc[:]  # single phase -> mole fractions equal to composition
            ph1 = phases

        return V, x, ph1

    # Initial guess with ideal K-values
    initial = 1
    # Initial guess with ideal K-values, with only one HC phase
    # needs to be discussed
    K, phases = k(phases)
    V, x, phases = flash(zc, K, phases)

    #print("V, x [initial]", V, x)  # [V1 V2 ...] [x1, x2, x...]
    # print(ph)
    if np.count_nonzero(V) == 1:  # single phase
        converged = 1
    else:
        converged = 0
    initial = 0

    iter = 0
    max_iter = 200
    # converged = 1  # use only initial guess

    while converged == 0:
        # Updated K-Values with ideal K-values
        K, phases = k(phases)
        # print(K, ph)
        # print("Updated K-values", K)
        if np.count_nonzero(K) == 0:  # only HC phases remain, but unable to calculate fugacity of two phases
            #print('this if/else statement')
            converged = 1
            V1 = [1, 0]
            x1 = np.zeros((NP, NC))
            x1[0, :] = zc
        else:
            V1, x1, phases = flash(zc, K, phases)
            test = np.amax(np.abs(V1 - V))
            #print('test',test)

        if np.size(V1) < np.size(V):
            V1 = np.append(V1, 0)

        if np.amax(np.abs(V1 - V)) < 1E-6:
            converged = 1
        elif np.count_nonzero(V1) == 1:  # single phase \here a 1 was missing
            converged = 1
        iter = iter + 1
        if iter>max_iter:
            print('Flash Warning')
        V, x = V1, x1
        # print("V, x [update]", V, x)
    #print("iterations: ", iter)
    NC = np.size(components)

    # V and x for each specified phase
    V1 = np.zeros(np.size(phases))
    x1 = np.zeros((np.size(phases), np.size(components)))
    for i in range(0, np.size(NP)):
        V1[phases.index(phases[i])] = V[i]
        x1[phases.index(phases[i]), :] = x[i, :]
    #print(K,V,x)
    return K, V, x

