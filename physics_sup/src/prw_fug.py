def fug_prw(components,z,X,T,p):
    dij = [
        [CO2, N2, H2O, C1, C2, C3, iC4, nC4, iC5, nC5, nC6, nC7, nC8, nC9, nC10, H2S],
        [0, 0, 0.1387397, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.10223, 0.096],
        [0, 0, 0.275, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.176],
        [0.1387397, 0.275, 0, 0.4907, 0.4911, 0.5469, 0.508, 0.508, 0.5, 0.5, 0.48, 0.48, 0.48, 0.5177695, 0.48, 0.12],
        [0.1, 0.1, 0.4907, 0, 0, 0, 0, 0, 0, 0, 0.028739997, 0.033919997, 0.036999997, 0.039659997, 0.0522, 0.85],
        [0.1, 0.1, 0.4911, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.07],
        [0.1, 0.1, 0.5469, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.07],
        [0.1, 0.1, 0.508, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.508, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.48, 0.02874, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.48, 0.03392, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.48, 0.037, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.1, 0.1, 0.48, 0.03966, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.10223, 0.1, 0.5177695, 0.0522, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.096, 0.176, 0.12, 0.085, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0, 0, 0, 0, 0, 0]]

    mix = Mix(components)
    NC = np.size(components)


    Kij = np.zeros((NC,NC))
    Tc=mix.Tcrit(z,components)
    Pc=mix.Pcrit(z,components)

    Tr=T/Tc
    Prw=p/221.2



    for i in range(0, NC):
        indexi = dij[0][:].index(components[i])+1 #
        for j in range(0, NC):
            indexj = dij[0][:].index(components[j])

            if components[i]==H2O:
                if components[j]==C1:
                    Kij[i][j]= 1.659*Tr*Prw-0.761
                elif components[j] == C2:
                    Kij[i][j]= 2.109*Tr*Prw-0.607
                elif components[j] == C3:
                    Kij[i][j]= -18.032*(Tr**2)*(Prw**2)+0.9441*Tr*Prw-1.208
                elif components[j] == nC4 or components[j] == iC4 :
                    Kij[i][j] = 2.8* Tr * Prw - 0.488
                elif components[j] ==N2:
                    Kij[i][j] = 0.402*Tr-1.586
                elif components[j] == H2S:
                    Kij[i][j] = 0.22 * Tr - 0.19
                elif components[j] == CO2:
                    Kij[i][j]= -0.074*Tr**2 +0.478*Tr-0.503
                elif components[j] == H2O:
                    Kij[i][j] = 0
                else:
                    #Kij[i][j] = (dij[indexi][indexj])
                    Kij[i][j]= (0.4* Tc/Pc)*Tr*Prw -0.8
            elif components[j]==H2O:
                if components[i] == C1:
                    Kij[i][j] = 1.659 * Tr * Prw - 0.761
                elif components[i] == C2:
                    Kij[i][j] = 2.109 * Tr * Prw - 0.607
                elif components[i] == C3:
                    Kij[i][j] = -18.032 * (Tr ** 2) * (Prw ** 2) + 0.9441 * Tr * Prw - 1.208
                elif components[i] == nC4 or components[j] == iC4:
                    Kij[i][j] = 2.8 * Tr * Prw - 0.488
                elif components[i] == N2:
                    Kij[i][j] = 0.402 * Tr - 1.586
                elif components[i] == H2S:
                    Kij[i][j] = 0.22 * Tr - 0.19
                elif components[i] == CO2:
                    Kij[i][j] = -0.074 * Tr ** 2 + 0.478 * Tr - 0.503
                elif components[i] == H2O:
                    Kij[i][j] = 0
                else:
                    #Kij[i][j] = (dij[indexi][indexj])
                    Kij[i][j] = (0.4 * Tc / Pc) * Tr * Prw - 0.8
            else:
                Kij[i][j]=(dij[indexi][indexj])

    '''
    for i in range(0, NC):
        indexi = dij[0][:].index(components[i]) + 1  #
        for j in range(0, NC):
            indexj = dij[0][:].index(components[j])
            Kij[i][j] = (dij[indexi][indexj])
    '''
    #print(Kij)

    mix.kij_cubic(Kij)
    eos=preos(mix,mixrule = 'qmr',volume_translation = False, water = True)

    fugaq, v1 = eos.logfugef(X, T, p, 'L')
    #print ('fugprw',fugaq)
    return fugaq