

from __future__ import division, print_function, absolute_import
import numpy as np

def alpha_vdw():
    return 1.

#Redlich Kwong's alphas function
def alpha_rk(T, Tc):
    return np.sqrt(T/Tc)**-0.5

#Soaves's alpha function
def alpha_soave(T, k, Tc):
    #print('a3', (1+k*(1-np.sqrt(T/Tc)))**2)
    return (1+k*(1-np.sqrt(T/Tc)))**2

#def alpha_prw(T, k, Tc):
    #print('a3', (1+k*(1-np.sqrt(T/Tc)))**2)
    #return (1+k*(1-np.sqrt(T/Tc)))**2


def alpha_prw(T, k, Tc,components):
    Tr = T/ Tc
    NC =len(components)
    alpha=np.zeros(NC)
    #print('NOW',(T/ Tc[0])**0.5)
    c=0
    for i in range(0, NC):
        c=c+1
        if components[i] == 'H2O':
            if (T/ Tc[i])**0.5 <0.85:
                alpha[i]=(1.0085677+0.82154*(1-(T/Tc[i])**0.5))**2
                #alpha[i] = (1 + k[i] * (1 - np.sqrt(T / Tc[i]))) ** 2
                #alpha[i] = (1 + k[i] * (1 - np.sqrt(T / Tc[i]))) ** 2

            else:
                alpha[i] = (1 + k[i] * (1 - np.sqrt(T / Tc[i]))) ** 2
        else:
            alpha[i] = (1 + k[i] * (1 - np.sqrt(T / Tc[i]))) ** 2
        #print('a2', T/ Tc)
    return alpha


#SV's alphas function
def alpha_sv(T, ksv, Tc):
    ksv = ksv.T
    k0 = ksv[0]
    k1 = ksv[1]
    Tr = T/ Tc
    sTr = np.sqrt(Tr)
    return (1+(k0+k1*(0.7-Tr)*(1+sTr))*(1-sTr))**2