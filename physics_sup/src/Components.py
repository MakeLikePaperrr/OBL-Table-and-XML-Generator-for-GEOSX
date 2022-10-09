from __future__ import division, print_function, absolute_import
import numpy as np

from src.constants import R



class component(object):
    '''
    Creates an object with pure component info

    Parameters
    ----------
    naime : str
        Name of the component
    Tc : float
        Critical temperature
    Pc : float
        Critical Pressure
    Zc : float
        critical compresibility factor
    Vc : float
        critical volume
    w  : float
        acentric factor
    Ant : list
        Antoine correlation parameters
    MW : float
        Molar Mass


    '''

    def __init__(self, name='None', Tc=0, Pc=0, Zc=0, Vc=0, w=0,MW=0, c=0, Ant=[0, 0, 0]):
        self.name = name
        self.Tc = Tc  # Critical Temperature in K
        self.Pc = Pc  # Critical Pressure in bar
        self.Zc = Zc  # Critical compresibility factor
        self.Vc = Vc  # Critical volume in cm3/mol
        self.w = w  # Acentric Factor
        self.MW = MW  # Molar mass
        self.Ant = Ant  # Antoine coefficeint, base e = 2.71 coeficientes de antoine, list or array
        self.c = c
        self.nc = 1


class mixture(object):
    '''

    Creates an object that cointains info about a mixture.

    Parameters

    name : list
        Name of the component
    Tc : list
        Critical temperature
    Pc : list
        Critical Pressure
    Zc : list
        critical compresibility factor
    Vc : list
        critical volume
    w  : list
        acentric factor
    Ant : list
        Antoine correlation parameters
    Methods
    -------
    add_component : adds a component to the mixture
    '''

    def __init__(self, component1, component2):
        self.names = [component1.name, component2.name]
        self.Tc = [component1.Tc, component2.Tc]
        self.Pc = [component1.Pc, component2.Pc]
        self.Zc = [component1.Zc, component2.Zc]
        self.w = [component1.w, component2.w]
        self.MW = [component1.MW, component2.MW] # Molar mass
        self.Ant = [component1.Ant, component2.Ant]
        self.Vc = [component1.Vc, component2.Vc]
        self.c = [component1.c, component2.c]
        self.nc = 2



    def add_component(self, component):
        """
        Method that add a component to the mixture
        """
        self.names.append(component.name)
        self.Tc.append(component.Tc)
        self.Pc.append(component.Pc)
        self.Zc.append(component.Zc)
        self.Vc.append(component.Vc)
        self.w.append(component.w)
        self.MW.append(component.MW)
        self.Ant.append(component.Ant)
        self.c.append(component.c)
        self.nc += 1

    def kij_cubic(self, k):
        '''
        Method that add kij matrix for QMR mixrule. M

        Parameters
        ----------
        k: array like
            matrix of interactions parameters

        '''

        self.kij = k

    def Peneloux(self,components):
        """
        Method that computes Peneloux volume shift parameter (1982)
        """
        NC = np.size(components)
        z_ra=np.zeros(NC)
        c = np.zeros(NC)
        for i in range(0, NC):
            z_ra[i] = 0.29056 - 0.08775 * self.w[i]
            c[i] = (0.50033 * R * self.Tc[i] / self.Pc[i] * (0.25969 - z_ra[i]))
            #print(z_ra[i])
        return c



    def Tcrit(self,z, components):
        '''
        Method that computes critical temparture of a mixture.

        Parameters
        ----------

        '''

        NC = np.size(components)
        Zvi = np.zeros(NC)
        for i in range(0,NC):
            Zvi[i]= z[i] * self.Vc[i]
        sum_Zvi=np.sum(Zvi)
        lamda=Zvi/sum_Zvi
        Tc_mix=np.sum(lamda*self.Tc)
        return Tc_mix

    def Pcrit(self,z,components):
        '''
        Method that computes critical pressure of a mixture.

        Parameters
        ----------

        '''
        NC = np.size(components)
        pTc=0
        pPc=0
        w_avg=0
        for i in range(0, NC):
            pTc =pTc+ z[i] * self.Tc[i]
            pPc = pPc+z[i] * self.Pc[i]
            w_avg =w_avg+ z[i] * self.w[i]
        Tc_mix = self.Tcrit(z,components)
        Pc_mix=pPc*(1+(5.808+4.93*w_avg)*(Tc_mix/pTc -1))
        return Pc_mix

    def Vcrit(self,z,components):
        '''
        Method that computes critical volume of a mixture.

        Parameters
        ----------

        '''
        NC = np.size(components)
        Zc=0
        for i in range(0, NC):
            Zc = Zc +z[i] * self.Zc[i]

        Tc = self.Tcrit(z,components)
        Pc = self.Pcrit(z, components)
        V_crit=Zc*Tc*R*10000/Pc
        #print('here',Zc)
        return V_crit























