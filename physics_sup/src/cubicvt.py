

from __future__ import division, print_function, absolute_import
from src.qmr import *
from src.alphas import alpha_soave, alpha_sv, alpha_rk
from src.constants import R

class vtcubicm():

    
    def __init__(self, mix, c1, c2, oma, omb, alpha_eos, mixrule):
        
        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        
        self.Tc = np.array(mix.Tc, ndmin = 1) 
        self.Pc = np.array(mix.Pc, ndmin = 1)
        self.w = np.array(mix.w, ndmin = 1)
        self.b = self.omb*R*self.Tc/self.Pc
        self.c =np.array(mix.c, ndmin = 1)
        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])
        
        if mixrule == 'qmr':
            self.mixrule = qmr  
            if hasattr(mix, 'kij'):
                self.kij = mix.kij
                self.mixruleparameter = (mix.kij,)
            else: 
                self.kij = np.zeros([self.nc, self.nc])
                self.mixruleparameter = (self.kij, )

        else: 
            raise Exception('Mixrule not valid')
            
            
    #Cubic EoS methods    
    def a_eos(self,T):

        alpha = self.alpha_eos(T,self.k,self.Tc)
        a  = self.oma*(R*self.Tc)**2*alpha/self.Pc

        return a
    
    def _Zroot(self, A, B, C):
    
        a1 = (self.c1+self.c2-1)*B-1 + 3 * C
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a2 += 3*C**2 + 2*C*(-1 + B*(-1 + self.c1 + self.c2))
        a3 = A*(-B + C) + (-1-B+C)* (C +self.c1 * B)*(C+self.c2*B)
        
        Zpol=[1,a1,a2,a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots>(B - C)]
        return Zroots
        
    def Zmix(self, X, T, P):

        a = self.a_eos(T)
        c = self.c
        am, bm, ep, ap, bp = self.mixrule(X,T, a, self.b,*self.mixruleparameter)
        cm = np.dot(X, c)
        RT  = R * T
        A = am*P/RT**2
        B = bm*P/RT
        C = cm*P/RT
        #
        return self._Zroot(A,B,C)

    def density(self, X, T, P, state):

        if state == 'L':
            Z=min(self.Zmix(X,T,P))
        elif state == 'V':
            Z=max(self.Zmix(X,T,P))
        return P/(R*T*Z)
    
    def logfugef(self, X, T, P, state, v0 = None):

        c1 = self.c1
        c2 = self.c2
        
        b = self.b
        a = self.a_eos(T)
        c = self.c
        cm = np.dot(X, c)
        
        am, bm, ep, ap, bp = self.mixrule(X, T, a, b, *self.mixruleparameter)
        
        if state == 'V':
            Z=max(self.Zmix(X,T,P))
        elif state == 'L':
            Z=min(self.Zmix(X,T,P))
        #print('nroots', Z)
        RT = R * T    
        v = (RT*Z)/P
        B = bm * P/RT
        C = cm*P/RT
        Cp = c * P/RT

        logfug = (Z + C - 1) * (bp/bm) - np.log(Z + C - B) - Cp
        logfug -= (ep/(c2-c1))*np.log((Z+C+c2*B)/(Z+C+c1*B))

        
        return logfug, v
        
    def logfugmix(self, X, T, P, state, v0 = None):


        c1 = self.c1
        c2 = self.c2
        
        b = self.b
        a = self.a_eos(T)
        c = self.c
        cm = np.dot(X, c)
        
        am, bm, ep, ap, bp = self.mixrule(X, T, a, b, *self.mixruleparameter)
        
        if state == 'V':
            Z=max(self.Zmix(X,T,P))
        elif state == 'L':
            Z=min(self.Zmix(X,T,P))
            
        RT = R * T
        v = (RT*Z)/P
        A = am * P / RT**2
        B = bm * P/RT
        C = cm * P/RT

        logfug = Z - 1 - np.log(Z + C -B)
        logfug -= (A/(c2-c1)/B)*np.log((Z + C +c2*B)/(Z + C +c1*B))
        
        return logfug, v
    

        
    def _lnphi0(self, T, P):
        
        nc = self.nc
        a_puros = self.a_eos(T)
        Ai = a_puros*P/(R*T)**2
        Bi = self.b*P/(R*T)
        pols = np.array([Bi-1,-3*Bi**2-2*Bi+Ai,(Bi**3+Bi**2-Ai*Bi)])
        Zs = np.zeros([nc,2])
        for i in range(nc):
            zroot = np.roots(np.hstack([1,pols[:,i]]))
            zroot = zroot[zroot>Bi[i]]
            Zs[i,:]=np.array([max(zroot),min(zroot)])
    
        logphi=Zs - 1 - np.log(Zs.T-Bi)
        logphi -= (Ai/(self.c2-self.c1)/Bi)*np.log((Zs.T+self.c2*Bi)/(Zs.T+self.c1*Bi))
        logphi = np.amin(logphi,axis=0)
    
        return logphi
    

                 
# Peng Robinson EoS 
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854
class vtprmix(vtcubicm):
    def __init__(self, mix, mixrule = 'qmr'):
        vtcubicm.__init__(self, mix,c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_soave, mixrule = mixrule )
        
        self.k =  0.37464 + 1.54226*self.w - 0.26992*self.w**2
        
# Peng Robinson SV EoS 
class vtprsvmix(vtcubicm):
    def __init__(self, mix, mixrule = 'qmr'):
        vtcubicm.__init__(self, mix, c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_sv, mixrule = mixrule )
        if np.all(mix.ksv == 0):
            self.k = np.zeros([self.nc,2])
            self.k[:,0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2+0.0196553*self.w**3
        else:
             self.k = np.array(mix.ksv) 

# RKS - EoS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664
class vtrksmix(vtcubicm):
    def __init__(self, mix, mixrule = 'qmr'):
        vtcubicm.__init__(self, mix, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos = alpha_soave, mixrule = mixrule)
        self.k =  0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3
      
#RK - EoS        
class vtrkmix(vtcubicm):
    def __init__(self, mix, mixrule = 'qmr'):
        vtcubicm.__init__(self, mix, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos = alpha_rk, mixrule = mixrule)
    def a_eos(self,T):
        alpha=self.alpha_eos(T, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

