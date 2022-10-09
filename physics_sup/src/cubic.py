
from __future__ import division, print_function, absolute_import
import numpy as np
from src.qmr import qmr
from src.alphas import alpha_soave, alpha_sv, alpha_rk
from src.constants import R

class cubicm():


    def __init__(self, mix, c1, c2, oma, omb, alpha_eos, mixrule):

        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos
        self.emin = 2+ self.c1 + self.c2 + 2 * np.sqrt((1 + self.c1) * (1 + self.c2))

        self.Tc = np.array(mix.Tc, ndmin=1)
        self.Pc = np.array(mix.Pc, ndmin=1)
        self.w = np.array(mix.w, ndmin=1)
        self.b = self.omb * R * self.Tc / self.Pc
        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])

        if mixrule == 'qmr':
            self.mixrule = qmr
            if hasattr(mix, 'kij'):
                self.kij = mix.kij
                self.mixruleparameter = (mix.kij,)
            else:
                self.kij = np.zeros([self.nc, self.nc])
                self.mixruleparameter = (self.kij,)



    # Cubic EoS methods
    def a_eos(self, T):


        alpha = self.alpha_eos(T, self.k, self.Tc)
        #print('AAA2', alpha)
        a = self.oma * (R * self.Tc) ** 2 * alpha / self.Pc
        return a

    def a_eos_w(self, T):


        alpha = self.alpha_eos(T, self.k, self.Tc)
        a = self.oma * (R * self.Tc) ** 2 * alpha / self.Pc
        return a

    def _Zroot(self, A, B):
        a1 = (self.c1 + self.c2 - 1) * B - 1
        a2 = self.c1 * self.c2 * B ** 2 - (self.c1 + self.c2) * (B ** 2 + B) + A
        a3 = -B * (self.c1 * self.c2 * (B ** 2 + B) + A)
        Zpol = [1, a1, a2, a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots > B]
        return Zroots

    def Zmix(self, X, T, P):

        a = self.a_eos(T)

        am, bm, ep, ap, bp = self.mixrule(X, T, a, self.b, *self.mixruleparameter)

        RT = R * T
        A = am * P / RT ** 2
        B = bm * P / RT
        #print('Z1', am, bm, ep, ap, bp)
        return self._Zroot(A, B)

    def density(self, X, T, P, state):

        if state == 'L':
            Z = min(self.Zmix(X, T, P))
        elif state == 'V':
            Z = max(self.Zmix(X, T, P))
        return P / (R * T * Z)

    def logfugef(self, X, T, P, state, v0=None):

        b = self.b
        a = self.a_eos(T)
        am, bm, ep, ap, bp = self.mixrule(X, T, a, b, *self.mixruleparameter)
        if state == 'V':
            Z = max(self.Zmix(X, T, P))
        elif state == 'L':
            Z = min(self.Zmix(X, T, P))

        RT = R * T
        v = (RT * Z) / P
        B = (bm * P) / (RT)
        #print('here',bm,bp,B)
        logfug = (Z - 1) * (bp / bm) - np.log(Z - B)
        logfug -= (ep / (self.c2 - self.c1)) * np.log((Z + self.c2 * B) / (Z + self.c1 * B))

        return logfug, v

    def logfugmix(self, X, T, P, state, v0=None):
        a = self.a_eos(T)
        am, bm, ep, ap, bp = self.mixrule(X, T, a, self.b, *self.mixruleparameter)
        if state == 'V':
            Z = max(self.Zmix(X, T, P))
        elif state == 'L':
            Z = min(self.Zmix(X, T, P))
        RT = R * T
        v = (RT * Z) / P
        B = (bm * P) / (RT)
        A = (am * P) / (RT) ** 2

        logfug = Z - 1 - np.log(Z - B)
        logfug -= (A / (self.c2 - self.c1) / B) * np.log((Z + self.c2 * B) / (Z + self.c1 * B))

        return logfug, v

    def _lnphi0(self, T, P):

        nc = self.nc
        a_puros = self.a_eos(T)
        Ai = a_puros * P / (R * T) ** 2
        Bi = self.b * P / (R * T)
        pols = np.array([Bi - 1, -3 * Bi ** 2 - 2 * Bi + Ai, (Bi ** 3 + Bi ** 2 - Ai * Bi)])
        Zs = np.zeros([nc, 2])
        for i in range(nc):
            zroot = np.roots(np.hstack([1, pols[:, i]]))
            zroot = zroot[zroot > Bi[i]]
            Zs[i, :] = np.array([max(zroot), min(zroot)])
        logphi = Zs - 1 - np.log(Zs.T - Bi)
        logphi -= (Ai / (self.c2 - self.c1) / Bi) * np.log((Zs.T + self.c2 * Bi) / (Zs.T + self.c1 * Bi))
        logphi = np.amin(logphi, axis=0)

        return logphi


# Peng Robinson EoS
c1pr = 1 - np.sqrt(2)
c2pr = 1 + np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854


class prmix(cubicm):
    def __init__(self, mix, mixrule='qmr'):
        cubicm.__init__(self, mix, c1=c1pr, c2=c2pr,
                        oma=omapr, omb=ombpr, alpha_eos=alpha_soave, mixrule=mixrule)

        self.k = 0.37464 + 1.54226 * self.w - 0.26992 * self.w ** 2


# Peng Robinson SV EoS
class prsvmix(cubicm):
    def __init__(self, mix, mixrule='qmr'):
        cubicm.__init__(self, mix, c1=c1pr, c2=c2pr,
                        oma=omapr, omb=ombpr, alpha_eos=alpha_sv, mixrule=mixrule)
        if np.all(mix.ksv == 0):
            self.k = np.zeros([self.nc, 2])
            self.k[:, 0] = 0.378893 + 1.4897153 * self.w - 0.17131838 * self.w ** 2 + 0.0196553 * self.w ** 3
        else:
            self.k = np.array(mix.ksv)

    # RK - EoS

c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664

class rksmix(cubicm):
    def __init__(self, mix, mixrule='qmr'):
        cubicm.__init__(self, mix, c1=c1rk, c2=c2rk,
                        oma=omark, omb=ombrk, alpha_eos=alpha_soave, mixrule=mixrule)
        self.k = 0.47979 + 1.5476 * self.w - 0.1925 * self.w ** 2 + 0.025 * self.w ** 3


# RKS- EoS
class rkmix(cubicm):
    def __init__(self, mix, mixrule='qmr'):
        cubicm.__init__(self, mix, c1=c1rk, c2=c2rk,
                        oma=omark, omb=ombrk, alpha_eos=alpha_rk, mixrule=mixrule)

    def a_eos(self, T):
        alpha = self.alpha_eos(T, self.Tc)
        return self.oma * (R * self.Tc) ** 2 * alpha / self.Pc

