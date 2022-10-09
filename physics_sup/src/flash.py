from src.equilibriumresult import EquilibriumResult
from rrsolver import RachfordRice
from src.Activity import *
from rrsolver import index_vector as i_vec
from rrsolver import value_vector as v_vec


def obj(z, K, beta):

    f = (z * (1.0 - K)) / ((1.0 + (K - 1.0) * beta)+1e-50)
    f = np.sum(f)
    return f

def rachfordrice(beta, K, z):
    '''
    Solves Rachford Rice equation by Halley's method
    '''
    K1 = K - 1.
    S1 = np.dot(z, K) - 1.
    S2 = 1. - np.dot(z, 1 / K)
    singlephase = False

    if S1 < 0:
        beta = 0.
        D = np.ones_like(z)
        singlephase = True
    elif S2 > 0:
        beta = 1.
        D = 1 + K1
        singlephase = True
    it = 0
    e = 1.

    maxk = np.amax(K)
    mink = np.amin(K)

    V_low = 1 / (1 - maxk) + 1e-8
    V_high= 1 / (1 - mink) + 1e-8
    V_mid = (V_low + V_high) / 2

    while e > 1e-8 and it < 1000 and not singlephase:
        it += 1

        if obj(z, K, V_mid) * obj(z, K, V_low) > 0:
            V_low = V_mid
        else:
            V_high = V_mid

        V_mid = (V_high + V_low) / 2

        e = np.absolute(obj(z, K, V_mid))

        beta=V_mid
        D = 1 + beta * K1 + 1e-49



    #print(D)

    return beta, D, singlephase

def Gibbs_obj(v, phase, Z, T, P, eos, v10, v20):
    '''
    Objective function to minimize Gibbs energy in biphasic flash
    '''
    l = Z - v
    v[v < 1e-8] = 1e-8
    l[l < 1e-8] = 1e-8
    X = l / l.sum()
    Y = v / v.sum()

    lnfugl, v1 = eos.logfugef(X, T, P, phase[0], v10)
    lnfugv, v2 = eos.logfugef(Y, T, P, phase[1], v20)
    fugl = np.log(X) + lnfugl
    fugv = np.log(Y) + lnfugv
    fo = v * fugv + l * fugl
    f = fo.sum()
    df = fugv - fugl
    return f, df

def gdem(X, X1, X2, X3):
    dX2 = X - X3
    dX1 = X - X2
    dX = X - X1
    b01 = dX.dot(dX1)
    b02 = dX.dot(dX2)
    b12 = dX1.dot(dX2)
    b11 = dX1.dot(dX1)
    b22 = dX2.dot(dX2)
    den = b11 * b22 - b12 ** 2
    mu1 = (b02 * b12 - b01 * b22) / den
    mu2 = (b01 * b12 - b02 * b11) / den
    dacc = (dX - mu2 * dX1) / (1 + mu1 + mu2)
    return dacc


def flash(components, phase, Fc, z, T, P, K, eos, full_output=False):
    nc = eos.nc
    if len(z) != nc:
        raise Exception('Composition vector length must be equal to nc')

    e1 = 1
    itacc = 0
    it = 0
    it2 = 0
    n = 4
    ztemp = np.asarray(z)
    z = ztemp
    eps = 10e-12
    V = np.array([0, 0])
    X = np.zeros((2, nc))
    lnK = np.log(K)
    '''
    while e1 > 10e-8 and it2 < 1000:
        it += 1
        it2 += 1
        lnK_old = lnK


        rr = RachfordRice(2)
        ph_i = index_vector([0, 1])
        nPhases = len(ph_i)
        KK1 = K
        KK = value_vector(KK1)
        # KK=value_vector([3.08529149e+01, 5.07937049e-01 ,5.32085427e-01,1.31466045e-03, 2.09149452e-02 ,4.03032498e+01 ])
        # print(np.concatenate(K) )
        zz = value_vector(z)
        nu = value_vector([0] * nPhases)
        rr.solveRachfordRice(nPhases, ph_i, nc, KK, zz, nu, eps)

        # V,X,Fc=negative_flash(z,np.exp(lnK_old))

        V = np.array([nu[0], nu[1]])  # molar fraction of reference phase 0, phase 1, phase 2

        #print(V)

        X = z / (1 + V[1] * (K - 1))  # K[0] contains K for phase 1, K[1] = K for phase 2, etc.
        Y = K * X


        if np.all(V)==0:
            X=Y=z

        #print (V,X,Y)
        for i in range(0,np.size(phase)):
            if phase[i] == 'V':
                fugv, v2 = eos.logfugef(Y , T, P, phase[i])
                #print(phase[i], fugv)
            elif phase[i]== 'L':
                fugl, v1 = eos.logfugef(X, T, P, phase[i])
                #print(phase[i],fugl)
            else:
                molality=0
                'Activity'
                fugaq=np.log(Activity(P, T, components, molality))


                'Full-PR'
                #fugaq, v1 = eos.logfugef(Y, T, P, 'L')

                'Modified-PR'
                #fugaq=fug_prw(components,z,Y,T,P)


                #fugaq = np.exp(fugaq)
                #for i in range(0,nc):
                    #if fugaq[i]==0:
                        #fugaq[i]=10e-25



        if phase == ['V', 'L']:
            #K=np.exp(fugv)/np.exp(fugl)
            lnK =  fugl-fugv
        elif phase == ['L', 'V']:
            lnK =  fugl-fugv
        elif phase == ["V", "Aq"] or phase == ["Aq", "V"]:
            lnK=fugv-np.log(fugaq)
        elif phase == ['L', 'Aq']:
            # K=np.exp(fugl)/np.exp(fugaq)
            lnK = fugl - fugaq
        elif phase == ['Aq', 'L']:
            lnK = fugl - fugaq


        if it == (n - 3):
            lnK3 = lnK.flatten()
        elif it == (n - 2):
            lnK2 = lnK.flatten()
        elif it == (n - 1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0
            itacc += 1
            lnKf = lnK.flatten()
            dacc = gdem(lnKf, lnK1, lnK2, lnK3).reshape(lnK.shape)
            lnK += dacc



        K = np.exp(lnK)
        e1 = ((lnK - lnK_old) ** 2).sum()
        #print('e_LL',e1, K)
    '''

    if Fc == 2:

        bmin = max(np.hstack([((K * z - 1.) / (K - 1.))[K > 1], 0.]))
        bmax = min(np.hstack([((1. - z) / (1. - K))[K < 1], 1.]))
        beta = (bmin + bmax) / 2

        lnK = np.log(K)

        while e1 > 10e-8 and it2 < 1000:
            it += 1
            it2 += 1
            lnK_old = lnK

            beta, D, singlephase = rachfordrice(beta, K, z)

            x = z / D
            y = x * K

            x /= x.sum()
            y /= y.sum()
            X = np.block([[x], [y]])

            #print('X,Y',x,y,beta)

            for i in range(0, np.size(phase)):
                if phase[i] == 'V':
                    fugv, v2 = eos.logfugef(X[i], T, P, phase[i])
                elif phase[i] == 'L':
                    fugl, v1 = eos.logfugef(X[i], T, P, phase[i])
                else:
                    molality = 0
                    'Activity'
                    fugaq = np.log(Activity(P, T, components, molality))

                    'Full-PR'
                    #fugaq, v1 = eos.logfugef(X[i], T, P, 'L')

                    'Modified-PR'
                    #fugaq=fug_prw(components,z,X[i],T,P)


            if phase == ['V', 'L']:
                # K=np.exp(fugv)/np.exp(fugl)
                lnK = fugv - fugl
            elif phase == ['L', 'V']:
                lnK = fugl - fugv
            elif phase == ["V", "Aq"] or phase == ["Aq", "V"]:
                lnK = fugv - fugaq
            elif phase == ['L', 'Aq']:
                # K=np.exp(fugl)/np.exp(fugaq)
                lnK = fugl - fugaq
            elif phase == ['Aq', 'L']:
                lnK = fugl - fugaq

            # print(lnK)

            '''
            if it == (n - 3):
                lnK3 = lnK
            elif it == (n - 2):
                lnK2 = lnK
            elif it == (n - 1):
                lnK1 = lnK
            elif it == n:
                it = 0
                itacc += 1
                dacc = gdem(lnK, lnK1, lnK2, lnK3)
                lnK += dacc
            '''
            K = np.exp(lnK)
            # print(K)
            e1 = ((lnK - lnK_old) ** 2).sum()
            #print('e',e1 )
            V = [beta, 1 - beta]
            if phase == ['V', 'L']:
                V = [1-beta, beta]
                X=np.block([[y], [x]])

    if Fc == 0:  # gas
        V[1] = 1.0
        V[0] = 0.0
        X[1] = np.zeros(nc)
        X[0] = z
        fugv, v2 = eos.logfugef(X[0], T, P, 'V')
        v1 = 0
    elif Fc == 1:  # liquid
        if phase == ['L', 'Aq'] or phase ==  ['V', 'Aq']:
            V[0] = 0.0
            V[1] = 1.0
            X[0] = z
            X[1] = np.zeros(nc)
        else:
            V[0] = 1.0
            V[1] = 0.0
            X[0] = np.zeros(nc)
            X[1] = z
            fugl, v1 = eos.logfugef(X[1], T, P, 'L')
            v2 = 0
    v1 = 1
    v2 = 1

    # V[0]=np.round(V[0],5)
    # V[1] = np.round(V[1], 5)

    if full_output:
        sol = {'T': T, 'P': P, phase[0]: V[1], phase[1]: V[0], 'error': e1, 'iter': it2,
               'state1': phase[0], 'X1': X[0], 'v1': v1,
               'state2': phase[1], 'X2': X[1], 'v2': v2}
        out = EquilibriumResult(sol)
        return out

    return X[0], X[1], V[0], v1, v2, K


def vlle(components, phase, z, T, P, K, eos, full_output=False):
    nc = eos.nc
    if len(z) != nc:
        raise Exception('Composition vector length must be equal to nc')
    e1 = 1
    itacc = 0
    it = 0
    it2 = 0
    n = 4
    ztemp = np.asarray(z)
    z = ztemp
    ktemp = np.asarray(K)
    K = ktemp
    X = np.zeros((3, nc))
    fug = np.zeros_like(X)
    V = np.zeros((1, 3))
    # lnK=np.zeros((2, nc))
    Fc = 3
    eps = 1e-8
    lnK = np.log(K)

    while e1 > 1e-8 and it2 < 500:
        it += 1
        it2 += 1

        lnK_old = lnK
        V_old = V
        #print('previos',np.concatenate(np.exp(lnK)))
        # K[0]=1/K[0]

        cc = 0
        for l in range(0, nc):
            if z[l] <= 10e-5:
                cc = cc + 1
                # print(cc,it2)

        # if nc-cc <=2 and it2>2:
        # print('21321321312312312312312321321dsfdddddddddddddddddddddddddddddddddd')
        # break

        # print(cc, it2)

        rr = RachfordRice(3)
        ph_i = i_vec([0, 1, 2])
        nPhases = len(ph_i)
        KK1 = np.concatenate(K)
        KK = v_vec(KK1)
        # KK=value_vector([3.08529149e+01, 5.07937049e-01 ,5.32085427e-01,1.31466045e-03, 2.09149452e-02 ,4.03032498e+01 ])
        # print(np.concatenate(K) )
        zz = v_vec(z)
        nu = v_vec([0] * nPhases)
        rr.solveRachfordRice(nPhases, ph_i, nc, KK, zz, nu, eps)

        # V,X,Fc=negative_flash(z,np.exp(lnK_old))

        V = np.array([nu[0], nu[1], nu[2]])  # molar fraction of reference phase 0, phase 1, phase 2
        #print(V)

        x0 = z / (1 + V[1] * (K[0] - 1) + V[2] * (K[1] - 1))  # K[0] contains K for phase 1, K[1] = K for phase 2, etc.
        x1 = K[0] * x0
        x2 = K[1] * x0
        X = np.block([[x0], [x1], [x2]])

        if V.all() == 10:
            K[0] = 1 / (K[0])
            KK1 = np.concatenate(K)
            KK = v_vec(KK1)
            rr.solveRachfordRice(nPhases, ph_i, nc, KK, zz, nu, eps)
            V = np.array([nu[0], nu[1], nu[2]])
            x0 = z / (1 + V[1] * (K[0] - 1) + V[2] * (
                    K[1] - 1))  # K[0] contains K for phase 1, K[1] = K for phase 2, etc.
            x1 = K[0] * x0
            x2 = K[1] * x0
            V = np.array([nu[1], nu[0], nu[2]])
            X = np.block([[x1], [x0], [x2]])
            # print(V)

        # print("FC",Fc,it2,e1)

        # if Fc== 3 and it2>100:
        # break

        # if Fc==-2 and it2>2 :
        # e1=0
        # break

        # if np.all(V)==0:
        # X[0,:]=X[1,:]=X[2,:]=z

        # print('LN1', np.exp(lnK))

        #print('V',  V)

        for i in range(0, np.size(phase)):
            # print(i, phase[i])
            if phase[i] == 'L' or phase[i] == 'V':
                fug[i, :], v2 = eos.logfugef(X[i, :], T, P, phase[i])
                # print(phase[i], fug[i, :], X[i, :])
            # elif  phase[i] == 'V' :
            # fug[i,:], v2 = eos.logfugef(X[i,:], T, P, phase[i])
            # print(phase[i], fug[i, :], X[i, :])
            elif phase[i] == 'Aq':
                molality = 0

                #fug[i, :], v2 = eos.logfugef(X[i,:], T, P,'L')
                #fug[i,:] = fug_prw(components,z,X[i,:],T,P)

                fug[i, :] = np.log(Activity(P, T, components, molality))
                # print(phase[i], fug[i, :], X[i, :])

            #print(i,phase[i], fug[i,:])

            # print(np.log(Activity(P, T, components, 0)), fug[i, :])

        # print('lnkold')

        for m in range(1, np.size(phase)):
            lnK[0, :] = fug[0, :] - fug[1, :]  # L-V
            lnK[1, :] = fug[0, :] - fug[2, :]  # L-Aq
            # print('result',phase[0],phase[m], lnK)

        # print('k_before', rr)
        # print('k_new', lnK)

        '''
        if it == (n - 3):
            lnK3 = lnK.flatten()
        elif it == (n - 2):
            lnK2 = lnK.flatten()
        elif it == (n - 1):
            lnK1 = lnK.flatten()
        elif it == n:
            it = 0
            itacc += 1
            lnKf = lnK.flatten()
            dacc = gdem(lnKf, lnK1, lnK2, lnK3).reshape(lnK.shape)
            lnK += dacc
        '''

        K = np.exp(lnK)
        #print(lnK)

        #e1 = ((lnK.flatten() - lnK_old.flatten()) ** 2).sum()

        e1 = ((V - V_old) ** 2).sum()

        #print('ee', it2,e1)

    #print('here', np.exp(lnK))

    return V, X, Fc


def multiphase_flash(components, phase, z, T, P, K, eos, full_output=False):
    NP = np.size(phase)
    if (NP == 2):  # perform 2 phase flash
        "do nothing"
    return
