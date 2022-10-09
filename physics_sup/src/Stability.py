from src.Kval import *
from src.Activity import *
from scipy.optimize import minimize


def tpd(X, state, Z, T, P, model,components, v0=[None, None]):
    v1, v2 = v0
    if state == 'L' or state == 'V':
        logfugX, v1 = model.logfugef(X, T, P, state, v1)
    else:
        molality = 0
        logfugX = Activity(P, T, components, molality=0)

    logfugZ, v2 = model.logfugef(Z, T, P, 'L', v2)
    di = np.log(Z) + logfugZ
    tpdi = X * (np.log(X) + logfugX - di)
    return np.sum(np.nan_to_num(tpdi))


def tpd_obj(a, T, P, di, model, state, v0):
    W = a ** 2 / 2  # cambio de variable a numero de moles
    w = W / W.sum()  # normalizacion a fraccion molar

    logfugW, _ = model.logfugef(w, T, P, state, v0)

    dtpd = np.log(W) + logfugW - di
    tpdi = np.nan_to_num(W * (dtpd - 1.))
    tpd = 1. + tpdi.sum()
    dtpd *= a / 2
    return tpd, dtpd


def tpd_min(W, Z, T, P, model, stateW, stateZ, vw=None, vz=None):
    """

    Found a minimiun of Michelsen's Adimentional tangent plane function

    tpd_min (W, Z, T, P, model, stateW, stateZ)

    Parameters
    ----------
    W : array_like
        mole fraction array of trial fase
    Z : array_like
        mole fraction array of overall mixture
    T :  absolute temperature, in K
    P:  absolute pressure in bar
    model : object create from mixture, eos and mixrule
    stateW : string
        'L' for liquid phase, 'V' for vapor phase
    stateZ : string
        'L' for liquid phase, 'V' for vapor phase
    vw, vz: float, optional
        initial volume value to compute fugacities of phases

    Returns
    -------
    w : array_like
        molar fraction of minimum
    f : float
        minimized tpd distance

    """
    nc = model.nc
    if len(W) != nc or len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')
    # valores de la fase de global
    Z=np.asarray(Z)
    Z[Z < 1e-8] = 1e-8
    logfugZ, vz = model.logfugef(Z, T, P, stateZ, vz)
    di = np.log(Z) + logfugZ

    alpha0 = 2 * W ** 0.5
    alpha0[alpha0 < 1e-8] = 1e-8  # hay que asegurarse que ninguna composicion
    # sea negativa
    alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw)
                     , jac=True, method='BFGS')
    W = alpha.x ** 2 / 2
    w = W / W.sum()  # composicion normalizada
    tpd = alpha.fun
    return w, tpd


def tpd_minimas(nmin, Z, T, P, model, stateW, stateZ, vw=None, vz=None):
    """

    Found nmin minimuns of Michelsen's Adimentional tangent plane function

    tpd_minimas (nmin, Z, T, P, model, stateW, stateZ)

    Parameters
    ----------
    nmin: int
        number of minimiuns to be founded
    Z : array_like
        mole fraction array of overall mixture
    T : float
        absolute temperature, in K
    P:  float
        absolute pressure in bar
    model : object
        create from mixture, eos and mixrule
    stateW : string
        'L' for liquid phase, 'V' for vapour phase
    stateZ : string
        'L' for liquid phase, 'V' for vapour phase
    vw, vz : float, optional
        if supplied volume used as initial value to compute fugacities

    Returns
    -------
    all_minima: tuple
        molar fractions arrays of minimums
    f_minima: tuple
        minimized tpd distance

    """
    nc = model.nc
    if len(Z) != nc:
        raise Exception('Composition vector lenght must be equal to nc')

    Z[Z < 1e-8] = 1e-8
    logfugZ, vz = model.logfugef(Z, T, P, stateZ, vz)
    di = np.log(Z) + logfugZ

    nc = model.nc
    all_minima = []
    f_minima = []

    # search from pures
    Id = np.eye(nc)
    alpha0 = 2 * Id[0] ** 0.5
    alpha0[alpha0 < 1e-5] = 1e-5  # no negative or zero compositions
    alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw)
                     , jac=True, method='BFGS')
    W = alpha.x ** 2 / 2
    w = W / W.sum()  # normalized composition
    tpd = alpha.fun
    all_minima.append(w)
    f_minima.append(tpd)

    for i in range(1, nc):
        alpha0 = 2 * Id[i] ** 0.5
        alpha0[alpha0 < 1e-5] = 1e-5  # hay que asegurarse que
        # ninguna composicion sea negativa
        alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw)
                         , jac=True, method='BFGS')
        W = alpha.x ** 2 / 2
        w = W / W.sum()  # normalized composition
        tpd = alpha.fun
        if alpha.success:
            if not np.any(np.all(np.isclose(all_minima, w, atol=1e-3), axis=1)):
                f_minima.append(tpd)
                all_minima.append(w)
                if len(f_minima) == nmin:
                    return tuple(all_minima), np.array(f_minima)

    # busqueda aleatoria
    niter = 0
    while len(f_minima) < nmin and niter < (nmin + 1):
        niter += 1
        Al = np.random.rand(nc)
        Al = Al / np.sum(Al)
        alpha0 = 2 * Al ** 0.5
        alpha0[alpha0 < 1e-5] = 1e-5  # hay que asegurarse que
        # ninguna composicion sea negativa
        alpha = minimize(tpd_obj, alpha0, args=(T, P, di, model, stateW, vw)
                         , jac=True, method='BFGS')
        W = alpha.x ** 2 / 2
        w = W / W.sum()  # normalized composition
        tpd = alpha.fun
        if alpha.success:
            if not np.any(np.all(np.isclose(all_minima, w, atol=1e-3), axis=1)):
                f_minima.append(tpd)
                all_minima.append(w)
                if len(f_minima) == nmin:
                    return tuple(all_minima), np.array(f_minima)

    while len(f_minima) < nmin:
        all_minima.append(all_minima[0])
        f_minima.append(f_minima[0])

    return tuple(all_minima), np.array(f_minima)

def simple_fluid_state(eos,z,T,p,components):
    mix = Mix(components)
    roots=eos.Zmix(z, T, p)
    n_roots=np.size(roots)

    if n_roots ==3:
        Zl = min(eos.Zmix(z, T, p))
        Zv = max(eos.Zmix(z, T, p))

        lnfugl, v1 = eos.logfugmix(z, T, p, 'L')
        lnfugv, v2 = eos.logfugmix(z, T, p, 'V')


        Delta_G= R*T*(np.exp(lnfugv)-np.exp(lnfugl))
        #print('G',Delta_G)
        if Delta_G>0:

            Fluid_State = "Subcooled Liquid"

        elif Delta_G<0:
            Fluid_State = "Superheated Vapor"
    else:
        Tc= mix.Tcrit(z, components)
        #Z = eos.Zmix(z, T, p)
        #RT=R*T
        #V=Z*RT*10000/p
        #Vc=mix.Vcrit(z,components)
        #print('G', V,Vc)
        if T < Tc:
            Fluid_State = "Subcooled Liquid"

        elif T>Tc:
            Fluid_State = "Superheated Vapor"

    return Fluid_State

def VaporLike(eos,z,T,p,components):

    e1 = 1
    e2 = 1

    '1. Calculate the mixture fugacity (fzi) using overall composition zi.'

    lnfugz, v0 = eos.logfugef(z, T, p, 'L')

    '2.Create a vapor-like second phase'

    'a) Use Wilson’s correlation to obtain initial Ki-values.'


    K = vapour_liquid(p, T, components)

    max_it = 10000
    eps_SSI = 10E-12
    eps_tpd=10E-4
    eps_K=10E-4
    it = 0
    convg = 'True'
    t=0
    while e1>eps_SSI:

        it=it+1

        'b) Calculate second-phase mole numbers, Yi'
        Yi = z *K
        #print('Yi', Yi)
        'c) Obtain the sum of the mole numbers,'
        Sv = np.sum(Yi)
        #print('SUMV', Sv)
        'd) Normalize the second-phase mole numbers to get mole fractions'
        yi = Yi/Sv
        'e) Calculate the second-phase fugacity (fyi) using the corresponding EOS and the previous composition.'
        lnfug1, v1 = eos.logfugef(yi, T, p, 'V')

        'f) Calculate corrections for the K-values:'

        R=((np.exp(lnfugz)*z) / (np.exp(lnfug1)*yi))*(1/Sv)

        K=K*R

        #K= (np.exp(lnfugz) / np.exp(lnfug1))

        #Sv_new=np.sum(K*z)
        'g) Check if:'
        'i) Convergence is achieved:'
        e1 = ((R-1)**2).sum()
        'ii) A trivial solution is approached'
        e2= (np.log(K)**2).sum()

        'If a trivial solution is approached, stop the procedure.'

        if (e2 < eps_K):

            convg='False'
            #print('trivial-v')
            break;
        elif (it>max_it):
            Sv = -1 # simple status
            break;
        #print('TPDv', t, -np.log(Sv))
        #print ('Sv', Sv , e1 ,e2)
        'If convergence has not been attained, use the new K-values and go back to step (b).'

        'Calculate TPD'

        t=tpd(yi, 'V', z, T, p, eos)

        #print('error', t, e1 , e2)

    if convg=='False':
        Sv = -1  # trivial solution

    else:
        if Sv>1:
            Sv=2 # stable gas phase
        elif Sv<=1:
            Sv=-1 # two phase
        #else:
            #Sv=-1 #check liquid phase

    return Sv,it

def LiquidLike(eos,z,T,p,components):
    roots = eos.Zmix(z, T, p)
    n_roots = np.size(roots)
    #print('roots',roots)
    e1 = 1
    e2 = 1
    '1. Calculate the mixture fugacity (fzi) using overall composition zi.'

    lnfugz, v0 = eos.logfugef(z, T, p, 'V')

    '2.Create a Liquid-like second phase'

    'a) Use Wilson’s correlation to obtain initial Ki-values.'


    K = vapour_liquid(p, T, components)
    max_it = 10000
    eps_SSI = 10E-12
    eps_tpd = 10E-4
    eps_K = 10E-4
    convg = 'True'
    it = 0
    it=0
    while e1>eps_SSI:

        it=it+1

        'b) Calculate second-phase mole numbers, Yi'

        Yi = z / K
        #print('Yi', Yi)

        'c) Obtain the sum of the mole numbers,'

        Sl = np.sum(Yi)
        #print('SUMV', Sv)

        'd) Normalize the second-phase mole numbers to get mole fractions'

        yi = Yi/Sl


        'e) Calculate the second-phase fugacity (fyi) using the corresponding EOS and the previous composition.'

        lnfugL, v1 = eos.logfugef(yi, T, p, 'L')


        'f) Calculate corrections for the K-values:'

        R = ((np.exp(lnfugL) * yi) / (np.exp(lnfugz) * z)) * (Sl)
        K = K * R


        'g) Check if:'

        'i) Convergence is achieved:'
        e1 = ((R-1.0)**2).sum()
        'ii) A trivial solution is approached'
        e2= (np.log(K)**2).sum()


        'If a trivial solution is approached, stop the procedure.'
        if (e2 < eps_K):
            convg='False'
            Sl = -1 #trivial solution
            #print('trivial-l')

            break;
        elif (it > max_it):
            Sl = -1 #simple status
            break;

        #print('Sl', Sl, e1, e2)
        'If convergence has not been attained, use the new K-values and go back to step (b).'

        'Calculate TPD'
        t = tpd(yi, 'L', z, T, p, eos)
        #print('TPDl',t)
    if convg =='False':
        Sl=-1
    else:
        if Sl>1:
            Sl=2 # stable liquid phase
            print(Sl)
        elif Sl<=1:
            Sl=-1 # two phase
        else:
            Sl = -1 # simple status

    return Sl,it

def stability(eos,z,T,p,components):
    it_v=0
    it_l=0
    K=vapour_liquid(p,T,components)

    Sv,it_v=VaporLike(eos, z, T, p, components)
    if Sv==0:
        state = 'Single Phase Vapor'
        Fc=0
    elif Sv==2:
        state = 'Vapor-Like Two Phase'
        Fc = 2
    elif Sv==-1: # check liquid phase
        Sl,it_l = LiquidLike(eos, z, T, p, components)

        if Sl==1:
            state = 'Single Phase Liquid'
            Fc = 1
        elif Sl==2:
            state = 'Liquid-Like Two Phase'
            Fc = 2
        elif Sl==-1: #trivial - simple state
            state = simple_fluid_state(eos, z, T, p, components)
            if state=="Subcooled Liquid":
                Fc=1 #liquid
            else:
                Fc=0 # gas

    it = it_v+it_l

    'Fc : Flash condition'
    return Fc,K,state,it

def TrialPhase(eos,z,T,p,components,phase):

    e1 = 1
    e2 = 1

    '1. Calculate the mixture fugacity (fzi) using overall composition zi.'

    print(z)

    '2.Create a vapor-like second phase'

    'a) Use Wilson’s correlation to obtain initial Ki-values.'

    if phase == 'L':
        K = vapour_liquid(p, T, components) # create L
        lnfugz, v0 = eos.logfugef(z, T, p, 'L')
    elif phase == 'V':
        K = 1/vapour_liquid(p, T, components) # create V # lasrt changed
        lnfugz, v0 = eos.logfugef(z, T, p, 'V')


    max_it = 10000
    eps_SSI = 10E-12
    eps_tpd=10E-4
    eps_K=10E-4
    it = 0
    convg = 'True'
    t=0

    while e1>eps_SSI:

        it=it+1

        'b) Calculate second-phase mole numbers, Yi'
        Yi = z * K
        #print('Yi', Yi)
        'c) Obtain the sum of the mole numbers,'
        Sv = np.sum(Yi)
        #print('SUMV', Sv)
        'd) Normalize the second-phase mole numbers to get mole fractions'
        yi = Yi/Sv
        'e) Calculate the second-phase fugacity (fyi) using the corresponding EOS and the previous composition.'
        if phase == 'L':
            lnfugV, v1 = eos.logfugef(yi, T, p, 'V')
            R = ((np.exp(lnfugz) * z) / (np.exp(lnfugV) * yi)) * (1/Sv)

        elif phase == 'V':
            lnfugL, v1 = eos.logfugef(yi, T, p, 'L')
            R = ((np.exp(lnfugL) * yi) / (np.exp(lnfugz) * z)) * (Sv)

        'f) Calculate corrections for the K-values:'

        if phase == 'L':
            K=K*R
            #print(K)
        elif phase == 'V':
            K=(K)/R
            #print(K)



        #K= (np.exp(lnfugz) / np.exp(lnfug1))

        #Sv_new=np.sum(K*z)
        'g) Check if:'
        'i) Convergence is achieved:'
        e1 = ((R-1)**2).sum()
        'ii) A trivial solution is approached'
        e2= (np.log(K)**2).sum()
        #print('e2',e2)
        'If a trivial solution is approached, stop the procedure.'

        if (e2 < eps_K):
            convg='False'
            #print('trivial-v')
            break;
        elif (it>max_it):
            print('Max Iterations of Stability Test Reached')
            convg='False' # simple status
            break;
        #print('TPDv', t, -np.log(Sv))
       # print ('Sv', Sv , e1 ,e2)
        'If convergence has not been attained, use the new K-values and go back to step (b).'

    'Calculate TPD'
    if convg =='False':
        Sv=-1
        t=0
    else:
        t=tpd(yi, phase, z, T, p, eos,components)




        #print('error', t, e1 , e2)

    '''if convg=='False':
        Sv = -1  # trivial solution
    else:
        if Sv>1:
            Sv=2 # stable gas phase
        elif Sv<=1:
            Sv=-1 # two phase
        #else:
            #Sv=-1 #check liquid phase
    '''
    return Sv,t,it,K

def VAq(eos,z,T,p,K,components,phase):

    e1 = 1
    e2 = 1

    '1. Calculate the mixture fugacity (fzi) using overall composition zi.'



    '2.Create a vapor-like second phase'

    'a) Use Wilson’s correlation to obtain initial Ki-values.'
    NC=np.size(components)


    if phase == 'L': #If trial phase is Liquid (-> tested phase z is Aqeous)
        #lnfugz, v0 = eos.logfugef(z, T, p, phase)
        lnfugz = np.log(Activity(p, T, components, molality=0))
    elif phase == 'Aq': #If trial phase is Aqeous (-> tested phase z is Liquid)
        molality=0
        lnfugz, v0 = eos.logfugef(z, T, p, 'L')
        #lnfugz = np.log(Activity(p, T, components, molality=0))


    max_it = 1000
    eps_SSI = 10E-12
    eps_tpd= 0
    eps_K=10E-4
    it = 0
    convg = 'True'
    t=0

    while e1>eps_SSI:

        it=it+1

        'b) Calculate second-phase mole numbers, Yi'
        if phase == 'Aq':
            Yi = z * K
        elif phase == 'L':
            Yi = z / K

        #print('Yi', Yi)
        'c) Obtain the sum of the mole numbers,'
        Sv = np.sum(Yi)

        #print('SUMV', Sv)
        'd) Normalize the second-phase mole numbers to get mole fractions'
        yi = Yi / Sv

        #yi[yi == 0] = 10e-49
        'e) Calculate the second-phase fugacity (fyi) using the corresponding EOS and the previous composition.'
        if phase == 'Aq': #If trial phase is Aqeous (-> tested phase z is Liquid)

            lnfugL =np.log( Activity(p, T, components, molality=0))
            #lnfugL, v1 = eos.logfugef(yi, T, p, 'L')
            R = ((np.exp(lnfugz) * z) / (np.exp(lnfugL) * yi)) * (1/Sv)
            #print('residual',np.log(Yi)+lnfugL - np.log(z)-lnfugz,np.linalg.norm(R))
            #R = ((np.exp(lnfugL) * yi) / (np.exp(lnfugz) * z)) * (Sv)

        elif phase == 'L': #If trial phase is Liquid (-> tested phase z is Aqeous)
            molality = 0
            lnfugL, v1 = eos.logfugef(yi, T, p, 'L')
            #lnfugL = np.log(Activity(p, T, components, molality=0))
            #R = ((np.exp(lnfugz) * z) / (np.exp(lnfugL) * yi)) * (1 / Sv)
            R = ((np.exp(lnfugL) * yi) / ((np.exp(lnfugz) * z))) * (Sv)
            #print(R)

        'f) Calculate corrections for the K-values:'
        #R[R == 0] = 10e-25
        if phase == 'L': # Trial is Aq
            #print('K', K)
            K=K*R

            #print('R1',R)
            #print('R2',((np.exp(lnfugL) * yi) / ((np.exp(lnfugz) * z))) * (Sv))
            #print('R2', ((np.exp(lnfugL) * yi) / ((np.exp(lnfugz) * z))) * (Sv))
            #print('Knew', K)
        elif phase == 'Aq': #Trial is L
            #print('K',K)
            K=K*R




        #K= (np.exp(lnfugz) / np.exp(lnfugL))

        #Sv_new=np.sum(K*z)
        'g) Check if:'
        'i) Convergence is achieved:'
        e1 = ((R-1)**2).sum()
        'ii) A trivial solution is approached'
        e2= (np.log(K)**2).sum()
       #if math.isinf(e2):
            #e2= 10e-5

        'If a trivial solution is approached, stop the procedure.'

        if (e2 < eps_K):
            convg='False'
            #print('trivial-v')
            break;
        elif (it>max_it):
            convg='False' # simple status
            break;

        #print ('Sv', Sv , e1 ,e2)
        'If convergence has not been attained, use the new K-values and go back to step (b).'

    'Calculate TPD'
    if convg =='False':
        Sv=-1
        #t=tpd(yi, phase, z, T, p, eos,components)
        t = 0
    else:
        #print('error', yi, phase)
        t=tpd(yi, phase, z, T, p, eos,components)
        t=-np.log(Sv)
    #print('TPDv',t, -np.log(Sv))







    '''if convg=='False':
        Sv = -1  # trivial solution
    else:
        if Sv>1:
            Sv=2 # stable gas phase
        elif Sv<=1:
            Sv=-1 # two phase
        #else:
            #Sv=-1 #check liquid phase
    '''
    return Sv,t,it,K

def VL(eos,z,T,p,K,components,phase):

    e1 = 1
    e2 = 1

    '1. Calculate the mixture fugacity (fzi) using overall composition zi.'



    '2.Create a vapor-like second phase'

    'a) Use Wilson’s correlation to obtain initial Ki-values.'

    if phase == 'V': #If trial phase is vap (-> tested phase z is liquid)
        lnfugz, v0 = eos.logfugef(z, T, p, 'L')
    elif phase == 'L': #If trial phase is liq (-> tested phase z is vapor)
        lnfugz, v0 = eos.logfugef(z, T, p, 'V')

    #print(lnfugz,z, T, p)

    max_it = 10000
    eps_SSI = 10E-12
    eps_tpd=10E-4
    eps_K=10E-4
    it = 0
    convg = 'True'
    t=0
    while e1>eps_SSI:

        it=it+1

        'b) Calculate second-phase mole numbers, Yi'
        if phase == 'V':
            Yi = z * K #-> tested phase z is liquid
        elif phase == 'L':
            Yi = z / K #-> tested phase z is vapor
        #print('Yi', Yi)
        'c) Obtain the sum of the mole numbers,'
        Sv = np.sum(Yi)
        #print('SUMV', Sv)
        'd) Normalize the second-phase mole numbers to get mole fractions'
        yi = Yi/Sv
        'e) Calculate the second-phase fugacity (fyi) using the corresponding EOS and the previous composition.'
        if phase == 'V': #If trial phase (y) is vap (-> tested phase z is liquid)
            lnfugV, v1 = eos.logfugef(yi, T, p, phase) #trial phase
            R = ((np.exp(lnfugz) * z) / (np.exp(lnfugV) * yi)) * (1/Sv)

        elif phase == 'L': #If trial phase (y) is liq (-> tested phase z is vapor)
            lnfugL, v1 = eos.logfugef(yi, T, p, phase) #trial phase
            R = ((np.exp(lnfugL) * yi) / (np.exp(lnfugz) * z)) * (Sv)
        'f) Calculate corrections for the K-values:'

        K = K * R

        #if phase == 'V':
            #K=K*R
            #print(K)
       # elif phase == 'L':
            #K=K*R
            #print(K)



        #K= (np.exp(lnfugz) / np.exp(lnfug1))

        #Sv_new=np.sum(K*z)
        'g) Check if:'
        'i) Convergence is achieved:'
        e1 = ((R-1)**2).sum()
        'ii) A trivial solution is approached'
        e2= (np.log(K)**2).sum()
        #print('e2',e2,e1)
        'If a trivial solution is approached, stop the procedure.'

        if (e2 < eps_K):
            convg='False'
            #print('trivial-v')
            break;
        elif (it>max_it):
            print('Max Iterations of Stability Test Reached')
            convg='False' # simple status
            break;
        #print('TPDv', t, -np.log(Sv))
       #print ('Sv', Sv , e1 ,e2)
        'If convergence has not been attained, use the new K-values and go back to step (b).'

    'Calculate TPD'
    if convg =='False':
        #t=-np.log(Sv)
        Sv=-1
        t = 0
        #ss, w = tpd_min(yi, z, T, p, eos, 'L', 'V')
        #print(t)

    else:
        t=tpd(yi, phase, z, T, p, eos,components)
        #print('star',1 - np.sum(Yi),-np.log(Sv))
        #ss,w = tpd_min(yi, z, T, p, eos, 'L', 'L')
        #print(t,-np.log(Sv))
        #print(t)


        #print('error', t, e1 , e2)

    '''if convg=='False':
        Sv = -1  # trivial solution
    else:
        if Sv>1:
            Sv=2 # stable gas phase
        elif Sv<=1:
            Sv=-1 # two phase
        #else:
            #Sv=-1 #check liquid phase
    '''
    return Sv,t,it,K

def stabVL(eos,z,T,p,components):
    it_v=0
    it_l=0
    eps_tpd=10e-8

    K = wilsonK(p, T, components)
    #Sv,tpd_v,it_v,K1=TrialPhase(eos,z,T,p,components,'V') # Liquid Phase is Trial
    #print(tpd_v)
    #Sl, tpd_l, it_l, K2 = TrialPhase(eos, z, T, p, components, 'L') # Vapor Phase is Trial

    K11 = K

    Sv, tpd_l, it1, K1 = VL(eos, z, T, p, K11, components, 'V')  # 1/K
    # print('S2')

    K22 = K # K
    Sl, tpd_v, it2, K2 = VL(eos, z, T, p, K22, components, 'L')  # K

    '''
    if tpd_v< 0 and tpd_l<0 and tpd_v<tpd_l:
        K= K1
        #print('chosenK_VL','K1K2')
    elif tpd_v< 0 and tpd_l<0 and tpd_l<tpd_v:
        K= 1/K2
        #print('chosenK_VL','K2K1')
    elif tpd_v < 0 and tpd_l>=0:
        K=K1
        #print('chosenK_VL','K1')
    elif tpd_l < 0 and tpd_v>=0:
        K= 1/K2
        #print('chosenK_VL','K2')
    #if tpd_v==0 and tpd_l==0:
        #print('BOTH')
        #K = vapour_liquid(p, T, components)
    '''
    #print('SV', tpd_v, tpd_l)
    '''
    if tpd_l==0 and tpd_v==0:
        state = simple_fluid_state(eos, z, T, p, components)
        #print('BOTH', state)
        if state == "Subcooled Liquid":
            Fc = 1  # liquid
            K = 1/wilsonK(p, T, components)
        else:
            Fc = 0  # gas
            K = wilsonK(p, T, components)
            

    if tpd_v > eps_tpd:
        if tpd_l==0:
            Fc=0
            K=K2
        elif tpd_l >eps_tpd:
            if tpd_v>tpd_l:
                Fc=0
                K=K2
            elif tpd_l>tpd_v:
                Fc=1
                K=1/K1
    elif tpd_l > eps_tpd:
        if tpd_v==0:
            Fc=1
            K=1/K1


    if tpd_v < -eps_tpd:
        if tpd_l>eps_tpd or tpd_l==0:
            Fc=2
            K=K2
        if tpd_v==tpd_l:
            Fc=2
            K = K2

    if tpd_l < -eps_tpd:
        if tpd_v>eps_tpd or tpd_v==0:
            Fc=2
            K=1/K1

    if tpd_l<-eps_tpd and tpd_v<-eps_tpd:
        if tpd_l<tpd_v:
            Fc=2
            K=1/K1
        elif tpd_v<tpd_l:
            Fc=2
            K=K2

    '''
    if tpd_v > eps_tpd:
        Fc = 0
        state = 'Single Phase Vapor'
        K= 1/vapour_liquid(p, T, components)
    elif tpd_v < -eps_tpd:
        state = 'Vapor-Like Two Phase'
        #print(state)
        Fc = 2
        K=K
        #K = 1/vapour_liquid(p, T, components)
    else: # check liquid phase
        #print(tpd_l)
        if tpd_l > eps_tpd:
            state = 'Single Phase Liquid'
            K = vapour_liquid(p, T, components)
            Fc = 1
        elif tpd_l < -eps_tpd:
            state = 'Liquid-Like Two Phase'
            #print(state)
            Fc = 2
            K=1/K
            #K = 1/vapour_liquid(p, T, components)
        else: #trivial - simple state
            state = simple_fluid_state(eos, z, T, p, components)
            #print('BOTH', state)
            if state=="Subcooled Liquid":
                Fc=1 #liquid
                K = 1/vapour_liquid(p, T, components)
            else:
                Fc=0 # gas
                K = vapour_liquid(p, T, components)
    state='empty'






    '''
    if tpd_v > eps_tpd:
        Fc = 0
        state = 'Single Phase Vapor'
        K= 1/vapour_liquid(p, T, components)
    elif tpd_v < -eps_tpd:
        state = 'Vapor-Like Two Phase'
        #print(state)
        Fc = 2
        K=K1
        #K = 1/vapour_liquid(p, T, components)
    else: # check liquid phase
        #print(tpd_l)
        if tpd_l > eps_tpd:
            state = 'Single Phase Liquid'
            K = vapour_liquid(p, T, components)
            Fc = 1
        elif tpd_l < -eps_tpd:
            state = 'Liquid-Like Two Phase'
            #print(state)
            Fc = 2
            K=1/K2
            #K = 1/vapour_liquid(p, T, components)
        else: #trivial - simple state
            state = simple_fluid_state(eos, z, T, p, components)
            #print('BOTH', state)
            if state=="Subcooled Liquid":
                Fc=1 #liquid
                K = 1/vapour_liquid(p, T, components)
            else:
                Fc=0 # gas
                K = vapour_liquid(p, T, components)

    if tpd_l<0 and tpd_v<0:
        if tpd_l<tpd_v:
            K=K2
        elif tpd_v<tpd_l:
            K=1/K1

    print('here',Fc)
    if tpd_l==0 and tpd_v==0:
        state = simple_fluid_state(eos, z, T, p, components)
        print('BOTH', state)
        if state == "Subcooled Liquid":
            Fc = 1  # liquid
            K = 1 / vapour_liquid(p, T, components)
        else:
            Fc = 0  # gas
            K = vapour_liquid(p, T, components)
        #print('here')
        #if tpd_l>tpd_v:
            #Fc=1
        #elif tpd_v>tpd_l:
            #Fc=0

    if tpd_v > 0.0 and tpd_l > 0.0:
        if tpd_l < tpd_v:
            Fc=1
        elif tpd_v < tpd_l:
            Fc=0
            print('here2', Fc)
    '''




    it = it_v+it_l
    #print(state,Fc)
    'Fc : Flash condition'
    return Fc,K,state,it

def stabLL(eos,z,T,p,components,phase):
    it1 = 0
    it2 = 0
    it3 = 0
    it4 = 0
    it5=0
    it6=0
    tpd4=10e5
    tpd3=10e5
    S4=0
    eps_tpd=10e-8
    NC=np.size(components)
    K = np.zeros(NC)
    K1 = np.zeros(NC)
    K2 = np.zeros(NC)
    K33 = np.zeros(NC)
    K44 = np.zeros(NC)
    K5= np.zeros(NC)
    K6 = np.zeros(NC)
    K3 = np.zeros(NC)
    K4 = np.zeros(NC)



    z_index=components.index(H2O)
    if z[z_index]<=0.5:
        phase='Aq'
        #print('ref phase is L')
    elif z[z_index]>0.5:
        phase = 'L'
        #print('ref phase is L')

    K=wilsonK(p,T,components)

    K11 = K
    K22 = 1/K  # K

    #print('S1')

    #K11 = vapour_liquid(p, T, components)

    #if phase == 'L':
        #K11 =1/K11

    S1, tpd1, it1,K1 = VAq(eos, z, T, p, K11, components,  phase) # 1/K

    #print('S2')

    #K22 = 1/ vapour_liquid(p, T, components)  # K
    S2, tpd2, it2,K2 = VAq(eos, z, T, p, K22, components, phase) # K
    #print('S5')

    #K55 = (vapour_liquid(p, T, components)) ** (1 / 3)

    K55= wilsonK(p,T,components)** (1 / 3)
    #if phase == 'L':
        #K55 = K55

    S5, tpd5, it5,K5 = VAq(eos, z, T, p, K55, components, phase)  # K
    #print('S6')
    #K66 = (1 / (vapour_liquid(p, T, components)) ** (1 / 3))

    K66=(1/(wilsonK(p,T,components)** (1 / 3)))
    S6, tpd6, it6,K6 = VAq(eos, z, T, p, K66, components, phase)  # K
    #print('S3')


    for i in range(0, NC):
        if components[i] == H2O:
            K33[i] = (0.99 / z[i])
        else:
            K33[i] = (0.01 / (NC - 1)) / z[i]


    #if z[z_index] > 0.5:
        #K33 = 1 / K33
        #print('ref phase is L')

    S3, tpd3, it3,K3 = VAq(eos, z, T, p, K33, components, phase) # K_H2O
   # print('1/Kw', tpd3)
    #Sw, tpdw, itw, Kw = VAq(eos, z, T, p, K33, components, 'Aq')  # K_H2O
    #print('1/Kw', tpdw)
    #print('S4')
    for i in range(0, NC):
        if components[i] == CO2:
            for j in range(0, NC):
                if components[j] == CO2:
                    K44[j] = 0.99 / z[j]
                else:
                    K44[j] = (0.01 /(NC - 1)/ z[j])
            S4, tpd4, it4,K4 = VAq(eos, z, T, p, K44, components,phase)  # K_CO2

    #print('S7')

    K77 = vapour_aqueous(p, T, components) # MAYBE FLIP 1/K7
    #if phase=='L':
        #K77=1/K77
    S7, tpd7, it7,K7 = VAq(eos, z, T, p, K77, components, phase)  # K
    #print('S7', tpd7)
    #tpd7=10000
    it = it1 + it2 + it3 + it4+it5+it6

    #print('TPDs', tpd1, tpd2,tpd5,tpd6,tpd7,tpd3,tpd4,it)
    #print(K1,K7)
    #tpd4=0
    '''
    if tpd4==0: 
        TPD=[tpd1,tpd2,tpd5,tpd6,tpd7,tpd3]
        #TPD = [tpd7]
        S=[S1,S2,S3]
    elif tpd3==0:
        TPD = [tpd1, tpd2, tpd5, tpd6, tpd7,tpd4]
    else:
    '''
    TPD = [tpd1, tpd2,tpd5,tpd6,tpd7,tpd3,tpd4]
        #TPD = [tpd7, tpd3, tpd4]

    index = np.argmin(TPD)

    if index==0:
        K=K1
    elif index==1:
        K=K2
    elif index == 2:
        K = K7 # CHANGE 0.48,0.1,0.42
    elif index == 3:
        K = K6
    elif index == 4:
        K = K7
    elif index == 5:
        K = 1/K3
    elif index == 6:
        K = K4 # CHANGED


    #K=K7
    KK=[K1,K2,K5,K6,K7,K3,K4]
    K=KK[index]

    if K.all()==1.0:
        KVL = vapour_liquid(p, T, components)
        KVAq = vapour_aqueous(p, T, components)
        K = KVL / KVAq

    #print(np.min(TPD),np.min(TPD)<-eps_tpd)
    if np.min(TPD)<-eps_tpd:
        Fc=2
    elif np.min(TPD)>eps_tpd:
        Fc = 1
        if phase=='L':
            Fc=3
            #K = vapour_liquid(p, T, components)
        elif phase=='Aq':
            Fc=1
    else:
        Fc=2

    if z[z_index] >= 1. - 1.E-8:
            Fc=3
    #print('chosenK_LL', index,Fc,phase)
    'Fc : Flash condition'
    return Fc,K,it

def solveTPD_SSI(eos,components, z, T,p,phasez):

    nc=len(z)
    print_info = 'false'
    be_negative = -1.E-8  # Define what is negative numerically
    be_pure_substance = 1. - 1.E-8  # Define what we mean by pure substance
    return_if_negative = 'true'  # return immediately if negativity is detected
    # discard other trial phase compositions

    if any(z >= be_pure_substance):
        TPDmin = 10.
        stable = 'true'
        K_init_min = 0.

    ntrial = 4# Number of trial compostions we will use to find the
    # global minium of the TPD function. Note that this
    # approach does not guarantee to locate the global
    # minumum of the TPD.

    tol = 1.E-9# Residuum for convergence as in REF2
    iter_max = 10000# Maximum number of iterations
    y_sp = np.zeros(ntrial, nc)# Stationary points of TPD

    lnfugz, v0 = eos.logfugef(z, T, p, phasez)

    d= np.log(z) + lnfugz

    K=wilsonK(p,T,components)
    Y_init=[]
    Y=[]

    if phasez=='L':
        phasey='V'
        Y_init[0, :]= z*K           # If trial phase is vap (-> tested phase z is liquid)
        Y_init[1, :]= z*K**(1/3)    # If trial phase is vap (-> tested phase z is liquid)
        Y_init[2, :]= z/K           # If trial phase is liq (-> tested phase z is vapor)
        Y_init[3, :]= z/(K**(1/3))  # If trial phase is liq (-> tested phase z is vapor)
    else:
        phasey = 'L'
        Y_init[0, :]= z/K           # If trial phase is liq (-> tested phase z is vapor)
        Y_init[1, :]= z/(K**(1/3))  # If trial phase is liq (-> tested phase z is vapor)
        Y_init[2, :]= z*K           # If trial phase is vap (-> tested phase z is liquid)
        Y_init[3, :]= z*K**(1/3)    # If trial phase is vap (-> tested phase z is liquid)

    # See App. A in REF2 for details on the SSI-Algorithm (here without Newton)
    TPD_star=[]
    for k in range(0,ntrial-1):

        Y[k]=Y_init[K,:]
        break_eps = 'false'

        for iter in range (0,iter_max):

            y_trial = Y/np.sum(Y)

            lnfugy, v1 = eos.logfugef(y_trial, T, p, phasey)

            Yn = np.exp(d - lnfugy)

            dY = Yn - Y
            eps = np.sqrt(abs(dY * np.transpoe(dY)))
            Y = Yn

            if eps <= tol:
                y_sp[K,:]=Y/sum(Y)
                TPD_star[K]=1.-np.sum(Y)
                break_eps = 'true'
                break
