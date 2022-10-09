from select_para import *
import numpy as np
from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *


def update_k_value_flash(ki, P, T, zc, Cm, components, phases):
    (f_H2O, rho_H2O) = Fugacity_Density_H2O(P, T)  # water fugacity in the liquid phase (only use for Henry-Constant)
    
    nc = len(components)
    nph = len(phases)
    prop_gas = compositional_prop_evaluator(nc, nph, T, components, 0)

    for j in range(0, 50):  # loop for convergence of K-values
        k_old = ki

        # run RR for initial x and y mol fractions
        (x, y, V) = RR(zc, ki)

        model = 'PR-EOS'

        if model == 'PR-EOS':
            # run fugacity coeff.
            #phi, x_g = fugacity_coeff_eval(P, y, T, components)
            #This part is from fugacity_coeff_eval to allow creating less objects
            y_list = y.tolist()  # convert to list for flash!
            xt = np.zeros(nc + 3)
            x = value_vector(xt)
            x2 = value_vector(xt)

            state = value_vector([P] + y_list)
            prop_gas.evaluate(state, x)
            x_g = np.array(x).copy()
            f_gasses = x_g[3:(3 + nc)]
            phi = f_gasses / (np.asarray(y) * P)
            # run activity coeff.
            gamma = Activity_Coef(P, T, Cm, components)
            # run henry's constant
            Kh = Henrys_Constant(T, f_H2O, rho_H2O, components)
            ## Calculate K-values for gas components
            K = (Kh * gamma) / (P * phi[:-1])
            # K-value of water H2O
            f_H2O_EoS = x_g[-1]  # water fugacity in the gas phase
            # print(state, 'in vapor', y)
            phi_H2O = f_H2O_EoS / (y[-1] * P)
            K_H2O = K_value_H2O(P, T, phi_H2O)
            ki = np.append(K, [K_H2O])

        else :
            #Michiels model

            """ Using PR EoS Michiel library """
            phi_c, Vm = vapour_liquid(P, T, zc, components)
            gamma = Activity_Coef(P,T, Cm, components)
            Kh = Henrys_Constant(T,f_H2O, rho_H2O, components)
            K = (Kh * gamma) / (P * phi_c[:-1])
            (f_H2O, rho_H2O) = Fugacity_Density_H2O(P, T)
            # phi_h2o = f_H2O/(y[-1]*P)
            # #phi_H2O  = f_h2o(P,T, x, y)
            K_H2O = K_value_H2O(P, T, phi_c[-1])

            # compile k-values
            ki = np.append(K, [K_H2O])

        res = max(abs(ki - k_old) / k_old)

        if abs(res) < 1e-5:
            break

    return ki, phi[-1]

def fugacity_coeff_eval(p, z,T,components):
    phases = ['Aq', 'Gas']
    z_list = z.tolist()  # convert to list for flash!
    nc = len(components)
    nph = len(phases)
    prop_gas = compositional_prop_evaluator(nc,nph, T, components, 0)
    xt = np.zeros(nc + 3)  # +3 = [density, viscosity, enthalpy]
    # xt = np.zeros((nc + 4)* self.n_phases)
    x = value_vector(xt)
    x2 = value_vector(xt)

    state = value_vector([p] + z_list)

    prop_gas.evaluate(state, x)
    x_g = np.array(x).copy()

    f_gasses = x_g[3:(3 + nc)]  # fugacity's of the gas components from PR EoS

    phi = f_gasses / (np.asarray(z)* p)
    return phi,x_g

def RR(zc, k):
    """
    Own RR implementation
    :param k_values: array of partition coefficients for each component in each phase

    :return: xi, yi e.g. liquid and vapor mol fractions
    """
    # zc = [zc]
    # zc = np.append(zc, [1 - sum(zc)])

    eps = 1e-12

    # Negative flash, if between a and b solution converges
    # a = 1 / (1 - np.max(k[np.nonzero(k)]) ) + eps
    # b = 1 / (1 - np.min(k[np.nonzero(k)]) ) - eps

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 100  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)

        r = sum(zc * (k - 1) / (V * (k - 1) + 1))

        if r > 0:
            a = V
        else:
            b = V

        if abs(r) < 1 * 10 ** -12:
            break

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (x, y, V)

def Fugacity_Density_H2O(P, T):
    Tc = props('H2O', "Tc")
    Pc = props('H2O', "Pc")

    R = 83.144598

    # density of pure water. Ref--> Fine and Millero (1973)
    tetha = T - 273.15

    V0 = (1 + 18.1597e-3 * tetha) / (0.9998396 + 18.224944e-3 * tetha - 7.922210e-6 * tetha ** 2 - \
                                     55.44846e-9 * tetha ** 3 + 149.7562e-12 * tetha ** 4 - \
                                     393.2952e-15 * tetha ** 5)

    B = 19654.32 + 147.037 * tetha - 2.21554 * tetha ** 2 + 1.0478e-2 * tetha ** 3 - 2.2789e-5 * tetha ** 4
    A1 = 3.2891 - 2.391e-3 * tetha + 2.8446e-4 * tetha ** 2 - 2.82e-6 * tetha ** 3 + 8.477e-9 * tetha ** 4
    A2 = 6.245e-5 - 3.913e-6 * tetha - 3.499e-8 * tetha ** 2 + 7.942e-10 * tetha ** 3 - 3.299e-12 * tetha ** 4

    V = V0 - ((V0 * P) / (B + A1 * P + A2 * P ** 2))
    rho = 1 / V  # rho in [g/cm3] or in [cm3/g]???

    # Fugacity of H2O. Ref--> King et al. (1992)
    a = np.array([-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502])
    aT = (1 - T / Tc)
    ans = (Tc / T) * (a[0] * aT + a[1] * aT ** 1.5 + a[2] * aT ** 3 + a[3] * aT ** 3.5 + a[4] * aT ** 4 + a[
        5] * aT ** 7.5)  # shibue et al.
    Ps = Pc * np.exp(ans)

    Vm = 18.0152

    f_H2O = Ps * np.exp((Vm * (P - Ps)) / (rho * R * T))  # fugacity of water in liquid phase

    return (f_H2O, rho)

def Activity_Coef(P, T, Cm, components):
    m_a = Cm  # molality of anions
    m_c = Cm  # molality of cations
    nc = len(components)

    ##### Calculate Activity coeff. #####

    order2, order3 = select_parameter_activity(components)
    # print('order2:', order2,', order3:', order3)

    lambda_i_Na = np.zeros(nc - 1)  # second order
    lambda_i_Cl = np.zeros(nc - 1)  # second order
    lambda_i_Na_Cl = np.zeros(nc - 1)  # third order

    for i in range(0, nc - 1):
        lambda_i_Na[i] = (order2[i, 0] + order2[i, 1] * T + order2[i, 2] / T + order2[i, 3] * P +
                          order2[i, 4] / P + order2[i, 5] * (P / T) + order2[i, 6] * (T / (P ** 2))
                          + (order2[i, 7] * P) / (630 - T) + order2[i, 8] * T * np.log(P) +
                          order2[i, 9] * (P / T ** 2))
    for i in range(0, nc - 1):
        lambda_i_Na_Cl[i] = (
                order3[i, 0] + order3[i, 1] * T + order3[i, 2] / T + order3[i, 3] * P + order3[i, 4]
                / P + order3[i, 5] * (P / T) + order3[i, 6] * (T / (P ** 2)) + (order3[i, 7] * P) / (
                        630 - T) + order3[i, 8] * T * np.log(P) + order3[i, 9] * (P / T ** 2))

    # gamma_ans = 2 * m_c * lambda_i_Na + 2 * m_a * lambda_i_Cl + 2 * m_c * m_a * lambda_i_Na_Cl # old
    gamma_ans = 2 * m_c * lambda_i_Na + 2 * m_a * lambda_i_Cl + m_c * m_a * lambda_i_Na_Cl  # new

    gamma = np.exp(gamma_ans)

    return gamma

def Henrys_Constant(T, f_H2O, rho_H2O, components):
    Mw_H2O = props('H2O', "Mw")
    R = 83.144598

    # Henry's parameters, from -> Ziabakhsh-ganji & Kooi
    mu, tau, beta = select_parameter_Henry(components)
    # print('mu:', mu, ', tau:', tau, ', beta:', beta)

    dB = tau + beta * (1000 / T) ** 0.5

    kh_ans = (1 - mu) * np.log(f_H2O) + mu * (np.log((R * T * rho_H2O) / (Mw_H2O))) + 2 * rho_H2O * dB
    Kh = np.exp(kh_ans)

    return Kh

def K_value_H2O(P, T, phi_H2O):
    tetha = T - 273.15
    Vm_h2o = 18.5
    R = 83.144598
    P_ref = 1  # reference pressure (in this case 1 bar Spycher&Pruess)
    K0_H2O = 10**(
        -2.209 + 3.097e-2 * tetha - 1.098e-4 * tetha ** 2 + 2.048e-7 * tetha ** 3)  # Kooi + shabani&vilcaez
    #K_H2O = (K0_H2O / (P * 1 / phi_H2O)) * np.exp(((P - P_ref) * Vm_h2o) / (R * T))
    K_H2O = (K0_H2O/(phi_H2O*P) * np.exp(((P - P_ref) * Vm_h2o) / (R * T)))
    #print(phi_H2O,K_H2O)

    return K_H2O

def vapour_liquid(p, T, x, components):
    """"
    inputs: - x: vapour mol fraction y
            - components: vapour components excl. H2O

            Peneloux correction is correction for Vm, provides more accurate density
            Phase is always assumed to be vapour here, so x is always y """
    Peneloux = False

    phase = "V"
    # Peng-Robinson EoS
    NC = np.size(x)
    R = 8.3145e-5

    # attraction parameter
    ai = np.zeros(NC)

    ai_forward = np.zeros(NC)  # Added for Enthalpy departure
    dt = 0.001 * T
    T_forward = T + dt  # Added for Enthalpy departure

    for i in range(0, np.size(ai)):
        C_a = 1
        Tc = props(components[i], "Tc")
        Pc = props(components[i], "Pc")
        ac = props(components[i], "ac")
        kappa = 0.37646 + 1.4522 * ac - 0.26992 * ac ** 2
        alpha = (1 + C_a * kappa * (1 - np.sqrt(T / Tc))) ** 2
        ai[i] = 0.45724 * R ** 2 * Tc ** 2 * alpha / Pc

        alpha_forward = (1 + kappa * (1 - np.sqrt(T_forward / Tc))) ** 2  #Added for Enthalpy departure
        ai_forward[i] = 0.45724 * R ** 2 * Tc ** 2 * alpha_forward / Pc         #Added for Enthalpy departure

    dij = [["H2O", "CO2", "C1", "N2", "H2S"],  # from Aspen plus (DOI 10.1016/j.fluid.2016.06.012)
           [0       , 0.19014   , 0.47893   , 0.32547   , 0.105],
           [0.19014 , 0         , 0.100     , -0.017    , 0.0974],
           [0.47893 , 0.100     , 0         , 0.0311    , 0.0503],
           [0.32547 , -0.017    , 0.0311    , 0         , 0.1767],
           [-1.10329, 0.0974    , 0.0503    , 0.1767    , 0]]  # binary interaction parameters for CH4, CO2, H2O. dij == dji??
    aij = np.zeros((NC, NC))
    a = aij

    aij_forward = np.zeros((NC, NC))    #Added for Enthalpy departure
    a_forward = aij_forward             #Added for Enthalpy departure

    for i in range(0, NC):
        indexi = dij[0][:].index(components[i]) + 1  #
        for j in range(0, NC):
            indexj = dij[0][:].index(components[j])
            aij[i, j] = np.sqrt(ai[i] * ai[j]) * (1 - dij[indexi][indexj])
            a[i, j] = aij[i, j] * x[i] * x[j]

            aij_forward[i, j] = np.sqrt(ai_forward[i] * ai_forward[j]) * (1 - dij[indexi][indexj])  #Added for Enthalpy departure
            a_forward[i, j] = aij_forward[i, j] * x[i] * x[j]                                       #Added for Enthalpy departure

    a = np.sum(a)
    a_forward = np.sum(a_forward)   #Added for Enthalpy departure
    dadT = (a_forward-a)/dt         #Added for Enthalpy departure

    A = a * p / (R ** 2 * T ** 2)


    # repulsion parameter
    bi = np.zeros(NC)
    for i in range(0, NC):
        C_b = 1
        Tc = props(components[i], "Tc")
        Pc = props(components[i], "Pc")
        bi[i] = C_b * 0.0778 * R * Tc / Pc * x[i]
    b = np.sum(bi)

    if Peneloux:
        # Peneloux volume shift parameter (1982)
        c = 0
        for i in range(0, NC):
            z_ra = 0.29056 - 0.08775 * props(components[i], "ac")
            c += x[i] * (0.50033 * R * props(components[i], "Tc") / props(components[i], "Pc") * (
                    0.25969 - z_ra))

        b = b - c  # Peneloux correction for b and V

    B = b * p / (R * T)

    # solve for compressibility Z
    Z = np.roots([1, -(1 - B), A - 3 * B ** 2 - 2 * B,
                  -(A * B - B ** 2 - B ** 3)])  # 3 real roots: 2-phase region; 1 real root: supercritical??

    # interpret solution for Z and calculate fugacity coefficients
    if np.sum(np.imag(Z) == 0) == 1:
        index = np.nonzero(np.imag(Z) == 0)  # find real root
        Z = np.real(Z[index])  # Z reduces to only the real root
    elif phase == "V":
        Z = np.amax(Z)
    elif phase == "L":
        Z = np.amin(Z)

    # print("phase, Z", phase, Z)

    # fugacity coefficients for either V or L phase
    phi_c = np.zeros(NC)
    for i in range(0, NC):  # for 2-phase: fugacity coefficient for each phase, based on x and y input

        phi_c[i] = np.exp(bi[i] / b * (Z - 1) - np.log(Z - B) + A / (2.828 * B) * ( bi[i] / b -
                2 * np.sum(x * aij[i, :]) / a) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))

        # phi_c[i] = np.exp(bi[i] / b * (Z - 1) - np.log(Z - B) + A / (2.828 * B) * (
        #         bi[i] / b - 2 * np.sum(x * aij[i, :]) / a) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))

    Vm = Z * R * T / p  # molar volume
    hg_dev = R * T * (Z - 1) + (T * dadT - a) / (2 * np.sqrt(2) * b) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)) * 1e5

    if Peneloux:
        Vm = Vm - c  # Peneloux correction

    return phi_c, Vm, hg_dev

# def vapour_liquid(p, T, x, components):
#     """"
#     inputs: - x: vapour mol fraction y
#             - components: vapour components excl. H2O
#
#             Peneloux correction is correction for Vm, provides more accurate density
#             Phase is always assumed to be vapour here, so x is always y """
#     Peneloux = False
#
#     phase = "V"
#     # Peng-Robinson EoS
#     NC = np.size(x)
#     R = 8.3145e-5
#
#     # attraction parameter
#     ai = np.zeros(NC)
#     for i in range(0, np.size(ai)):
#         C_a = 1
#         Tc = props(components[i], "Tc")
#         Pc = props(components[i], "Pc")
#         ac = props(components[i], "ac")
#         kappa = 0.37646 + 1.4522 * ac - 0.26992 * ac ** 2
#         alpha = (1 + C_a * kappa * (1 - np.sqrt(T / Tc))) ** 2
#         ai[i] = 0.45724 * R ** 2 * Tc ** 2 * alpha / Pc
#     dij = [["H2O", "CO2", "C1", "N2", "H2S"],  # from Aspen plus (DOI 10.1016/j.fluid.2016.06.012)
#            [0       , 0.19014   , 0.47893   , 0.32547   , 0.105],
#            [0.19014 , 0         , 0.100     , -0.017    , 0.0974],
#            [0.47893 , 0.100     , 0         , 0.0311    , 0.0503],
#            [0.32547 , -0.017    , 0.0311    , 0         , 0.1767],
#            [-1.10329, 0.0974    , 0.0503    , 0.1767    , 0]]  # binary interaction parameters for CH4, CO2, H2O. dij == dji??
#     aij = np.zeros((NC, NC))
#     a = aij
#     for i in range(0, NC):
#         indexi = dij[0][:].index(components[i]) + 1  #
#         for j in range(0, NC):
#             indexj = dij[0][:].index(components[j])
#             aij[i, j] = np.sqrt(ai[i] * ai[j]) * (1 - dij[indexi][indexj])
#             a[i, j] = aij[i, j] * x[i] * x[j]
#     a = np.sum(a)
#     A = a * p / (R ** 2 * T ** 2)
#
#     # repulsion parameter
#     bi = np.zeros(NC)
#     for i in range(0, NC):
#         C_b = 1
#         Tc = props(components[i], "Tc")
#         Pc = props(components[i], "Pc")
#         bi[i] = C_b * 0.0778 * R * Tc / Pc * x[i]
#     b = np.sum(bi)
#
#     if Peneloux:
#         # Peneloux volume shift parameter (1982)
#         c = 0
#         for i in range(0, NC):
#             z_ra = 0.29056 - 0.08775 * props(components[i], "ac")
#             c += x[i] * (0.50033 * R * props(components[i], "Tc") / props(components[i], "Pc") * (
#                     0.25969 - z_ra))
#
#         b = b - c  # Peneloux correction for b and V
#
#     B = b * p / (R * T)
#
#     # solve for compressibility Z
#     Z = np.roots([1, -(1 - B), A - 3 * B ** 2 - 2 * B,
#                   -(A * B - B ** 2 - B ** 3)])  # 3 real roots: 2-phase region; 1 real root: supercritical??
#
#     # interpret solution for Z and calculate fugacity coefficients
#     if np.sum(np.imag(Z) == 0) == 1:
#         index = np.nonzero(np.imag(Z) == 0)  # find real root
#         Z = np.real(Z[index])  # Z reduces to only the real root
#     elif phase == "V":
#         Z = np.amax(Z)
#     elif phase == "L":
#         Z = np.amin(Z)
#
#     # print("phase, Z", phase, Z)
#
#     # fugacity coefficients for either V or L phase
#     phi_c = np.zeros(NC)
#     for i in range(0, NC):  # for 2-phase: fugacity coefficient for each phase, based on x and y input
#
#         phi_c[i] = np.exp(bi[i] / b * (Z - 1) - np.log(Z - B) + A / (2.828 * B) * ( bi[i] / b -
#                 2 * np.sum(x * aij[i, :]) / a) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))
#
#         # phi_c[i] = np.exp(bi[i] / b * (Z - 1) - np.log(Z - B) + A / (2.828 * B) * (
#         #         bi[i] / b - 2 * np.sum(x * aij[i, :]) / a) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))
#
#     Vm = Z * R * T / p  # molar volume
#     if Peneloux:
#         Vm = Vm - c  # Peneloux correction
#
#    return phi_c, Vm

def flash_main(P,T, zc, Cm, components,phases):
    components = components
    nc = len(components)
    #zc = np.append(zc, 1 - sum(zc))
    if (min(zc) <= 0):  # fix for when zc <= 0
        print('-----> fix:   min(zc) <= 0')
        V = 1
        x = zc
        y = zc

    else:
        # print('use Niek Flash')

        #ki = evaluate_flash.wilson_eq(P)  # Initial k-value with wilson eq.

        # if P < 60:
        #     ki = evaluate_flash.wilson_eq(P)
        # else:
        #ki = np.array([20, 0.1])

        """ FIXED Initial K-VALUES """
        if nc == 2:
            ki = np.array([20, 0.01])
        elif nc == 3:
            ki = np.array([10, 20, 0.01])
        else:
            ki = np.array([10, 8, 3, 0.01])

        ki,l = update_k_value_flash(ki, P,T, zc, Cm, components, phases)

        (x, y, V) = RR(zc, ki)

    return (x, y, V)


# x,y,V = flash_main(200,350,[0.99,0.005],0)
# print(x,y,V)