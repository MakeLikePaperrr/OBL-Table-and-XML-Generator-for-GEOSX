from src.Fugacity import *
from src.Props import *
import matplotlib.pyplot as plt
from src.WhatIfAnalysis import GoalSeek
import time


def Tsat(p,components):

    F = Mix(components)
    NC = np.size(components)

    Tc = np.zeros(NC)
    Pc = np.zeros(NC)
    w = np.zeros(NC)
    Tsat = np.zeros(NC)
    T = 0

    Tc = F.Tc
    Pc = F.Pc
    w = F.w

    for i in range(0, NC):
        Tsat[i] = Tc[i] / (1 - 3 * np.log(p / Pc[i]) / (np.log(10) * (7 + 7 * w[i])))
    return Tsat

def DewT(p, y, components,T_guess,EOS):
    # normalize to non-salt components


    NC = np.size(components)  # number of components (normalized)
    F = Mix(components)
    Tc=np.zeros(NC)
    Pc = np.zeros(NC)
    w = np.zeros(NC)
    Tsat = np.zeros(NC)
    T=0
    K=np.zeros(NC)
    x_sum = np.zeros(NC)
    x_sum1 = np.zeros(NC)
    x=np.zeros(NC)
    eps=1E-10

    Tc = F.Tc
    Pc = F.Pc
    w = F.w
    for i in range(0,NC):
        K[i]=eps

    T = T_guess
    x_sum=0

    def objective(T):
     for i in range(0, NC):
        K[i] = Pc[i] / p * np.exp(5.373*(1 + w[i])*(1 - Tc[i]/T))
        x_sum=y/K



     x=x_sum/np.sum(x_sum)

     Vm=0
     error=1
     b=0
     com=F.names
     while error > 1E-5:
        phi_L ,Vm= fugacity(p, T, x, com, "L", 0,F,EOS)
        phi_V ,Vm= fugacity(p, T, y, com, "V", 0,F,EOS)

        #print("phi", b)
        x_sum1 = y * (phi_V / phi_L)


             #print("y_sum", np.sum(y_sum1))

        x = x_sum1 / np.sum(x_sum1)

        error=np.abs(np.sum(x_sum)-np.sum(x_sum1))
        x_sum=x_sum1
        #print("error",b, np.sum(x_sum1),error, T)
        b=b+1
     return np.sum(x_sum1)

    goal = 1
    #f=objective(T)
    #print("OUT OF LOOP-before",T)

    T = GoalSeek(objective, goal, T)
    #print('Result of Example 1 is                            = ', p,"        ",T)
    #print("OUT OF LOOP-after", T)



    return T

def BubbleT(p, x, components,T_guess,EOS):
    # normalize to non-salt components


    NC = np.size(components)  # number of components (normalized)
    F = Mix(components)
    Tc=np.zeros(NC)
    Pc = np.zeros(NC)
    w = np.zeros(NC)
    Tsat = np.zeros(NC)
    T=0
    K=np.zeros(NC)
    y_sum = np.zeros(NC)
    y_sum1 = np.zeros(NC)
    y=np.zeros(NC)

    Tc = F.Tc
    Pc = F.Pc
    w = F.w



    T=T_guess


    def objective(T):
     for i in range(0, NC):
        K[i] = Pc[i] / p * np.exp(5.373*(1 + w[i])*(1 - Tc[i]/T))
        y_sum=K*x

     y=y_sum/np.sum(y_sum)

     error=1
     b=0
     Vm=0
     comp = F.names
     while error > 1E-5:
        phi_L,Vm = fugacity(p, T, x, comp, "L", 0,F,EOS)
        phi_V,Vm = fugacity(p, T, y, comp, "V", 0,F,EOS)

        y_sum1 = x * (phi_L / phi_V)

        #print("phi", b)
             #print("y_sum", np.sum(y_sum1))

        y = y_sum1 / np.sum(y_sum1)

        error=np.abs(np.sum(y_sum)-np.sum(y_sum1))
        y_sum=y_sum1
        #print("error",b, np.sum(y_sum1),T)
        b=b+1
     return np.sum(y_sum1)

    goal = 1
    #f=objective(T)
    #print("OUT OF LOOP-before",T)

    T = GoalSeek(objective, goal, T)
    #print('Result of Example 1 is                            = ', p,"        ",T)
    #print("OUT OF LOOP-after", T)



    return T

def phaseD(components,z , pmax ,res,EOS):
    t0 = time.clock()
    p0 = 1
    pstep = (pmax - p0) / res
    p=0
    data = np.zeros([res, 2])
    data2 = np.zeros([res, 2])
    for i in range(0, res):
        Temp = Tsat(p0, components)
        T_guess = np.sum(Temp * z)

    T_guessB=T_guess
    T_guessD = T_guess + 50

    for i in range(0, res):
        p = p0 + pstep * i
        # print("p",p)
        T1 = BubbleT(p, z, components, T_guessB, EOS)
        #print(p, "   ", T1)
        T_guessB = T1
        data[i, 0] = T1
        data[i, 1] = p
        T2 = DewT(p, z, components, T_guessD, EOS)
        print(p, "   ", T1, T2)
        T_guessD = T2
        data2[i, 0] = T2
        data2[i, 1] = p

    t1 = time.clock() - t0
    print("Time elapsed: ", (t1 - t0) / 60)  # CPU seconds elapsed (floating point)
    return data,data2



components=[C1,C2,C3,nC4,nC5,nC6]
z=[0.5,0.1,0.1,0.1,0.1,0.1]
p=10


#print(data)
EOS=1
p0=0.0001
pmax=118
res=50

data,data2=phaseD(components,z , pmax ,res,EOS)


fig, ax1 = plt.subplots(figsize=(8, 5))

#for i in range(0, res):
    # for j in range(0, np.size(m)):
l1,=ax1.plot(data[:,0],data[:,1])
l2,=ax1.plot(data2[:,0],data2[:,1])
ax1.legend((l1,l2),('Bubble Line','Dew Line'))


print(data)
ax1.set_ylabel('Pressure [bar]')
ax1.set_xlabel('Temperature [K]')
# ax1.set_ylabel('y_H2O [-]')
ax1.set_ylim(0, 140)
ax1.set_xlim(120, 450)
plt.show()

