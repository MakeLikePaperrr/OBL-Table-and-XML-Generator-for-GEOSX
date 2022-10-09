from src.flash import *
from src.Stability import *
from src.Props import *


def three_phase_flash(components, phase, z, T, p,eos):

    z = np.array(z)

    z[z == 0.] = float(10e-49)
    if np.sum(z)>1+1e-8:
       print('Error:Sum of Feed > 1')
    NC=np.size(components)
    X = np.zeros((3, NC))
    V = np.zeros((1, 3))
    Tz = np.zeros((1, NC))

    Kx = Kval(p, T, components, phase)
    # status = '3p'
    # Kx=[1/K_VL,K_LL]

    molality = 0
    #K3 = Henry(p, T, components, molality)
    #Kw = [Kx[0], K3]
    V, X, Fc = vlle(components, phase, z, T, p, Kx, eos, full_output=False)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    #print('Fv', V * 100)
    # print('x',X*100)

    cc = 0

    for l in range(0, 3):
        if V[l] <= 0:
            cc = cc + 1

    if cc == 0:
        final_status = 3

    elif cc == 1 or cc == 2 or cc == 3:

        phase_LL = [phase[1], phase[2]]
        Fc1, K, it = stabLL(eos, z, T, p, components, 'L')
        X, W, Fl, v1, v2, K_LL = flash(components, phase_LL, Fc1, z, T, p, K, eos, full_output=False)
        #print(flash(components, phase_LL, Fc1, z, T, p, K, eos, full_output=True))

        if Fl <= 0.0 or Fl >= 1.0 or Fc1 == 1 :
            status1 = '1p'
            X = z


        phase_VL = [phase[0], phase[1]]
        Fc2, K, state, it = stabVL(eos, X, T, p, components)
        Y, X2, Fv, v1, v2, K_VL = flash(components, phase_VL, Fc2, X, T, p, K, eos, full_output=False)
        #print(flash(components, phase_VL, Fc2, X, T, p, K, eos, full_output=True))

        #print(Fl,Fv,Fc2)
        status2 = '2p'
        if status2 == '2p':
            if Fl == 1.0 or Fc1 == 3:  # or Fc1==3:# Aq
                final_status = 0
                V = np.block([0.0, 0, 1])
                X = np.block([[Tz], [Tz], [z]])
            elif Fl == 0.0:  # V or L
                if Fv == 0.0:  # V
                    final_status = 1
                    V = np.block([1 - Fv, Fv, 0.0])
                    X = np.block([[Y], [X2], [Tz]])
                elif Fv == 1.0:  # L
                    final_status = 2
                    V = np.block([1 - Fv, Fv, 0.0])
                    X = np.block([[Y], [X2], [Tz]])
                else:  # V-L
                    final_status = 4
                    V = np.block([1 - Fv, Fv, 0.0])
                    X = np.block([[Y], [X2], [Tz]])
            else:  # V-W or L-W
                if Fv == 0.0 or Fc2==0.0:  # V-W
                    final_status = 5
                    V = np.block([1 - Fl, 0, Fl])
                    X = np.block([[X], [Tz], [W]])
                elif Fv == 1.0 or Fc2==1:  # L-W
                    final_status = 6
                    V = np.block([0, 1 - Fl, Fl])
                    X = np.block([[Tz], [X], [W]])
                else:  # V-L-W supposedly # stabilty contradicts negative flash # prioritize negative flash if possible
                    #print('Fv <1e-3', Fl)
                    final_status = 7
                    if V[1] < 0:  # Fv <1e-2: #V-W
                        final_status = 5
                        V = np.block([1 - Fl, 0, Fl])
                        X = np.block([[X], [Tz], [W]])
                        # print('Fv <1e-3')
                    elif V[2] < 0:  # V-L
                        final_status = 4
                        V = np.block([1 - Fv, Fv, 0.0])
                        X = np.block([[Y], [X2], [Tz]])
                    elif V[0]<0: # L-W
                        final_status = 6
                        V = np.block([0, 1 - Fl, Fl])
                        X = np.block([[Tz], [X], [W]])
                    elif V[0]==0 and V[1]==0 and V[2]==0:
                        if (1-Fv) > 0.99: #V-W
                            final_status = 5
                            V = np.block([1 - Fl, 0, Fl])
                            X = np.block([[X], [Tz], [W]])
                            # print('Fv <1e-3')
                        elif Fv >0.99: #L-W
                            final_status = 6
                            V = np.block([0, 1 - Fl, Fl])
                            X = np.block([[Tz], [X], [W]])
                        else:
                            final_status = 4
                            V = np.block([1 - Fv, Fv, 0.0])
                            X = np.block([[Y], [X2], [Tz]])






    return V, X,final_status

def two_phase_flash(components, phase, z, T, p,eos):
    z = np.array(z)

    z[z == 0.] = float(10e-49)
    if np.sum(z) > 1.0:
        print('Error:Sum of Feed > 1')
    NC = np.size(components)
    X = np.zeros((2, NC))
    V = np.zeros((1, 2))

    phase_VL = [phase[0], phase[1]]
    Fc2, K, state, it = stabVL(eos, z, T, p, components)
    Y, X2, Fv, v1, v2, K_VL = flash(components, phase_VL, Fc2, z, T, p, K, eos, full_output=False)
    #print(flash(components, phase_VL, Fc2, z, T, p, K, preos, full_output=True))


    if Fv == 0.0:  # V
        final_status = 0
        V = np.block([1 - Fv, Fv])
        X = np.block([[Y], [X2]])
    elif Fv == 1.0:  # L
        final_status = 1
        V = np.block([1 - Fv, Fv])
        X = np.block([[Y], [X2]])
    else:  # V-L
        final_status = 2
        V = np.block([1 - Fv, Fv])
        X = np.block([[Y], [X2]])



    return V, X, final_status

def multiphase_flash(components,z, T, p,eos):

    if any(x == H2O for x in components):
        phase = ['V', 'L', 'Aq']
        V, X, final_status = three_phase_flash(components, phase, z, T, p,eos)
    else:
        phase = ['V', 'L']
        V, X, final_status = two_phase_flash(components, phase, z, T, p,eos)

    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    return V, X, final_status

def ternary_data(res, components, T, p, eos):

    def ternary(res):
        Z = np.zeros(shape=(res * res, 3))
        for i in range(0, res):
            for j in range(0, res):
                z1 = np.round(i * (1 / res), 2)
                z2 = np.round(j * (1 / res), 2)
                z3 = np.round(1 - (z1 + z2), 2)
                # print('z', z1 ,z2 ,z3)
                if z3 >= 0:
                    Z[res * i + j, 0] = z1
                    Z[res * i + j, 1] = z2
                    Z[res * i + j, 2] = z3

        Z = Z[~np.all(Z == 0, axis=1)]
        Z[Z == 0] = 1E-49
        return Z

    t0 = time.clock()
    Z = ternary(res)
    c=0
    s=[]
    mixture = Mix(components)
    print('==================================')
    print('Components:', mixture.names)
    print('==================================')

    for i in range(len(Z)):
        s.append("")
    for i in range(0, len(Z)):
        c = c + 1

        z = Z[i, :]
        #z=[ 0.96000 , 0.04000 , 0.00000]
        print('==================================')
        print('Feed:', z)
        print('==================================')
        V, X, final_status = multiphase_flash(components, z, T, p,eos)
        s[i] = final_status
        print(c, 'Phase Index', final_status)
        t1 = time.clock() - t0

    print('******************************************')
    print("Time elapsed (min): ", (t1 - t0) / 60)  # CPU seconds elapsed (floating point)
    print('******************************************')
    return Z,s

def ternary_plot(Z, s, res):

    import matplotlib.pyplot as plt
    import ternary

    #Prepare Data
    temp=Z*100
    result2 =np.column_stack((temp, s))

    z1 = np.zeros(shape=(res*res,4))
    z2 = np.zeros(shape=(res*res,4))
    z3 = np.zeros(shape=(res*res,4))
    z4 = np.zeros(shape=(res*res,4))
    z5 = np.zeros(shape=(res*res,4))
    z6 = np.zeros(shape=(res*res,4))
    z7= np.zeros(shape=(res*res,4))
    D = np.zeros(shape=(res*res,5))
    m=-1
    n=-1
    l=-1
    mm=-1
    nn=-1
    ll=-1
    mmm=-1
    for i in range(0,len(Z)):
        D[i, 0] = result2[i, 0]
        D[i, 1] = result2[i, 1]
        D[i, 2] = result2[i, 2]
        D[i, 3] = result2[i, 3]
        D[i, 4] = 1

        if D[i, 3]==0.0:
            m=m+1
            z1[m,0]=result2[i,0]
            z1[m, 1] = result2[i, 1]
            z1[m, 2] = result2[i, 2]
            z1[m,3]=result2[i, 3]
        elif D[i,3]==1.0:
            n=n+1
            z2[n, 0] = result2[i, 0]
            z2[n, 1] = result2[i, 1]
            z2[n, 2] = result2[i, 2]
            z2[n, 3] = result2[i, 3]
        elif D[i,3]==2.0:
            l = l + 1
            z3[l, 0] = result2[i, 0]
            z3[l, 1] = result2[i, 1]
            z3[l, 2] = result2[i, 2]
            z3[l, 3] = result2[i, 3]
        elif D[i,3]==3.0:
            mm = mm + 1
            z4[mm, 0] = result2[i, 0]
            z4[mm, 1] = result2[i, 1]
            z4[mm, 2] = result2[i, 2]
            z4[mm, 3] = result2[i, 3]
        elif D[i,3]==4.0:
            nn = nn + 1
            z5[nn, 0] = result2[i, 0]
            z5[nn, 1] = result2[i, 1]
            z5[nn, 2] = result2[i, 2]
            z5[nn, 3] = result2[i, 3]
        elif D[i,3]==5.0:
            ll = ll + 1
            z6[ll, 0] = result2[i, 0]
            z6[ll, 1] = result2[i, 1]
            z6[ll, 2] = result2[i, 2]
            z6[ll, 3] = result2[i, 3]
        elif D[i,3]==6.0:
            mmm = mmm + 1
            z7[mmm, 0] = result2[i, 0]
            z7[mmm, 1] = result2[i, 1]
            z7[mmm, 2] = result2[i, 2]
            z7[mmm, 3] = result2[i, 3]


    f1= z1.tolist()
    f2= z2.tolist()
    f3= z3.tolist()
    f4= z4.tolist()
    f5= z5.tolist()
    f6= z6.tolist()
    f7= z7.tolist()



    def generate_heatmap_data(scale=100):
        status = dict()
        loge=dict()
        for i in range(0, len(Z)):
            status[(D[i, 0], D[i, 1])] = D[i, 3]
            loge[(D[i, 0], D[i, 1], D[i, 2])] = 1 #np.log(D[i, 4])
        return status,loge


    ## Boundary and Gridlines
    scale = 100
    fig1, tax2 = ternary.figure(scale=scale)


    tax2.boundary(linewidth=0.01)
    #tax2.gridlines(color="black", multiple=10)
    fontsize = 10

    # Set Axis labels and Title
    offset = 0.14
    tax2.set_title("Hexagonal Interpolation\n\n", fontsize=fontsize)
    tax2.right_corner_label("A", fontsize=fontsize)
    tax2.top_corner_label("B", fontsize=fontsize)
    tax2.left_corner_label("C  ", fontsize=fontsize)

    # Set ticks
    tax2.ticks(axis='lbr', multiple=20, linewidth=0.1, offset=0.025)

    # Remove default Matplotlib Axes
    tax2.clear_matplotlib_ticks()
    tax2.get_axes().axis('off')

    #print fig
    status, loge = generate_heatmap_data(scale)
    tax2.heatmap(status, scale=res, style="triangular", colorbar=True, cmap="jet")


    fig2, ax = ternary.figure(scale=scale)

    # Draw Boundary and Gridlines
    ax.boundary(linewidth=0.5)
    ax.gridlines(color="black", multiple=10)
    #ax.gridlines(color="blue", multiple=1, linewidth=0.5)
    fontsize = 10
    # Set Axis labels and Title

    offset = 0.14
    ax.set_title("Scatter Plot\n\n", fontsize=fontsize)
    ax.right_corner_label("A", fontsize=fontsize)
    ax.top_corner_label("B", fontsize=fontsize)
    ax.left_corner_label("C  ", fontsize=fontsize)
    #ax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize, offset=offset)
    #ax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize, offset=offset)
    #ax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize, offset=offset)

    # Set ticks
    ax.ticks(axis='lbr', multiple=20, linewidth=0.5, offset=0.025)

    #Remove default Matplotlib Axes
    ax.clear_matplotlib_ticks()
    ax.get_axes().axis('off')


    '''
    ax.scatter(z1[:,0],z1[:,1], marker='o',s=5, color='navy',  label="1-P",edgecolors='none')
    ax.scatter(z2[:,0],z2[:,1], marker='o', s=5,color='blue', label="3-P",edgecolors='none')
    ax.scatter(z3[:,0],z3[:,1], marker='o', s=5,color='cyan',  label="2-P",edgecolors='none')
    ax.scatter(z4[:,0],z4[:,1], marker='o',s=5, color='springgreen',  label="1-P",edgecolors='none')
    ax.scatter(z5[:,0],z5[:,1], marker='o', s=5,color='gold', label="3-P",edgecolors='none')
    ax.scatter(z6[:,0],z6[:,1], marker='o', s=5,color='orangered',  label="2-P",edgecolors='none')
    ax.scatter(z7[:,0],z7[:,1], marker='o', s=5,color='darkred',  label="2-P",edgecolors='none')
    '''
    size=8
    ax.scatter(f1, marker='o', s=size, color='navy',  label="1-P",edgecolors='none')
    ax.scatter(f2, marker='o', s=size, color='blue', label="3-P",edgecolors='none')
    ax.scatter(f3, marker='o', s=size, color='cyan',  label="2-P",edgecolors='none')
    ax.scatter(f4, marker='o', s=size,  color='springgreen',  label="1-P",edgecolors='none')
    ax.scatter(f5, marker='o', s=size, color='gold', label="3-P",edgecolors='none')
    ax.scatter(f6, marker='o', s=size, color='orangered',  label="2-P",edgecolors='none')
    ax.scatter(f7, marker='o', s=size, color='darkred',  label="2-P",edgecolors='none')



    plt.show()