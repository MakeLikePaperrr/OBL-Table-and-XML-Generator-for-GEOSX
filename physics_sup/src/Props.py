from src.Components import *




H2O	=	component(	name='H2O'	,	Tc=647.3,	Pc=221.2,	w=0.344	,	Zc=0.229,	MW=18.0152,                 c=-8.326592583006173e-05)
N2	=	component(	name='N2'	,	Tc=126.192,	Pc=33.958,	w=0.0372,	Zc=0.29,	MW=28.0135,                 c=-0.0042675558428042736	)
CO2	=	component(	name='CO2'	,	Tc=304.203,	Pc=73.7797,	w=0.235	,	Zc=0.991,	MW=44.0098, Vc=0.094,       c=-0.001938185708501956	)
H2S	=	component(	name='H2S'	,	Tc=373.1,	Pc=90,  	w=0.1005,	Zc=0.274,	MW=34.0809,                 c=-0.00380280643318156	)
C1	=	component(	name='C1'	,	Tc=190.564,	Pc=45.992,	w=0.0104,	Zc=0.285,	MW=16.0428, Vc= 0.098,      c=-0.005163609748818847)
C2	=	component(	name='C2'	,	Tc=305.33,	Pc=48.718,	w=0.0991,	Zc=0.284,	MW=30.07,   Vc= 0.148,      c=-0.0057811509517632755)
C3	=	component(	name='C3'	,	Tc=369.85,	Pc=42.4766,	w=0.152	,	Zc=0.281,	MW=44.0956, Vc= 0.2,        c=-0.006350355926973509)
iC4	=	component(	name='iC4'	,	Tc=407.85,	Pc=36.4	,   w=0.1844,	Zc=0.28	,	MW=58.1222,                c=-0.006846665119890271)
nC4	=	component(	name='nC4'	,	Tc=425.16,	Pc=37.96,	w=0.1985,	Zc=0.274,	MW=58.1222, Vc= 0.258,      c=-0.006267461509947329)
iC5	=	component(	name='iC5'	,	Tc=460.45,	Pc=33.77,	w=0.227	,	Zc=0.268,	MW=72.1488,                 c=-0.006211350121213367	)
nC5	=	component(	name='nC5'	,	Tc=469.7,	Pc=33.665,	w=0.2513,	Zc=0.262,	MW=72.1488, Vc= 0.31,       c=-0.005118275940536239)
nC6	=	component(	name='nC6'	,	Tc=507.82,	Pc=30.181,	w=0.2979,	Zc=0.263,	MW=86.1754, Vc= 0.351,      c=-0.0033102551759978884)
nC7	=	component(	name='nC7'	,	Tc=540.13,	Pc=27.27,	w=0.3498,	Zc=0.263,   MW=100.202,                 c=-0.0001442334547211825	)
nC8	=	component(	name='nC8'	,	Tc=569.32,	Pc=24.97,	w=0.396	,	Zc=0.259,	MW=114.231,                 c=0.003679157731481079)
nC9	=	component(	name='nC9'	,	Tc=594.6,	Pc=22.88,	w=0.445	,	Zc=0.251,	MW=128.257,                 c=0.008841919309612239	)
nC10=	component(	name='nC10'	,	Tc=617.7,	Pc=21.2,    w=0.489	,	Zc=0.347,	MW=142.2848,Vc=0.534,       c=0.0145931799792744)


def Mix(components):
    NC = np.size(components)
    mix1 = mixture(components[0], components[1])
    for i in range(2, NC):
        mix1.add_component(components[i])
    return mix1







