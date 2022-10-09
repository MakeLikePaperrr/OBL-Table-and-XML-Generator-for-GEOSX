from rrsolver import check_rr, RachfordRice
from rrsolver import index_vector as i_vec
from rrsolver import value_vector as v_vec

# check C++ interface (former main() function)
#print ('C++ output:')
#check_rr()

#check Python interface
rr = RachfordRice(3)
ph_i = i_vec([0,1,2])
nPhases = len(ph_i)
print(nPhases)
component_K = v_vec([3.08529149e+01, 5.07937049e-01 ,5.32085427e-01,1.31466045e-03, 2.09149452e-02 ,4.03032498e+01 ])
nc = 3
eps = 1e-12
zc = v_vec([0.32, 0.6 , 0.08])
nu = v_vec([0] * nPhases)
rr.solveRachfordRice(nPhases, ph_i, nc, component_K, zc, nu, eps)
print ('Python interface output:')
print ('%.5f' % nu[0], nu[1],nu[2])