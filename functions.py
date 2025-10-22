import numpy as np
import matplotlib.pyplot as plt


# common varables
eV = 1.602E-19
q = 1.602E-19
k_J = 1.38E-23
k_eV = k_J/eV
m0 = 9.11E-31
perm = 8.85E-14 # cm
h_J = 6.63E-34
h_eV = h_J/eV
h_bar_J = 1.055E-34
h_bar_eV = h_bar_J/eV
c = 3E8
thermal_V_at_300 = (k_J*300)/q

# silicion
si_mn = .26*m0
si_mp = .39*m0
si_Eg = 1.12
si_ue = 1417
si_uh = 471
si_lattice_cosnt = 5.43
si_e_af = 4.05

# dieletrics
si_di = 11.8*perm
siO2_di = 3.9*perm
# functions
def effective_dos(m,T):
    if m == si_mn and T == 300:
        return 2.8E19
    if m == si_mp and T == 300:
        return 1.04E19
    return 2*((2*np.pi*m*k_J*T)/(h_J**2))**(3/2)

def ni(T=300,Eg = si_Eg):
    if T == 300:
        return 1E10
    Nc = effective_dos(si_mn, T)
    Nv = effective_dos(si_mp, T)
    return np.sqrt(Nc*Nv)*np.exp(-Eg/(2*k_eV*T))

def find_enstien(u, T=300):
    return k_eV*T*u

def find_drift(car, mob, efield):
    return q*car*mob*efield

def find_carrier(toEf, m, T=300):
    return effective_dos(m, T)*np.exp(-(toEf/(k_eV*T)))

def dos(m, EtoEcorEv):
    return (8*np.pi*m*np.sqrt(2*m*EtoEcorEv))/(h_eV**3)

def fermi(EtoEf, T=300):
    return 1/(1+np.exp(-EtoEf/(k_eV*T)))

def find_car_inj(init_car, V, T=300):
    return init_car*np.exp(V/(k_eV*T))

def find_L(D, tau):
    return np.sqrt(D*tau)

def find_inverse_current(A, un, up, Na, Nd, T=300):
    return A*q*ni(T)**2(find_enstien(up,T)/(find_L(find_enstien(up,T))*Nd) + find_enstien(un,T)/(find_L(find_enstien(un,T))*Na))


def bulk_E(Eg, Ef, Ev):
    return (Eg/2 - (Ef-Ev))/q
 
def bulk(Na, T=300):
    return k_eV*T*np.log(Na/ni(T))

def surf_pot(Na, Nd, T=300):
    return (q*Na*Wdep(Na, Nd, T)**2)/(2*si_di)

def surf_pot_t(Na, T=300):
    return 2*bulk(Na, T)

def min_carrier(V, c, T=300):
    return c*np.exp(V/(k_eV*T))

def ex_carrier(V, c, T=300):
    min_car = min_carrier(V, T, c)
    return min_car - c

def bi_V(Na, Nd, T=300):
    if Na == 0:
        Na = ni(T)**2/Nd
    if Nd == 0:
        Nd = ni(T)**2/Na
    return k_eV*T*np.log((Na*Nd)/ni(T)**2)

def Wdep(Na=0, Nd=0, T=300):
    if Na != 0 and Nd != 0:
        return np.sqrt(((2*si_di*bi_V(Na, Nd, T))/(q))*(1/Na + 1/Nd))
    if Na != 0 and Nd == 0:
        return np.sqrt(((2*si_di*bi_V(Na, Nd, T))/(q*Na)))
    if Nd != 0 and Nd == 0:
        return np.sqrt(((2*si_di*bi_V(Na, Nd, T))/(q*Nd)))

def find_Wdmax(Na, Nd = 0, T=300):
    if Nd == 0:
        return np.sqrt(((2*si_di*surf_pot_t(Na, T))/(q*Na)))
    return np.sqrt(((2*si_di*surf_pot(Na, Nd, T))/(q*Na)))

def find_Vfb(Na=0, T=300, EmEf = None, e_af = si_e_af, N_type = True):
    if EmEf == None and N_type:
        # figure 5-5 adds them
        EmEf = si_Eg/2 + bulk(Na, T)
        return e_af - (e_af + (EmEf))
    elif EmEf == None and not N_type:
        # figure #5-6 subs them
        EmEf = si_Eg/2 - bulk(Na, T)
        return (e_af + (EmEf)) - e_af
    else:
        return e_af - (e_af + (EmEf))
    

def C_dep(A, T=300, Na=0, Nd=0):
    return A*((si_di)/(Wdep(Na, Nd, T)))

def cap(di, d):
    return di/d

def find_Vt(Na, l, T=300, Nd = 0, N_type=True):
    if Nd == 0:
        return find_Vfb(Na, T, N_type=N_type) + surf_pot_t(Na, T) + np.sqrt(2*q*Na*si_di*surf_pot_t(Na, T))/(cap(siO2_di,l))
    return find_Vfb(Na, T, N_type=N_type) + surf_pot(Na, Nd,  T) + np.sqrt(2*q*Na*si_di*surf_pot(Na, Nd, T))/(cap(siO2_di,l))

def find_Vt_cox(Na, cox, T=300, Nd = 0, N_type=True):
    if Nd == 0:
        return find_Vfb(Na, T, N_type=N_type) + surf_pot_t(Na, T) + np.sqrt(2*q*Na*si_di*surf_pot_t(Na, T))/(cox)
    return find_Vfb(Na, T, N_type=N_type) + surf_pot(Na, Nd,  T) + np.sqrt(2*q*Na*si_di*surf_pot(Na, Nd, T))/(cox)

def find_Vox(Na, T=300, l=0, c=None):
    if c == None:
        c = cap(siO2_di,l)
    return np.sqrt(q*Na*si_di*surf_pot_t(Na, T))/(c)

def find_Vox_Q(Qs, co=None , l=0):
    if co==None:
        co = cap(siO2_di,l)
    return Qs/co

def find_Vg(Vfb, sur_pot, Vox):
    return Vfb + sur_pot + Vox

def efield_dep_layer(Na, x):
    return -((q*Na)/si_di)*x

def Vfield_dep_layer_p(N, xc,  x):
    return ((q*N)/(2*si_di))*(xc - x)**2

def Vfield_dep_layer_n(N, xc,  x):
    return -((q*N)/(2*si_di))*(x - xc)**2

def find_bias_V_pn(Na, Nd, x):
    Wde = Wdep(Na, Nd)
    Xp = Wde/2
    Xn_ratio = (Na)/Nd
    Xn = -(Wde/2)*Xn_ratio
    Xp = Wde + Xn
    bi = bi_V(Na, Nd)
    print(Wde)
    print(Xp)
    print(Xn)
    print(Xp - Xn)

    V = np.zeros(len(x))
    for i in range(len(x)): 
        if x[i] < Xn:
            V[i] = bi
        if Xn <= x[i] <= 0:
            V[i] = bi + Vfield_dep_layer_n(Nd, Xn, x[i])
        if 0 <= x[i] <= Xp:
            V[i] = Vfield_dep_layer_p(Na, Xp, x[i])
        if V[i] > bi:
            V[i] = bi
    return V

def dio_G(I0, V, T=300):
    return ((q)/(k_J*T))*I0*np.exp(V/(k_eV*T))

def dio_C(G, tau):
    return tau*G
