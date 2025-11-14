import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


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
si_un = 1417
si_up = 471
si_lattice_cosnt = 5.43
si_e_af = 4.05

# dieletrics
si_di = 11.8*perm
siO2_di = 3.9*perm

class Doping(Enum):
    N_to_P = 0
    P_to_N = 1
    P_to_P = 2
    N_to_N = 3
    MET = 4

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
    return A*q*ni(T)**2*(find_enstien(up,T)/(find_L(find_enstien(up,T))*Nd) + find_enstien(un,T)/(find_L(find_enstien(un,T))*Na))


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
        return np.sqrt(((2*si_di*surf_pot_t(Na, T))/(q*Na)))
    if Nd != 0 and Nd == 0:
        return np.sqrt(((2*si_di*surf_pot_t(Nd, T))/(q*Nd)))

def find_Wdmax(Na, Nd = 0, T=300):
    if Nd == 0:
        return np.sqrt(((2*si_di*surf_pot_t(Na, T))/(q*Na)))
    return np.sqrt(((2*si_di*surf_pot(Na, Nd, T))/(q*Na)))

def find_Vfb(Na=0, T=300, EmEf = None, e_af = si_e_af, doping = Doping.N_to_P):
    if EmEf is None:
        if doping == Doping.N_to_P:
            # figure 5-5 adds them
            EmEf = si_Eg/2 + bulk(Na, T)
            return e_af - (e_af + (EmEf))
        elif doping == Doping.N_to_N:
            # figure #5-6 subs them
            EmEf = si_Eg/2 - bulk(Na, T)
            return e_af - (e_af + (EmEf))
        elif doping == Doping.P_to_P:

            EmEf = -si_Eg/2 + bulk(Na, T)
            return e_af - (e_af + (EmEf))
        elif doping == Doping.P_to_N:

            EmEf = -si_Eg/2 - bulk(Na, T)
            return e_af - (e_af + (EmEf))
    else:
        return e_af - (e_af + (EmEf))
    

def C_dep(A, T=300, Na=0, Nd=0):
    return A*((si_di)/(Wdep(Na, Nd, T)))

def cap(di, d):
    return di/d

def find_Vt(Na, l=0, cox=0, T=300, Nd = 0, doping=Doping.N_to_P):
    if cox == 0:
        cox = cap(siO2_di, l)
    if Nd == 0:
        if doping==Doping.N_to_P or doping==Doping.P_to_P:
            return find_Vfb(Na, T, doping=doping) + surf_pot_t(Na, T) + np.sqrt(2*q*Na*si_di*surf_pot_t(Na, T))/(cox)
    
        elif doping==Doping.P_to_N or doping==Doping.N_to_N:
            return find_Vfb(Na, T, doping=doping) - surf_pot_t(Na, T) - np.sqrt(2*q*Na*si_di*surf_pot_t(Na, T))/(cox)
    
    if doping==Doping.N_to_P or doping==Doping.P_to_P:
        return find_Vfb(Na, T, doping=doping) + surf_pot(Na, Nd,  T) + np.sqrt(2*q*Na*si_di*surf_pot(Na, Nd, T))/(cox)
    
    elif doping==Doping.N_to_N or doping==Doping.P_to_N:
        return find_Vfb(Na, T, doping=doping) - surf_pot(Na, Nd,  T) - np.sqrt(2*q*Na*si_di*surf_pot(Na, Nd, T))/(cox)
    

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

def find_Ids_deeplin(WdL, Vgs, Vt, n, T=300):
    return WdL*np.exp(q*(Vgs - Vt)/(n*k_J*T))*1E-7

def find_Vth(alpha, Vto, Vsb):
    return Vto + alpha*Vsb

def find_gamma(N, tox=0, cox=0):
    if cox == 0:
        cox = cap(siO2_di, tox)
    return np.sqrt(q*N*2*si_di)/cox

def find_Vth(gamma, Vt0, N, Vsb, T=300):
    if Vt0 < 0:
        return Vt0 - gamma*(np.sqrt(surf_pot_t(N, T) + Vsb) - np.sqrt(surf_pot_t(N, T)))
    return Vt0 + gamma*(np.sqrt(surf_pot_t(N, T) + Vsb) - np.sqrt(surf_pot_t(N, T)))

def alpha(Cdep=0, Cox=0, Tox=0, Wdmax=0):
    if Cdep and Cox != 0:
        return Cdep/Cox
    if Tox and Wdmax != 0:
        return (3*Tox)/Wdmax

def find_m(alpha):
    return 1 + alpha

def solve_iteration(desired_value ,desired_value_ind, init_jump, accuracy, func, *args, **kargs):

    debug = False
    print_num = 10000
    if "debug" in kargs:
        debug = kargs["debug"]
        del kargs["debug"]
    if "print_amt" in kargs:
        print_num = kargs["print_amt"]
        del kargs["print_amt"]

    aerr = accuracy
    cur_Val = 0
    perror = lambda E, A:(E - A)/A
    er = 1
    ind = 0

    dn = init_jump
    args = list(args)
    # not optimal but works
    while(np.abs(er) > aerr):
        cur_Val = func(*args, **kargs)
        er = -perror(cur_Val, desired_value)
        args[desired_value_ind] += dn*er
        if debug and ind%print_num == 0:
            print(f"ind: {ind}")
            print(f"cur value: {cur_Val}")
            print(f"error: {er}")
            print(f"desired value: {desired_value}")
            print(f"current input: {args[desired_value_ind]}")
        
        ind += 1
    
    return args[desired_value_ind]

def find_Ids_lin(WdL, Cox, u, vgs, vds, vt, m=1.2):
    return WdL*Cox*u*(vgs - vt - (m/2)*vds)*vds

def find_Vdsat(vgs, vt, m=1.2):
    return (vgs-vt)/m

def find_Idsat(WdL, cox, u, vgs, vt, m=1.2):
    return (WdL/(2*m))*cox*u*(vgs - vt)**2

def find_Ids(u, vg, vs, vd, vt, WdL=0, m = 1.2, Na = 0, Nd=0, W=0, L=0, cox=0, t=0, T=300):
    vgs = vg - vs
    vds = vd - vs
    if WdL == 0:
        WdL = W/L
    if t != 0:
        cox = cap(siO2_di, t)
    if Na != 0:
        Vto = find_Vt_cox(Na, cox, T)
        Cdep = find_cdep(W*L, Na, Nd, T)
        nu = find_n(Cdep, cox, t)
        if vt > vgs:
            return find_Ids_deeplin(WdL, vgs, vt, nu, T)
    
    else:
        if vt > vgs:
            return 0
    
    Vd_sat = find_Vdsat(vgs, vt, m)
    if  vgs >= vt and vds < Vd_sat:
        return find_Ids_lin(WdL, cox, u, vgs, vds, vt, m)
    
    if vgs >= vt and vds >= Vd_sat:
        return find_Idsat(WdL, cox, u, vgs, vt, m)


def find_Isd(u, vg, vs, vd, vt, WdL=0, m = 1.2, Na = 0, Nd=0, W=0, L=0, cox=0, t=0, T=300, doping=Doping.N_to_N):
    vsg =  vs - vg
    vsd = vs - vd
    if WdL == 0:
        WdL = W/L
    if t != 0:
        cox = cap(siO2_di, t)
    if Na != 0:
        Vto = find_Vt_cox(Na, cox, T)
        Cdep = find_cdep(W*L, Na, Nd, T)
        nu = find_n(Cdep, cox, t)
        if np.abs(vt) > vsg:
            return find_Ids_deeplin(WdL, vsg, vt, nu, T)
    
    else:
        if np.abs(vt) > vsg:
            return 0
    
    Vd_sat = find_Vdsat(vsg, np.abs(vt), m)
    if  vsg >= np.abs(vt) and vsd < Vd_sat:
        return find_Ids_lin(WdL, cox, u, vsg, vsd, np.abs(vt), m)
    
    if vsg >= np.abs(vt) and vsd >= Vd_sat:
        return find_Idsat(WdL, cox, u, -vsg, vt, m)

def find_crossing(line1, line2, x, max_distance=.1):
    diff = np.abs(line1[0] - line2[0])
    ret = 0
    for i in range(len(line1)):
        if np.abs(line1[i] - line2[i]) < diff:
            ret = x[i]
            diff = np.abs(line1[i] - line2[i])
    
    if diff <= max_distance: 
        return ret
    
    return None

def ytox(Yvalue, Y, X, dist=.1):
    ret = 0
    diff = np.abs(Y[0] - Yvalue)
    for i in range(len(Y)):
        if np.abs(Y[i] - Yvalue) < diff:
            ret = X[i]
            diff = np.abs(Y[i] - Yvalue)
    
    if diff <= dist:
        return ret

    return None

def xtoy(Xvalue, Y, X, dist=.1):
    ret = 0
    diff = np.abs(X[0] - Xvalue)
    for i in range(len(X)):
        if np.abs(X[i] - Xvalue) < diff:
            ret = Y[i]
            diff = np.abs(X[i] - Xvalue)
    
    if diff <= dist:
        return ret

    return None

def find_n(Cdep, Cox=0, l=0):
    if Cox == 0:
        Cox = cap(si_di, l)
    
    return 1 + Cdep/Cox

def find_cdep(A=1, Na=0, Nd=0, T=300):
    return A*(si_di/Wdep(Na, Nd, T=T))

