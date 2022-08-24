import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import newton


def TimeIndependentSE(S, x, V, E):
    psi, phi = S
    return np.asarray([phi, (V-E)*psi])

def rk4(func, psi0, x, V, E):
    n = len(x)
    psi = np.array([psi0]*n)
    for i in range(n-1):
        h = x[i+1] - x[i]
        k1 = h*func(psi[i], x[i], V[i], E)
        k2 = h*func(psi[i] + 0.5*k1, x[i] + 0.5*h, V[i], E)
        k3 = h*func(psi[i] + 0.5*k2, x[i] + 0.5*h, V[i], E)
        k4 = h*func(psi[i] + k3, x[i+1], V[i], E)
        psi[i+1] = psi[i] + (k1 + 2.0*(k2+k3)+k4)/6.0
    return psi

def shoot(func, psi0, x, V, E_arr):
    psi_end = []
    for E in E_arr:
        psi = rk4(func, psi0, x, V, E)
        psi_end.append(psi[len(psi)-1][0])
    return psi_end

def shootOneE(E, func, psi0, x, V):
    psi = rk4(func, psi0, x, V, E)
    return psi[len(psi)-1][0]

def findZeroIntervals(values):
    signs = np.signbit(values)
    return np.where(np.diff(signs))[0]

def normalize(func):
    func_max = max(func)
    return func/func_max

def findEnergyEigenValues(func, psi0, x, V, E_arr):
    first_shoot = shoot(func, psi0, x, V, E_arr)
    zeroInterval = findZeroIntervals(first_shoot)
    eigenEnergies = []
    for z in zeroInterval:
        eigenEnergies.append(newton(shootOneE, E_arr[z], args = (func, psi0, x, V)))
    return np.asarray(eigenEnergies)


def shootInfinitePotentialWell(psi0, dx, a, V0, E_arr):
    x_ipw = np.arange(0.0, a + dx, dx)
    V_ipw = []
    for _ in x_ipw:
        V_ipw.append(V0)
    eigE = findEnergyEigenValues(TimeIndependentSE, psi0, x_ipw, V_ipw, E_arr)
    ipw_out = []
    for E in eigE:
        out = rk4(TimeIndependentSE, psi0, x_ipw, V_ipw, E)
        ipw_out.append(normalize(out [: , 0]))
    out_arr = np.asarray(ipw_out)
    return x_ipw, out_arr

def shootFinitePotentialWell(psi0, dx, x_range, a, V0, E_arr):
    x_pw = np.arange(-x_range, x_range + dx, dx)
    V_pw = []
    for x in x_pw:
        if x < -a/2.0 or x > a/2.0:
            V_pw.append(0) 
        else:
            V_pw.append(V0)
    eigE = findEnergyEigenValues(TimeIndependentSE, psi0, x_pw, V_pw, E_arr)
    pw_out = []
    for E in eigE:
        out = rk4(TimeIndependentSE, psi0, x_pw, V_pw, E)
        pw_out.append(normalize(out[: , 0]))
    out_arr = np.asarray(pw_out)
    return x_pw, out_arr

def shootQuantumHarmonicOscillator(psi0, dx, x_range, V_offset, E_arr):
    x_qho = np.arange(-x_range, x_range + dx, dx)
    V_qho = []
    for x in x_qho:
        V_qho.append(V_offset + x**2)
    eigE = findEnergyEigenValues(TimeIndependentSE, psi0, x_qho, V_qho, E_arr)
    pw_out = []
    for E in eigE:
        out = rk4(TimeIndependentSE, psi0, x_qho, V_qho, E)
        pw_out.append(normalize(out[: , 0]))
    out_arr = np.asarray(pw_out)
    return x_qho, out_arr

def main():
    psi_0 = 0.0
    phi_0 = 1.0
    psi0 = np.array([psi_0, phi_0])
    dx = 0.001
    E = np.arange(-5, 0, 1.0)
    x, states = shootQuantumHarmonicOscillator(psi0, dx, 5, -5, E)
    for state in states:
        plt.plot(x, state)
    plt.show()

if __name__ == "__main__":
    main()

