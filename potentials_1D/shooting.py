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
    x = np.arange(0.0, a + dx, dx)
    V = []
    for _ in x:
        V.append(V0)
    eigE = findEnergyEigenValues(TimeIndependentSE, psi0, x, V, E_arr)
    ipw_out = []
    for E in eigE:
        out = rk4(TimeIndependentSE, psi0, x, V, E)
        ipw_out.append(normalize(out [: , 0]))
    out_arr = np.asarray(ipw_out)
    return x, out_arr

def main():
    print("Hello World")

if __name__ == "__main__":
    main()

