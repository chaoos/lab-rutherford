# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from decimal import Decimal
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from subprocess import call
from datetime import date, datetime
import os
import helpers as hp
import sys
import re

#font = {'family' : 'normal',
#        'size'   :  12}
#matplotlib.rc('font', **font)

# Custom code starts here

hp.replace("Name", "Roman Gruber")
hp.replace("Experiment", "Rutherford scattering")

time_readoff_error = 0.1 # seconds
x_readoff_error = 3*10**-3 # meters
delta_turns = 0.1
pressure = hp.physical(20, 5, 2)
hp.replace("pressure", pressure)
rad = 57.29577951

####################################
### Part I - Discriminator Curve ###
####################################

# data for the discriminator curve
turns = hp.fetch2('data/discriminator_curve.xlsx', 'turns', delta_turns)
#delta_counts = hp.fetch2('data/discriminator_curve.xlsx', 'counts err')
delta_counts = hp.pnumpy.sqrt(hp.fetch2('data/discriminator_curve.xlsx', 'counts'))
counts = hp.fetch2('data/discriminator_curve.xlsx', 'counts', delta_counts)
seconds = hp.fetch2('data/discriminator_curve.xlsx', 'seconds', time_readoff_error)

ctps = counts/seconds
U_DA = 1 # turns

hp.replace("time_readoff_error", hp.physical(time_readoff_error, 0, 1))
hp.replace("x_readoff_error", hp.physical(x_readoff_error*100, 0, 1))
hp.replace("delta_turns", hp.physical(delta_turns, 0, 1))

# erf fit
def erf_param(x, a, b, c, d):
   return ( np.vectorize( lambda x, a, b, c, d: a*erf(b*x + c) + d )(x, a, b, c, d) )

guess_a, guess_b, guess_c, guess_d = -20.0, 1.2, -3.5, 20.0 # erf()
guesses = np.array([guess_a, guess_b, guess_c, guess_d])
popt, pcov = curve_fit(erf_param, hp.nominal(turns), hp.nominal(ctps), p0=guesses, sigma=hp.stddev(ctps), absolute_sigma=True)
aerr, berr, cerr, derr = list(np.sqrt(np.diag(pcov)))
a, b, c, d = list(popt)

erf_fit = lambda x: np.vectorize( lambda x: a*erf(b*x + c) + d )(x)

# parameters of the erf fit
hp.replace("fit:erf_a", hp.physical(a, aerr, 4))
hp.replace("fit:erf_b", hp.physical(b, berr, 4))
hp.replace("fit:erf_c", hp.physical(c, cerr, 4))
hp.replace("fit:erf_d", hp.physical(d, derr, 4))
X = np.linspace(0, 6, 500)

gauss = lambda x: 2*a*b*np.exp(-(b*x + c)**2)/np.sqrt(np.pi)

# plot for the discriminator curve and its fit
plt.figure(0)
plt.errorbar(hp.nominal(turns), hp.nominal(ctps), xerr=hp.stddev(turns), yerr=hp.stddev(ctps), fmt='x', label=r'measured data')
plt.plot(X, erf_fit(X), label=r'$f(x)$')
plt.plot(X, -gauss(X), label=r"$-f'(x)$")

#plt.plot(np.array([1, 1]), np.array([0, 40]), color='b', label=r"$-f'(x)$")
plt.axvline(U_DA, linewidth=2, color='black')
plt.text(U_DA+0.05, 20, r'Operating point $U_{DA} = '+hp.fmt_number(U_DA)+'$ turn', rotation=90, verticalalignment='center')

plt.ylabel(r"counts per sec $\,[s^{-1}]$")
plt.xlabel(r"turns $U_D \, [a.u.]$")
plt.xlim(0, 6)
plt.ylim(-1, 40)
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/discriminator_curve.eps")

# the data measurement table for the discriminator curve
turns = np.concatenate((np.array([r"$turns\,[a.u.]$"]), turns))
counts = np.concatenate((np.array([r"$counts\,[\#]$"]), counts))
seconds = np.concatenate((np.array([r"$time\,[s]$"]), seconds))
ctps   = np.concatenate((np.array([r"$counts/sec\,[s^{-1}]$"]), ctps))
arr = np.array([turns, counts, seconds, ctps]).T
hp.replace("table:discriminatorCurve", arr)


####################################
### Part II - Angle Distribution ###
####################################

# a bunch of constants
x_min, x_max = 5*10**-2, 15*10**-2 # meter
R_1, R_2 = 23*10**-3, 27*10**-3 # meter
R_A = R = 0.5*(R_1 + R_2)
A_D = 50*10**-6 # m^2
d_F = 120*10**-6
delta = 73*10**-3 # meter, TODO: error
c = 3*10**-3
R_D = np.sqrt(A_D/np.pi)
mean_energy = 3.65*10**6 # eV

# Activity of the source
x_activity = hp.physical(2*10**-2, x_readoff_error, 2) # 2cm
hp.replace("x_activity", x_activity*100)

omega_tot = 4*np.pi
omega_D1 = lambda x: np.pi*R_D**2*x/(x**2 + R_A**2)**(3/2)
#omega_D1 = lambda x: A_D/(delta + x)**2 # not working
R_hole = 0.5*10**-3 # TODO: error
A_hole = np.pi*R_hole**2
omega_hole = A_hole/delta**2

#print(omega_hole, A_D/(delta + x_activity)**2)

n = hp.physical(3562, sqrt(3562), 4)
t = hp.physical(90.09, 0.3, 4)
rate_activity = n/t
hp.replace("rate_activity", rate_activity)
hp.replace("counts_activity", n)
hp.replace("seconds_activity", t)

I_S1 = rate_activity * (omega_tot/omega_hole)
hp.replace("I_S1MBq", I_S1*10**-6) # MBq
hp.replace("I_S1muCi", I_S1/(3.7*10**10)*10**6) # micro Ci

# activity accoring to the law of radioactive decay
I_then = hp.physical(3.7 * 10**6, 0, 2) # Bq @ 1.1.1962
t_half = hp.physical(432.2*365*86400, 0, 3) # 432.2 years
elapsed_time = (datetime(2018, 10, 15) - datetime(1962, 1, 1)).days*86400 # time in sec since 1.1.1962
tau = t_half/np.log(2)
I_now = I_then*hp.pnumpy.exp(-elapsed_time/tau)
hp.replace("I_then", I_then*10**-6) # MBq
hp.replace("I_now_radl", I_now*10**-6) # MBq


# determining uniform thetas (T) between x_min and x_max and get the corresponding 
# x values from them (X)
theta_1 = np.arctan(R/delta)
theta_2 = lambda x: hp.pnumpy.arctan((R + np.sqrt(A_D/np.pi))/(x + c + d_F))
theta = lambda x: theta_1 + theta_2(x)
x_twiggle = lambda t: (R + np.sqrt(A_D/np.pi))/(hp.pnumpy.tan(t - theta_1)) - c - d_F
theta_max = theta(x_min)
theta_min = theta(x_max)
T = np.linspace(theta_max, theta_min, 12)
X_twiggle = x_twiggle(T)
X_twiggle = hp.pharray(X_twiggle, x_readoff_error, 3)
T = theta(X_twiggle)
#T = theta(X_twiggle)/np.pi

delta_z_counts = hp.pnumpy.sqrt(hp.fetch2('data/angle_distribution.xlsx', 'z counts [#]'))
z_counts = hp.fetch2('data/angle_distribution.xlsx', 'z counts [#]', delta_z_counts)
z_time = hp.fetch2('data/angle_distribution.xlsx', 'z time [s]', time_readoff_error)
ctps = z_counts/z_time

fx = ctps/omega_D1(X_twiggle)
Y = ctps/omega_D1(X_twiggle)

logY = hp.pnumpy.log(Y)
logX = hp.pnumpy.log(hp.pnumpy.sin(T/2))
coeffs = hp.phpolyfit(logX, logY, 1,)
a = coeffs[0]
C = hp.pnumpy.exp(coeffs[1])
#C = hp.physical(np.exp(0.09), 0.001)
b = coeffs[1]

hp.replace("fit:b", b)

hp.replace("a", a)
hp.replace("C", C)

fitline2 = lambda theta: hp.pnumpy.log(C) + a*hp.pnumpy.log(hp.pnumpy.sin(0.5*theta))
fitline = lambda x: np.log(C.n) + a.n*x
Xs = fitline(T)

coeffs2 = hp.phpolyfit(T, Y, 2)
fitline3 = lambda x: np.polyval(coeffs2, x)


# erf fit
def exp_param(x, a, b, c, d):
   return ( np.vectorize( lambda x, a, b, c, d: a*np.exp(b*x + c) + d )(x, a, b, c, d) )

guess_a, guess_b, guess_c, guess_d = 1.0, -1.0, 0.0, 0.0
guesses = np.array([guess_a, guess_b, guess_c, guess_d])
popt, pcov = curve_fit(exp_param, hp.nominal(T), hp.nominal(Y), p0=guesses, sigma=hp.stddev(Y), absolute_sigma=True)
a2err, b2err, c2err, d2err = list(np.sqrt(np.diag(pcov)))
a2, b2, c2, d2 = list(popt)

exp_fit = lambda x: np.vectorize( lambda x: a2*np.exp(b2*x + c2) + d2 )(x)

hp.replace("fit:exp_a", hp.physical(a2, 0, 4))
hp.replace("fit:exp_b", hp.physical(b2, 0, 4))
hp.replace("fit:exp_c", hp.physical(c2, 0, 4))
hp.replace("fit:exp_d", hp.physical(d2, 0, 4))


X1 = hp.pnumpy.log(hp.pnumpy.sin(0.5*T))
X = np.linspace(np.min(hp.nominal(X1)) - 0.1, np.max(hp.nominal(X1)) + 0.1, 100)
#X = np.linspace(np.min(hp.nominal(X1)) - 0.1, 0.1, 4)
Ts = np.linspace(np.min(hp.nominal(T))-0.1, np.max(hp.nominal(T))+0.1, 300)

# plot for the angle distribution
plt.figure(1)
plt.errorbar(hp.nominal(T), hp.nominal(Y), xerr=hp.stddev(T), yerr=hp.stddev(Y), fmt='x', label=r'angle distribution')
#plt.plot(Ts, hp.nominal(fitline3(Ts)), label=r'quadratic fit')
plt.plot(Ts, hp.nominal(exp_fit(Ts)), label=r'exponential fit')
#plt.plot(X, fitline(X), label=r'linear fit')
plt.xlabel(r"angle $\theta \, [rad]$")
plt.ylabel(r"counts per sec into solid angle $\frac{N_a}{\Omega_D t} \, [s^{-1} sr^{-1}]$")
plt.grid(True)
plt.xlim(0.5, 0.9)
plt.ylim(0, 350)
plt.legend(loc="best")
plt.savefig("plots/angle_distribution2.eps")

def chisquared(fx, Y, Sigma):
	return np.sum( ((Y - fx)/Sigma)**2 )

n = ctps.size
m = coeffs.size
chi_sq = chisquared(hp.nominal(logY), hp.nominal(fitline(logX)), hp.stddev(logY))
chi_r_sq = chi_sq/(n-m)

#print(chi_sq)
#print(chi_r_sq)

# linear fit for 9.5 Tab. A1 for v=10
alphas = np.array([0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.001])
chi_rs = np.array([0.256, 0.306, 0.394, 0.487, 0.618, 0.727, 0.830, 0.934, 1.047, 1.178, 1.344, 1.599, 1.831, 2.116, 2.321, 2.959])
coeffs4 = np.polyfit(chi_rs, alphas, 1)
chi_dist = lambda chi: np.polyval(coeffs4, chi)

alpha = chi_dist(chi_r_sq)

hp.replace("chi_sq", chi_sq)
hp.replace("chi_r_sq", chi_r_sq)
hp.replace("alpha", alpha*100)

hp.replace("quadFit", hp.fmt_fit(coeffs2, None, r'\theta'))

# plot for the angle distribution
plt.figure(2)
plt.errorbar(hp.nominal(X1), hp.nominal(logY), xerr=hp.stddev(X1), yerr=hp.stddev(logY), fmt='x', label=r'measured data')
plt.plot(X, fitline(X), label=r'linear fit')
plt.xlabel(r"$\log{\left( \sin{\left(\theta / 2 \right)} \right)} \, [a. u.]$")
plt.ylabel(r"$\log{\left( \frac{N_a}{\Omega_D t} \right)} \, [a. u.]$")
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/angle_distribution.eps")

hp.replace("linearFit", hp.fmt_fit(coeffs))

# the data measurement table for the discriminator curve
T_grad = T*rad
T_grad = np.concatenate((np.array([r"$\theta \, [^{\circ}]$"]), T_grad))
X_twiggle = np.concatenate((np.array([r"$\tilde{x} \, [cm]$"]), X_twiggle*10**2))
z_counts = np.concatenate((np.array([r"$counts \, [\#]$"]), z_counts))
z_time = np.concatenate((np.array([r"$time \, [s]$"]), z_time))
ctps = np.concatenate((np.array([r"$counts/sec \, [s^{-1}]$"]), ctps))
arr = np.array([X_twiggle, T_grad, z_counts, z_time, ctps]).T
hp.replace("table:angleDistribution", arr)

####################################
### Part III - Exercises         ###
####################################

# Definitions
eV = hp.physical(1.6021766208*10**-19, 0, 11) # Joule
MeV = eV*10**6
keV = eV*10**3
ucsquared = hp.physical(931.49432, 0, 8)*MeV
u = ucsquared/c**2
c = hp.physical(299792458, 0, 9)
m_alpha = hp.physical(4.001487900, 0, 10) # in units
m_Np = hp.physical(237.048167253, 0, 12) # in units
m_Am = hp.physical(241.056822944, 0, 12) # in units
m_He = hp.physical(4.002603250, 0, 10) # in units
Q_0 = (m_Am - m_Np - m_He)*ucsquared
E_A = hp.pharray([0, 33.20, 59.54, 102.96, 158.51], 0, [4, 4, 4, 5, 5])*keV
I_alpha = hp.pharray([0.34, 0.22, 84.5, 13.0, 1.6], 0, [2, 2, 3, 3, 2]) # %
Z_1, Z_2 = 2, 79
eps_0 = 8.854187*10**-12
epsilon_squared = eV**2/(4*np.pi*eps_0)
#epsilon_squared = 1.44*eV*10**-9

m_alpha *= u
m_Np *= u
m_Am *= u
m_He *= u
m_Au = hp.physical(196.96657, .000004, 8)*u
K_1 = m_alpha/m_Au


# 2.1.1
########

# kinetic Energies
T_alpha = (Q_0 - E_A)/(1 + m_alpha/m_Np)

# average energy
T_m = np.sum(T_alpha*(I_alpha/100))/np.sum(I_alpha/100)

# average energy in center of mass
v_r = v_m = hp.pnumpy.sqrt(2*T_m/m_alpha)
m_r = m_alpha*m_Au/(m_alpha + m_Au)
#T_mc = 0.5*(m_Au/(m_alpha + m_Au))*m_r*v_r**2
T_mc = T_m/(1 + K_1)**2

# 2.1.2
########

# minimal distrance to core
E_0 = T_m
D = Z_1*Z_2*epsilon_squared*(1 + K_1)/E_0

# differential cross section 
sigma = lambda theta, E: 1.296*(Z_1*Z_2/(E/MeV))**2*(np.sin(0.5*theta)**(-4) - 2*(m_alpha/m_Au)**2) # mb/sr
omega_central = sigma(np.pi, E_0)
omega_central.s = 0
#omega_central.sf = 4
#print(omega_central)

# 2.1.3
########

# impact parameters
b_max = Z_1*Z_2*epsilon_squared/(2*E_0*np.tan(0.5*theta_min))
b_min = Z_1*Z_2*epsilon_squared/(2*E_0*np.tan(0.5*theta_max))

# 2.1.4
########

# specific energy loss
m_e = hp.physical(9.10938356*10**-31, 0, 9)
N = 5.91*10**22*10**6 # m^-3
N_au_orig = 5.91*10**22 # cm^-3
N_air = 2.55*10**19 # cm^-3 TODO: find out
E_B_mean_Au = hp.physical(1059.81, 0, 6)*eV
K_Au = hp.physical(-1.037, 0, 4)
E_B_mean_Air = hp.physical(94.22, 0, 4)*eV
K_Air = hp.physical(-0.710, 0, 3)
E_alpha = E_T = E_0


epsilon_1 = lambda E_B, K, Z: (3.80/(E_alpha/MeV)) * Z*hp.pnumpy.log((548.58*(E_alpha/MeV))/(E_B/eV) - K) # Gl. 9.11

# 2.1.4.1 & 2.1.4.2
##########

# in gold foil
epsilon_au_1 = epsilon_1(E_B_mean_Au, K_Au, Z_2) # eV/(10^15 atoms cm^-2)
epsilon_au_2 = epsilon_au_1/1000*N_au_orig*10**(-15)/10**4 # keV/(micro m)
epsilon_au_si = epsilon_au_1*N_au_orig*10**(-15)/10**4*10**6*eV # J/m

# in air
epsilon_air_1 = epsilon_1(E_B_mean_Air, K_Air, 7) #eV/(10^15 atoms cm^-2)
epsilon_air_2  = epsilon_air_1/1000*N_air*10**(-15)/10 # keV/mm
epsilon_air_si = epsilon_air_2*keV/1000 # J/m
#epsilon_air_si = epsilon_air_1*N_air*10**(-15)/10*1000*eV # J/m

# middle hole open
d_air = delta + x_max
d_au = 10**-6 # m

#remaining_energy = lambda E0, d1, d2: (E0/MeV - epsilon_au_2/1000*(d1*10**6) - epsilon_air_2/1000*d2/1000)*MeV
remaining_energy = lambda E0, d1, d2: E0 - epsilon_au_si*d1 - epsilon_air_si*d2

E_signal = remaining_energy(E_0, d_au, d_air)
hp.replace("E_signal", hp.physical((E_signal/MeV).n, 0, 3))

# E_min / E_max
R_det = np.sqrt(A_D/np.pi)
theta_1_min = np.arctan(R_1/delta)
theta_1_max = np.arctan(R_2/delta)
theta_2_min = np.arctan((R_1 - R_det)/x_min)
theta_2_max = np.arctan((R_2 + R_det)/x_min)
N_A = 6.022*10**23 # Avogadro constant [1/mol]
M = 196.96657*10**-3 # molar mass of Au [kg/mol]
N = 5.91*10**22*10**6 # #-density of gold [m^-3]
density = 40*10**-9*100**2 # kg/m^2
mass = density*A_D
rho = N*M/N_A
V = mass/rho
d_sensible = V/A_D # sensible thickness of the detector
d_au_min = 0.5*d_au/np.cos(theta_1_min) + 0.5*d_au/np.cos(theta_2_min) + d_sensible/np.cos(theta_2_min)
d_au_max = 0.5*d_au/np.cos(theta_1_max) + 0.5*d_au/np.cos(theta_2_max) + d_sensible/np.cos(theta_2_min)
d_air_min = np.sqrt(delta**2 + R_1**2) + np.sqrt((R_1 - R_det)**2 + x_min**2)
d_air_max = np.sqrt(delta**2 + R_2**2) + np.sqrt((R_2 + R_det)**2 + x_min**2)

#K_2 = lambda theta: ((K_1*np.cos(theta) + hp.pnumpy.sqrt(1 - K_1**2*np.sin(theta)**2))/(1 + K_1))**2
#E_scatter = lambda E0, theta: (1 - K_2(theta))/E0

E_0_min = remaining_energy(E_0, 0.5*d_au/np.cos(theta_1_min), np.sqrt(delta**2 + R_1**2))
E_min = remaining_energy(E_0, d_au_min, d_air_min)
E_max = remaining_energy(E_0, d_au_max, d_air_max)
hp.replace("E_0", hp.physical((E_0/MeV).n, 0, 3))
hp.replace("x_min", hp.physical(x_min*100, 0, 1))
hp.replace("E_min", hp.physical((E_min/MeV).n, 0, 3))
hp.replace("E_max", hp.physical((E_max/MeV).n, 0, 3))


# 2.1.5
########

# differential cross section
#T_max = theta(x_min).n
#T_min = theta(x_max).n
#print(theta_max, theta_min, T_min)
m_1 = m_alpha.n
m_2 = 196.96657*u.n

# in labor system
sigma_1 = lambda theta, E: (Z_1*Z_2*epsilon_squared/(4*E))**2 * np.sin(0.5*theta)**(-4)

# in center of mass system of gold foil
sigma_2 = lambda theta, E: (Z_1*Z_2*epsilon_squared/(4*E))**2 * (4/(np.sin(theta))**4) * (np.cos(theta) + np.sqrt(1 - ((m_1/m_2)*np.sin(theta))**2))**2/np.sqrt(1 - ((m_1/m_2)*np.sin(theta))**2)

#print(sigma_1(T_min, E_0.n))
#print(sigma_2(T_min, E_0.n))

sigma_1_min = sigma_1(theta_max, E_0.n)
sigma_1_max = sigma_1(theta_min, E_0.n)
sigma_2_min = sigma_2(theta_max, T_mc.n)
sigma_2_max = sigma_2(theta_min, T_mc.n)
sigma_21_min = sigma_2_min/sigma_1_min
sigma_21_max = sigma_2_max/sigma_1_max

hp.replace("sigma_1_min", sigma_1_min)
hp.replace("sigma_1_max", sigma_1_max)
hp.replace("sigma_2_min", sigma_2_min)
hp.replace("sigma_2_max", sigma_2_max)
hp.replace("sigma_21_min", sigma_21_min)
hp.replace("sigma_21_max", sigma_21_max)

# activity exp
Omega_F = 0.0998 # sr 9.7
N_A = 6.022*10**23 # Avogadro constant [1/mol]
M = 196.96657*10**-3 # molar mass of Au [kg/mol]
N = 5.91*10**22*10**6 # #-density of gold [m^-3]
n_AK = N*M/N_A # density [kg/m^3]
d = 10**-6 # m
#I_S2 = 16*E_0**2*4*np.pi*C/(n_AK*d*Omega_F*Z_1**2*Z_2**2*epsilon_squared**2)
#I_S2 = C*4*np.pi/(n_AK*d*Omega_F*((Z_1*Z_2*epsilon_squared)/(4*E_0))**2)

I_numerator = C*4*np.pi*(4*E_0*MeV)**2
I_denumerator = n_AK*d*Omega_F*(Z_1*Z_2*epsilon_squared)**2

I_S2 = I_numerator/I_denumerator

hp.replace("d", hp.physical(d*10**6, 0, 1)) # micro m
hp.replace("I_S2MBq", I_S2*10**-6) # MBq

arr = hp.to_table(
	r'$i$', np.array([0, 1, 2, 3, 4]),
	r'$E_{Ai} [keV]$', E_A/keV,
	r'$I_{Ai} [\%]$', I_alpha,
	r'$T_{\alpha i} \, [MeV]$', T_alpha/MeV
).T

hp.replace("table:Talphai", arr)
hp.replace("T_m", hp.physical((T_m/MeV).n, 0, 3))
hp.replace("T_mc", hp.physical((T_mc/MeV).n, 0, 3))

hp.replace("D", D)
hp.replace("omega_central", omega_central)

hp.replace("theta_max", theta_max*rad)
hp.replace("theta_min", theta_min*rad)
hp.replace("b_max", b_max.n)
hp.replace("b_min", b_min.n)

epsilon_au_1.sf = epsilon_air_1.sf = 3
hp.replace("epsilon_au_1", epsilon_au_1)
hp.replace("epsilon_air_1", epsilon_air_1)
hp.replace("epsilon_au_2", epsilon_au_2)
hp.replace("epsilon_air_2", epsilon_air_2)

hp.compile()