import numpy as np
import matplotlib.pyplot as plt
import Parameters as par
import time
import scipy.constants as sc
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']
# ===================================
# FUNCTIONS
def Populations():
    Pop = np.zeros((par.nData, par.Nt), dtype = np.float64)
    for k in range(par.nData):
        Pop[k,:] = np.diag(ρRe[k,:].reshape((par.Nt, par.Nt)))
    return Pop

def LoadData():
    R1t = np.zeros((par.nData), dtype = np.complex128)
    ρt  = np.zeros((par.nData, par.Nt * par.Nt), dtype = np.complex128)
    for j in range(len(par.μMat_sp)):
        ρt[:,:] = 0.0
        for i in range(par.Cpus):
            ρt += np.loadtxt(f'./Data/rhoRe_{i}_{par.μMat_id[j,0]}_{par.μMat_id[j,1]}.txt', dtype = np.complex128)
        ρt /= par.Cpus
        for k in range(par.nData):
            R1t[k] += 1j * np.trace(ρt[k,:].reshape((par.Nt, par.Nt)) @ μ0)
        R1t *= par.μMat_sp[j] 
    return R1t
    
def Fourier1D():
    Δω   = 750 #* par.cm2au
    ωmin = (par.E0/par.cm2au - Δω) * (2 * np.pi) * sc.c/(10**13)  
    ωmax = (par.E0/par.cm2au + Δω) * (2 * np.pi) * sc.c/(10**13)  
    lenω = 1001
    ω    = np.linspace(ωmin, ωmax, lenω)
    δω   = ω[1] - ω[0]

    τ    = time * par.fs2au
    δtN  = τ[1] - τ[0]

    R1ω  = np.zeros((lenω), dtype = np.complex128)
    for k in range(lenω):
        ExpFact = np.exp(1j * ω[k] * τ / par.fs2au)
        CosFact = np.cos(np.pi * τ / (2 * np.max(τ)))
        IntFunc = ExpFact * CosFact * R1t
        R1ω[k]  = -2 * np.sum(IntFunc) * δtN
    
    R1ω_Area = np.trapz(R1ω, dx = δω)
    return R1ω / R1ω_Area, ω / ((2 * np.pi) * sc.c/(10**13))
    
# ===================================
time = par.Sim_time[::par.nskip]/par.fs2au 

μ0   = (par.μMat @ par.ρ0) - (par.ρ0 @ par.μMat)
R1t = LoadData()
R1ω, ω = Fourier1D()

fig, ax = plt.subplots(figsize = (5,3))
ax.plot(time, np.real(R1t)/np.max(np.abs(R1t)), lw = 2, c = col[0], label = 'Real')
ax.plot(time, np.imag(R1t)/np.max(np.abs(R1t)), lw = 2, c = col[1], label = 'Imaginary')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('R1(t)')
ax.set_xlim(0,time[-1])
ax.legend(frameon=False, fontsize = 5)
plt.savefig('./Images/R1t.png', dpi = 300, bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(figsize = (5,3))
# ax.axvline(par.E0/par.cm2au, lw = 1, c = 'k', ls = '--')
ax.plot(ω, np.real(R1ω), lw = 2, c = col[0], label = 'Real')
ax.plot(ω, np.imag(R1ω), lw = 2, c = col[1], label = 'Imaginary')
ax.set_xlabel('Frequency (cm$^{-1}$)')
ax.set_ylabel('R1(ω)')
ax.legend(frameon=False, fontsize = 5)
plt.savefig('./Images/R1ω.png', dpi = 300, bbox_inches = 'tight')
plt.close()



# # LOAD DATA
# ρRe    = np.zeros((par.nData, par.Nt * par.Nt), dtype = np.complex128)

# for k in range(par.Cpus):
#     ρRe += np.loadtxt(f'./Data/rhoRe_{k}.txt', dtype = np.complex128)

# ρRe /= par.Cpus

# Pop = Populations()

# fig, ax = plt.subplots(figsize = (5,3))
# ax.plot(time, Pop[:,0], lw = 2, c = col[0], label = '|0>')
# ax.plot(time, Pop[:,1], lw = 2, c = col[1], label = '|1>')
# ax.plot(time, Pop[:,2], lw = 2, c = col[2], label = '|2>')
# ax.plot(time, Pop[:,3], lw = 2, c = col[3], label = '|3>')
# ax.set_xlabel('Time (fs)')
# ax.set_ylabel('Population')
# ax.legend(frameon=False, fontsize = 5)
# plt.savefig('./Images/Populations.png', dpi = 300, bbox_inches = 'tight')
# plt.close()
