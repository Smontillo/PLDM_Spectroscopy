import numpy as np
import matplotlib.pyplot as plt
import Parameters as par
import time
import scipy.constants as sc
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# print(f"[Rank {rank}] started on {socket.gethostname()} out of {size} total ranks", flush=True)
c_cm_per_fs = sc.c * 100.0 / 1e15
# =================================== 
# FUNCTIONS
# ===================================
def fourier2DRephasing(t1,t3,R3t):
    Δω   = 600
    ω1_min, ω1_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    ω3_min, ω3_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    lenω1, lenω3 = 1001, 1001
    ω1, ω3   = np.linspace(ω1_min, ω1_max, lenω1), np.linspace(ω3_min, ω3_max, lenω3)
    δω1, δω3 = ω1[1] - ω1[0], ω3[1] - ω3[0]

    R3ω = np.zeros((len(ω1),len(ω3)),dtype=np.complex128)    

    E1 = np.exp(1j * np.outer(ω1, t1))          # (lenω1, Nt1)
    E3 = np.exp(1j * np.outer(ω3, t3))          # (lenω3, Nt3)

    # Rω = E1 @ R(t1,t3) @ E3^T   (note: E3 is (w3,t3), so use transpose to get (t3,w3))
    R3ω = (E1 @ (R3t) @ E3.T) * (par.dtN/par.fs2au * par.dtN/par.fs2au)
    # for ω1Index in range(len(ω1)):
    #     for ω3Index in range(len(ω3)):
    #         for t1Index in range(len(t1)):
    #             exp1 = np.exp(1j * ω1[ω1Index] * t1[t1Index])
    #             for t3Index in range(len(t3)):
    #                 exp3 = np.exp(1j * ω3[ω3Index] * t3[t3Index])
    #                 R3ω[ω1Index,ω3Index] += R3t[t1Index,t3Index]*exp1*exp3*par.dtN*par.dtN
    return R3ω, ω1/(2 * np.pi * c_cm_per_fs), ω3/(2 * np.pi * c_cm_per_fs)
# ===================================

R3t = np.loadtxt("Data/R3t.txt", dtype = np.complex128)
print(np.shape(R3t), par.nData_L1, par.nData_L3)
timeL1 = par.Sim_time_L1[::par.nskip]/par.fs2au 
timeL3 = par.Sim_time_L3[::par.nskip]/par.fs2au 
print(np.shape(timeL3))
R3ω, ω1, ω3 = fourier2DRephasing(timeL1,timeL3,R3t)
np.savetxt('./Data/FT2D.dat', R3ω)
np.savetxt('./Data/freq.dat',np.c_[ω1,ω3])
print(par.nData_L1,par.nData_L2,par.nData_L3)



lo, hi = - 1, 1
levels = np.linspace(lo, hi, 500)
norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

X, Y = np.meshgrid(ω1-par.ω0/par.cm2au, ω3-par.ω0/par.cm2au)
fig, ax = plt.subplots(figsize=(5.6,5))
cntr_left  = ax.contourf(X, Y, np.clip((R3ω.real)/np.max(np.abs(R3ω)), lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
cbar = fig.colorbar(cntr_left, ax=ax, ticks=[lo, 0, hi])
ax.axvline(0, color='black', ls = '--', lw=1, alpha=0.5)
ax.axhline(0, color='black', ls = '--', lw=1, alpha=0.5)
ax.set_ylabel("ω$_3$ (cm$^{-1}$)")
ax.set_xlabel("ω$_1$ (cm$^{-1}$)")
plt.tight_layout()
plt.savefig('./Data/Spectra3D_Real.png',dpi=500)
plt.close()

fig, ax = plt.subplots(figsize=(5.6,5))
cntr_left  = ax.contourf(X, Y, np.clip((R3ω.imag)/np.max(np.abs(R3ω)), lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
cbar = fig.colorbar(cntr_left, ax=ax, ticks=[lo, 0, hi])
ax.axvline(0, color='black', ls = '--', lw=1, alpha=0.5)
ax.axhline(0, color='black', ls = '--', lw=1, alpha=0.5)
ax.set_ylabel("ω$_3$ (cm$^{-1}$)")
ax.set_xlabel("ω$_1$ (cm$^{-1}$)")
plt.tight_layout()
plt.savefig('./Data/Spectra3D_Imag.png',dpi=500)
plt.close()
