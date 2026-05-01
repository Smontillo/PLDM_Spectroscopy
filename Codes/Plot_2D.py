import numpy as np
import matplotlib.pyplot as plt
import Codes.Parameters as par
import time
import scipy.constants as sc
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from scipy.optimize import curve_fit
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# print(f"[Rank {rank}] started on {socket.gethostname()} out of {size} total ranks", flush=True)
c_cm_per_fs = sc.c * 100.0 / 1e15
t2 = int(sys.argv[1])
# =================================== 
# FUNCTIONS
# ===================================
def LinFun(x,m):
    return m * x

def extract_cls(w1, w3, S, *, use='abs',  # 'positive' or 'abs'
                w1_max,                # only use |w1| <= w1_max for CLS
                frac_thresh=0.15,      # ignore weak slices
                w3_window,             # window around slice peak
                power=1.0):            # weights = signal**power for COM

    # S expected shape (len(w3), len(w1))
    if use == 'positive':
        T = np.maximum(S, 0.0)
    elif use == 'abs':
        T = np.abs(S)
    else:
        raise ValueError("use must be 'positive' or 'abs'")

    assert T.shape == (len(w3), len(w1))

    ridge_w1, ridge_w3 = [], []
    global_max = T.max()

    for j, x in enumerate(w1):
        if abs(x) > w1_max:
            continue

        col = T[:, j]
        if col.max() < frac_thresh * global_max:
            continue

        k0 = np.argmax(col)
        y0 = w3[k0]
        mask = (w3 >= y0 - w3_window) & (w3 <= y0 + w3_window)

        y = w3[mask]
        s = col[mask]
        w = s**power

        if w.sum() <= 0:
            continue

        y_star = (y * w).sum() / w.sum()

        ridge_w1.append(x)
        ridge_w3.append(y_star)

    ridge_w1 = np.array(ridge_w1)
    ridge_w3 = np.array(ridge_w3)

    # linear fit: w3 = m*w1 + b
    # m, b = np.polyfit(ridge_w1, ridge_w3, 1)
    popt, pcov = curve_fit(LinFun, ridge_w1, ridge_w3)
    return popt[0], 0, ridge_w1, ridge_w3

def fourier2DRephasing(t1,t3,R3t,Δω):
    ω1_min, ω1_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    ω3_min, ω3_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    lenω1, lenω3 = 1001, 1001
    ω1, ω3   = np.linspace(ω1_min, ω1_max, lenω1), np.linspace(ω3_min, ω3_max, lenω3)
    δω1, δω3 = ω1[1] - ω1[0], ω3[1] - ω3[0]

    δt1, δt3 = t1[1] - t1[0], t3[1] - t3[0]
    T1, T3   = t1.max(), t3.max()
    W1 = np.cos(0.5 * np.pi * t1 / T1)
    W3 = np.cos(0.5 * np.pi * t3 / T3)
    R3t = R3t * np.outer(W1, W3)

    R3ω = np.zeros((len(ω1),len(ω3)),dtype=np.complex128)    

    E1 = np.exp(1j * np.outer(-ω1, t1))          # (lenω1, Nt1)
    E3 = np.exp(1j * np.outer(ω3, t3))          # (lenω3, Nt3)

    R3ω = (E1 @ (R3t) @ E3.T) * δt1 * δt3 * 1
    return R3ω, ω1/(2 * np.pi * c_cm_per_fs), ω3/(2 * np.pi * c_cm_per_fs)

def fourier2DNonRephasing(t1,t3,R3t,Δω):
    ω1_min, ω1_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    ω3_min, ω3_max = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs , (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs 
    lenω1, lenω3 = 1001, 1001
    ω1, ω3   = np.linspace(ω1_min, ω1_max, lenω1), np.linspace(ω3_min, ω3_max, lenω3)
    δω1, δω3 = ω1[1] - ω1[0], ω3[1] - ω3[0]

    δt1, δt3 = t1[1] - t1[0], t3[1] - t3[0]
    T1, T3   = t1.max(), t3.max()
    W1 = np.cos(0.5 * np.pi * t1 / T1)
    W3 = np.cos(0.5 * np.pi * t3 / T3)
    R3t = R3t * np.outer(W1, W3)

    R3ω = np.zeros((len(ω1),len(ω3)),dtype=np.complex128)    

    E1 = np.exp(1j * np.outer(ω1, t1))          # (lenω1, Nt1)
    E3 = np.exp(1j * np.outer(ω3, t3))          # (lenω3, Nt3)

    # Rω = E1 @ R(t1,t3) @ E3^T   (note: E3 is (w3,t3), so use transpose to get (t3,w3))
    R3ω = (E1 @ (R3t) @ E3.T) * δt1 * δt3 * 1
    return R3ω, ω1/(2 * np.pi * c_cm_per_fs), ω3/(2 * np.pi * c_cm_per_fs)
# ===================================
timeL1 = par.Sim_time_L1[::par.nskip]/par.fs2au 
timeL3 = par.Sim_time_L3[::par.nskip]/par.fs2au 

R1  = np.load(f"R1/Data/R3t_t2_{t2}_rank_0.npy")
R2  = np.load(f"R2/Data/R3t_t2_{t2}_rank_0.npy")
R3  = np.load(f"R3/Data/R3t_t2_{t2}_rank_0.npy")
R4  = np.load(f"R4/Data/R3t_t2_{t2}_rank_0.npy")

for rank in range(1,par.Cpus):
    R1 += np.load(f"R1/Data/R3t_t2_{t2}_rank_{rank}.npy")
    R2 += np.load(f"R2/Data/R3t_t2_{t2}_rank_{rank}.npy")
    R3 += np.load(f"R3/Data/R3t_t2_{t2}_rank_{rank}.npy")
    R4 += np.load(f"R4/Data/R3t_t2_{t2}_rank_{rank}.npy")

R1 /= par.Cpus
R2 /= par.Cpus
R3 /= par.Cpus
R4 /= par.Cpus

Δω = 60

R3t = R2 + R3 + np.conjugate(R1)
R3ω_Rephasing, ω1, ω3 = fourier2DRephasing(timeL1,timeL3,R3t,Δω)

R3t = np.conjugate(R2) + R4 + R1
R3ω_NonRephasing, ω1, ω3 = fourier2DNonRephasing(timeL1,timeL3,R3t,Δω)

np.savetxt(f'./Data/FT2D_Rephasing_t2_{t2}.dat', R3ω_Rephasing)
np.savetxt(f'./Data/FT2D_NonRephasing_t2_{t2}.dat', R3ω_NonRephasing)
np.savetxt(f'./Data/freq_t2_{t2}.dat',np.c_[ω1,ω3])

R3ω = R3ω_Rephasing + R3ω_NonRephasing
win = 20
m, b, rw1, rw3 = extract_cls(ω1-par.ω0/par.cm2au, ω3-par.ω0/par.cm2au, (R3ω.real)/np.max(np.real(R3ω)),
                            use='positive',   # try 'abs' too if needed
                            w1_max=win/2,
                            frac_thresh=0.15,
                            w3_window=win,
                            power=1.5)

Δx   = par.ω0D/par.cm2au /2
print(Δx, 'here')
test1 = np.min(np.abs(R3ω_Rephasing - R3ω_NonRephasing))
test2 = np.max(np.abs(R3ω_Rephasing - R3ω_NonRephasing))
print(test1/test2, 'this')
diag = np.linspace(-Δω,Δω,100)
lo, hi = -1, 1
levels = np.linspace(lo, hi, 500)
norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
X, Y = np.meshgrid(ω1-par.ω0/par.cm2au, ω3-par.ω0/par.cm2au)

# =================================== 
# REPHASING + NON-REPHASING
# ===================================
fig, ax = plt.subplots(figsize=(5,5), sharey=True)
cntr_left  = ax.contourf(X, Y, np.clip((R3ω.real)/np.max(np.abs(R3ω)), lo,hi).T, levels=levels, cmap='seismic', norm=norm, extend='neither')
ax.axvline(Δx, color='black', ls = '--', lw=1, alpha=0.3)
ax.axvline(-Δx, color='black', ls = '--', lw=1, alpha=0.3)
ax.axhline(Δx, color='black', ls = '--', lw=1, alpha=0.3)
ax.axhline(-Δx, color='black', ls = '--', lw=1, alpha=0.3)
ax.plot(diag,diag, color='black', ls = '--', lw=1, alpha=0.3)
ax.set_ylabel("ω$_3$ (cm$^{-1}$)")
ax.set_xlabel("ω$_1$ (cm$^{-1}$)")

div1 = make_axes_locatable(ax)
cax  = div1.append_axes("right", size="3.5%", pad=0.03)
cbar = fig.colorbar(cntr_left, cax=cax, ticks=[lo, 0, hi])
ax.set_title(f't₂ = {t2} fs')

xline = np.array([-Δω, Δω])
yline = m*xline + b
ax.plot(xline, yline, lw=2, color='black', label=f"CLS: {m:.3f}")
# ax.plot(rw1, rw3, ms=4, c = 'white')
ax.legend(frameon = False)
plt.tight_layout()
plt.savefig(f'./Data/2D_Spectra_t2_{t2}.png',dpi=500)
plt.close()

np.savetxt(f'./Data/CLS_t2_{t2}.dat', np.c_[t2,m,b])

# =================================== 
# REPHASING
# ===================================
# fig, ax = plt.subplots(1,2,figsize=(9,5), sharey=True)
# cntr_left  = ax[0].contourf(X, Y, np.clip((R3ω_Rephasing.real)/np.max((R3ω_Rephasing.real)).T, lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
# ax[0].axvline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axvline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axhline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axhline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].plot(diag,diag, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].set_ylabel("ω$_3$ (cm$^{-1}$)")
# ax[0].set_xlabel("ω$_1$ (cm$^{-1}$)")
# cntr_right  = ax[1].contourf(X, Y, np.clip((R3ω_Rephasing.imag)/np.max((R3ω_Rephasing.imag)), lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
# ax[1].axvline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axvline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axhline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axhline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].plot(diag,diag, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].set_xlabel("ω$_1$ (cm$^{-1}$)")

# div1 = make_axes_locatable(ax[1])
# cax  = div1.append_axes("right", size="3.5%", pad=0.03)
# cbar = fig.colorbar(cntr_right, cax=cax, ticks=[lo, 0, hi])

# ax[0].set_title("Real", fontsize=8)
# ax[1].set_title("Imag", fontsize=8)

# plt.tight_layout()
# plt.savefig(f'./Data/2D_Rephasing_t2_{t2}.png',dpi=500)
# plt.close()

# # =================================== 
# # NON-REPHASING
# # ===================================
# fig, ax = plt.subplots(1,2,figsize=(9,5), sharey=True)
# cntr_left  = ax[0].contourf(X, Y, np.clip((R3ω_NonRephasing.real)/np.max(np.real(R3ω_NonRephasing)), lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
# ax[0].axvline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axvline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axhline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].axhline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].plot(diag,diag, color='black', ls = '--', lw=1, alpha=0.5)
# ax[0].set_ylabel("ω$_3$ (cm$^{-1}$)")
# ax[0].set_xlabel("ω$_1$ (cm$^{-1}$)")
# cntr_right  = ax[1].contourf(X, Y, np.clip((R3ω_NonRephasing.imag)/np.max(np.abs(R3ω_NonRephasing)), lo,hi), levels=levels, cmap='seismic', norm=norm, extend='neither')
# ax[1].axvline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axvline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axhline(-Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].axhline(Δx, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].plot(diag,diag, color='black', ls = '--', lw=1, alpha=0.5)
# ax[1].set_xlabel("ω$_1$ (cm$^{-1}$)")

# div1 = make_axes_locatable(ax[1])
# cax  = div1.append_axes("right", size="3.5%", pad=0.03)
# cbar = fig.colorbar(cntr_right, cax=cax, ticks=[lo, 0, hi])

# ax[0].set_title("Real", fontsize=8)
# ax[1].set_title("Imag", fontsize=8)

# plt.tight_layout()
# plt.savefig(f'./Data/2D_NonRephasing_t2_{t2}.png',dpi=500)
# plt.close()

# # ===================================
# # Build grid
# # ===================================
# X, Y = np.meshgrid(ω1 - par.ω0/par.cm2au,
#                    ω3 - par.ω0/par.cm2au)

# # ===================================
# # Data (normalized properly)
# # ===================================
# Zr = R3ω.real
# Zi = R3ω.imag

# # normalize to global max abs (important!)
# global_max = np.max(np.real(np.concatenate([Zr.ravel(), Zi.ravel()])))
# Zr = Zr / global_max
# Zi = Zi / global_max

# # ===================================
# # Thresholds from paper
# # ===================================
# vmax = 0.85  * np.max([Zr.max()])
# vmin = 0.425 * np.min([Zr.min()])

# norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# # ===================================
# # Custom black → blue → green → yellow → red → white
# # ===================================
# colors = [
#     (0.0,  "black"),
#     (0.15, "navy"),
#     (0.30, "blue"),
#     (0.45, "limegreen"),
#     (0.60, "yellow"),
#     (0.80, "red"),
#     (1.0,  "white"),
# ]

# cmap = mcolors.LinearSegmentedColormap.from_list(
#     "custom_spectrum",
#     colors,
#     N=512
# )

# # enforce saturation outside limits
# cmap = cmap.copy()
# cmap.set_under("black")
# cmap.set_over("white")

# levels = np.linspace(vmin, vmax, 50)

# # ===================================
# # Plot
# # ===================================
# fig, ax = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

# cntr_left = ax[0].contourf(
#     X, Y, Zr.T,
#     levels=levels,
#     cmap=cmap,
#     norm=norm,
#     extend="both"
# )

# cntr_right = ax[1].contourf(
#     X, Y, Zi,
#     levels=levels,
#     cmap=cmap,
#     norm=norm,
#     extend="both"
# )

# # guide lines
# for a in ax:
#     a.axvline(-Δx, color="black", ls="--", lw=1, alpha=0.5)
#     a.axvline( Δx, color="black", ls="--", lw=1, alpha=0.5)
#     a.axhline(-Δx, color="black", ls="--", lw=1, alpha=0.5)
#     a.axhline( Δx, color="black", ls="--", lw=1, alpha=0.5)
#     a.plot(diag, diag, color="black", ls="--", lw=1, alpha=0.5)

# ax[0].set_ylabel("ω$_3$ (cm$^{-1}$)")
# ax[0].set_xlabel("ω$_1$ (cm$^{-1}$)")
# ax[1].set_xlabel("ω$_1$ (cm$^{-1}$)")

# ax[0].set_title("Real", fontsize=8)
# ax[1].set_title("Imag", fontsize=8)

# # ===================================
# # Colorbar
# # ===================================
# div1 = make_axes_locatable(ax[1])
# cax  = div1.append_axes("right", size="3.5%", pad=0.03)

# cbar = fig.colorbar(cntr_right, cax=cax,
#                     ticks=np.linspace(vmin, vmax, 6),
#                     extend='both')

# cbar.set_label("Scaled response")

# plt.tight_layout()
# plt.savefig(f'./Data/test_{t2}.png', dpi=500)
# plt.close()