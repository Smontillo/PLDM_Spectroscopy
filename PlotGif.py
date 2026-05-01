import imageio
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']
# =================================== 
# FUNCTIONS
# ===================================
def ExpDec(x,a,k,c):
    return a * np.exp(-1/k * x) #+ c
# =================================== 
# LOADING FILES
# ===================================
Data = np.loadtxt("./ExpData.dat")
t2_Exp = Data[:,0]
CLS_Exp = Data[:,1]

folder = "./Data"
files = [f for f in os.listdir(folder) if f.startswith("CLS_t2_") and f.endswith(".dat")]
t2 = []
m = []
b = []
for file in files:
    data = np.loadtxt(os.path.join(folder, file))
    t2.append(data[0])
    m.append(data[1])
    b.append(data[2])
t2 = np.array(t2)
m, b = np.array(m), np.array(b)
idx = np.argsort(t2)   # indices that would sort t2
t2_sorted = t2[idx]
m_sorted = m[idx]
b_sorted = b[idx]
print(t2)
print(t2_sorted)
# =================================== 
# CLS FITTING
# ===================================
p0         = [m_sorted[0] - m_sorted[-1], 1e3, m_sorted[-1]]
popt, pcov = curve_fit(ExpDec, t2_sorted, m_sorted, p0 = p0,maxfev=10000)
popt_Exp, pcov_Exp = curve_fit(ExpDec, t2_Exp, CLS_Exp, p0 = p0,maxfev=10000)
# =================================== 
# PLOTTING CLS
# ===================================
t2_con = np.linspace(t2_sorted.min(),t2_Exp.max(),100)
fig, ax = plt.subplots(figsize = (3,3))
# ax.plot(t2_Exp,CLS_Exp, ls = '', lw = 1, marker = 'o', markersize = 5, c = col[0], label = "Experiment", alpha = 0.5)
# ax.plot(t2_con,ExpDec(t2_con,*popt_Exp), ls = '-', lw = 3, c = col[0], alpha = 0.5)
ax.plot(t2_con,ExpDec(t2_con,*popt), ls = '-', lw = 2, c = col[1], label = r"Fit $\propto e^{-1/\tau \cdot t}$")
ax.plot(t2_sorted,m_sorted, ls = '', lw = 1, marker = 'o', markersize = 5, c = col[1], label = "Simulation")
ax.set_xlim(t2_sorted.min(),t2_Exp.max())
ax.set_xlabel("$t_2$ (fs)")
ax.set_ylabel("CLS")
ax.text(0.5, 0.65, r"$\tau_{sim} = $" + str(round(popt[1]/1000,2)) + " ps", transform=ax.transAxes, fontsize=8, verticalalignment='top')
# ax.text(0.5, 0.58, r"$\tau_{exp} = $" + str(round(popt_Exp[1]/1000,2)) + " ps", transform=ax.transAxes, fontsize=8, verticalalignment='top')
ax.legend(frameon = False, fontsize = 8)
plt.savefig("./Data/CLS.png", dpi = 300, bbox_inches='tight')
plt.close()
# =================================== 
# 2D SPECTRA GIF
# ===================================
# collect all files matching your pattern
files = [f for f in os.listdir(folder) if f.startswith("2D_Spectra_t2_") and f.endswith(".png")]

# sort numerically by t2
def extract_t2(filename):
    return int(re.search(r"t2_(\d+)", filename).group(1))

files = sorted(files, key=extract_t2)

images = []
for file in files:
    images.append(imageio.imread(os.path.join(folder, file)))

# save GIF
imageio.mimsave("./Data/2D_Spectra_animation.gif",
                images,
                duration=0.4)   # seconds per frame

print("GIF saved successfully.")


    