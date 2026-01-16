import numpy as np
import numba as nb
from numpy.random import normal as nm
import matplotlib.pyplot as plt
from scipy.optimize import root
# ==================================

# FUNCTIONS
# ==================================
# DRUDE - LORENTZ SPECTRAL DENSITY
def J_DrudeL(λ, γ, ω):
    return (2 * γ * λ * ω) / (ω**2 + γ**2)

# BATH PARAMETERS
def BathParam(λD, γD, N, num):    
    ωj = np.zeros((N))
    cj = np.zeros((N), dtype = np.complex128)

    if num == False:
    # ANALYTIC DISCRETIZATION OF THE DRUDE - LORENTZ SPECTRAL DENSITY 
    # Huo. P., et al (Mol. Phys. 2012, 110, 1035–1052)
        arr = np.arange(0,N,1) + 1
        ω_max = 3 * γD #1.30 * γD
        ωj[:] = γD * np.tan(arr/N * np.arctan(ω_max/γD))
        cj[:] = 2 * ωj[:] * np.sqrt(λD * np.arctan(ω_max/γD)/(np.pi * N))
    
    else:
        ω  = np.linspace(1E-10,100*γD,50000)                   # FREQUENCY SCAN FOR BATH FREQUENCIES
        dω = ω[1] - ω[0]
    # NUMERICAL DISCRETIZATION OF SPECTRAL DENSITY
    # Walters, P. L.; et al.  J Comput Chem 2017, 38 (2), 110–115. https://doi.org/10.1002/jcc.24527.

        J = J_DrudeL(λD, γD, ω)  
    
        Fω = np.zeros(len(ω))
        for i in range(len(ω)):
            Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

        λs =  Fω[-1]
        for i in range(N):
            costfunc = np.abs(Fω-(((float(i)+0.5)/float(N))*λs))
            m = np.argmin((costfunc))
            ωj[i] = ω[m]
        cj[:] = ωj[:] * ((λs/(2*float(N)))**0.5)
    return cj, ωj

# ELECTRONIC HAMILTONIAN 
def Hel_cons():
    H  = np.zeros((Nt, Nt), dtype = np.complex128)
    H[0,0]         = 0            # Ground State
    H[1,1]         = -ϵ + E0          # Cavity
    H[2,2]         = ϵ + E0            # Cavity
    H[3,3]         = 0
    H[1,2], H[2,1] = J12, J12
    return H 

# PHYSICAL CONSTANTS
# ==================================
eV2au  = 0.036749405469679
meV2au = 0.036749405469679 / 1000
fs2au  = 41.341                           # 1 fs = 41.341 a.u.
ps2au  = 41.341 * 1000                          # 1 fs = 41.341 a.u.
cm2au  = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 
temp   = 300 / autoK
β      = 1 / temp 
# ==================================

# =====================================
# VARIABLES
# =====================================
NSites     = 2
M          = 1     # Mass
Nt         = 4
Ω          = 30 * cm2au 
g          = 12.8 * cm2au/(Nt**0.5)
print('Coupling per molecule -> ', g/cm2au)
E0         = 1050 * cm2au
ϵ          = 100 / 2 * cm2au
J12        = 100 * cm2au

μMat_sp = np.array([1.0, -0.2, 1.0, -0.2])
μMat_id = np.array([[0, 1], [0, 2], [1, 0], [2, 0]])
μMat = np.zeros((Nt,Nt))
μMat[1,0], μMat[0,1] = 1, 1
μMat[2,0], μMat[0,2] = -0.2, -0.2

# PLDM PARAMETERS
stype      = 0

# RATES
# nB   = 1 / (np.exp(ωc/temp)-1)
# Γexc = nB
# τc   = 1.5 * ps2au
# τA   = 4.6 * ps2au
# τD   = 4.5 * ps2au
# ΓCav = 1 / (τc)
# ΓAcp = 1 / (τA)
# ΓDon = 1 / (τD)
# LID  = np.loadtxt('Lind_data.txt')

ρ0 = np.zeros((Nt,Nt), dtype = np.complex128)
ρ0[0,0] = 1

# SIMULATION PARAMETERS ==============================
parallel = True                                          # DO PARALLELIZATION
# parallel = False                                          # DO PARALLELIZATION
Cpus     = 50                                              # NUMBER THE CPUS USE FOR PARALLELIZATION
NTraj    = 500                                          # NUMBER OF TRAJECTORIES
ntj      = 50
tf       = 300 * fs2au                                    # SIMULATION TIME IN FEMTOSECONDS
if parallel == False:
    Cpus     = 1                                              # NUMBER THE CPUS USE FOR PARALLELIZATION
    NTraj    = 1                                          # NUMBER OF TRAJECTORIES
    tf       = 200 * fs2au                                     # SIMULATION TIME IN FEMTOSECONDS
    ntj      = 2
dtN      = 10                                    # NUCLEAR TIME STEP
NSteps   = int(tf/dtN)                                     # NUMBER OF SIMULATION STEPS
Sim_time = np.array([(x * dtN) for x in range(NSteps)])    # SIMULATION TIMES ARRAY
Estep    = 20                                             # NUMBER OF ELECTRONIC STEPS PER NUCLEAR TIME STEP ⇒ MUST BE EVEN!!!!
dtE      = dtN/Estep                                       # ELECTRONIC TIME STEP
nskip    = 2                                           # FRAME SAVING RATE
print(np.round(dtN/fs2au ,2), '-> Nuclear Time step (fs)')
print(np.round(dtE/fs2au ,2), '-> Electronic Time step (fs)')

if NSteps%nskip == 0:
    nData = NSteps // nskip + 0
else :
    nData = NSteps // nskip + 1

# BATH PARAMETERS ==============================
ndof   = 100                                         # NUMBER OF BATH OSCILLATORS
γD     = 100 * cm2au                                  # BATH CHARACTERISTIC FREQUENCY   
λD     = 100 * cm2au                                  # BATH REORGANIZATION ENERGY  
num    = False                                      # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical
ci, ωi = BathParam(λD, γD, ndof, num)               # BATH COUPLINGS AND FREQUENCIES
cj, ωj = np.tile(ci, 2), np.tile(ωi, 2)
# TIME INDEPENDENT FUNCTIONS ==============================
Hel        = Hel_cons()                                          # ELECTRONIC HAMILTONIAN | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
if __name__ == '__main__':
    ω = np.arange(1E-8,1000, 0.1) * cm2au 
    J = J_DrudeL(λD, γD, ω)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(ω/cm2au, J, lw = 2, c = 'black')
    ax.set_xlabel('$\omega$ (cm$^{-1}$)')
    ax.set_ylabel('J($\omega$)')
    for k in range(ndof):
        ax.axvline(ωi[k]/cm2au, ls='-', lw = 1, c = 'blue')
    ax.set_xlim(0,200)
    plt.savefig('./Images/Spectral.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

    