import numpy as np
import numba as nb
from numpy.random import normal as nm
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import jv
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
        ω_max = 200 * cm2au #5 * γD #1.30 * γD
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
    H[0,0]          = 0            # Ground State
    H[1,1]          = ω0 
    return H 

# CHEBYSHEV SCAILING:
def ChebyScaling(H,λ):
    Hc       = H.copy()
    Hc[1,1] += 2 * λ
    E,V      = np.linalg.eigh(Hc)
    a        = (np.max(E) - np.min(E)) / 2
    b        = (np.max(E) + np.min(E)) / 2
    Hscl     = (Hc - b * np.identity(len(E))) / a
    return a, b

# CHEBYSHEV ORDER
def ChebyOrder(a,Δt,ϵ=1e-5,Kmax=1000):
    # EVALUATE THE CHEBYSHEV ORDER
    τ = a * Δt
    for k in range(2,Kmax):
        if ( 2 * np.abs(jv(k,τ)) < ϵ):
            KCheby = k
            break
    # CREATE THE CHEBYSHEV COEFFICIENTS
    Coeff_Bessel = np.zeros((KCheby+1), dtype = np.complex128)
    Coeff_Bessel[0] = jv(0,τ)
    for k in range(1,KCheby+1):
        Coeff_Bessel[k] = 2 * (-1j)**k * jv(k,τ)
    return KCheby, Coeff_Bessel, τ
    
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
NSites     = 1
M          = 1     # Mass
Nt         = NSites + 1
ω0         = 1730 * cm2au
ϵ          = 100 / 2 * cm2au
J12        = 100 * cm2au

μMat_sp = np.array([1])
μMat_id = np.array([[1,0]])
μMat = np.zeros((Nt,Nt), dtype = np.complex128)
μMat[1,0], μMat[0,1] = μMat_sp[0], μMat_sp[0]

# PLDM PARAMETERS
stype      = 0

### INITIAL CONDITIONS ###
ρ0             = np.zeros((Nt,Nt), dtype = np.complex128)
ρ0[0,0]        = 1
ψF0, ψB0       = np.zeros((Nt), dtype = np.complex128), np.zeros((Nt), dtype = np.complex128)
ψF0[0], ψB0[0] = 1, 1
μF0, μB0       = μMat @ ψF0, μMat @ ψB0
### =================== ###
Sqrt2_inv = 1 / np.sqrt(2)
### ket → 1, bra → 0 ###
R1,R2,R3,R4 = [1,0,0], [0,1,0], [0,0,1], [1,1,1]
Louville_Paths = np.array([R1,R2,R3,R4])
Path = 0
k_dir1 = Louville_Paths[Path][0]
k_dir2 = Louville_Paths[Path][1]
k_dir3 = Louville_Paths[Path][2]
t2     = 0
# SIMULATION PARAMETERS ==============================
Spectra  = True
LaserStage = 3
parallel = True                                          # DO PARALLELIZATION
# parallel = False                                          # DO PARALLELIZATION
Cpus     = 50                                              # NUMBER THE CPUS USE FOR PARALLELIZATION
NTraj    = 100                                          # NUMBER OF TRAJECTORIES
ntj      = 40
# === LASER TIMES === #
tf_Las1  = 1500 * fs2au                                    # SIMULATION TIME FOR LASER 1
tf_Las2  = t2 * fs2au                                    # SIMULATION TIME FOR LASER 2
tf_Las3  = 1500 * fs2au                                    # SIMULATION TIME FOR LASER 3
if parallel == False:
    Cpus     = 1                                              # NUMBER THE CPUS USE FOR PARALLELIZATION
    NTraj    = 1                                          # NUMBER OF TRAJECTORIES
    tf_Las1  = 50 * fs2au                                    # SIMULATION TIME FOR LASER 1
    tf_Las2  = 0 * fs2au                                    # SIMULATION TIME FOR LASER 2
    tf_Las3  = 50 * fs2au                                    # SIMULATION TIME FOR LASER 3
    ntj      = 10
dtN      = 5 * fs2au                                    # NUCLEAR TIME STEP
# === SIMULATION STEPS === #
NSteps_L1   = int(tf_Las1/dtN)                                     # NUMBER OF SIMULATION STEPS
NSteps_L2   = int(tf_Las2/dtN)                                     # NUMBER OF SIMULATION STEPS
NSteps_L3   = int(tf_Las3/dtN)                                     # NUMBER OF SIMULATION STEPS
# === SIMULATION TIMES === #
Sim_time_L1 = np.array([(x * dtN) for x in range(NSteps_L1)])    # SIMULATION TIMES ARRAY
Sim_time_L2 = np.array([(x * dtN) for x in range(NSteps_L2)])    # SIMULATION TIMES ARRAY
Sim_time_L3 = np.array([(x * dtN) for x in range(NSteps_L3)])    # SIMULATION TIMES ARRAY
# === ELECTRONIC STEPS === #
Estep    = 10                                             # NUMBER OF ELECTRONIC STEPS PER NUCLEAR TIME STEP ⇒ MUST BE EVEN!!!!
dtE      = dtN/Estep                                       # ELECTRONIC TIME STEP
nskip    = 1                                          # FRAME SAVING RATE
# print(f'{dtN/fs2au:.2e}, -> Nuclear Time step (fs)')
# print(f'{dtE/fs2au:.2e}, -> Electronic Time step (fs)')

# === SIMULATION DATA === #
if NSteps_L1%nskip == 0:
    nData_L1 = NSteps_L1 // nskip + 0
else :
    nData_L1 = NSteps_L1 // nskip + 1

if NSteps_L2%nskip == 0:
    nData_L2 = NSteps_L2 // nskip + 0
else :
    nData_L2 = NSteps_L2 // nskip + 1

if nData_L2 ==0:
    nData_L2 = 1

if NSteps_L3%nskip == 0:
    nData_L3 = NSteps_L3 // nskip + 0
else :
    nData_L3 = NSteps_L3 // nskip + 1

# BATH PARAMETERS ==============================
ndof   = 100                                         # NUMBER OF BATH OSCILLATORS
γD     = 0.95 * cm2au                                  # BATH CHARACTERISTIC FREQUENCY   
λD     = 0.2586 * cm2au                                  # BATH REORGANIZATION ENERGY  
num    = False                                      # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical
cj, ωj = BathParam(λD, γD, ndof, num)               # BATH COUPLINGS AND FREQUENCIES
# cj, ωj = np.concatenate((ci, ci)), np.concatenate((ωi, ωi))
# TIME INDEPENDENT FUNCTIONS ==============================
Hel          = Hel_cons()                                          # ELECTRONIC HAMILTONIAN | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
aC, bC = ChebyScaling(Hel,λD)
KCheby, Coeff_Bessel, τ = ChebyOrder(aC,dtE)
τ = bC * dtE
exp_mtau = np.exp(-1j * τ)
exp_ptau = np.exp( 1j * τ)
inv_aC   = 1.0 / aC
b_over_a = bC / aC
bC_Identity = bC * np.identity(Nt)
# print(f'{KCheby:.0f}, -> Chebyshev Order')
E, V = np.linalg.eigh(Hel)
ω0D = E[1] - ω0
if __name__ == '__main__':
    print((E)/cm2au, ω0D/cm2au)




    
