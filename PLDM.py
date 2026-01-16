import numpy as np
import numba as nb
import Parameters as par
# ===================================
# INITIAL BATH PARAMETERS
@nb.jit(nopython=True)
def InitBath(data):
    σR = 1/np.sqrt(2 * par.ωj * np.tanh(par.β * par.ωj * 0.5))
    σP = np.sqrt(par.ωj/(2 * np.tanh(par.β * par.ωj * 0.5)))
    for k in range(par.ntj):
        data.R[:,k] = np.random.normal(loc=0.0, scale=1.0, size=len(par.ωj)) * σR
        data.P[:,k] = np.random.normal(loc=0.0, scale=1.0, size=len(par.ωj)) * σP
# ===================================
# MAPPING VARIABLES INITIALIZATION
@nb.jit(nopython=True)
def Init_ψ(data):
    data.ψF[:,:] = 0.0
    data.ψB[:,:] = 0.0
    for k in range(par.ntj):
        data.ψF[data.intiStateF,k] = (1 + 1j) / np.sqrt(2)
        data.ψB[data.intiStateB,k] = (1 - 1j) / np.sqrt(2)
# ===================================
# UPDATE MAPPING VARIABLES
@nb.jit(nopython=True)
def Evolve_ψ(data):
    ψFt, ψBt = data.ψF[:,:].copy(), data.ψB[:,:].copy()
    VMat     = par.Hel[:,:].copy()
    for k in range(par.ntj):
        VMat[1,1]    += np.sum(par.cj[:par.ndof] * data.R[:par.ndof,k])
        VMat[2,2]    += np.sum(par.cj[par.ndof:] * data.R[par.ndof:,k])
        # VMat[3,3]    += np.sum(par.cj * data.R[:,k])
        E, U          = np.linalg.eigh(VMat)
        UF            = U @ np.diag(np.exp(1j * par.dtE * E)) @ np.conjugate(U.T)
        UB            = U @ np.diag(np.exp(-1j * par.dtE * E)) @ np.conjugate(U.T)
        data.ψF[:,k]  = UF @ ψFt[:,k]
        data.ψB[:,k]  = ψBt[:,k] @ UB 
# ===================================
# COMPUTE FORCES
@nb.jit(nopython=True)
def Force1(data):
    data.Force1[:,:] = 0.0
    ψF_2 = np.absolute(data.ψF)**2
    ψB_2 = np.absolute(data.ψB)**2
    for k in range(par.ntj):
        data.Force1[:,k] -= par.ωj**2 * data.R[:,k]
        data.Force1[:par.ndof,k] -= par.cj[:par.ndof] * (ψF_2[1,k] + ψB_2[1,k])/2
        data.Force1[par.ndof:,k] -= par.cj[par.ndof:] * (ψF_2[2,k] + ψB_2[2,k])/2
        # data.Force1[:,k]         -= par.cj * np.sum(ψF_2[:,k] + ψB_2[:,k])/2
# ===================================
@nb.jit(nopython=True)
def Force2(data):
    data.Force2[:,:] = 0.0
    ψF_2 = np.absolute(data.ψF)**2
    ψB_2 = np.absolute(data.ψB)**2
    for k in range(par.ntj):
        data.Force2[:,k] -= par.ωj**2 * data.R[:,k]
        data.Force2[:par.ndof,k] -= par.cj[:par.ndof] * (ψF_2[1,k] + ψB_2[1,k])/2
        data.Force2[par.ndof:,k] -= par.cj[par.ndof:] * (ψF_2[2,k] + ψB_2[2,k])/2
        # data.Force2[:,k]         -= par.cj * np.sum(ψF_2[:,k] + ψB_2[:,k])/2
# ===================================
# VELOCITY VERLET
@nb.jit(nopython=True)
def VelVerlet(data):
    data.v[:,:] = data.P[:,:] / par.M * 1.0 
    # HALF STEP MAPPING
    for t in range(int(par.Estep/2)):
        Evolve_ψ(data)
    # NUCLEAR STEP
    Force1(data)
    data.R[:,:]     += par.dtN * data.v + 0.5 * (par.dtN**2/par.M) * data.Force1
    for t in range(int(par.Estep/2)):
        Evolve_ψ(data)
    Force2(data)
    data.v[:,:]     += 0.5 * (data.Force1 + data.Force2) * par.dtN / par.M
    data.P[:,:]      = data.v * par.M
# ===================================
@nb.jit(nopython=True)
def RunTraj(data):
    InitBath(data)
    Init_ψ(data)
    data.ρRe[:,:] = 0
    iskip = 0
    # ===================================
    for j in range(data.NSteps):
        # ESTIMATOR
        if (j % par.nskip == 0):
            for k in range(par.ntj):
                data.ρRe[iskip,:] += (np.outer(data.ψF[:,k], data.ψB[:,k])).flatten()
            iskip += 1
    # ===================================
        VelVerlet(data)
    data.ρRe = data.ρRe / par.ntj
    
