import numpy as np
import numba as nb
import Parameters as par
# ===================================
# MONTE CARLO SCHEME
@nb.njit
def phase_diff(z1, z2):
    return np.arctan2(z1.imag, z1.real) - np.arctan2(z2.imag, z2.real)

@nb.njit
def Focus1D(data):
    UniRan = np.random.uniform(0,1,data.ntj)
    data.ψF[:,:] = 0.0
    data.ψB[:,:] = 0.0
    for traj in range(data.ntj):
        CumSum1D(data,UniRan[traj],traj)
        data.ψF[data.ind_FL[traj,0],traj] = (1 + 1j) * par.Sqrt2_inv
        data.ψB[data.ind_FL[traj,1],traj] = (1 - 1j) * par.Sqrt2_inv

@nb.njit
def CumSum1D(data,UniRandom,traj):
    CS  = np.zeros((par.Nt * par.Nt), dtype = np.float64)
    s = 0
    if data.Dir == 1:
        ψF_dir = par.μMat @ par.ψF0
        ψB_dir = np.conjugate(par.ψB0)
    elif data.Dir == 0:
        ψF_dir = par.ψF0
        ψB_dir = np.conjugate(par.μMat @ par.ψB0)
    if UniRandom < 0.5:
        k = 1
    elif UniRandom < 1.0:
        k = 2
    # for i in range(par.Nt):
    #     for j in range(par.Nt):
    #         s += np.abs(ψF_dir[i]) * np.abs(ψB_dir[j])
    #         CS[i * par.Nt + j] = s               # THIS STEP IS NOT NEEDED as it can be done in a single step
    # CS = np.real(CS)/np.max(np.real(CS))         # THIS STEP IS NOT NEEDED as it can be done in a single step
    # for k in range(par.Nt * par.Nt):             # THIS STEP IS NOT NEEDED as it can be done in a single step
    #     if CS[k] >= UniRandom:                   # THIS STEP IS NOT NEEDED as it can be done in a single step
    #         break                                # THIS STEP IS NOT NEEDED as it can be done in a single step
    data.ind_FL[traj, 0] = int(k // par.Nt)
    data.ind_FL[traj, 1] = int(k % par.Nt)
    data.w_FL[traj] =  np.abs(ψF_dir[data.ind_FL[traj, 0]]) * np.abs(ψB_dir[data.ind_FL[traj, 1]])

@nb.njit
def Focus2D(data):
    data.ψF[:,:] = 0.0
    data.ψB[:,:] = 0.0
    UniRan = np.random.uniform(0,1,data.ntj)
    if data.Dir == 1:
        ψF_dir = par.μMat @ data.ψF0.copy()
        ψB_dir = np.conjugate(data.ψB0.copy())
    elif data.Dir == 0:
        ψF_dir = data.ψF0.copy()
        ψB_dir = np.conjugate(par.μMat @ data.ψB0.copy())
    absF = np.abs(ψF_dir)  # float array
    absB = np.abs(ψB_dir)
    for traj in range(data.ntj):
        CumSum2D(data,UniRan[traj],traj, ψF_dir, ψB_dir, absF, absB)
        data.ψF[data.ind_SL[traj,0],traj] = (1 + 1j) * par.Sqrt2_inv
        data.ψB[data.ind_SL[traj,1],traj] = (1 - 1j) * par.Sqrt2_inv

@nb.njit
def CumSum2D(data,UniRandom,traj, ψF_dir, ψB_dir, absF, absB):
    CS  = np.zeros((par.Nt * par.Nt), dtype = np.float64)
    s = 0
    for i in range(par.Nt):
        for j in range(par.Nt):
            s += absF[i,traj] * absB[j,traj]
            CS[i * par.Nt + j] = np.real(s)
    CS = CS/np.max(CS)
    for k in range(par.Nt * par.Nt):
        if CS[k] >= UniRandom:
            break
    data.ind_SL[traj, 0] = int(k // par.Nt)
    data.ind_SL[traj, 1] = int(k % par.Nt)
    data.θ_SL[traj,data.Laser-2] = phase_diff(ψF_dir[data.ind_SL[traj, 0],traj],ψB_dir[data.ind_SL[traj, 1],traj])
    data.r_SL[traj,data.Laser-2] = np.absolute(ψF_dir[data.ind_SL[traj, 0],traj]) * np.absolute(ψB_dir[data.ind_SL[traj, 1],traj])
# ===================================
# INITIAL BATH PARAMETERS
@nb.njit
def InitBath(data):
    σR = 1/np.sqrt(2 * par.ωj * np.tanh(par.β * par.ωj * 0.5))
    σP = np.sqrt(par.ωj/(2 * np.tanh(par.β * par.ωj * 0.5)))    
    for k in range(data.ntj):
        data.R[:,k] = np.random.normal(loc=0.0, scale=1.0, size=len(par.ωj)) * σR
        data.P[:,k] = np.random.normal(loc=0.0, scale=1.0, size=len(par.ωj)) * σP
# ===================================
# MAPPING VARIABLES INITIALIZATION
@nb.njit
def Init_ψ(data):
    data.ψF[:,:] = 0.0
    data.ψB[:,:] = 0.0
    for k in range(data.ntj):
        data.ψF[data.intiStateF,k] = (1 + 1j) / np.sqrt(2)
        data.ψB[data.intiStateB,k] = (1 - 1j) / np.sqrt(2)
# ===================================
# UPDATE MAPPING VARIABLES
@nb.njit
def Evolve_ψ(data):
    ψFt, ψBt = data.ψF[:,:].copy(), data.ψB[:,:].copy()
    for k in range(data.ntj):
        VMat          = par.Hel[:,:].copy()
        for Site in range(par.NSites):
            VMat[Site+1,Site+1] += np.sum(par.cj[Site * par.ndof:(Site + 1) * par.ndof] * data.R[Site * par.ndof:(Site + 1) * par.ndof,k])
        Hscl          = (VMat - par.bC_Identity) * par.inv_aC   
        # CHEBYSHEV PROPAGATION
        ϕ0F,ϕ1F = ψFt[:,k], Hscl @ ψFt[:,k]
        ϕ0B,ϕ1B = ψBt[:,k], Hscl @ ψBt[:,k]
        JkF, JkB = par.Coeff_Bessel[0] * ϕ0F + par.Coeff_Bessel[1] * ϕ1F, par.Coeff_Bessel[0] * ϕ0B + np.conjugate(par.Coeff_Bessel[1]) * ϕ1B
        for kCheby in range(2,par.KCheby):
            ϕkF, ϕkB = 2 * Hscl @ ϕ1F - ϕ0F, 2 * Hscl @ ϕ1B - ϕ0B
            JkF    += par.Coeff_Bessel[kCheby] * ϕkF
            JkB    += np.conjugate(par.Coeff_Bessel[kCheby]) * ϕkB
            ϕ0F,ϕ1F = ϕ1F, ϕkF
            ϕ0B,ϕ1B = ϕ1B, ϕkB
        data.ψF[:,k]  = par.exp_mtau * JkF 
        data.ψB[:,k]  = par.exp_ptau * JkB 
# ===================================
# COMPUTE FORCES
@nb.njit
def Force1(data):
    data.Force1[:,:] = 0.0
    ψF_2 = np.absolute(data.ψF)**2
    ψB_2 = np.absolute(data.ψB)**2
    for k in range(data.ntj):
        F                = np.zeros((par.NSites * par.ndof), dtype = np.complex128)
        F               -= par.ωj**2 * data.R[:,k]
        for Site in range(par.NSites):
            F[Site * par.ndof:(Site + 1) * par.ndof] -= par.cj[Site * par.ndof:(Site + 1) * par.ndof] * (ψF_2[Site+1,k] + ψB_2[Site+1,k])/2     # GROUND STATE IS SITE 0
        data.Force1[:,k] = F
# ===================================
@nb.njit
def Force2(data):
    data.Force2[:,:] = 0.0
    ψF_2 = np.absolute(data.ψF)**2
    ψB_2 = np.absolute(data.ψB)**2
    for k in range(data.ntj):
        F                = np.zeros((par.NSites * par.ndof), dtype = np.complex128)
        F               -= par.ωj**2 * data.R[:,k]
        for Site in range(par.NSites):
            F[Site * par.ndof:(Site + 1) * par.ndof] -= par.cj[Site * par.ndof:(Site + 1) * par.ndof] * (ψF_2[Site+1,k] + ψB_2[Site+1,k])/2     # GROUND STATE IS SITE 0
        data.Force2[:,k] = F
# ===================================
# VELOCITY VERLET
@nb.njit
def VelVerlet(data):
    data.v[:,:]      = data.P[:,:] / par.M * 1.0 
    # HALF STEP MAPPING
    for t in range(int(par.Estep/2)):
        Evolve_ψ(data)
    # NUCLEAR STEP
    data.R[:,:]     += par.dtN * data.v[:,:] + 0.5 * (par.dtN**2/par.M) * data.Force1[:,:]
    for t in range(int(par.Estep/2)):
        Evolve_ψ(data)
    Force2(data)
    data.v[:,:]     += 0.5 * (data.Force1[:,:] + data.Force2[:,:]) * par.dtN / par.M
    data.P[:,:]      = data.v[:,:] * par.M
    data.Force1[:,:] = data.Force2[:,:]
# ===================================
@nb.njit
def RunTraj(data):
    if par.LaserStage == 0:
        InitBath(data)
        for k in range(data.ntj):
            data.ψF[1,k] = (1 + 1j) / np.sqrt(2)
            data.ψB[1,k] = (1 - 1j) / np.sqrt(2)
        Force1(data)
        iskip = 0
        for j in range(data.NSteps):
            # ESTIMATOR
            if (j % par.nskip == 0):
                data.ψFw[iskip,:,:] = data.ψF[:,:]
                data.ψBw[iskip,:,:] = data.ψB[:,:]
                data.Rw[iskip,:,:]  = data.R[:,:]
                data.Pw[iskip,:,:]  = data.P[:,:]
                iskip += 1
            VelVerlet(data)
# =========================================================
    elif par.LaserStage == 1:
        InitBath(data)
        data.Dir = par.k_dir1
        Focus1D(data)
        Force1(data)
        iskip = 0
        for j in range(data.NSteps):
            # ESTIMATOR
            if (j % par.nskip == 0):
                data.ψFw[iskip,:,:] = data.ψF[:,:]
                data.ψBw[iskip,:,:] = data.ψB[:,:]
                data.Rw[iskip,:,:]  = data.R[:,:]
                data.Pw[iskip,:,:]  = data.P[:,:]
                iskip += 1
            VelVerlet(data)
# =========================================================
    elif par.LaserStage == 3:
        data.Laser = 2
        data.Dir = par.k_dir2
        Focus2D(data)
        Force1(data)
        if par.NSteps_L2 != 0:
            for j in range(par.NSteps_L2):
                VelVerlet(data)
# =========================================================
        data.Laser = 3
        data.ψF0[:,:] = data.ψF[:,:].copy()
        data.ψB0[:,:] = data.ψB[:,:].copy()
        data.Dir = par.k_dir3
        Focus2D(data)
        Force1(data)
        iskip = 0
        for j in range(data.NSteps):
            if (j % par.nskip == 0):
                μF = par.μMat @ data.ψF[:,:]
                for traj in range(data.ntj):
                    w1 = data.w_FL[traj]
                    w2 = data.r_SL[traj,0] * np.exp(1j * data.θ_SL[traj,0])
                    w3 = data.r_SL[traj,1] * np.exp(1j * data.θ_SL[traj,1])
                    data.ψFw[iskip,:,traj] = w1 * w2 * w3 * μF[:,traj]
                data.ψBw[iskip,:,:] = data.ψB[:,:]
                iskip += 1
            VelVerlet(data)


