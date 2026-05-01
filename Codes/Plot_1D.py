import numpy as np
import matplotlib.pyplot as plt
import Parameters as par
import time
import scipy.constants as sc
from scipy.optimize import curve_fit
from mpi4py import MPI
import os
import socket
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']
ExpVal = [3.376, 9.855, 1724.4]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"[Rank {rank}] started on {socket.gethostname()} out of {size} total ranks", flush=True)
# ===================================
# FUNCTIONS
# ===================================
def Populations():
    Pop = np.zeros((par.nData, par.Nt), dtype = np.float64)
    ρt  = np.zeros((par.nData, par.Nt * par.Nt), dtype = np.complex128)
    for j in range(len(par.μMat_sp)):
        print('Mu parameter: ', par.μMat_sp[j], par.μMat_id[j])
        ρt[:,:] = 0.0
        for i in range(par.Cpus):
            ρt += np.loadtxt(f'./Data/rhoRe_{i}_{par.μMat_id[j,0]}_{par.μMat_id[j,1]}.txt', dtype = np.complex128)
        ρt /= (len(par.μMat_sp) * par.Cpus)
        for k in range(par.nData):
            Pop[k,:] += np.real(np.diag(ρt[k,:].reshape((par.Nt, par.Nt))))
    return Pop

def TrajPerProc():
    tot_Tasks = par.NTraj
    NTasks = tot_Tasks// par.Cpus
    return NTasks

def Pop_1D():
    NTasks = TrajPerProc()
    ρ = np.zeros((par.nData_L1, par.Nt), dtype = np.complex128)
    for nrank in range(par.Cpus):
        for task in range(NTasks):
            ψF_all = np.load(f'./Data_Laser1/psiF_{nrank}_Task_{task}.npy')
            ψB_all = np.load(f'./Data_Laser1/psiB_{nrank}_Task_{task}.npy')
            for traj in range(par.ntj):
                ψF    = ψF_all[:,:,traj]
                ψB    = ψB_all[:,:,traj]
                for time in range(par.nData_L1):
                    # Rt1 COMPUTATION #
                    # term1 = sum_k ψB(t,k) ψB0(k)  *  sum_p ψF(t,p) (μψF0)(p)
                    ρ[time,:] += ψF[time,:] * ψB[time,:]
    ρ /= par.Cpus * par.ntj
    return ρ      

def build_jobs():
    # Build the same logical jobs your serial code iterates over.
    # If your data exists for nrank in [0..par.Cpus-1], tasks in [0..NTasks-1], traj in [0..par.ntj-1]
    NTasks = TrajPerProc()
    jobs = []
    for task in range(NTasks):
        for traj in range(par.ntj):
            jobs.append((task, traj))
    return jobs

def LoadData_1D_mpi(data_dir="./Data_Laser1", root=0, reduce_to_root=True):

    # Precompute constants (same on all ranks)
    muF0 = par.μMat @ par.ψF0

    # Build and scatter work logically (each rank takes a strided subset)
    jobs = build_jobs()
    my_jobs = jobs#[rank::size]

    R1t_local = np.zeros((par.nData_L1,), dtype=np.complex128)
    ψB0 = par.ψB0
    ψF0 = par.ψF0
    μMat = par.μMat
    for (task,traj) in my_jobs:
        psiF_path = os.path.join(data_dir, f"psiF_{rank}_Task_{task}.npy")
        psiB_path = os.path.join(data_dir, f"psiB_{rank}_Task_{task}.npy")
        ind_path  = os.path.join(data_dir, f"Ind_{rank}_Task_{task}.npy")

        ψF  = np.load(psiF_path)
        ψB  = np.conjugate(np.load(psiB_path))
        Ind = np.load(ind_path)

        for time in range(par.nData_L1):     
            psiF_t = ψF[time, :, traj]
            psiB_t = (ψB[time, :, traj])

            mu = par.μMat
            i = int(Ind[traj,0]); j = int(Ind[traj,1])

            s1 = (
                np.vdot(psiB_t, mu @ par.ψF0) * np.vdot(par.ψB0, psiF_t)
                - np.vdot(psiB_t, par.ψF0)   * np.vdot(par.ψB0, mu @ psiF_t)
            )

            # --- Second trace: tr(ρ† μ0) = <ψF_t| μ0 |ψB_t>
            s2 = (
                np.vdot(psiF_t, mu @ par.ψF0) * np.vdot(par.ψB0, psiB_t)
                - np.vdot(psiF_t, par.ψF0)    * np.vdot(par.ψB0, mu @ psiB_t)
            )

            R1t_local[time] += 1j * s1 + 1j * s2 
    
    # scale this trajectory’s contribution once
            R1t_local *= np.real(par.μMat[int(Ind[traj,0]),int(Ind[traj,1])] )
        # R1t_local += mu_scalar * contrib
    # --- Reduce across ranks ---
    if reduce_to_root:
        R1t_global = None
        if rank == root:
            R1t_global = np.zeros_like(R1t_local)

        comm.Reduce(R1t_local,R1t_global, op=MPI.SUM, root=root)
        if rank == root:
            # Total number of jobs (i.e., total trajectories/files processed)
            n_total = len(jobs) * 2 * par.Cpus
            R1t_global /= n_total
            return R1t_global
        else:
            return None
    else:
        # Everyone gets the result
        R1t_global = np.zeros_like(R1t_local)
        comm.Allreduce([R1t_local, MPI.COMPLEX], [R1t_global, MPI.COMPLEX], op=MPI.SUM)
        n_total = len(jobs)
        # R1t_global /= n_total
        return R1t_global

def LoadData_1D():
    muF0 = par.μMat @ par.ψF0
    NTasks = TrajPerProc()
    R1t = np.zeros((par.nData_L1), dtype = np.complex128)
    for nrank in range(par.Cpus):
        for task in range(NTasks):
            for traj in range(par.ntj):
                ψF    = np.load(f'./Data_Laser1/psiF_{nrank}_Task_{task}_traj_{traj}.npy', dtype = np.complex128)   # RUN IN PARALLEL
                ψB    = np.load(f'./Data_Laser1/psiB_{nrank}_Task_{task}_traj_{traj}.npy', dtype = np.complex128)   # RUN IN PARALLEL
                Ind   = np.load(f'./Data_Laser1/Ind_{nrank}_Task_{task}_traj_{traj}.npy')   # RUN IN PARALLEL
                μB = (par.μMat @ ψB[time, :]).T
                for time in range(par.nData_L1):
                    # ρ = np.outer(ψF[time,:], ψB[time,:])
                    # R1t[time] += 1j * np.trace(ρ @ μ0) + 1j * np.trace(np.conjugate(ρ.T) @ μ0)  * par.μMat[int(Ind[0]),int(Ind[1])]
                    # Rt1 COMPUTATION #
                    # term1 = sum_k ψB(t,k) ψB0(k)  *  sum_p ψF(t,p) (μψF0)(p)
                    s1 = np.dot(par.ψB0, ψB[time, :])               # B0^T B
                    s2 = np.dot(par.ψB0, μB[time,:])    # B0^T (μ B)

                    a  = np.dot(ψF[time, :], muF0)                  # F^T (μ F0)
                    b  = np.dot(ψF[time, :], par.ψF0)               # F^T F0

                    R1t[time] -= (a*s1 - b*s2) * 1j + np.conjugate((a*s1 - b*s2)) * 1j
                R1t *= np.real(par.μMat[int(Ind[0]),int(Ind[1])] )
    R1t /= par.Cpus * par.ntj
    return R1t    
                    
def LoadData():
    R1t = np.zeros((par.nData_L1), dtype = np.complex128)
    ρt  = np.zeros((par.nData_L1, par.Nt * par.Nt), dtype = np.complex128)
    ρt[:,:] = 0.0
    ρt += np.loadtxt(f'./Data_Laser1/rhoRe.txt', dtype = np.complex128)
    ρt /= par.Cpus
    for k in range(par.nData_L1):
            ρNt = ρt[k,:].reshape((par.Nt, par.Nt)) # * par.μMat_sp[j]
            R1t[k] += 1j * np.trace(ρNt @ μ0) + 1j * np.trace(np.conjugate(ρNt.T) @ μ0) 
    return R1t
    
def Fourier1D(R1t):
    c_cm_per_fs = sc.c * 100.0 / 1e15 
    Δω   = 800
    ωmin = (par.ω0/par.cm2au - Δω) * 2 * np.pi * c_cm_per_fs  
    ωmax = (par.ω0/par.cm2au + Δω) * 2 * np.pi * c_cm_per_fs  
    lenω = 1001
    ω    = np.linspace(ωmin, ωmax, lenω)
    δω   = ω[1] - ω[0]

    τ    = time 
    T    = τ.max()
    δtN  = τ[1] - τ[0]

    window = np.cos(np.pi * τ / (2 * T))
    phase  = np.exp(1j * ω[:, None] * τ[None, :])
    R1ω    = np.sum((R1t * window)[None, :] * phase, axis=1) * δtN
    R1ω_Area = np.trapz(np.abs(R1ω), dx = δω)

    return -np.imag(R1ω)/np.max(np.abs(R1ω)), np.real(R1ω)/np.max(np.abs(R1ω)), ω / ((2 * np.pi) * c_cm_per_fs)

def Lorentzian(x,A,Γ,ω0):
    return A * Γ / (np.pi * (Γ**2 + (x - ω0)**2))
# ===================================
time = par.Sim_time_L1[::par.nskip]/par.fs2au 
Δx = 100
if par.LaserStage != 0:
    An = np.loadtxt('/scratch/smontill/UT_Texas_Projects/Baiz_Collab/AnSpectra/R1ω.dat')
    # μ0   = (par.μMat @ par.ρ0) - (par.ρ0 @ par.μMat)
    R1t = LoadData_1D_mpi()
    if rank == 0:
        R1ω_Im, R1ω_Re, ω = Fourier1D(R1t)

        np.savetxt('R1ω.dat', np.c_[ω, R1ω_Im, R1ω_Re])
        
        fig, ax = plt.subplots(1,2, figsize = (6,3))
        ax[0].plot(time, np.real(R1t), lw = 1, c = col[0], label = 'Correlation Function')
        ax[0].plot(time, np.imag(R1t), lw = 1, c = col[1], label = 'Correlation Function')
        ax[1].plot(ω-par.ω0/par.cm2au, R1ω_Im, lw = 3, c = col[0], label = 'Simulation')
        # ax[1].plot(ω-par.ω0/par.cm2au, R1ω_Re, lw = 1, c = col[1], label = 'Real')
        ax[1].plot(An[:,0]-par.ω0/par.cm2au, An[:,1], lw = 0.5, c = col[1], label = 'Fitted')
        ax[0].set_ylabel('R1(t)')
        ax[1].set_ylabel('R1(ω)')
        ax[0].set_xlabel('Time (fs)')
        ax[1].set_xlabel('Frequency (cm$^{-1}$)')
        ax[0].set_xlim(0,time[-1])
        ax[1].set_xlim(-Δx, +Δx)
        ax[0].legend(frameon=False, fontsize = 5)
        ax[1].legend(frameon=False, fontsize = 5)
        fig.tight_layout()
        plt.savefig('./Data/Spectra1D.png', dpi = 300, bbox_inches = 'tight')
        plt.close()

else:
    Pop = Pop_1D()
    fig, ax = plt.subplots(figsize = (5,3))
    ax.plot(time, Pop[:,0], lw = 3, c = col[0], label = '|0>')
    ax.plot(time, Pop[:,1], lw = 3, c = col[1], label = '|1>')
    # ax.plot(time, Pop[:,2], lw = 3, c = col[2], label = '|2>')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Population')
    ax.set_xlim(0,time[-1])
    ax.legend(frameon=False, fontsize = 5)
    plt.savefig('./Data/Populations.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

