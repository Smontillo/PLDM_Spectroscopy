import numpy as np
import matplotlib.pyplot as plt
import Parameters as par
import time
import scipy.constants as sc
from scipy.optimize import curve_fit
import os
import socket
import sys
col = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#34495e']
parallel = par.parallel
if (parallel == True):
    JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"])               # GET ID OF THIS JOB
    TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"])             # GET ID OF THIS TASK WITHIN THE ARRAY 

    rank = int(TASKID)                                         # JOD ID FOR A JOB 
    size  = par.Cpus                                            # TOTAL NUMBER OF PROCESSOR AVAILABLE
else:
    rank = 0
    size  = 1

tot_Tasks = par.NTraj
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)   
t2 = int(sys.argv[1])
# ===================================
# FUNCTIONS
# ===================================
def TrajPerProc():
    NTasks = par.NTraj// par.Cpus
    return NTasks

def build_jobs():
    # Build the same logical jobs your serial code iterates over.
    # If your data exists for rank in [0..par.Cpus-1], tasks in [0..NTasks-1], traj in [0..par.ntj-1]
    NTasks = TrajPerProc()
    jobs = []
    for task in range(NTasks):
        for traj in range(par.ntj):
            jobs.append((task, traj))
    return jobs

def LoadData_1D_mpi():
    # Build and scatter work logically (each rank takes a strided subset)
    jobs = build_jobs()
    my_jobs = jobs#[rank::size]
    R1t_local = np.zeros((par.nData_L1, par.nData_L3), dtype=np.complex128)

    R1t_local = np.zeros((par.nData_L1, par.nData_L3), dtype=np.complex128)

    # for task in TaskArray:
    for task in range(par.NTraj// par.Cpus):
        # We don't need traj anymore if we sum over all trajectories at once
        for t1 in range(par.nData_L1):
            psiF_all = np.load(f"t2_{t2}/Data_Laser3/psiF_{rank}_Task_{task}_t1_{t1}.npy")  # (L3,Nt,ntj)
            psiB_all = np.load(f"t2_{t2}/Data_Laser3/psiB_{rank}_Task_{task}_t1_{t1}.npy")  # (L3,Nt,ntj)

            tet_it  = np.einsum('ijt,ijt->it', psiF_all, np.conjugate(psiB_all))  # (L3,ntj)
            tet_sum = tet_it.sum(axis=1)  # (L3,)

            R1t_local[t1, :] -= 1j * (tet_sum)

    n_total = par.NTraj// par.Cpus * par.ntj
    R1t_local /= n_total
    return R1t_local 
    
# ===================================
timeL3 = par.Sim_time_L3[::par.nskip]/par.fs2au
timeL1 = par.Sim_time_L1[::par.nskip]/par.fs2au

R1tL1 = LoadData_1D_mpi()
sl = 30

np.save(f"Data/R3t_t2_{t2}_rank_{rank}.npy", R1tL1)
print(f"Saved R3t_t2_{t2}_rank_{rank}.npy")
fig, ax = plt.subplots(1,2,figsize=(6,3))
ax[0].plot(timeL3, np.real(R1tL1[sl,:]))
ax[0].plot(timeL3, np.imag(R1tL1[sl,:]))
ax[1].plot(timeL1, np.real(R1tL1[:,sl]))
ax[1].plot(timeL1, np.imag(R1tL1[:,sl]))
# ax[0].set_xlim(timeL3[0], timeL3[-1])
# ax[1].set_xlim(timeL1[0], timeL1[-1])
# ax.set_ylim([-1, 1])
ax[0].set_xlabel("Time (fs)")
ax[1].set_xlabel("Time (fs)")
ax[0].set_ylabel("R3(t)")
plt.savefig(f"Data/R3t_Slice_{sl}_t2_{t2}.png", dpi=300, bbox_inches='tight')
plt.close()
