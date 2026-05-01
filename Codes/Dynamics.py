#!/software/anaconda3/2020.11/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qjob.out
#SBATCH --error=qjob.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import time as tm
import sys, os
import PLDM as method
import Parameters as par
import ClassFile as tc
# =========================
# Parallelization
# =========================
# RUN PARALLEL TRAJECTORIES
# THE NUMBER OF TRAJECTORIES PER JOB (j) IS DETERMINED BASED ON THE NUMBER OF CPUS (par.Cpus) AND TOTAL TRAJECTORIES (par.NTraj)
# j = NTraj / Cpus
parallel = par.parallel
if (parallel == True):
    sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # INCLUDE PARENT DIRECTORY WHICH HAS METHOD AND MODEL FILES
    JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"])               # GET ID OF THIS JOB
    TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"])             # GET ID OF THIS TASK WITHIN THE ARRAY 

    nrank = int(TASKID)                                         # JOD ID FOR A JOB 
    size  = par.Cpus                                            # TOTAL NUMBER OF PROCESSOR AVAILABLE
else:
    nrank = 0
    size  = 1

# =================================
# COMPILATION 
# =================================
# WITH JIT, THE CODE MUST BE COMPILE FIRST. RUN THE CODE FOR ONLY TWO TIME STEPS FIRST
com_ti = tm.time()
nsteps_dummy       = 2
ntj_dummy          = 2
nData_dummy        = 2
data_dummy         = tc.TrajData(nsteps_dummy, ntj_dummy, nData_dummy)
# MODEL FUNCTIONS =================
if par.LaserStage == 3:
    base     = "../../Data_Laser1" if os.path.exists("../../Data_Laser1") else "../Data_Laser1"
    ψF_all   = np.load(f'{base}/psiF_{nrank}_Task_{0}.npy')
    ψB_all   = np.load(f'{base}/psiB_{nrank}_Task_{0}.npy')
    data_dummy.ψF0[:,:]    = ψF_all[0,:,:ntj_dummy]  
    data_dummy.ψB0[:,:]    = ψB_all[0,:,:ntj_dummy]
method.RunTraj(data_dummy)

com_tf = tm.time()
print(f'Compilation time --> {np.round(com_tf - com_ti,2)} s or {np.round((com_tf - com_ti)/60,2)} min')

# =================================
# SIMULATION
# =================================
# DIVIDE THE NUMBER OF TRAJECTORIES PER JOB BASE ON THE NUMBER OF PROCESSORS AND TOTAL TRAJECTORIES
tot_Tasks = par.NTraj
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
for i in range(NRem):
    if i == nrank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)                                  # CONTAINS THE NUMBER OF TRAJECTORIES ASSIGNED TO EACH JOB
# =================================
sim_ti = tm.time()

for i in range(len(TaskArray)):
# ===================================================
# FIRST LASER SET UP
# ===================================================
    if par.LaserStage == 1 or par.LaserStage == 0:
        base = "../Data_Laser1" if os.path.exists("../Data_Laser1") else "./Data_Laser1"
        # === INITIATE THE TIME DEPENDENT DATA ===
        trajData = tc.TrajData(par.NSteps_L1, par.ntj, par.nData_L1) 
        # === RUN TRAJECTORIES ===
        method.RunTraj(trajData)
        # === SAVING DATA ===
        np.save(f'{base}/psiF_{nrank}_Task_{i}.npy', trajData.ψFw[:,:,:])   # RUN IN PARALLEL
        np.save(f'{base}/psiB_{nrank}_Task_{i}.npy', trajData.ψBw[:,:,:])   # RUN IN PARALLEL
        np.save(f'{base}/R_{nrank}_Task_{i}.npy', trajData.Rw[:,:,:])   # RUN IN PARALLEL
        np.save(f'{base}/P_{nrank}_Task_{i}.npy', trajData.Pw[:,:,:])   # RUN IN PARALLEL
        np.save(f'{base}/w1_{nrank}_Task_{i}.npy', trajData.w_FL[:])   # RUN IN PARALLEL
        np.save(f'{base}/Ind_{nrank}_Task_{i}.npy', trajData.ind_FL[:,:])   # RUN IN PARALLEL

# ===================================================
# SECOND AND THIRD LASER SET UP
# ===================================================
    if par.LaserStage == 3:
        # === INITIATE THE TIME DEPENDENT DATA ===
        trajData = tc.TrajData(par.NSteps_L3, par.ntj, par.nData_L3) 
        # === LOAD DATA ===
        base     = "../../Data_Laser1" if os.path.exists("../../Data_Laser1") else "../Data_Laser1"
        base3    = f"../Data_Laser3" if os.path.exists(f"../Data_Laser3") else f"./Data_Laser3"
        ψF_all   = np.load(f'{base}/psiF_{nrank}_Task_{i}.npy')
        ψB_all   = np.load(f'{base}/psiB_{nrank}_Task_{i}.npy')
        R_all    = np.load(f'{base}/R_{nrank}_Task_{i}.npy')
        P_all    = np.load(f'{base}/P_{nrank}_Task_{i}.npy')
        w1       = np.load(f'{base}/w1_{nrank}_Task_{i}.npy')
        trajData.w_FL[:] = w1[:]
        for t1 in range(par.nData_L1):
            trajData.ψF0[:,:]    = ψF_all[t1,:,:]  
            trajData.ψB0[:,:]    = ψB_all[t1,:,:]
            trajData.R[:,:]      = R_all[t1,:,:]   # RUN IN PARALLEL
            trajData.P[:,:]      = P_all[t1,:,:]   # RUN IN PARALLEL
            # === RUN TRAJECTORIES ===
            method.RunTraj(trajData)
            np.save(f'{base3}/psiF_{nrank}_Task_{i}_t1_{t1}.npy', trajData.ψFw[:,:,:])   # RUN IN PARALLEL
            np.save(f'{base3}/psiB_{nrank}_Task_{i}_t1_{t1}.npy', trajData.ψBw[:,:,:])   # RUN IN PARALLEL
sim_tf = tm.time()
t = (sim_tf - sim_ti)
times = [t, t/60, t/(60*60)]
time_tj = (sim_tf - sim_ti)/len(TaskArray)
print(f'Simulation time --> {np.round(times[0],6)} s, {np.round(times[1],2)} min, {np.round(times[2],2)} h')
print(f'Time per trajectory -> {np.round(time_tj/par.ntj,2)} s')
print(' ================================================================================================= ')


