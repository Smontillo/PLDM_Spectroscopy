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
data_dummy         = tc.TrajData(nsteps_dummy)
# MODEL FUNCTIONS =================
# method.InitBath(data_dummy)
# method.InitMapping(data_dummy)
# method.Force1(data_dummy)
# method.Force2(data_dummy)
# method.VelVerlet(data_dummy)
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
trajData = tc.TrajData(par.NSteps) # INITIATE THE TIME DEPENDENT DATA

sim_ti = tm.time()
for j in range(len(par.μMat_sp)):
    trajData.intiStateF = par.μMat_id[j,0]
    trajData.intiStateB = par.μMat_id[j,1]
    ρRe       = np.zeros((par.nData, par.Nt * par.Nt), dtype = np.complex128)                              # DENSITY MATRIX AVERAGED OVER THE NUMBER OF TRAJECTORIES ASSIGNED TO THIS JOB
    for i in range(len(TaskArray)):
        method.RunTraj(trajData)
        ρRe    += trajData.ρRe
    if (parallel == True):
        np.savetxt(f'../Data/rhoRe_{nrank}_{trajData.intiStateF}_{trajData.intiStateB}.txt', ρRe/len(TaskArray))   # RUN IN PARALLEL
    else:
        np.savetxt(f'./Data/rhoRe_{nrank}_{trajData.intiStateF}_{trajData.intiStateB}.txt', ρRe/len(TaskArray))    # RUN IN SERIES
sim_tf = tm.time()
t = (sim_tf - sim_ti)
times = [t, t/60, t/(60*60)]
time_tj = (sim_tf - sim_ti)/len(TaskArray)
print(f'Simulation time --> {np.round(times[0],6)} s, {np.round(times[1],2)} min, {np.round(times[2],2)} h')
print(f'Time per trajectory -> {np.round(time_tj/par.ntj,2)} s')
print(' ================================================================================================= ')


