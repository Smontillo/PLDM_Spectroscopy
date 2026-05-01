# Linear and Non-Linear Spectroscopy 

Code that implements the algorithm describe by Mondal et. al. for the linear and non-linear spectroscopy of a vibrational monomer.

#### **Quantum dynamics simulations of the 2D spectroscopy for exciton polaritons.** 

[J. Chem. Phys. 7 September 2023; 159 (9): 094102.](https://pubs.aip.org/aip/jcp/article/159/9/094102/2908708)  

---

The code has 4 main files located in the *Codes* folder:

+ **Parameters.py** &rarr; Contains all the system and simulation parameters (Simulation time for different laser, Hamiltonian and bath details, number of trajectories, number of Cpu's, etc.).  
+ **Multipar.py** &rarr; Divides the jobs based on the number of Cpu's stablished for parallelization.
+ **Dynamics.py** &rarr; Submit simulations based on the different laser stages.
+ **PLDM.py** &rarr; Contains all the function for the PLDM dynamics for spectra simulation. Trajectories are vectorized for computing speed.