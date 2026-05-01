mkdir R1 R2 R3 R4
for Path in 0 1 2 3; do
    cd "R$((Path + 1))"
    mkdir Data_Laser1
    cp ../Codes/ClassFile.py .
    cp ../Codes/Multipar.py .
    cp ../Codes/Parameters.py .
    cp ../Codes/PLDM.py .
    cp ../Codes/runDynamics.sh . 
    cp ../Codes/Dynamics.py .
    sed -i "s/^Path = 0$/Path = $Path/" Parameters.py
    sed -i "s/^LaserStage = 3$/LaserStage = 1/" Parameters.py
    sbatch Multipar.py
    cd ..
done

