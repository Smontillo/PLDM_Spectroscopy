for t2 in 2500 3000 3500 4000; do
    for Path in 0 1 2 3; do
        cd "R$((Path + 1))"
        mkdir "t2_$t2"
        cd "t2_$t2"
        mkdir "Data_Laser3"
        cp ../../Codes/ClassFile.py .
        cp ../../Codes/Multipar.py .
        cp ../../Codes/Parameters.py .
        cp ../../Codes/PLDM.py .
        cp ../../Codes/runDynamics.sh . 
        cp ../../Codes/Dynamics.py . 
        sed -i "s/^Path = 0$/Path = $Path/" Parameters.py   # Stablish Path
        sed -i "s/^t2     = 0$/t2     = $t2/" Parameters.py   # Stablish t2
        sed -i "s/^LaserStage = 3$/LaserStage = 3/" Parameters.py # Stablish Laser Stage
        sbatch Multipar.py
        cd ../..
    done
done
