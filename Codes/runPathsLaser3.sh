for t2 in 0 500 1000 1500 2000; do
    for Path in 0 1 2 3; do
        cd "R$((Path + 1))"
        mkdir "t2_$t2"
        mkdir "t2_$t2/Data_Laser3"
        cp ../Codes/Parameters.py .
        cp ../Codes/Dynamics.py .
        sed -i "132c LaserStage = 3" Parameters.py
        sed -i "129c t2     = $t2" Parameters.py
        sbatch Multipar.py
        cd ..
    done
done
