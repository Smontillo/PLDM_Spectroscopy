for Path in 0 ; do
    cd "R$((Path + 1))"
    mkdir Data
    cp ../Codes/runGather1D.sh .
    cp ../Codes/Plot_1D.py .
    sbatch runGather1D.sh
    cd ..
done

