for t2 in 2500 3000 3500 4000; do
    for Path in 0 1 2 3; do
        cd "R$((Path + 1))"
        # cp ../Codes/ConstructResponseF.py .
        # cp ../Codes/runGatherR3.sh .
        cp ../Codes/ConstructResponseF2.py .
	    cp ../Codes/runResponse.sh .
	    cp ../Codes/Parameters.py .
	    cp ../Codes/MultiResponse.py .
        mkdir Data
        sbatch MultiResponse.py $t2
        cd ..
    done
done

