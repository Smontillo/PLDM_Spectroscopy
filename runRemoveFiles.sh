for Path in 0 1 2 3; do
	cd "R$((Path + 1))"
	cp ../Codes/runRemoveFiles_Laser1.sh .
	sbatch runRemoveFiles_Laser1.sh
	for t2 in 0 500 1000 1500 2000 2500 3000 3500 4000; do
		cd "t2_$t2"
		cp ../../Codes/runRemoveFiles_Laser3.sh .
		sbatch runRemoveFiles_Laser3.sh
		cd ../
	done
	cd ..
done
