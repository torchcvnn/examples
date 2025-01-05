#!/bin/bash

# Run the training script for all the acceleration factors ACC4, ACC8 and ACC10
# and both views SAX and LAX
# The results are saved in "./results" for every run , we copy the results in a dedicated directory after each run 

# Example call : 
# python nir_miccai2023.py  --rootdir /mounts/Datasets4/MICCAIChallenge2023/ChallegeData/ --acc_factor ACC10 --view SAX

for acc in ACC4 ACC8 ACC10
do
    for view in SAX LAX
    do
        echo "=================================================="
        echo "Running for $acc $view"
        echo "=================================================="
        python nir_miccai2023.py  --rootdir /mounts/Datasets4/MICCAIChallenge2023/ChallegeData/ --acc_factor $acc --view $view
        mkdir -p ./all_results/$acc/$view
        cp -r ./results/* ./all_results/$acc/$view
    done
done
