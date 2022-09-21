#!/usr/bin/env bash

echo -e "Dynamic Routing log file created during the automated script for performing experiments on routing.\nWe either do the classical dynamic routing or choose a winner child capsule for each parent and pass this vector to \
the parent\n\n" > general_log.logs

echo -e "Experiments on various datasets.\n" >> general_log.logs
for DATASET in MNIST
do
    echo -e "Running on $DATASET ... with reconstruction decoder and with argmax.\n" >> general_log.logs
    python3 net.py --dataset $DATASET --with_reconstruction --with_argmax --epochs 100

    echo -e "Running on $DATASET ... with reconstruction decoder and with argmax but without ones.\n" >> general_log.logs
    python3 net.py --dataset $DATASET --with_reconstruction --with_argmax --without_argmax_one --epochs 100
 
    echo -e "Runed on $DATASET succesfully.\n" >> general_log.logs
done
echo -e "Runed dataset experiments succesfully.\n\n" >> general_log.logs


echo -e "Finished. " >> general_log.logs