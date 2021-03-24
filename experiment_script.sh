#!/usr/bin/env bash

echo -e "Dynamic Routing log file created during the automated script for performing experiments.\n\n" > general_log.logs

echo -e "Starting with the execution of the first experiment: test on various datasets.\n" >> general_log.logs
for DATASET in MNIST Fashion-MNIST CIFAR10
do
    echo -e "Running on $DATASET ... with no reconstruction decoder.\n" >> general_log.logs
    python3 net.py --dataset $DATASET

    echo -e "Running on $DATASET ... with reconstruction decoder.\n" >> general_log.logs
    python3 net.py --dataset $DATASET --with_reconstruction

    echo -e "Runed on $DATASET succesfully.\n" >> general_log.logs
done
echo -e "Runed dataset experiments on default values succesfully.\n\n" >> general_log.logs


echo -e "Moving on with experiments on routing iterations.\n" >> general_log.logs
for i in 1 2 4
do
    echo -e "Running on MNIST ... with $i routing iterations & no reconstruction decoder.\n" >> general_log.logs
    python3 net.py --dataset MNIST --routing_iterations $i

    echo -e "Running on MNIST ... with $i routing iterations & reconstruction decoder.\n" >> general_log.logs
    python3 net.py --dataset MNIST --routing_iterations $i --with_reconstruction

    echo -e "Running on CIFAR10 ... with $i routing iterations.\n" >> general_log.logs
    python3 net.py --dataset CIFAR10 --routing_iterations $i

done
echo -e "Runed routing iteration experiments on the two datasets succesfully.\n\n" >> general_log.logs

echo -e "Finally, we perform experiments testing different learning rates.\n" >> general_log.logs
for lr in 0.002 0.005 0.05 0.1
do
    echo -e "Running on MNIST ... with $lr learning rate & reconstruction decoder.\n" >> general_log.logs
    python3 net.py --dataset MNIST --lr $lr --with_reconstruction
done
echo -e "Runed learning rate experiments on the MNIST dataset succesfully.\n" >> general_log.logs

echo -e "Finished. " >> general_log.logs