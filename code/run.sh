#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python main.py --prefix ../data/D1/ --dataset lap14 --seed 1000 --batch_size 16 --epochs 100 --learning_rate 1e-3 --fusion add --pm_model_class bert