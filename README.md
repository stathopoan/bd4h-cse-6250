# bd4h-cse-6250
Big Data Health - CSE-6250-O01 (part of the MSc in Computer Science)

Steps to run the code for model training pipeline-
1. All data required for running the pipeline in code module (deep learning train/test) is available in the file: data.tar.gz in a private link. Extract the file into **data** folder.
2. When running for the first time, run:  **python code/main.py -t -m <model>**, this will do data preprocessing required to create data loaders, store the loader objects as pkl files and train a model and save the model file. *For training CNN pass 'cnn' as argument. 'cnn_attn' is default argument.*
3. For subsequent runs, run: **python code/main.py -t -dp -m lr** if you want to train and evaluate. Run: **python code/main.py -m lr** if you only want to evaluate
Available models: ['lr', 'cnn_attn', 'rnn', 'vanilla_cnn']

For preprocessing stage you need to copy the tables entitled: DIAGNOSES_ICD.csv, NOTEEVENTS.csv, PROCEDURES_ICD.csv from MIMIC III database into pre_processing\data folder. To start preprocessing please run pre_processing\src\main\scala\main.scala
