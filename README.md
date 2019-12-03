# bd4h-cse-6250
Big Data Health - CSE-6250-O01 (part of the MSc in Computer Science)

Steps to run the code for model training pipeline-
1. All data required for running the pipeline is available at **http://34.93.169.254:5000/tree/data_processed/my_results**, copy the data files to **data** folder
2. When running for the first time, run:  **python code/main.py -dp -t -m lr**, this will do data preprocessing required to create data loaders, store the loader objects as pkl files and train an lr model and save the model file. *For training CNN pass 'cnn' as argument. 'lr' is default argument.*
3. For subsequent runs, run: **python code/main.py -t -m lr** if you want to train and evaluate. Run: **python code/main.py -m lr** if you only want to evaluate
