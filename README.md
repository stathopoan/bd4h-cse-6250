# bd4h-cse-6250
Big Data Health - CSE-6250-O01 (part of the MSc in Computer Science)

Steps to run the code for model training pipeline-
1. All data required for running the pipeline is available at **http://34.93.169.254:5000/tree/data_processed/my_results**, copy the data files to **data** folder
2. When running for the first time, run:  **python code/main.py -dp**, this will do data preprocessing required to create data loaders and store the loader objects as pkl files
3. For subsequent runs, run: **python code/main.py**
