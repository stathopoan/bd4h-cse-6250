# bd4h-cse-6250
Big Data Health - CSE-6250-O01 (part of the MSc in Computer Science)

Steps to run the code for model training pipeline:
1. All data required for running the pipeline in code module (deep learning train/test) is available in the file: data.tar.gz from this link: . Extract the file into **data** folder.
2. Navigate to the **\code** folder. When running for the first time, run:  **python main.py -t -m \<model\>**, this will do data preprocessing required to create data loaders, store the loader objects as pkl files and train a model and save the model file. For example: for training vanilla CNN pass 'vanilla_cnn' as model argument. 'cnn_attn' is the default argument.
3. For subsequent runs, run: **python main.py -t -dp -m \<model\>** if you want to train and evaluate. It considers data loaders are already saved. 
4. For evaluating run: **python main.py -m \<model\>** if you only want to evaluate
Available models: ['lr', 'cnn_attn', 'rnn', 'vanilla_cnn']

Examples:
1. Navigate to the **\code** folder.
2. For training cnn_attn model type:  **python main.py -t -m cnn_attn**
3. For evaluating cnn_attn model type: **python main.py -m cnn_attn**

For preprocessing stage you need to copy the tables entitled: DIAGNOSES_ICD.csv, NOTEEVENTS.csv, PROCEDURES_ICD.csv from MIMIC III database into pre_processing\data folder. To start preprocessing please run pre_processing\src\main\scala\main.scala
Manual work is required to assemble the csvs since duw to multiple workers spark separates the csv creation to different parts.