# NLP_FINAL

#### Welcome to my final project for NLP course. If you would like to test-run the project, you can use the main.py:

The methods in main are from the following scripts:

1. HandMadeMain.py : Stratified 10-Fold Cross Validation, KNN Classification with Cosine Similarity and TF-IDF.
2. sklearn_main.py : Stratified 10-Fold Cross Validation, KNN Classification with Cosine Similarity and TF-IDF using sklearn library.
3. BERT.py : Basic implementation of BERT 
4. RoBERTa.py : Basic implementation of RoBERTa

The config for each script are inside them. You can adjust them to your hearts content. The total run should take at 
most 3 hours.

## HandMadeMain.py
HandMadeMain.py is the version that I created myself (not actually inventing the methods but hard-coding the theory).
This is some kind of improved version of my previous homework.

### BEFORE RUNNING, Please adjust the max_worker value in stratified_kfolds_config accordingly to your CPU cores where each fold will be processed in parallel.
```
'max_workers': 5 # amount of cpu cores
```

### For tokenization for TF-IDF, try to use nltk because my BPE takes longer. In TF_IDF_Config:
```
"use_nltk_tokenizer": True,
```

#### For BERT and RoBERTa, the script saves the best model for that training fold for the sake of reusability because how long it takes to train/evaluate.
Currently I did not add a function to check if the folders for the models exists, so it may give an error at the end.
Please be sure that RoBERTa_Models and BERT_Models directories exist. 

#### You can also check how long it took for me to run the code in profile_outputs folder, or running the Profile_Output_Reader.py script.

### BEFORE RUNNING BERT and RoBERTa, please adjust batch_size and max_length or model_name appropriate to your VRAM. I've got 16GB of VRAM and it was full all the time.
Also you can lower the max_length and increase batch_size for a faster run.


Hope I have covered at least a bit of the project and you won't have trouble understanding it.

Have a great day!