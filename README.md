# 🧩 Ensemble-Knowledge-Tracing 

This project implements a Deep Knowledge Tracing model using LSTM to predict students’ future performance based on their historical interactions with problems.



## Project Summary

**Task:**  Model students' knowledge state by predicting their correctness on future questions based on their previous performance.

**Model:**  Ensemble Deep Knowledge Tracing (LSTM-based)

**Evaluation Metric:**  AUC (Area Under the ROC Curve)



## Dataset

This project uses several public datasets, including **ASSIST0910**, **ASSIST2017**, and **OLIES**, along with a newly collected dataset from the **HKU Moodle** Programming Course, which contains student performance records.

Below is a description of the files in the `Dataset/` folder:

\- `generate_sequence_embeddings.py` and all `*_seq_*.csv` files

These files are templates and outputs for **exercise sequence embeddings**, which serve as the **input to Model-I**. Each `*_seq_*.csv` file stores processed student question-answer sequences. Every three rows correspond to a single student's data in the following format: [Number of questions answered, Sequence of question IDs, Sequence of correctness labels (1 = correct, 0 = incorrect)]

\- `generate_feature_embeddings.py`

This script is a preprocessing template for **behavioral (feature) sequence embeddings**, which serve as the **input to Model-II**.

Due to slight differences in processing requirements across datasets and models, the preprocessed feature embedding files are **not included**. You may modify this script based on your specific dataset and feature definitions.

\- `train_test_split.py`

This script splits the dataset into **training** and **testing** sets and saves the results in the same directory.



## Model Structure

<br><br>

<img src="https://github.com/user-attachments/assets/1a7669b1-cea6-4e1f-9f8c-741962f3719f" alt="frame" width="700" />


**Fig. 1.** Overview of the Dual-LSTM Knowledge Tracing Framework with Ensemble Voting. The model processes both exercise sequence and behavior feature data through two separate LSTM structures, and combines their outputs via a weighted ensemble.

<br>

<img src="https://github.com/user-attachments/assets/5a7a21eb-73df-4d91-b9f2-a0141df5a2ee" alt="frame2" width="700" />


**Fig. 2.** Embedding Layer and Sequential Processing in LSTM



## Miscellaneous

The code also supports testing other RNN variants on the specified datasets, such as BiLSTM and GRU.

