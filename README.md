# Threat-detection-using-ML-in-Network-Traffic
Project Overview

This project implements a machine learning–based Intrusion Detection System (IDS) to detect malicious network traffic using the CICIDS2017 dataset. The goal is to classify network flows as either benign or malicious by applying and comparing multiple machine learning and deep learning models.

Traditional signature-based IDS struggle to detect new or unknown attacks. To address this limitation, this project explores data-driven techniques capable of learning patterns from historical network traffic and identifying abnormal behaviors automatically.

        Dataset
The project uses the CICIDS2017 dataset, a widely used benchmark dataset for intrusion detection research.
It contains realistic network traffic records labeled as:

         Benign traffic
Multiple attack types such as DDoS, DoS, Port Scan, Brute Force, Web attacks, etc.
For this project, the dataset was transformed into a binary classification task:

0 – Benign traffic
1 – Malicious traffic

Models Implemented

Three models were implemented and compared:

Random Forest (RF)

      - Ensemble learning method
      - Robust to noise and high-dimensional data
      - Implemented using Scikit-learn
Artificial Neural Network (ANN)

        - Feedforward neural network
        - Two hidden layers with ReLU activation
        - Implemented using TensorFlow/Keras
Convolutional Neural Network (CNN)

     - Deep learning model adapted for network traffic data
     - Captures spatial relationships among features
     - Implemented using TensorFlow/Keras
Implementation Environment

   - Platform: Google Colab
   - Programming Language: Python
   - Main Libraries Used:

                 -  Pandas, NumPy – data processing
                - Scikit-learn – preprocessing and Random Forest
                - TensorFlow/Keras – ANN and CNN
                - Matplotlib/Seaborn – visualization
Methodology

The project followed these main steps:
Data loading from CICIDS2017 Parquet files.
Data cleaning and preprocessing.
Handling missing and infinite values.
Label encoding to binary classes.
Feature scaling using standardization.
Train-test split (80% training, 20% testing).
Model training and evaluation.
Evaluation Metrics.
Model performance was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

These metrics allowed a fair comparison of detection effectiveness among the three models.

Results

All three models were trained and tested on the same dataset. Their performance was compared to determine the most effective approach for network intrusion detection.

The comparative analysis demonstrated that machine learning–based IDS can effectively identify malicious traffic and provide an intelligent alternative to traditional rule-based systems.

How to Run the Project

Open the provided Google Colab notebook.
Upload or mount the CICIDS2017 dataset files.
Run the cells sequentially:
Data preprocessing
Model training
Evaluation

No additional configuration is required beyond standard Python libraries.

Future Work

Possible improvements include:
Multi-class attack classification
Real-time intrusion detection
Hyperparameter optimization
Deployment as a live IDS system
