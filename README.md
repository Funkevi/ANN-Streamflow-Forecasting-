# 🌊 Streamflow Forecasting using Artificial Neural Networks

### **Project Title:** Application of Artificial Neural Network in Streamflow Forecasting

---

## 📄 Project Overview

This project implements and compares two neural network architectures—a Standard Feed-Forward Neural Network (ANN) and a Long Short-Term Memory (LSTM) network—for a daily streamflow forecasting task. The methodology is a replication of the procedure outlined in the reference study, "Application of Artificial Neural Network in Streamflow Forecasting."

The core objective is to model the non-linear relationship between historical precipitation and runoff to predict future streamflow, with a specific focus on evaluating model performance during extreme (peak flow) events.



---

## 🧩 Problem Statement

To replicate the comparative analysis from the reference study by developing and evaluating two distinct neural network models for rainfall-runoff forecasting. The models are tasked with predicting the current day's streamflow based on a sequence of historical precipitation and runoff data. The performance of the Standard ANN is directly compared against the LSTM to determine the more effective architecture for this time-series problem on the chosen dataset.

---

## ⚙️ Implementation Overview

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `xarray`, `netCDF4`

#### **Workflow Steps:**

1.  **Data Sourcing & Processing:** The **CAMELS (Catchment Attributes and Meteorology for Large-sample Studies)** dataset was used. Helper scripts (`src/`) were executed to download the raw data and process it into a single, clean NetCDF file containing daily precipitation and streamflow for a single watershed over a 10-year period.
2.  **Feature Engineering:** The time series data was transformed into a supervised learning format. Input sequences of 30 days of historical data were created to predict the runoff on the 31st day.
3.  **Model Building:** Both a Standard ANN and an LSTM were built using TensorFlow/Keras, adhering to the optimal hyperparameters (hidden size, learning rate, activation functions) described in the reference study.
4.  **Training & Validation:** Both models were trained for 200 epochs using a chronological 80/20 train-test split.
5.  **Evaluation:** Model performance was primarily evaluated using the **Root Mean Squared Error (RMSE)**. A detailed analysis was conducted to compare the RMSE on "peak" vs. "non-peak" flow days, mirroring the evaluation strategy of the reference study.

---



