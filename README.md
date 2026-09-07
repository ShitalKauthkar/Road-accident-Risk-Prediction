# 🚦 Road Accident Risk Prediction

### AI/ML Based System for Road Accident Risk and Severity Prediction

This project is an AI/ML-based road accident risk assessment system that combines structured accident data and road-scene image analysis.

The system uses a **Deep Neural Network (DNN)** to predict accident severity from accident-related data and **YOLO** to detect objects from road-scene images. **SHAP** is used to understand the important features contributing to the model prediction. The outputs are combined using a weighted risk-fusion approach to generate an overall **Low, Medium, or High risk level**.

---

## 🎯 Project Objective

The main objective of this project is to develop an intelligent system that can:

- Predict accident severity using accident-related data.
- Analyze road-scene images using YOLO object detection.
- Identify important factors influencing the DNN prediction using SHAP.
- Combine data-based and image-based risk information.
- Generate an overall accident risk score.
- Display the result through an interactive Streamlit web application.

---

## 🧠 Technologies Used

- **Python**
- **Machine Learning**
- **Deep Learning**
- **TensorFlow / Keras**
- **Scikit-learn**
- **YOLO / Ultralytics**
- **SHAP**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Kaggle Dataset**

---

## 📊 Dataset

The project uses a **Kaggle Road Accident dataset**.

- Records: **12,316**
- Features: **32 columns**
- Target variable: **Accident Severity**
- Target classes:
  - Fatal Injury
  - Serious Injury
  - Slight Injury

---

## 🔄 Project Workflow

```text
                USER INPUT
                    |
          +---------+---------+
          |                   |
     Accident Data        Road Image
          |                   |
   Data Preprocessing        YOLO
          |                   |
         DNN            Object Detection
          |                   |
    DNN Risk Score       YOLO Risk Score
          |                   |
          +---------+---------+
                    |
               Risk Fusion
                    |
             Final Risk Score
                    |
          +---------+---------+
          |         |         |
         Low      Medium     High
                    |
             Streamlit UI
