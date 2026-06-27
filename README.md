# IMU-Based Assessment of Duchenne Muscular Dystrophy (DMD)

## Overview

This project presents an end-to-end pipeline for automatic assessment of motor function in children with Duchenne Muscular Dystrophy (DMD) using wearable Inertial Measurement Units (IMUs) and machine learning.

The proposed system records motion data from five wearable IMU sensors during functional tasks derived from the North Star Ambulatory Assessment (NSAA). The collected signals are processed, transformed into statistical features, reduced using Principal Component Analysis (PCA), and finally used to train machine learning classifiers for predicting patients' functional performance.

---

## Project Pipeline

1. Motion acquisition using five TTGO T-Wrist IMU sensors
2. Bluetooth communication with MATLAB
3. Signal recording and storage
4. Butterworth filtering
5. Motion segmentation (cropping)
6. Feature extraction
7. Feature normalization
8. PCA dimensionality reduction
9. Machine learning training and evaluation

---

## Hardware

- 5 × TTGO T-Wrist
- ESP32 Pico-D4
- 3-axis Accelerometer
- 3-axis Gyroscope
- Bluetooth communication
- Sampling rate: 90–100 Hz

Sensor placement:

- Forehead
- Left wrist
- Right wrist
- Left ankle
- Right ankle

---

## Software

### MATLAB

- Bluetooth communication
- Real-time visualization
- Data acquisition
- Data storage

### Python

- Data preprocessing
- Feature extraction
- PCA
- Machine learning
- Performance evaluation

---

## Machine Learning Workflow

Raw Data

↓

Filtering

↓

Action Segmentation

↓

Feature Extraction

↓

Standardization

↓

PCA

↓

Classifier Training

↓

Performance Evaluation

---

## Repository Structure
MATLAB/
Arduino/
Python/
Dataset/
Models/
Figures/
Report/

---

## Research Goal

To develop a low-cost and objective framework for assessing motor performance in children with Duchenne Muscular Dystrophy using wearable IMU sensors and machine learning techniques.

---

## Author

Mohammad
Bachelor Project
Mechanical Engineering