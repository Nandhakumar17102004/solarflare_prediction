**Built a Convolutional Neural Network (CNN) in Keras to classify solar magnetogram images into flare intensity classes (e.g., A, B, C, M).
Performed image preprocessing and label conversion for binary classification (flare vs. no-flare) using class mapping and statistical balancing.
Trained and evaluated the model using accuracy and confusion matrix; applied ImageDataGenerator for augmentation and ReduceLROnPlateau for adaptive learning.
Exported the trained model and representative test images for UI integration in a potential deployment phase.**

#  Solar Flare Classification using CNN + MLP

---

## 📌 Project Overview

Solar flares are sudden eruptions of electromagnetic radiation from the Sun’s surface.  
They can disrupt satellites, power grids, communication systems, and space missions.

This project develops a **deep learning model (CNN + MLP)** to predict the occurrence of major solar flares using magnetogram images.

The approach combines:

- Spatial feature extraction using Convolutional Neural Networks (CNN)  
- Classification using Multi-Layer Perceptron (Fully Connected Layers)  

---

##  Problem Statement

Traditional physics-based simulations for solar flare prediction are computationally expensive.

This project explores:

- Can CNN extract meaningful spatial features from solar magnetogram images?
- Can fully connected layers effectively classify flare intensity?
- Can a CNN + MLP architecture accurately detect major (M/X) solar flares?

---

## 🎯 Objectives

- Develop a CNN + MLP model for solar flare classification  
- Preprocess and analyze solar magnetogram images  
- Handle class imbalance for rare major flare events  
- Evaluate performance using standard classification metrics  

---

## 🏗 System Architecture

### 1️⃣ CNN Component
- Extracts spatial features such as:
  - Sunspots  
  - Active magnetic regions  
  - Magnetic field patterns  

### 2️⃣ MLP (Fully Connected Layer)
- Receives flattened feature maps from CNN  
- Performs non-linear classification  
- Outputs binary prediction (Major / Non-Major flare)

---

##  Data Collection & Preprocessing

### Solar Images
- NASA SDO Magnetogram Images  

### Flare Data
- NOAA SWPC Solar Flare Reports  

### Preprocessing Steps
- Image cropping  
- Resizing to 28×28  
- Normalization to [0,1]  
- Reshaping to (28, 28, 1)  
- Label mapping to GOES classes  
- Binary conversion (M/X = 1, others = 0)  

---

##  Training & Evaluation

### Dataset Split
- 80% Training  
- 20% Testing  

### Loss & Optimizer
- Binary Cross-Entropy  
- Adam Optimizer  

### Regularization
- Dropout layers to prevent overfitting  
- Data augmentation (shifts and flips)  
- Learning rate reduction using ReduceLROnPlateau  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix Analysis  

---

## Outcomes

- Trained CNN + MLP classification model  
- Accurate detection of major solar flares  
- Improved detection of rare flare events using class weighting  
 
---

##  Tools & Libraries

- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Flask   

---

##  Learning Outcomes

- Deep learning architecture design (CNN + MLP)  
- Image preprocessing for scientific data  
- Handling imbalanced datasets  
- Model evaluation and tuning  

---

## 📌 Conclusion

By combining CNN (for spatial feature extraction) and MLP (for classification),  
this project provides an efficient deep learning-based system for detecting major solar flares.

Such predictive systems contribute to:

- Satellite protection  
- Communication infrastructure safety  
- Space weather forecasting  
- Scientific research in heliophysics  

---

##  Author  

Nandhakumar  
CB.EN.U4CSE22530
