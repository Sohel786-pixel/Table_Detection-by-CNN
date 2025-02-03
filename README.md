# Table_Detection-by-CNN

Overview
This repository contains a machine learning pipeline to detect tables within images and PDFs. It employs a Convolutional Neural Network (CNN) model trained to detect tables and generate bounding boxes around them. The model outputs both bounding boxes and classification labels (table or not) to enable automated table detection in document images. The pipeline also includes tools for evaluating the model, visualizing predictions, and processing PDF documents to extract tables using Optical Character Recognition (OCR).

Features
Table Detection: The model detects tables in images and PDF pages, drawing bounding boxes around identified tables.
Bounding Box Regression: The model performs bounding box regression to localize tables within the images.
Classification: Classifies the presence of a table in the image (binary classification: table or not).
PDF Processing: The pipeline can process PDF files, extracting images from each page and predicting tables with OCR support for extracting column headers.
Data Augmentation: Includes data augmentation techniques to help the model generalize better by applying transformations like rotations, shifts, and brightness changes.
Evaluation: Evaluate model performance using Mean Absolute Error (MAE) for bounding boxes and classification accuracy.
Requirements
TensorFlow
OpenCV
scikit-learn
tqdm
pdf2image
reportlab
pytesseract (for OCR)
matplotlib
For Google Colab users, the necessary libraries are automatically installed.

Usage
Preprocessing & Data Preparation:

Download and unzip the dataset (ensure to include images and annotations in XML or JSON format).
The annotations should contain bounding boxes of the tables (either in XML or JSON format).
Training:

The model is trained using a custom CNN built with TensorFlow's Keras API. You can modify the architecture or hyperparameters as needed.
Training occurs in batches, and the data is shuffled to improve model robustness.
Evaluation:

Evaluate the trained model on test data and visualize performance metrics such as loss and accuracy.
Plot training and validation loss/accuracy during training.
Model Prediction & Visualization:

Use the model to predict tables in images and visualize the results.
For PDF documents, tables are identified by extracting images from the document and running the model on each page.
Bounding Box and Classification Evaluation:

The model's performance can be evaluated by calculating the mean absolute error (MAE) for bounding box predictions and classification accuracy for detecting tables.
