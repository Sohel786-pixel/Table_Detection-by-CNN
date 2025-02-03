# Table Detection Model README

This project implements a **table detection model** using a **Convolutional Neural Network (CNN)**. The model is designed to detect tables within images, extract their bounding boxes, and classify the presence of tables in the image. .

## Features

- **Data preprocessing**: Loads and processes image data along with corresponding annotations (XML/JSON).
- **Batch processing**: Efficiently processes data in batches to save memory.
- **CNN model for table detection**: Trains a CNN to detect tables and output bounding boxes and classification labels.
- **Evaluation**: Evaluates the model performance on test data using Mean Absolute Error (MAE) and classification accuracy.
- **Visualization**: Displays images with predicted bounding boxes and extracted table content.
- **PDF Table Detection**: Detects tables in PDF files by converting each page to an image, predicting tables, and extracting column headers using OCR.
- **Custom Loss Function**: Implements Intersection over Union (IoU) loss to refine bounding box predictions.

## Dependencies

The following Python libraries are required to run this project:

- TensorFlow
- OpenCV
- scikit-learn
- matplotlib
- tqdm
- pdf2image
- pytesseract
- reportlab

## Key Metrics:
Test Loss: 486.9518
Test BBox Loss: 486.9518
Test Class Loss: 0.0000
Test Classification Accuracy: 1.0000

#### This perfect accuracy could be indicative of overfitting—especially if the model is achieving near-perfect results on the training set and not generalizing well to unseen data.
