# Table Detection Model README

## Introduction

This project implements a **table detection model** using a **Convolutional Neural Network (CNN)**. The model is designed to detect tables within images, extract their bounding boxes, and classify the presence of tables in the image. It employs various techniques such as **data augmentation**, **object detection**, and **optical character recognition (OCR)** to improve prediction accuracy and table recognition.

The project utilizes **TensorFlow**, **OpenCV**, **scikit-learn**, and several other libraries to build, train, evaluate, and visualize the model's predictions. It supports PDF table detection by converting PDF pages to images and performing OCR to extract column headers.

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


## File Structure

- **subset.zip**: Contains the dataset with images and annotations.
- **train, val, test**: Directories for training, validation, and test data, containing images and their corresponding annotations.
- **table_detection_model.h5**: The saved model after training.

## How It Works

### Step 1: Data Preparation
The dataset is a collection of images with corresponding annotation files in XML or JSON format. These annotations contain the bounding box information for each table in the image.

### Step 2: Model Architecture
A Convolutional Neural Network (CNN) is built using TensorFlow/Keras with the following components:
- **Convolutional Layers**: Extract features from the images.
- **MaxPooling Layers**: Reduce spatial dimensions of the feature maps.
- **Fully Connected Layers**: Output the bounding box coordinates and classification labels (table or not).

The model is compiled with a custom loss function:
- **Bounding Box Loss**: Mean squared error (MSE) for bounding box predictions.
- **Classification Loss**: Binary cross-entropy for table classification (presence or absence of a table).

### Step 3: Model Training
The model is trained using data generators to load images and annotations in batches. Training includes:
- **Data augmentation**: Random transformations such as rotation, width/height shifts, and brightness variations to enhance model generalization.
- **Early Stopping**: Monitors validation loss to prevent overfitting.

### Step 4: Model Evaluation
The model is evaluated on test data, and the following metrics are reported:
- **Test Loss**: Overall loss (sum of bounding box and classification loss).
- **Test Classification Accuracy**: Accuracy of the classification task (table or no table).
- **Bounding Box MAE**: Mean Absolute Error for bounding box predictions.

### Step 5: Prediction Visualization
Predicted bounding boxes are visualized on the test images. The model's predictions are compared with ground truth, and column headers are extracted using **OCR** on detected table regions.

### Step 6: PDF Table Detection
For PDF files, the model converts each page to an image and performs table detection. The extracted tables' bounding boxes are drawn on the images, and the model performs OCR to extract column headers.

## How to Use

1. **Prepare the Data**: Ensure the dataset (`subset.zip`) is available, containing images and annotations (XML/JSON) for training, validation, and test data.
2. **Training**: Run the script to build and train the model using `train_gen` and `val_gen` data generators.
3. **Save and Load the Model**: The trained model is saved as `table_detection_model.h5`. You can load this model for inference and predictions.
4. **Prediction**: Use the model to predict tables in images or PDF files. Visualize predictions using the `plot_predictions()` and `visualize_predictions()` functions.

### Example

To train and evaluate the model:

```python
history = model.fit(train_gen, steps_per_epoch=train_steps, validation_data=val_gen, validation_steps=val_steps, epochs=5)
```

To make predictions on a PDF:

```python
visualize_predictions("/path/to/table.pdf", model, min_bbox_area=2000)
```

## Conclusion

This project demonstrates an efficient pipeline for detecting tables in images and PDF documents using a CNN-based model. By leveraging batch processing, data augmentation, and OCR, the system can handle a variety of input formats and provide accurate results. The model can be further improved by adding additional data, refining the model architecture, or using more advanced techniques like Region Proposal Networks (RPN) for better object localization.
