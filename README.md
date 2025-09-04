# A Real-Time Predictive Emotion and Behavioral Analysis with Wellness Recommendations.


## Project Overview:

This project aims to develop a deep learning-based system for real-time emotion and behavioral analysis using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The model identifies emotions such as anger, happiness, fear, and surprise through body language and facial expressions and provides wellness recommendations based on the detected emotions.

## Key Features:

Emotion Detection: Recognizes various emotions using a camera feed and image processing techniques.
Behavioral Analysis: Provides insights into emotional patterns over time, helping users manage mental well-being.
Wellness Recommendations: Offers suggestions based on the identified emotions, e.g., stress-relief activities.
Real-Time Feedback: Uses the model to monitor emotions continuously.
Deep Learning Models: Employs CNNs for emotion detection and LSTMs to track behavioral changes over time.

## Dataset: 
The dataset consists of grayscale images representing different emotions:
Dataset link: https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition

Classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised.
Image Size: 48x48 pixels, grayscale images.
Ensure the dataset is structured in subdirectories for each emotion label, where each folder contains images associated with that emotion.

## Model Architecture:

### CNN Layers:
3 Convolutional layers with MaxPooling for feature extraction.
Flattening layer to prepare for fully connected layers.
Dense layers with ReLU activation.
Dropout for regularization.
Output Layer: Softmax activation for multi-class emotion prediction.

## Tech Stack:
Programming Language: Python
Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn, Scikit-learn
Deep Learning Models: CNN (Convolutional Neural Networks), LSTM (Long Short-Term Memory)
Training Platform: Google Colab, GPU-enabled for faster training.

## Performance:

Metrics – Accuracy, Categorical Cross Entropy Loss.

val_loss: 0.2984 - val_accuracy: 0.8462

Test Loss: 0.29843610525131226
Test Accuracy: 0.8461538553237915

## Google collab page: https://colab.research.google.com/drive/1ozkoEGtZzgDFYbT0mUx5nlBo3T5g84xW#scrollTo=hlUNSKaT5WmX

 ## linuxone server : http://148.100.108.81:38888/lab
