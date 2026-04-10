# Real-Time Predictive Emotion and Behavioral Analysis with Wellness Recommendations

> Deep learning system that reads emotions in real time and suggests what to do about them.

A CNN + LSTM pipeline that detects emotions from facial expressions via live camera feed, tracks behavioral patterns over time, and generates wellness recommendations based on detected emotional states. Built as a competitive entry for the IBM Z Datathon.

**Model accuracy: 84.6% on held-out test set**
**Google Colab notebook:** [Open in Colab](https://colab.research.google.com/drive/1ozkoEGtZzgDFYbT0mUx5nlBo3T5g84xW)

---

## How It Works

```
Camera feed
  |
  v
Frame capture (OpenCV)
  |
  v
CNN — Spatial feature extraction
  |    3x Conv2D + MaxPooling layers
  |    Flatten + Dense + Dropout
  |
  v
LSTM — Temporal behavioral tracking
  |    Tracks emotion sequences over time
  |    Identifies patterns (e.g., sustained stress)
  |
  v
Softmax classifier
  |    8 emotion classes
  |
  v
Wellness recommendation engine
       Maps detected emotion -> actionable suggestion
```

---

## Emotion Classes

| Class | Example wellness recommendation |
|-------|--------------------------------|
| Angry | Breathing exercises, 5-minute walk |
| Sad | Mood journaling, uplifting playlist |
| Fear | Grounding technique (5-4-3-2-1 method) |
| Disgust | Reframing exercise, distraction activity |
| Happy | Log the moment, share with someone |
| Surprised | Pause and assess, mindful breathing |
| Contempt | Perspective-taking prompt |
| Neutral | Check-in prompt, hydration reminder |

---

## Model Architecture

### CNN (feature extractor)

| Layer | Details |
|-------|---------|
| Conv2D (32 filters) | 3x3 kernel, ReLU, input: 48x48x1 |
| MaxPooling2D | 2x2 |
| Conv2D (64 filters) | 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Conv2D (128 filters) | 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Flatten | — |
| Dense (256) | ReLU + Dropout |
| Dense (8) | Softmax — emotion class probabilities |

### LSTM (behavioral tracker)

Sequences of CNN predictions are fed to an LSTM to capture temporal patterns — distinguishing a brief surprised expression from sustained fear, or tracking emotional drift over a session.

---

## Performance

| Metric | Value |
|--------|-------|
| Validation accuracy | **84.62%** |
| Validation loss | 0.2984 |
| Test accuracy | **84.62%** |
| Test loss | 0.2984 |
| Loss function | Categorical Cross-Entropy |
| Optimizer | Adam |

---

## Dataset

**Source:** [Facial Emotion Recognition — Kaggle](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition)

- 48x48 pixel grayscale images
- 8 emotion classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised
- Organized in subdirectories per class label

Download and place under `data/` with the following structure:

```
data/
├── train/
│   ├── angry/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    └── (same structure)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended (Google Colab with T4 runtime works well)

### Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn, Scikit-learn

### Train the model

Open `model.ipynb` in Jupyter or Google Colab and run all cells. The notebook:
1. Loads and preprocesses the dataset
2. Builds the CNN + LSTM architecture
3. Trains with data augmentation
4. Evaluates on the test set
5. Saves the best model to `best_model.keras` and `best_model.h5`

### Run predictions on an image

```python
python model.py
```

By default this runs inference on `predicition.jpg`. Modify `model.py` to point to any image or enable the webcam feed.

### Real-time webcam inference

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('best_model.keras')
emotions = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48)) / 255.0
    pred = model.predict(face.reshape(1, 48, 48, 1))
    emotion = emotions[np.argmax(pred)]
    cv2.putText(frame, emotion, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('EmotionCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data processing | NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Training platform | Google Colab (GPU) |

---

## Project Files

| File | Description |
|------|-------------|
| `model.ipynb` | Full training notebook with EDA, training, evaluation |
| `model.py` | Standalone inference script |
| `best_model.keras` | Saved model (Keras native format) |
| `best_model.h5` | Saved model (HDF5 format, for compatibility) |
| `requirements.txt` | Python dependencies |
| `predicition.jpg` | Sample image for testing inference |

---

## License

MIT
