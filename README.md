# 🧠 Face Mask Detection using SimpleCNN

This project implements a deep learning model to classify whether a person is wearing a face mask, wearing it incorrectly, or not wearing one at all. The model is built from scratch using **PyTorch**, with a custom-designed 7-layer Convolutional Neural Network (CNN) followed by multiple fully connected layers.

---

## 📐 1. Architecture Overview

### 🔹 Feature Extractor: `self.features`

The feature extractor consists of **7 convolutional blocks**, each built using a loop from the following configuration:

```
specs = [(3,16), (16,32), (32,64), (64,128), (128,256), (256,512), (512,1024)]
```

Each block contains:
- A 2D Convolution (`Conv2d`)
- A ReLU activation
- A MaxPooling layer (2×2)

Followed by:
- **Adaptive Average Pooling** to reduce output to (1×1)

---

### 🔹 Classifier: `self.classifier`

After the convolutional layers, the feature map is flattened and passed through a **multi-layer perceptron (MLP)** with the following configuration:

```
hidden_units = [512, 256, 128, 64, 32]
```

Each layer includes:
- Fully connected (`Linear`)
- ReLU activation
- Dropout (p=0.5)

The final output layer produces logits for **3 classes**:
- `with_mask`
- `mask_weared_incorrect`
- `no_mask`

---

## 📚 2. Libraries Used

- torch  
- torchvision  
- opencv-python  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

Install them via:

```
pip install -r requirements.txt
```

---

## 🗃️ 3. Dataset

- Classes:
  - `with_mask`
  - `mask_weared_incorrect`
  - `no_mask`
- Preprocessing:
  - Resizing to 224×224
  - Normalization
  - Optional augmentations: horizontal flip, random rotation, etc.

---

## 🚀 4. How to Run

### 🏋️‍♂️ Training

```bash
python train.py
```

### 🧪 Inference on an image

```bash
python detect_mask_image.py --image path/to/image.jpg
```

### 🎥 Real-Time Detection with Webcam

```bash
python detect_mask_video.py
```

---

## 📊 5. Model Performance

| Metric             | Value  |
|--------------------|--------|
| Training Accuracy  | ~99%   |
| Validation Accuracy| ~98%   |
| Loss               | < 0.1  |

You can further evaluate the model using:
- Confusion Matrix
- Precision, Recall, F1-Score

---

## 📁 6. Project Structure

```
├── train.py                  # Training script
├── detect_mask_image.py     # Inference on images
├── detect_mask_video.py     # Webcam-based detection
├── SimpleCNN.py             # Model architecture
├── dataset/                 # Dataset directory
│   ├── with_mask/
│   ├── without_mask/
│   └── mask_weared_incorrect/
├── requirements.txt
└── README.md
```

---

## 🤝 7. Contributing

Contributions are welcome!  
Feel free to fork this repo and open pull requests for any improvements.

---

## 📄 8. License

This project is licensed under the **MIT License**.
