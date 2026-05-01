# 👗 Fashion Image Classifier

A clothing classification system that identifies fashion items from images using two different machine learning approaches — a CNN and a Random Forest — with an interactive Streamlit web app to compare them side by side.

---

## 🧠 What It Does

- Classifies clothing images into **8 categories**: T-shirts, Shirts, Dresses, Pants, Shoes, Jackets, Sweaters, and Tops
- Offers **two model options**: a fast Random Forest or a more accurate CNN
- Shows **confidence scores** for each prediction
- Extracts basic **visual attributes**: dominant color, texture, and pattern

---

## 📊 Model Performance

| Metric           | Random Forest | CNN      |
|------------------|---------------|----------|
| Test Accuracy    | 87.56%        | 92.25%   |
| Training Time    | ~10 minutes   | ~44 minutes |
| Prediction Speed | ~0.1 sec      | ~0.3 sec |

---

## 🗂️ Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle — 44,000 real product images. We used 8,000 images across 8 clothing categories, split into training and test sets.

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras (Custom CNN)
- **Machine Learning:** scikit-learn (Random Forest)
- **Computer Vision:** OpenCV, scikit-image (HOG, LBP, color histograms)
- **Web App:** Streamlit
- **Other:** NumPy, Pandas, Pillow, joblib

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aliamirchoudhary/Fashion-Image-Classifier.git
cd fashion-image-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🖼️ App Features

- Upload a clothing image and get an instant prediction
- Switch between **Random Forest** and **CNN** to compare results
- View confidence scores per class
- See extracted visual attributes like color and texture

---

## ⚙️ How It Works

### CNN
Custom architecture with 3 convolutional blocks, batch normalization, and dropout for regularization. Learns features directly from images.

### Random Forest
Uses handcrafted features — HOG for shape/texture and HSV color histograms — fed into a Random Forest classifier. Faster to train and easy to interpret.

---

## 📌 Known Limitations

- Trained on clean product photos — accuracy may drop on casual or real-world images
- No GPU was used during training, so larger architectures (ResNet, EfficientNet) weren't explored

---

## 📄 License

Built for academic purposes. Feel free to fork and build on it.
