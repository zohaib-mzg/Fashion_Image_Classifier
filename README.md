```markdown
# Fashion Image Classifier

A clothing classification system using **CNN** and **Random Forest**, with an interactive **Streamlit** web app to compare both models in real time.

## 📊 Model Performance

| Metric           | Random Forest | CNN        |
|------------------|---------------|------------|
| Test Accuracy    | 87.56%        | 92.25%     |
| Training Time    | ~10 min       | ~44 min    |
| Prediction Speed | ~0.1 sec      | ~0.3 sec   |

## 🗂️ Categories
T-Shirt · Shirt · Dress · Pants · Shoes · Jacket · Sweater · Top

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras (Custom CNN)
- **Machine Learning:** scikit-learn (Random Forest)
- **Computer Vision:** OpenCV, scikit-image (HOG, LBP)
- **Web App:** Streamlit

## 🚀 Getting Started

```bash
git clone https://github.com/zohaib-mzg/Fashion_Image_Classifier.git
cd Fashion_Image_Classifier
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
├── data_preparation.ipynb       # Preprocessing & cleaning
├── cnn_training.ipynb           # CNN training & evaluation
├── random_forest_training.ipynb # Feature extraction & RF training
├── app.py                       # Streamlit web app
└── requirements.txt
```

## 📌 Dataset
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — 8,000 images across 8 categories.

## 👨‍💻 Author
**Muhammad Zohaib** — BS Data Science, FAST-NUCES Lahore

## 📄 License
[MIT](LICENSE)
```

---

Short, scannable, professional — everything a recruiter or developer needs at a glance. Drop your next project whenever you're ready! 🚀
