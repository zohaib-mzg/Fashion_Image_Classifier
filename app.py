import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from skimage.feature import hog
import tensorflow as tf
import time



st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="wide"
)


st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# PREPROCESSING
def preprocess_image(image, target_size=(80, 60)):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size)
    return image

# MODEL FUNCTIONS
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Load Random Forest
        rf_data = joblib.load('models/random_forest_model.pkl')
        rf_model = rf_data['model']
        rf_accuracy = rf_data.get('test_accuracy', 0.0)
        
        # Load CNN 
        try:
            cnn_model = tf.keras.models.load_model('models/cnn_model.keras')
            cnn_available = True
        except:
            cnn_model = None
            cnn_available = False
            
        cnn_accuracy = 0.914 
        
        return {
            'label_encoder': label_encoder,
            'rf_model': rf_model,
            'rf_accuracy': rf_accuracy,
            'cnn_model': cnn_model,
            'cnn_available': cnn_available,
            'cnn_accuracy': cnn_accuracy,
            'categories': label_encoder.classes_
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def predict_rf(image, model_data):

    label_encoder = model_data['label_encoder']
    rf_model = model_data['rf_model']
    
    # Preprocess
    processed = preprocess_image(image)
    image_array = np.array(processed) / 255.0
    
    # Extract HOG features
    gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    
    # Extract color features
    hsv = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    
    # Combine features
    features = np.concatenate([hog_features, color_features]).reshape(1, -1)
    
    # Predict
    probabilities = rf_model.predict_proba(features)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(probabilities[predicted_idx])  
    
    # All probabilities
    all_probs = {
        label_encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(probabilities)
    }
    
    return predicted_class, confidence, all_probs, processed

def predict_cnn(image, model_data):

    label_encoder = model_data['label_encoder']
    cnn_model = model_data['cnn_model']
    
    # Preprocess
    processed = preprocess_image(image)
    image_array = np.array(processed) / 255.0
    image_array = image_array.reshape(1, 60, 80, 3)
    
    # Predict
    probabilities = cnn_model.predict(image_array, verbose=0)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(probabilities[predicted_idx])  # Convert to Python float
    
    # All probabilities
    all_probs = {
        label_encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(probabilities)
    }
    
    return predicted_class, confidence, all_probs, processed



# SIDEBAR
with st.sidebar:
    st.title("⚙️ Model Settings")
    
    # Model selection
    model_option = st.radio(
        "Select Model:",
        ["Random Forest", "CNN"],
        index=0
    )
    
    st.markdown("---")
    
    # Model info
    st.subheader("Model Information")
    
    if model_option == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "87.6%")
        with col2:
            st.metric("Speed", "Fast")
        st.info("Uses HOG + Color features")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "91.4%")
        with col2:
            st.metric("Speed", "Slower")
        st.info("Deep Learning CNN")
        

        model_data = load_models()
        if model_data and not model_data['cnn_available']:
            st.warning("CNN model not loaded properly")

# MAIN CONTENT
st.title("👗 Fashion Image Classifier")
st.markdown("Upload an image to classify clothing type")

# Load models
model_data = load_models()
if model_data is None:
    st.error("Failed to load models. Please check the models folder.")
    st.stop()

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"Dimensions: {image.size[0]} × {image.size[1]}")
    
    with col2:
        # Show what model sees
        processed = preprocess_image(image)
        st.subheader("🔍 Model Input")
        st.image(processed, use_container_width=True)
        st.caption(f"Resized to: 80 × 60")
    
    # Predict button
    if st.button(f"🚀 Predict with {model_option}", type="primary", use_container_width=True):
        with st.spinner(f"{model_option} is analyzing the image..."):
            time.sleep(0.3)
            
            try:
                # Make prediction based on selected model
                if model_option == "Random Forest":
                    pred_class, confidence, all_probs, proc_img = predict_rf(image, model_data)
                else:
                    if not model_data['cnn_available']:
                        st.error("CNN model is not available. Using Random Forest instead.")
                        pred_class, confidence, all_probs, proc_img = predict_rf(image, model_data)
                    else:
                        pred_class, confidence, all_probs, proc_img = predict_cnn(image, model_data)
                
                # Display results
                st.markdown("---")
                
                # Main prediction
                col_pred, col_conf = st.columns([2, 1])
                with col_pred:
                    st.markdown(f"### 🎯 Prediction: **{pred_class}**")
                with col_conf:
                    st.markdown(f"### 🔍 {confidence:.1%}")
                
                # Confidence bar - FIXED: use float value
                st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
                
                # Price estimation
                price_map = {
                    'Tshirts': '$20-40', 'Shirts': '$30-60', 'Dresses': '$50-120',
                    'Pants': '$40-80', 'Shoes': '$60-150', 'Jackets': '$80-200',
                    'Sweaters': '$50-100', 'Tops': '$25-60'
                }
                estimated_price = price_map.get(pred_class, '$30-80')
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>💰 Estimated Price: {estimated_price}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # All probabilities 
                st.subheader("📊 Category Probabilities")
                
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    col_name, col_bar, col_val = st.columns([2, 6, 1])
                    
                    with col_name:
                        if class_name == pred_class:
                            st.markdown(f"**🏆 {class_name}**")
                        else:
                            st.markdown(class_name)
                    
                    with col_bar:
                        st.progress(float(prob))  
                    
                    with col_val:
                        st.markdown(f"**{prob:.1%}**")
                
                # Model comparison toggle
                with st.expander("🔬 Compare Both Models"):
                    col_rf, col_cnn = st.columns(2)
                    
                    with col_rf:
                        st.subheader("Random Forest")
                        rf_pred, rf_conf, rf_probs, _ = predict_rf(image, model_data)
                        st.metric("Prediction", rf_pred)
                        st.metric("Confidence", f"{rf_conf:.1%}")
                        st.caption("Uses HOG + Color features")
                    
                    with col_cnn:
                        st.subheader("CNN")
                        if model_data['cnn_available']:
                            cnn_pred, cnn_conf, cnn_probs, _ = predict_cnn(image, model_data)
                            st.metric("Prediction", cnn_pred)
                            st.metric("Confidence", f"{cnn_conf:.1%}")
                            st.caption("Deep Learning model")
                        else:
                            st.warning("CNN model not available")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                st.info("Try a different image or check model files.")

else:
    # Empty state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h3>📤 Upload an Image</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
            <br>
            <div style='display: inline-block; padding: 20px; background: #f8f9fa; border-radius: 10px;'>
                <p>👕 <strong>Tshirts</strong> | 👔 <strong>Shirts</strong></p>
                <p>👗 <strong>Dresses</strong> | 👖 <strong>Pants</strong></p>
                <p>👟 <strong>Shoes</strong> | 🧥 <strong>Jackets</strong></p>
                <p>🧣 <strong>Sweaters</strong> | 👚 <strong>Tops</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Fashion Image Classifier | Random Forest & CNN Models | Data Science Project"
    "</div>", 
    unsafe_allow_html=True
)