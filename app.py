import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2  # Added for Face Validation
from tensorflow.keras import layers, models

# 1. Page Config
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️")
st.title("🛡️ Deepfake Detector (With Face Validation)")


st.info("📢 **NOTE:** Please upload close-up face images. The system will reject non-human images.")

# 2. Load Model
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.build((None, 224, 224, 3))
    
    try:
        # Make sure your file is named exactly this in the folder!
        model.load_weights('deepfake_v2_140k.keras') 
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        
    return model

with st.spinner("Loading AI Brain..."):
    model = load_model()

# --- STRICT FACE VALIDATION FUNCTION ---
def contains_face(image):
    # Convert PIL Image to OpenCV format (numpy array)
    img_array = np.array(image.convert('RGB'))
    # Convert to Grayscale for the detector
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load the Haar Cascade Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # ⚠️ STRICT MODE TRIGGERED ⚠️
    # minNeighbors=12 (Default is 5. Higher means it requires more overlapping detections)
    # minSize=(100, 100) (Ignores tiny background faces or small animal features)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=12,  
        minSize=(100, 100)
    )
    
    return len(faces) > 0

# 3. File Uploader
file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)
    
    # --- NEW: THE SECURITY GATE ---
    with st.spinner("Scanning for human features..."):
        has_face = contains_face(image)
        
    if not has_face:
        # STOP EXECUTION IF NO FACE IS FOUND
        st.error("🛑 **ERROR: No human face detected!**")
        st.warning("Please upload a clear picture of a person's face. Images of objects, animals, or distant landscapes are not supported.")
    else:
        # PROCEED ONLY IF A FACE IS FOUND
        st.success("✅ Face detected. Analyzing for Deepfake artifacts...")
        
        # Preprocessing
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = np.array(image_resized)
        normalized_image_array = img_array.astype(np.float32) / 255.0
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Prediction
        prediction_score = model.predict(data)[0][0]
        st.write(f"🔍 **Raw Confidence Score:** {prediction_score:.4f}")

        # Strict Mode Logic
        threshold = 0.75
        
        if prediction_score > threshold:
            st.success(f"Result: REAL Person 🟢 ({prediction_score*100:.1f}%)")
        elif prediction_score < 0.5:
            st.error(f"Result: FAKE / AI Image 🔴 ({(1-prediction_score)*100:.1f}%)")
        else:
            st.error(f"Result: SUSPICIOUS / POTENTIAL FAKE 🔴")
            st.warning(f"⚠️ Low confidence. Flagged as FAKE for safety.")