import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="Hair Type Classifier", layout="wide")

import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 7)  # Adjust this based on your number of classes
model.load_state_dict(torch.load('best_hair_classification_model22.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = [
    "Coily hair",  
    "Curly hair",  
    "Long Hair",   
    "Medium Hair",
    "Short hair",  
    "Straight Hair",
    "Wavy Hair"    
]

segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    
    # Apply segmentation pipeline to remove background
    segmentation_results = segmentation_pipeline(image)
    mask = segmentation_results[0]['mask']
    image = Image.composite(image, Image.new('RGB', image.size), mask)
    
    # Preprocess image for classification
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        font-size: 16px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .result-container {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .result-header {
        color: #4CAF50;
        font-size: 24px;
        font-weight: bold;
    }
    .result-text {
        font-size: 18px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header and Description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Hair Type Classifier</h1>", unsafe_allow_html=True)
st.write("""
    This application uses your webcam to capture an image, detects faces in the image, 
    removes the background, and classifies the type of hair based on the captured image.
    Use the controls below to capture an image and get results.
""")

# Create a column layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Capture Image")
    st.write("**Take a Picture**")
    
    # Capture image from webcam
    img_file_buffer = st.camera_input("Capture an image")
    
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the captured image
        st.image(img_rgb, caption="Captured Image", use_column_width=True)
        
        if detect_face(img_rgb):
            # Process the image and display results
            predicted_class = classify_image(img_rgb)
            
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown('<p class="result-header">Analysis Results</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">Predicted Hair Type: {predicted_class}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No face detected in the image.")

with col2:
    st.header("Instructions")
    st.write("""
        1. Allow access to your webcam when prompted.
        2. Click on **'Capture an image'** to take a picture.
        3. Wait for the analysis results to appear.
    """)

