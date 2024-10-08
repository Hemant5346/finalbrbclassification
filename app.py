import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline

# Set page config at the very beginning
st.set_page_config(page_title="Hair Type Classifier", layout="wide")

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
    
    pillow_mask = segmentation_pipeline(image, return_mask=True)  
    segmented_image = segmentation_pipeline(image) 
    
    image = np.array(segmented_image)
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return segmented_image, predicted_class

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
        transition-duration: 0.4s;
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

# Capture image from the webcam
st.write("Click 'Capture Image' to take a picture from the webcam.")
if st.button("Capture Image"):
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        ret, frame = cap.read()
        if ret:
            cap.release()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Captured Image", use_column_width=True)

            # Process the captured image
            if detect_face(img_rgb):
                background_removed_img, predicted_class = classify_image(img_rgb)
                st.image(background_removed_img, caption="Background Removed Image", use_column_width=True)

                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<p class="result-header">Analysis Results</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-text">Predicted Hair Type: {predicted_class}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No face detected in the image.")
        else:
            st.error("Error: Could not capture an image from the webcam.")
