import streamlit as st
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os
import random

# Load gender detection model
model = load_model('gender_detection.model')

def main():
    css = """
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url("https://images.unsplash.com/photo-1557682250-33bd709cbe85?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
        }     
        
        [data-testid="stNotificationContentSuccess"]{
            color: antiquewhite;
        }
        
        [data-testid="StyledLinkIconContainer"]{
            color: antiquewhite;
            margin-left: 180px;
        }
        
        [data-testid="stMarkdownContainer"]{
            color: antiquewhite;            
        }
        
        img{
            height: 300px;
            width: 300px;
            object-fit: cover;
            border: 5px solid black;
            border-radius: 15px;
        }
        
        [data-testid="caption"]{
            color: antiquewhite;            
            font-size: 32px;            
        }

        
        [data-testid="baseButton-secondary"]{
            background-color: black;
        }        
        
    </style>
    """    
    
    st.write(css, unsafe_allow_html=True)      
    st.title("Your Lucky Actor")
    
    # Button for new prediction
    if st.button("Get New Prediction"):
        gender_prediction()

def gender_prediction():
    # Open webcam
    webcam = cv2.VideoCapture(0)

    classes = ['man', 'woman']
    num_frames = 5  # Number of frames to average predictions over
    predictions = []

    # Define directories for male actors and female actresses
    male_actors_dir = 'male_actors'
    female_actresses_dir = 'female_actors'

    # Load images in directories
    male_actor_images = os.listdir(male_actors_dir)
    female_actress_images = os.listdir(female_actresses_dir)

    # Initialize columns for layout
    col1, col2 = st.columns(2)

    # Loop through frames
    for _ in range(num_frames):
        # Read frame from webcam 
        status, frame = webcam.read()

        # Apply face detection
        face, confidence = cv.detect_face(frame)

        # Check if any faces are detected
        if not face:
            st.warning("No faces detected. Image may not be clear.")
            break  # Stop processing this frame
        
        # Loop through detected faces
        for idx, f in enumerate(face):

            # Get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply gender detection on face
            conf = model.predict(face_crop)[0] # Model.predict returns a 2D matrix

            # Get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]

            # Store prediction
            predictions.append(label)

        # Display output for the first frame only
        if _ == 0:
            # Display webcam feed in the first column
            col1.image(frame, channels="BGR", caption="Look Here")

    # Determine final prediction based on majority vote
    if predictions:
        final_prediction = max(set(predictions), key=predictions.count)
        st.success(f"Predicted Gender: {final_prediction}")

        # Display random actor/actress image based on predicted gender
        if final_prediction == 'man':
            random_actor_image = random.choice(male_actor_images)
            col2.image(os.path.join(male_actors_dir, random_actor_image), caption="Actor")
        else:
            random_actress_image = random.choice(female_actress_images)
            col2.image(os.path.join(female_actresses_dir, random_actress_image), caption="Actress")
    else:
        st.error("Image not clear. No gender prediction available.")

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
