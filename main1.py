import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import pytesseract
import csv
from PIL import Image
import pandas as pd
import easyocr
def yolo(image):
    # Load the YOLO model with the custom weights
    model = YOLO('best.pt')
    
    # Predict using the model
    results = model.predict(image, show=False, stream=True, imgsz=640)
    
    # Required classes for the PAN card
    required_classes = ["DOB", "Emblem", "Income Logo", "Name", "Pan Number"]
    
    # Keep track of detected classes
    detected_classes = set()
    
    # Draw annotations on the image
    annotated_image = image.copy()
    
    # Iterate over the results
    for result in results:
        for box in result.boxes:
            # Get the class name for the detected object
            class_id = result.names[box.cls[0].item()]
            
            # If the detected class is one of the required classes, process it
            if class_id in required_classes:
                # Get the bounding box coordinates and confidence
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                
                # Add the class to the detected set
                detected_classes.add(class_id)

                # Annotate the image
                annotated_image = cv2.rectangle(annotated_image, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{class_id} {conf}", (cords[0], cords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Print the details
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")
    
    # Check if all required classes have been detected
    verified = detected_classes == set(required_classes)
    
    # Return the annotated image and verification status
    return annotated_image, verified 

def Sticker(image):
    model = YOLO('best1.pt')
    results = model.predict(image, show=False, stream=True, imgsz=640)
    required_classes = ["sticker"]
    detected_classes = set()
    annotated_image = image.copy()
    max_similarity = 0  # Initialize max_similarity outside the loop
    tampered = False  # Initialize tampered outside the loop
    cropped_region=0
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            if class_id in required_classes and box.conf[0].item() > 0.9:  # Check confidence level
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                detected_classes.add(class_id)

                annotated_image = cv2.rectangle(annotated_image, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{class_id} {conf}", (cords[0], cords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Crop the detected region
                cropped_region = image[cords[1]:cords[3], cords[0]:cords[2]]
                gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
                # Resize the image to a fixed size
                size = (100, 100)
                orginal = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)

                for i in range(1, 9):  # Assuming you have 9 reference images named ref1.jpeg, ref2.jpeg, ..., ref9.jpeg
                    ref_image = cv2.imread(f'Sticker/ref{i}.jpeg', cv2.IMREAD_GRAYSCALE)
                    ref_image = cv2.resize(ref_image, (100, 100), interpolation=cv2.INTER_CUBIC)
                     # Calculate SSIM
                    ssim_index = ssim(ref_image, orginal, multichannel=True)
                    print(ssim_index)
                    max_similarity=max(max_similarity, ssim_index)
                
                # Check if the sticker is tampered based on similarity threshold
                threshold = 0.1
                tampered = max_similarity < threshold

    # Check if sticker is detected and verified
    verified = detected_classes == set(required_classes)
    
    return annotated_image, verified, cropped_region, max_similarity, tampered



def remove_background(image):
    _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    foreground = image[y:y+h, x:x+w]
    return foreground 

def process_image(image, reference,reference2,reference3):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size
    size = (600, 400)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
    # Compute the structural similarity index
    ssim_index = ssim(resized, reference)

    # Compute the structural similarity index for refrence 2
    ssim_index2 = ssim(resized, reference2)
    # Compute the structural similarity index for refrence 2
    ssim_index3 = ssim(resized, reference3)
    max_ssim = max(ssim_index ,ssim_index2)
    max_ssim = max(max_ssim,ssim_index3)
    # Apply thresholding to detect tampering
    threshold = 0.4
    tampering_detected = max_ssim < threshold
    # Find contours in the image
    contours, hierarchy = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return tampering_detected, ssim_index, contours

def draw_contours(image, contours):
    # Draw the contours on the image
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 100:
            cv2.drawContours(image, contours, i, (0, 0, 255), 2)

def yolo_ocr_integration(image):
    # Load the YOLO model with custom weights
    model = YOLO('best.pt')
    
    # Read the image using OpenC
    
    # Predict using the model
    results = model.predict(image, show=False, stream=True, imgsz=640)
    
    # Initialize EasyOCR reader for English
    reader = easyocr.Reader(['en'])
    
    # Define required classes for the PAN card
    required_classes = ["DOB", "Emblem", "Income Logo", "Name", "Pan Number"]
    
    # Detected classes and texts
    detected_info = {}
    
    # Iterate over the results
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            # Process only the required classes
            if class_id in required_classes:
                # Extract bounding box coordinates
                cords = box.xyxy[0].tolist()
                cords = [int(x) for x in cords]  # Convert coordinates to integers
                
                # Crop the detected region from the image
                cropped_image = image[cords[1]:cords[3], cords[0]:cords[2]]
                
                # Use EasyOCR to read text from the cropped image
                ocr_result = reader.readtext(cropped_image)
                detected_text = ' '.join([text[1] for text in ocr_result])
                
                # Store the detected text with its class
                detected_info[class_id] = detected_text
                
                # For demonstration, let's print the detected text for each class
                print(f"{class_id}: {detected_text}")
                
    return detected_info

def main():
    st.set_page_config(page_title='PAN Card Tampering Detection')
    st.title('PAN Card Tampering Detection')
    st.write('Upload an image of a PAN card to detect tampering and validate the ID.')

    # Load the reference image 1
    reference_image = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)
    reference_image_resized = cv2.resize(reference_image, (600, 400), interpolation=cv2.INTER_CUBIC)
    # Load the reference image 2
    reference_image2 = cv2.imread('reference3.jpg', cv2.IMREAD_GRAYSCALE)
    reference_image_resized2 = cv2.resize(reference_image2, (600, 400), interpolation=cv2.INTER_CUBIC)
    # Load the reference image 3
    reference_image3 = cv2.imread('reference4.jpg', cv2.IMREAD_GRAYSCALE)
    reference_image_resized3 = cv2.resize(reference_image3, (600, 400), interpolation=cv2.INTER_CUBIC)
    # Add a file uploader to get the image from the user
    uploaded_front_image = st.file_uploader('Upload front side of the PAN card', type=['jpg', 'jpeg', 'png'])

    if uploaded_front_image is not None:
        # Read the front image
        front_image = cv2.imdecode(np.frombuffer(uploaded_front_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Remove background from the front image (assuming you have a function for this)
        front_image = remove_background(front_image)

        # Resize the front image
        front_image = cv2.resize(front_image, (600, 400), interpolation=cv2.INTER_CUBIC)

        # Display the front image
        st.image(front_image, channels='BGR', use_column_width=True)

        # Prompt user to upload the back image
        uploaded_back_image = st.file_uploader('Upload back side of the PAN card', type=['jpg', 'jpeg', 'png'])

        if uploaded_back_image is not None:
            # Read the back image
            back_image = cv2.imdecode(np.frombuffer(uploaded_back_image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Remove background from the back image (assuming you have a function for this)
            back_image = remove_background(back_image)

            # Resize the back image
            back_image = cv2.resize(back_image, (600, 400), interpolation=cv2.INTER_CUBIC)

            st.image(back_image, channels='BGR', use_column_width=True)

            # Detect tampering and validate ID with both front and back images
            tampering_detected, ssim_index, contours = process_image(front_image, reference_image_resized, reference_image_resized2, reference_image_resized3)
            annotated_image, verified = yolo(front_image)
            
            #for font image
            annotated_font, verified_font,cropped_font,max_similarity_font,tampered_sticker_font = Sticker(front_image)
            #for back side of image

            annotated_back, verified_back,cropped_back,max_similarity_back,tampered_sticker_back=Sticker(back_image)

            # Draw the contours on the front image 
            draw_contours(front_image, contours)

            

            # Display the front image and results
            st.image(front_image, channels='BGR', use_column_width=True)
            if tampering_detected:
                st.error('Tampering detected!')
            else:
                st.success('No tampering detected.')
            st.write(f'Structural similarity index: {ssim_index:.2f}')
            st.image(annotated_image, channels='BGR', caption='Processed Image')
            if verified:
                st.success("All classes have been detected.")
            else:
                st.error("Can't detect all the classes.")
            if(verified_font):
                if(tampered_sticker_font):
                    st.error("sorry your sticker cant be detected as genuine")
                    st.image(annotated_font, channels='BGR', use_column_width=True)
                    st.write(f'Structural similarity index: {max_similarity_font:.2f}')
                else:
                    st.success(" your sticker is detected as genuine")
                    st.image(annotated_font, channels='BGR', use_column_width=True)
                    st.write(f'Structural similarity index: {max_similarity_font:.2f}')
            else:
                if(tampered_sticker_back):
                    st.error("sorry your sticker cant be detected as genuine")
                    st.image(annotated_back, channels='BGR', use_column_width=True)
                    st.write(f'Structural similarity index: {max_similarity_back:.2f}')
                else:
                    st.success(" your sticker is detected as genuine")
                    st.image(annotated_back, channels='BGR', use_column_width=True)
                    st.write(f'Structural similarity index: {max_similarity_back:.2f}')

            # Placeholder for displaying detected information
            placeholder = st.empty()

            # Button to perform detection and extraction
            if st.button('Extract Information'):
                detected_info = yolo_ocr_integration(front_image)
                
                # Displaying detected information
                if detected_info:
                    placeholder.success("Information Extracted Successfully!")
                    for class_id, text in detected_info.items():
                        if class_id == "Emblem" or class_id == "Income Logo":
                            continue
                        st.subheader(class_id)
                        st.write(text)
                else:
                    placeholder.warning("No information could be extracted.")


if __name__ == '__main__':
    main()

