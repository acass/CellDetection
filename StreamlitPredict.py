# streamlit run StreamlitPredict.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("YOLO Object Detection App")


# Model loading with caching
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

if uploaded_file is not None:
    # Create columns for original and processed images
    col1, col2 = st.columns(2)

    # Display original image
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process image button
    if st.button("Detect Objects"):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Run inference
            results = model(tmp_path, conf=conf_threshold)

            # Display results
            with col2:
                st.subheader("Detected Objects")

                # Get the result image with annotations
                for result in results:
                    boxes = result.boxes
                    # Plot the image with detections
                    res_plotted = result.plot()
                    st.image(res_plotted, caption="Detected Objects", use_container_width=True)

                    # Display detection information
                    if len(boxes) > 0:
                        st.subheader("Detection Details:")
                        for box in boxes:
                            conf = box.conf.cpu().numpy()[0]
                            cls = box.cls.cpu().numpy()[0]
                            cls_name = model.names[int(cls)]
                            st.write(f"Object: {cls_name}, Confidence: {conf:.2f}")
                    else:
                        st.write("No objects detected.")

            # Clean up temporary file
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error during detection: {str(e)}")

# Add information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This app uses YOLO to detect objects in images. "
    "Upload an image and adjust the confidence threshold to see the results."
)

# Add usage instructions
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload an image using the file uploader
    2. Adjust the confidence threshold if needed
    3. Click 'Detect Objects' to run the detection
    4. View the results and detection details
    """
)








# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import tempfile
# import os
#
# st.set_page_config(page_title="YOLO Object Detection", layout="wide")
#
# st.title("YOLO Object Detection App")
#
#
# # Model loading with caching
# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")
#
#
# try:
#     model = load_model()
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {str(e)}")
#     st.stop()
#
# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
#
# # Confidence threshold slider
# conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
#
# if uploaded_file is not None:
#     # Create columns for original and processed images
#     col1, col2 = st.columns(2)
#
#     # Display original image
#     with col1:
#         st.subheader("Original Image")
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#
#     # Process image button
#     if st.button("Detect Objects"):
#         try:
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_path = tmp_file.name
#
#             # Run inference
#             results = model(tmp_path, conf=conf_threshold)
#
#             # Display results
#             with col2:
#                 st.subheader("Detected Objects")
#
#                 # Get the result image with annotations
#                 for result in results:
#                     boxes = result.boxes
#                     # Plot the image with detections
#                     res_plotted = result.plot()
#                     st.image(res_plotted, caption="Detected Objects", use_column_width=True)
#
#                     # Display detection information
#                     if len(boxes) > 0:
#                         st.subheader("Detection Details:")
#                         for box in boxes:
#                             conf = box.conf.cpu().numpy()[0]
#                             cls = box.cls.cpu().numpy()[0]
#                             cls_name = model.names[int(cls)]
#                             st.write(f"Object: {cls_name}, Confidence: {conf:.2f}")
#                     else:
#                         st.write("No objects detected.")
#
#             # Clean up temporary file
#             os.unlink(tmp_path)
#
#         except Exception as e:
#             st.error(f"Error during detection: {str(e)}")
#
# # Add information about the app
# st.sidebar.title("About")
# st.sidebar.info(
#     "This app uses YOLO to detect objects in images. "
#     "Upload an image and adjust the confidence threshold to see the results."
# )
#
# # Add usage instructions
# st.sidebar.title("Instructions")
# st.sidebar.markdown(
#     """
#     1. Upload an image using the file uploader
#     2. Adjust the confidence threshold if needed
#     3. Click 'Detect Objects' to run the detection
#     4. View the results and detection details
#     """
# )