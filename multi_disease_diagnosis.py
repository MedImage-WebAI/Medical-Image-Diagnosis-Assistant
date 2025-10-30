import streamlit as st
import numpy as np
import cv2  # OpenCV for resizing heatmap
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps like 'jet'
from PIL import Image
import io # To handle byte stream for uploaded file
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Saitama30/brain_tumor_model.h5",
    filename="brain_tumor_model.h5"
)

# Load TensorFlow model
model = tf.keras.models.load_model(model_path)

def clear_all():
    """Completely reset the Streamlit session and UI."""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()  # Restart the app to clear everything


# --- Configuration ---

# Path to the JSON file containing disease descriptions, models, labels, etc.
DISEASE_INFO_PATH = "disease_info.json"

# --- Load Disease Information ---
try:
    with open(DISEASE_INFO_PATH, "r") as f:
        disease_info = json.load(f)
    # Extract disease names for the selectbox
    DISEASE_NAMES = list(disease_info.keys())
except FileNotFoundError:
    st.error(f"STOP: Critical Error! Could not find the disease information file at '{DISEASE_INFO_PATH}'.")
    st.info("Please ensure 'disease_info.json' exists in the same directory as the script.")
    st.stop() # Stop execution if config file is missing
except json.JSONDecodeError:
    st.error(f"STOP: Critical Error! Could not decode the JSON file at '{DISEASE_INFO_PATH}'.")
    st.info("Please check the format of 'disease_info.json'. It might be corrupted.")
    st.stop() # Stop execution if config file is invalid
except Exception as e:
    st.error(f"STOP: Critical Error! An unexpected error occurred loading '{DISEASE_INFO_PATH}': {e}")
    st.stop()


# --- Grad-CAM Helper Function ---
# --- Grad-CAM Helper Function ---
def get_gradcam_heatmap(model, image_tensor, last_conv_layer_name, class_index):
    """
    Generates a Grad-CAM heatmap for a given model and image.
    """
    try:
        # Create a model that outputs the last conv layer output and the final prediction
        # --- !!! THIS IS THE CRITICAL LINE !!! ---
        grad_model = tf.keras.models.Model(
            inputs=model.inputs, # USE model.inputs HERE (returns a list)
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        # --- !!! END CRITICAL LINE !!! ---

        # Ensure the input tensor has a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)
            if predictions.shape[-1] > 1: # Multi-class (e.g., softmax)
                loss = predictions[:, class_index]
            else: # Binary classification (e.g., sigmoid)
                loss = predictions[:, 0]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            st.warning(f"Could not compute gradients for layer {last_conv_layer_name}. Grad-CAM cannot be generated.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0] # Remove batch dimension
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()

    except Exception as e:
        # Display the specific error within the function context
        st.error(f"An error occurred during Grad-CAM generation: {e}")
        # Optionally print traceback to console for debugging if running locally
        # import traceback
        # traceback.print_exc()
        return None

# --- Streamlit App UI Setup ---

# --- CONTRIBUTION 1: ADD CUSTOM CSS FOR A POLISHED UI ---
def load_css():
    """Injects custom CSS to improve the app's appearance."""
    css = """
    <style>
        /* --- General Page --- */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* --- Title --- */
        h1 {
            color: #E0E0E0;
        }

        /* --- Column Containers (Cards) --- */
        /* This targets the containers created by st.columns */
        [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
            background-color: #1E1E1E; /* Dark card background */
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            border: 1px solid #333;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* --- Headers --- */
        h2 {
            color: #4CAF50; /* Accent color */
            border-bottom: 2px solid #333;
            padding-bottom: 0.5rem;
        }
        
        /* --- Styled Alerts (Success, Info, Warning, Error) --- */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        .stAlert[data-alert-level="success"] {
            background-color: #1A3A1A; color: #90EE90;
        }
        .stAlert[data-alert-level="info"] {
            background-color: #1A314A; color: #ADD8E6;
        }
        .stAlert[data-alert-level="warning"] {
            background-color: #4A3A1A; color: #FFD700;
        }
        .stAlert[data-alert-level="error"] {
            background-color: #4A1A1A; color: #F08080;
        }

        /* --- Styled File Uploader --- */
        .stFileUploader {
            background-color: #252525;
            border-radius: 8px;
            border: 2px dashed #444;
        }
        .stFileUploader:hover {
            border-color: #4CAF50;
        }

        /* --- Styled Tabs --- */
        [data-baseweb="tab-list"] {
            background-color: #252525;
            border-radius: 8px 8px 0 0;
        }
        [data-baseweb="tab"] {
            background-color: #252525;
            color: #A0A0A0;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background-color: #333;
            color: #2196F3;
        }
        [data-baseweb="tab-panel"] {
            background-color: #252525;
            border-radius: 0 0 8px 8px;
            padding: 1.5rem;
            border: 1px solid #333;
            border-top: none;
        }

        /* --- Styled Expander --- */
        .stExpander {
            background-color: #252525;
            border-radius: 8px;
            border: 1px solid #444;
        }
        .stExpander summary {
            font-size: 1.1rem;
            font-weight: 600;
            color: #E0E0E0;
        }
        .stExpander summary:hover {
            color: #2196F3;
        }
        
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Streamlit App UI Setup ---

st.set_page_config(page_title="🩺 Medical Diagnosis AI", layout="wide", initial_sidebar_state="expanded")
load_css() # <-- CSS IS LOADED HERE
# --- Header with Title and Clear Button ---
# --- Header with Title and Clear Button ---
header_col1, header_col2 = st.columns([6, 1])

with header_col1:
    st.markdown("<h1 style='margin-bottom:0;'>🩺 Medical Image Diagnosis Assistant</h1>", unsafe_allow_html=True)

with header_col2:
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2196F3;
            color: white;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #1976D2;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔄 Clear / Reset"):
        clear_all()


# --- Clear Button in Header ---
# col_header1, col_header2 = st.columns([4, 1])
# with col_header1:
#     st.write("")  # placeholder for alignment
# with col_header2:
#     if st.button("🔄 Clear / Reset", use_container_width=True):
#         clear_all()


# Use columns for better layout
col1, col2 = st.columns([1, 2]) # Sidebar-like column on left, main area on right

with col1:
    st.header("Configuration")
    st.write("Select the condition and upload the corresponding medical image.")

    # --- Disease Selection ---
    selected_disease = st.selectbox("1. Select potential condition:", DISEASE_NAMES, key="disease_select")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("2. Upload Image:", type=["jpg", "jpeg", "png"], key="file_uploader")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This tool is for educational purposes ONLY and does not substitute professional medical advice. AI predictions can be incorrect.")

with col2:
    st.header("Analysis Results")
    # --- Main Processing Logic ---
    if uploaded_file is not None and selected_disease in disease_info:
        try:
            # Read image bytes and open with PIL
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure RGB

            st.subheader("Uploaded Image")
            st.image(image, caption=f"Uploaded Image for {selected_disease} Analysis", use_column_width=True)
            st.markdown("---")

            # --- Model Loading ---
            model_config = disease_info[selected_disease].get("model_config")
            if not model_config:
                st.error(f"Configuration Error: 'model_config' section not found for '{selected_disease}' in disease_info.json.")
                st.stop()

            model_path = model_config.get("path")
            image_size_tuple = tuple(model_config.get("size")) # Must have size defined

            if not model_path or not image_size_tuple:
                 st.error(f"Configuration Error: 'path' or 'size' missing in 'model_config' for '{selected_disease}'.")
                 st.stop()

            if not os.path.exists(model_path):
                st.error(f"File Not Found Error: The model file specified for '{selected_disease}' was not found at: '{model_path}'.")
                st.info("Please ensure the model file exists and the path in 'disease_info.json' is correct.")
                st.stop()

            with st.spinner(f"Loading {selected_disease} model..."):
                # Load model without compiling optimizer for faster inference
                model = load_model(model_path, compile=False)

            # --- Image Preprocessing ---
            with st.spinner("Preprocessing image..."):
                img_resized = image.resize(image_size_tuple)
                img_array = np.array(img_resized) / 255.0
                input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

            # --- Prediction ---
            with st.spinner("AI analyzing image..."):
                prediction = model.predict(input_tensor)[0] # Get prediction for the single image

                # Interpret prediction (handle binary vs multi-class based on JSON labels)
                label_list = disease_info[selected_disease]["labels"]
                num_classes = len(label_list)

                if num_classes == prediction.shape[0]: # Check if prediction length matches label list
                    if num_classes > 2 : # Multi-class (softmax assumed)
                        class_index = np.argmax(prediction)
                        confidence = prediction[class_index] * 100
                    elif num_classes == 2: # Binary (sigmoid or 2-output softmax)
                        # Handle both sigmoid (1 output) and 2-output softmax
                        if prediction.shape[0] == 1: # Sigmoid
                            pred_value = prediction[0]
                        else: # 2-output Softmax
                            pred_value = prediction[1] # Assume index 1 is the 'positive' class

                        if pred_value >= 0.5:
                            class_index = 1 # Positive class (second label)
                            confidence = pred_value * 100
                        else:
                            class_index = 0 # Negative class (first label)
                            confidence = (1 - pred_value) * 100
                    else: # Only 1 label defined, unusual case
                        class_index = 0
                        confidence = prediction[0] * 100

                    predicted_label = label_list[class_index]

                else: # Mismatch between labels in JSON and model output shape
                     st.error(f"Prediction Error: Model for '{selected_disease}' produced {prediction.shape[0]} outputs, but {num_classes} labels are defined in JSON.")
                     st.info("Please ensure the 'labels' list in 'disease_info.json' matches the model's output classes.")
                     st.stop()


            st.subheader("🔬 AI Prediction Result")
            st.success(f"Predicted Condition / Finding: **{predicted_label}**")
            st.info(f"Model Confidence: **{confidence:.2f}%**")
            st.caption("Confidence indicates the model's certainty. High confidence does not guarantee correctness.")
            st.markdown("---")


            # --- Display Disease Information ---
            st.subheader(f"ℹ️ Information about '{predicted_label}' ({selected_disease})")
            info = disease_info[selected_disease] # General info for the selected disease
            details = info.get("details", {}).get(predicted_label) # Specific info for the predicted label

            if details:
                st.markdown(f"**Description:** {details.get('info', info.get('description', 'N/A'))}") # Use specific info if available, else general description
                st.markdown(f"**Common Symptoms:** {', '.join(details.get('symptoms', ['N/A'])) if details.get('symptoms') else 'N/A'}")
                st.markdown(f"**Potential Causes:** {', '.join(details.get('causes', ['N/A'])) if details.get('causes') else 'N/A'}")
                st.markdown(f"**General Treatment Overview:** {details.get('treatment_overview', 'Consult a qualified medical professional.')}")
                st.markdown(f"**Common Diagnosis Methods:** {', '.join(details.get('diagnosis_methods', ['N/A'])) if details.get('diagnosis_methods') else 'N/A'}")
                st.markdown(f"**Prevention Tips:** {', '.join(details.get('prevention', ['N/A'])) if details.get('prevention') else 'N/A'}")
            else:
                # Fallback if specific details for the predicted label aren't found
                st.markdown(f"**General Description ({selected_disease}):** {info.get('description', 'N/A')}")
                st.warning(f"Specific details for the predicted finding '{predicted_label}' were not found in the configuration.")

            st.markdown("---")

            # --- Grad-CAM Visualization ---
            st.subheader("🎯 AI Focus Area (Grad-CAM Heatmap)")

            with st.spinner("Generating explanation heatmap..."):
                # Find the last convolutional layer name automatically
                last_conv_layer_name = None
                for layer in reversed(model.layers):
                    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        last_conv_layer_name = layer.name
                        break

                if last_conv_layer_name:
                    heatmap = get_gradcam_heatmap(model, input_tensor, last_conv_layer_name, class_index)

                    if heatmap is not None:
                        # Resize heatmap to original image size using OpenCV
                        heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

                        # Apply colormap
                        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3] # Get RGB channels from colormap
                        heatmap_colored = np.uint8(255 * heatmap_colored) # Scale to 0-255

                        # Overlay heatmap onto original image
                        original_img_array = np.array(image)
                        alpha = 0.4 # Heatmap transparency
                        overlay = cv2.addWeighted(original_img_array, 1 - alpha, heatmap_colored, alpha, 0)

                        st.image(overlay, caption=f"Heatmap Overlay (AI focus for '{predicted_label}')", use_column_width=True)
                        st.caption("Brighter areas (red/yellow) indicate regions the AI deemed more important for its prediction.")
                    # else: # Error message already shown in get_gradcam_heatmap
                    #    st.warning("Could not generate Grad-CAM heatmap due to an internal error.")
                else:
                    st.warning("Could not find a suitable convolutional layer in the loaded model for Grad-CAM visualization.")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing:")
            st.exception(e) # Show detailed traceback in the app for debugging

    elif uploaded_file is None and "disease_select" in st.session_state: # Check if disease was selected
        st.info("Please upload an image using the panel on the left to begin analysis.")
    else:
         st.info("Select a condition and upload an image using the panel on the left.")

# --- Footer ---
st.markdown("---")

st.markdown("Developed as an educational project .")