🧠 AI-Powered Multi-Disease Medical Diagnosis Assistant
A Community-Driven Platform for AI-Based Disease Detection & Model Contribution
🚀 Overview

This project is an open-source foundation for building a unified medical image diagnosis assistant.
It provides a ready-to-use Streamlit UI, Grad-CAM explainability, and modular disease configuration system — allowing contributors to add and train models for new diseases.

💡 Core Idea

🧩 The system starts as a base framework for diagnosing a few diseases.
🌍 Contributors train and add new deep-learning models for other diseases.
🧠 Over time, it evolves into a collective, intelligent assistant capable of diagnosing many medical conditions.

🌟 Project Vision

🏥 Build an AI assistant that can analyze diverse medical images (X-rays, CTs, MRIs, Blood Smears, Retina Scans, etc.)

🤝 Allow open-source contributors to integrate and share their trained models

📈 Create a large-scale, community-curated library of disease detection models

🔍 Promote explainable AI through Grad-CAM visualizations

💬 Enable transparent, modular, and ethical medical AI research

🧩 Features

✅ Unified Diagnosis Interface – A single Streamlit app that handles multiple diseases
✅ Explainable AI (Grad-CAM) – Visualize the regions influencing predictions
✅ Dynamic JSON Configuration – Easily add new diseases and models
✅ Contributor-Friendly Structure – Plug-and-play design for community submissions
✅ Open Data Integration – Use Kaggle or other public datasets for training
✅ Cross-Platform – Runs on Windows, macOS, and Linux

🧱 How It Works
User Uploads Image 
   ↓
Select Disease Model 
   ↓
Model Inference (User-Trained / Community-Contributed)
   ↓
Grad-CAM Heatmap Generation 
   ↓
Result Visualization in Streamlit UI

🧬 Base Repository Components
File / Folder	Purpose
multi_disease_diagnosis.py	Main Streamlit app with Grad-CAM visualization
disease_info.json	Configuration file for diseases, labels, model paths & metadata
models/	Contains trained .keras or .h5 models
uploads/	Temporary folder for uploaded medical images
requirements.txt	Dependencies list
README.md	Project documentation
💡 How to Contribute a New Disease

Follow these steps to add your disease model to the platform 👇

🧪 1️⃣ Train Your Model

Use TensorFlow/Keras, PyTorch, or any ML framework to train a classification model on a medical image dataset.

Example datasets:

Kaggle Pneumonia Dataset

Brain Tumor MRI Dataset

Malaria Cell Dataset

Diabetic Retinopathy Dataset

Save your model as:

models/your_disease_model.keras

⚙️ 2️⃣ Update disease_info.json

Add a new entry with your disease details:

"Your Disease Name": {
  "description": "Short explanation of the disease.",
  "model_config": {
    "path": "models/your_disease_model.keras",
    "size": [150, 150]
  },
  "labels": ["Healthy", "Affected"]
}

🧩 3️⃣ Test Locally

Run the Streamlit app:

streamlit run multi_disease_diagnosis.py


Upload a few test images and verify predictions + Grad-CAM output.

🔁 4️⃣ Submit Your Contribution

Fork the repository

Create a new branch (feature/add-your-disease)

Add your model and JSON entry

Commit changes with a clear message

Add [Disease Name] model and configuration


Open a Pull Request

✅ Your model will be reviewed and merged into the official disease collection.

📦 Installation
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run multi_disease_diagnosis.py

🧠 Example disease_info.json Entry
"Brain Tumor": {
  "description": "A brain tumor is an abnormal growth of cells within the brain or spinal canal.",
  "model_config": {
    "path": "models/brain_tumor_model.keras",
    "size": [128, 128]
  },
  "labels": ["glioma", "meningioma", "notumor", "pituitary"]
}

🔬 Explainability with Grad-CAM

The app automatically generates a Grad-CAM heatmap, showing which areas of the medical image influenced the model’s decision — increasing trust and interpretability in AI predictions.

☁️ Deployment Options

🟢 Streamlit Cloud – Easiest one-click deployment

🟣 Hugging Face Spaces – Free ML app hosting

⚙️ Local / Dockerized Deployment – For labs and institutions

🤝 Contribution Philosophy

Every contributor helps this assistant grow into a global open medical diagnosis framework.
Your contributions make the AI smarter, more reliable, and diverse in its diagnostic capabilities.

🧠 “One framework, many minds — one assistant, many diagnoses.”

📄 License

Licensed under the MIT License.
Free for research, academic, and open-source use with proper attribution.

🧾 requirements.txt
streamlit
tensorflow
keras
opencv-python
numpy
pandas
matplotlib
pillow
scikit-learn

🔖 Repository Short Description (for GitHub top line)

🧠 Open-source Streamlit framework for AI-based medical image diagnosis — contributors add new disease models to build a universal diagnostic assistant.
