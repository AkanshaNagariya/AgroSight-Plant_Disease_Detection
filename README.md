# 🌿 Plant Disease Detection using Deep Learning

A deep learning-based web application that detects plant leaf diseases using Convolutional Neural Networks (CNNs). This project was built using TensorFlow and deployed using Streamlit.

---

## 📌 Features

- Classifies images of plant leaves into diseases:
  - Tomato - Bacterial Spot
  - Potato - Early Blight
  - Corn - Common Rust
- Built with a custom CNN architecture
- Includes Streamlit-based Web UI for interactive predictions
- Visualization of training history and performance metrics

---

## 🧠 Model Architecture

- Conv2D → MaxPooling2D → Flatten → Dense layers
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Evaluation: Accuracy on test dataset

---

## 📁 Dataset

- The dataset is organized in folders per disease type.
- Each folder contains `.jpg` images of leaves affected by the respective disease.

Example:
/Dataset
├── Tomato-Bacterial_spot/
├── Potato-Early_blight/
├── Corn-Common_rust/

📚 Technologies Used

    Python, OpenCV, TensorFlow, Keras, NumPy

    Streamlit for frontend

    Matplotlib for visualization

📷 Sample UI
![Screenshot 2025-05-28 225502](https://github.com/user-attachments/assets/2bcc6b94-764e-43c7-8e54-75df641df6bd)
![Screenshot 2025-05-28 225552](https://github.com/user-attachments/assets/5296c8b3-59b5-4b89-b2ea-bc3df9e1a23d)

📊 Results

    Final test accuracy: ~99%
## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone  https://github.com/AkanshaNagariya/Plant_Disease_Detection.git
   cd Plant_Disease_Detection
2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py


