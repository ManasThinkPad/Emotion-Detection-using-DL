# 🧠 Emotion Detection Using Deep Learning  

### 👤 Author  
**Manas Mulchandani**  
**B13 - 64 | Thadomal Shahani Engineering College**

---

## 📘 Project Overview  
**Emotion Detection** is an advanced Artificial Intelligence application that recognizes and interprets human emotions using facial expressions.  
This project leverages **Computer Vision** and **Deep Learning** to build a **real-time emotion recognition system** capable of classifying seven key emotions from live webcam feeds.

---

## 🎯 Objective  
To develop a **real-time emotion detection system** using a pre-trained **CNN model** (Mini-XCEPTION) trained on the **FER2013** dataset.  
The system captures live video input, detects faces, analyzes expressions, and predicts emotional states accurately.

---

## 🧩 System Architecture  

1. **Face Detection** 🧍‍♂️  
   - Detects and isolates faces in real-time using the **Haar Cascade Classifier**.

2. **Image Preprocessing** 🖼️  
   - Converts frames to **grayscale** and resizes them to **48×48 pixels** for model input compatibility.

3. **CNN Classification** 🤖  
   - The **Mini-XCEPTION** model classifies facial features into seven emotional categories.

4. **Real-Time Output** 📸  
   - Displays bounding boxes and predicted emotion labels directly on the webcam feed.

---

## 🧰 Technology Stack  

| Component | Technology Used |
|------------|-----------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Computer Vision Library** | OpenCV |
| **Data Processing** | NumPy |
| **Dataset** | FER2013 (Facial Expression Recognition 2013) |

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/Emotion-Detection.git
cd Emotion-Detection
