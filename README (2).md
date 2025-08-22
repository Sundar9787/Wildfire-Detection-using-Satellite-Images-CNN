# 🔥 Wildfire Detection using Satellite Images – CNN Model

This project is a **deep learning-based wildfire detection system** built with **Python** and **TensorFlow/Keras**.  
It classifies satellite images into **Fire** 🔥 or **No Fire** 🌲, helping in early wildfire detection to reduce environmental damage.  

---

## 🚀 Features

- **Binary Classification**: Detects whether a satellite image contains wildfire or not.  
- **Convolutional Neural Network (CNN)** architecture for image feature extraction.  
- **Image Augmentation** (rotation, zoom, shear, shift, flip) for robust training.  
- **High Accuracy** model trained and validated on satellite image datasets.  
- **Model Saving**: Trained model is saved as `wildfire_detection_model.h5` for reuse.  

---

## 🛠 Tech Stack

- **Language**: Python 3.x  
- **Libraries**:  
  - `tensorflow` / `keras`  
  - `numpy`  
  - `os`  

---

## 📂 Project Structure

```
📦 wildfire-detection
 ┣ 📂 wildfire_dataset       # Dataset (train/test split)
 ┣ 📜 wildfire_cnn.py        # Main model training code
 ┣ 📜 wildfire_detection_model.h5  # Saved trained model
 ┣ 📜 requirements.txt       # Python dependencies
 ┗ 📜 README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/yourusername/wildfire-detection.git
cd wildfire-detection
```

2️⃣ **Create a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows
```

3️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

4️⃣ **Prepare Dataset**  
Organize dataset as:  
```
wildfire_dataset/
 ┣ 📂 train
 ┃ ┣ 📂 fire
 ┃ ┗ 📂 no_fire
 ┗ 📂 test
    ┣ 📂 fire
    ┗ 📂 no_fire
```

---

## ▶️ Usage

Run the training script:  
```bash
python wildfire_cnn.py
```

The script will:  
1. Preprocess and augment dataset images.  
2. Train CNN model for 25 epochs.  
3. Evaluate model accuracy on test data.  
4. Save the trained model as `wildfire_detection_model.h5`.  

---

## 📷 Demo

✅ Example Satellite Images:  
- Fire Image:  
![Fire Example](fire_example.jpg)  
- No Fire Image:  
![No Fire Example](no_fire_example.jpg)  

---

## 🧠 How It Works

1. **Data Preprocessing** – Images are resized (150x150), normalized, and augmented to improve generalization.  
2. **CNN Model** –  
   - Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling  
   - Flatten → Dense(128, relu) → Dense(1, sigmoid)  
3. **Training** – Model trained with Adam optimizer and Binary Crossentropy loss.  
4. **Evaluation** – Test accuracy is printed after training.  
5. **Deployment Ready** – Model saved as `.h5` file for future predictions.  
