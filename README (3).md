
# 🧠 Deepfake Voice Detection 🎙️  
**"Unmasking the Fake, One Voice at a Time."**

---

## 🔍 Overview

With the rapid growth of generative AI, **synthetic voices (deepfakes)** are now harder to distinguish from real human speech. This project offers a practical, ML-based solution to **detect deepfake voices**, aiming to strengthen trust in digital audio content.

> 🧠 *Built using CNNs and MFCCs, and extensible with GANs and Transformers.*

---

## 🌐 Real-World Applications

- 🔐 Voice-based Authentication Systems  
- 📱 Social Media Content Moderation  
- 📰 Fake News Prevention  
- 🎙️ Podcast & Media Authenticity Checks  
- 🚨 Forensics & Law Enforcement

---

## 🗂️ Dataset

Used from Kaggle:  
📦 [Deep Voice - Deepfake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)  
Contains two classes:
- `real/`: Authentic human voices  
- `fake/`: AI-generated deepfakes

---

## 🧠 Key Features

- 🎧 **Audio Preprocessing**  
- 🗂️ **Feature Extraction (MFCC, Spectrograms)**  
- 🤖 **ML Models**: CNN, RNN, Transformer (optional)  
- 📈 **Training & Evaluation Metrics**  
- 📊 **Visualization Dashboards**  
- 🕵️ **Real-time Prediction Support**

---

## 🖥️ Architecture Overview

### 🔁 Process Flow  
![Pipeline](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/2b0aea00-0f42-451f-a56b-5ce598418e33)

### 🧠 GAN Model  
![GAN](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/b27ac571-a358-4e22-b9d8-d3a154da4b58)

---

## 🚀 Installation & Usage

### 1. Clone the Repository  
```bash
git clone https://github.com/SankalpJumde/deepfake-voice-detection.git
cd deepfake-voice-detection
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## 🧪 Run the Project

### Run the Notebook
> Open the Jupyter notebook and execute the cells step-by-step.

### OR Train via Script
```bash
python train.py --config configs/training_config.yaml
```

### Evaluate Model
```bash
python evaluate.py --model_path models/model.pth --data_path datasets/test
```

### Predict Deepfake Audio
```bash
python detect.py --audio_file path/to/audio.wav --model_path models/model.pth
```

---

## 📊 Sample Output

```
Model Accuracy: 92.4%
Test Prediction: FAKE
Confidence: 0.94
```

![Results](https://github.com/user-attachments/assets/35c765ce-5c80-4f32-b85b-01326ffab338)

---

## 🔮 Future Scope

- 🌐 **Web Interface** (Streamlit/Flask)  
- 📱 **Mobile App Integration**  
- 🔁 **Advanced Models (LSTM/Transformer)**  
- 📊 **Voice Signature Dashboards**  
- 🌐 **Public REST API for Businesses**

---

## 💻 Tech Stack

- Python  
- TensorFlow / Keras  
- Librosa  
- NumPy, Matplotlib  
- (Optional) PyTorch, Streamlit

---

## 📚 References

1. [Audio Deepfakes: A Survey](https://www.frontiersin.org/articles/10.3389/fdata.2022.1001063/full)  
2. [GAN-based Deepfake Detection](https://dl.acm.org/doi/abs/10.1145/3595353.3595883)  
3. [IEEE GAN Overview](https://ieeexplore.ieee.org/abstract/document/9489160/)  
4. [Kaggle Dataset - Deep Voice](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)

---

## 🙌 Acknowledgements

Thanks to the open-source community, Kaggle, and ML researchers who continue to innovate in the field of deepfake detection and synthetic media analysis.

---

## 📬 Contact

Made with ❤️ by **Sankalp**  
🔗 [Add GitHub] • [LinkedIn] • [Email]

---

## 📄 License

This project is licensed under the **MIT License**.
