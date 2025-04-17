
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

🎭 How GANs Help Detect Deepfakes
Deepfake voice detection uses Generative Adversarial Networks (GANs)—a battle between two neural networks:

- 🛠️ Generator: Creates ultra-realistic fake voices

- 🕵️ Discriminator: Tries to catch the fakes

Through this tug-of-war, the system learns to detect even subtle, imperceptible audio patterns.
💡 This makes GANs powerful for spotting synthetic voices in security, media, and digital forensics!

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

## 📊 Sample Output 📌

```
Model Accuracy: 92.4%
Test Prediction: FAKE
Confidence: 0.94
```

![Results](https://github.com/user-attachments/assets/35c765ce-5c80-4f32-b85b-01326ffab338)

- Successfully Tests both Real Voices as well as Deepfake Generated Voices! 🎯
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

## Reference 📚
1. [Generative Adversarial Network (GAN): a general review on different variants of GAN and applications](https://ieeexplore.ieee.org/abstract/document/9489160/)
2. [GAN Discriminator based Audio Deepfake Detection](https://dl.acm.org/doi/abs/10.1145/3595353.3595883)/
3. [Deepfake detection using deep learning methods: A systematic and comprehensive review](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1520/)
4. [The Effect of Deep Learning Methods on Deepfake Audio Detection for Digital Investigation](https://www.sciencedirect.com/science/article/pii/S1877050923002910/)
5. [Audio deepfakes: A survey](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2022.1001063/full/)

---

## 🙌 Acknowledgements

Thanks to the open-source community, Kaggle, and ML researchers who continue to innovate in the field of deepfake detection and synthetic media analysis.

---

## 📬 Contact

Made with ❤️ by **Sankalp**  
🔗 ![[GitHub](https://github.com/SankalpJumde)] • [[LinkedIn](https://www.linkedin.com/in/sankalp-jumde/)("C:\Users\shruj\Downloads\linkedin.png")] • [[Email](sankalpkrishna1103@gmail.com)]

---

## License 📌
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the LICENSE file for more details.

--- 
