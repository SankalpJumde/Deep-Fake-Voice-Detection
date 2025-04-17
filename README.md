
# DeepFake Voice Detection 🔊🔎

Welcome to the Deepfake Voice Detection project! This repository contains the implementation of a system designed to detect deepfake audio, ensuring the authenticity of voice recordings. Leveraging state-of-the-art machine learning techniques, this project aims to provide a robust solution for identifying synthetic voices.


## Table of Contents

- **Introduction**

- **Data Set**

- **Features**

- **Code**

- **Installation**

- **Usage**

    - **Training a Model**

    - **Evaluating Model**

    - **Detecting Deepfakes in Audio**

- **Test Results**
  
- **Reference**

- **Contributing**

- **License**
## Introduction
Deepfakes are artificially generated media that can convincingly mimic real voices, posing significant threats to security, privacy, and trust in digital communications. This project addresses the growing need for reliable methods to detect such deepfake audio, providing tools and models to help mitigate these risks.

### Data Set
Link of Dataset: https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition

### Process Overview 🔁
![Screenshot 2024-06-26 140557](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/2b0aea00-0f42-451f-a56b-5ce598418e33)

Deepfake voice detection leverages Generative Adversarial Networks (GANs) to identify and differentiate between authentic and synthesized voices. GANs consist of two neural networks: a generator and a discriminator. The generator creates synthetic audio samples that mimic real human voices, while the discriminator evaluates these samples to determine their authenticity. During training, the generator continuously improves its ability to produce convincing fake voices, while the discriminator becomes more adept at identifying subtle differences between real and fake audio. This adversarial process enhances the overall detection capability, enabling the system to recognize even sophisticated deepfake voices by analyzing patterns, inconsistencies, and anomalies in the audio signals that are often imperceptible to human ears. As a result, GAN-based deepfake voice detection systems can effectively safeguard against the misuse of synthetic audio in various applications, from security to media integrity.

### GAN MODEL: 🖥️

![gan](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/b27ac571-a358-4e22-b9d8-d3a154da4b58)


## Features

- **Audio Preprocessing**: 🎧
- Tools for cleaning and preparing audio data for analysis.

- **Feature Extraction**: 🗂️
- Techniques to extract relevant features from audio clips, such as MFCCs (Mel-Frequency Cepstral Coefficients), spectrograms, and more.
- ![Screenshot 2024-06-26 140747](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/5394498e-eaab-4452-9327-638b11df355c)
![Screenshot 2024-06-26 140821](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/b87e4696-beed-4556-9124-59ad2ef687b9)


- **Machine Learning Models**: 🤖
- Implementation of various machine learning models, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers, tailored for audio deepfake detection.

- **Training and Evaluation**: 🧪
- Scripts and configurations for training models on datasets, along with evaluation metrics to assess performance.

- **Dataset Support**: 🔬
- Compatibility with multiple audio datasets, both synthetic and real, for comprehensive training and testing.

- **Visualization Tools**: 📊
- Utilities to visualize audio features, model performance, and detection results.

## Code: 👨‍💻
To view the code of this project, [Source Code](https://colab.research.google.com/drive/17z4BnxHi_PYOBmB4ezop4obrE4t1nHUL?usp=drive_link).

## Installation

To get started with the deepfake detection project, follow these steps:

    1. Clone the Repository: 
```bash
    git clone https://github.com/SankalpJumde/deepfake-voice-detection.git 
    cd deepfake-voice-detection
```

    2. Install Dependencies:

```bash
pip install -r requirements.txt
```
Download and Prepare Datasets:

• Instructions for downloading supported datasets will be provided in the datasets directory.

• Ensure datasets are placed in the correct directories as specified.    
## Usage

### Training a Model 🥷
To train a deepfake detection model, use the following command:
```bash
python train.py --config configs/training_config.yaml
```
Modify the configuration file to adjust parameters like dataset paths, model architecture, training epochs, etc.
![Screenshot 2024-06-26 140838](https://github.com/SankalpJumde/Deep-Fake-Voice-Detection/assets/135730661/d73077eb-7ff0-4e38-875f-311bc5ba6d85)


### Evaluating a Model 🔬
To evaluate a trained model, use:
```bash
python evaluate.py --model_path models/deep-fake-voice-detection.pth --data_path datasets/test_data
```
### Detecting Deepfakes in Audio 🕵️‍♂️✔️
For real-time detection on audio files:
```bash
python detect.py --audio_file path/to/audio.wav --model_path models/deep-fake-voice-detection.pth
```

## Test Reults 📋📌
![image](https://github.com/user-attachments/assets/35c765ce-5c80-4f32-b85b-01326ffab338)
Successfully Testes both Real voice as well as Deepfake generated voice! 🎯

## Reference 📚
1. [Generative Adversarial Network (GAN): a general review on different variants of GAN and applications](https://ieeexplore.ieee.org/abstract/document/9489160/)
2. [GAN Discriminator based Audio Deepfake Detection](https://dl.acm.org/doi/abs/10.1145/3595353.3595883)/
3. [Deepfake detection using deep learning methods: A systematic and comprehensive review](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1520/)
4. [The Effect of Deep Learning Methods on Deepfake Audio Detection for Digital Investigation](https://www.sciencedirect.com/science/article/pii/S1877050923002910/)
5. [Audio deepfakes: A survey](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2022.1001063/full/)

## Contributing 

We welcome contributions from the community! If you have ideas, bug fixes, or improvements, please open an issue or submit a pull request. Make sure to follow our contribution guidelines.
## License 📌
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the LICENSE file for more details.



## Acknowledgements

We would like to thank the developers of the libraries and datasets used in this project. Special thanks to the open-source community for their invaluable resources and support.

Feel free to reach out with any questions or feedback. Together, we can make digital communications safer and more secure!


