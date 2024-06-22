# Speech Emotion Recognition

This project aims to classify emotions from speech using machine learning techniques. The dataset used includes CREMA, RAVDESS, SAVEE, and TESS.

![SER](https://github.com/Shengwei0516/Speech-Emotion-Recognition/blob/main/imgs/SER.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/speech-emotion-recognition.git
    ```
2. Navigate to the project directory:
    ```bash
    cd speech-emotion-recognition
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the datasets in the correct directories as specified in the script.
2. Run the main script to start training and evaluating the model:
    ```bash
    python main.py
    ```

## Datasets

The project uses the following datasets:
- **CREMA**: Contains audio files labeled with emotions like sad, angry, disgust, fear, happy, and neutral.
- **RAVDESS**: Contains audio files labeled with emotions like neutral, happy, sad, angry, fear, disgust, and surprise.
- **SAVEE**: Contains audio files labeled with various emotions.
- **TESS**: Contains audio files labeled with various emotions.

## Features

- Data loading and preprocessing from CREMA, RAVDESS, SAVEE, and TESS datasets.
- Feature extraction using `librosa`.
- Model training using `TensorFlow` and `Keras`.
- Evaluation with confusion matrix and classification report.

## Model Architecture

The model is built using `TensorFlow` and `Keras`. The architecture includes:
- Convolutional layers for feature extraction.
- Dense layers for classification.
- Dropout layers for regularization.

## Training

The model is trained with:
- Early stopping to avoid overfitting.
- Reduce learning rate on plateau to adjust the learning rate dynamically.
- Model checkpoint to save the best model during training.

## Evaluation

Evaluation is performed using:
- Confusion matrix to visualize the performance.
- Classification report to provide precision, recall, and F1-score.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw