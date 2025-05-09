# CNN Image Classifier in PyTorch

This project demonstrates the implementation of an image classifier using a Convolutional Neural Network (CNN) in PyTorch. The model is trained on a custom image dataset (likely consisting of categories such as dogs and cats or similar classes).

## 📁 Contents

- **CNN.ipynb** – Main notebook containing the full pipeline: data loading, preprocessing, model training, and evaluation.
- Libraries used: `torch`, `torchvision`, `matplotlib`, `scikit-learn`, `seaborn`, `PIL`, `tqdm`.

## 🖼️ Dataset

The dataset was collected during an experiment using an accelerometer sensor. The raw sensor data was transformed into spectrogram images.

## 🧠 Model

The model is a custom Convolutional Neural Network (CNN) built using torch.nn, inspired by the Tiny VGG architecture. It was implemented from scratch and includes convolutional layers, ReLU activations, pooling layers, and fully connected layers.

## 📊 Evaluation

- Metrics computed: accuracy, precision, recall, F1-score.
- A confusion matrix and classification report are also generated.

## 📝 Author

This project is the result of my engineering thesis, combining signal processing and convolutional neural networks.