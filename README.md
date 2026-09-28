# 🐱🐶 Cat vs Dog Image Classification using CNN

## 📌 Project Overview

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras** to classify images into two categories:

- 🐱 **Cat**
- 🐶 **Dog**

The trained CNN model accepts an image as input, processes it using convolutional layers, and predicts whether the image contains a cat or a dog.

The model is also deployed as an interactive **Streamlit web application**, allowing users to upload an image and receive a prediction with confidence.

---

## 🎯 Project Objectives

- Build an image classification model using CNN.
- Perform image preprocessing and normalization.
- Train the model using the Cats vs Dogs dataset.
- Evaluate the model's classification performance.
- Save the trained model in `.keras` format.
- Create a user-friendly Streamlit application.
- Deploy the model for real-time image prediction.

---

## 📂 Dataset

The project uses the **Microsoft Cats vs Dogs Dataset**.

The dataset contains images belonging to two classes:

```text
Cat
Dog
```

Each image is resized to:

```text
128 × 128 × 3
```

where:

- `128` → Image height
- `128` → Image width
- `3` → RGB color channels

---

## 🛠️ Technologies Used

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Programming language       |
| TensorFlow       | Deep learning framework    |
| Keras            | CNN model development      |
| NumPy            | Numerical operations       |
| Pandas           | Data processing            |
| Matplotlib       | Data visualization         |
| PIL              | Image processing           |
| Streamlit        | Web application            |
| Jupyter Notebook | Model development          |
| Kaggle           | Model training environment |

---

## 🧠 CNN Architecture

The model contains multiple convolutional blocks.

```text
Input Image
(128 × 128 × 3)
       │
       ▼
Conv2D - 32 Filters
       │
Batch Normalization
       │
MaxPooling
       │
       ▼
Conv2D - 64 Filters
       │
Batch Normalization
       │
MaxPooling
       │
       ▼
Conv2D - 128 Filters
       │
Batch Normalization
       │
MaxPooling
       │
       ▼
Conv2D - 256 Filters
       │
Batch Normalization
       │
MaxPooling
       │
       ▼
Flatten
       │
       ▼
Dense - 256
       │
       ▼
Dropout - 0.5
       │
       ▼
Dense - 1
Sigmoid Activation
       │
       ▼
Cat / Dog
```

---

## 🔍 Model Components

### Conv2D

`Conv2D` extracts important visual features from the image such as:

- Edges
- Shapes
- Textures
- Patterns
- Object features

### Batch Normalization

Batch Normalization helps stabilize and speed up the training process.

### MaxPooling

MaxPooling reduces the spatial dimensions of feature maps while retaining important features.

### Flatten

Flatten converts the 2D feature maps into a one-dimensional vector before passing them to the fully connected layers.

### Dense Layer

The Dense layer learns the relationship between extracted image features and the target classes.

### Dropout

A dropout rate of `0.5` is used to reduce overfitting during training.

### Sigmoid

The final layer uses sigmoid activation because this is a **binary classification** problem.

```text
Output < 0.5  → Cat
Output >= 0.5 → Dog
```

---

## 🔄 Image Preprocessing

Before prediction, the uploaded image is:

1. Loaded using PIL.
2. Resized to `128 × 128`.
3. Converted to RGB.
4. Converted into a NumPy array.
5. Pixel values normalized from `0–255` to `0–1`.
6. Batch dimension added.
7. Passed to the CNN model.

Example:

```python
image = image.resize((128, 128))

image_array = np.array(image)

image_array = image_array / 255.0

image_array = np.expand_dims(image_array, axis=0)
```

---

## 🚀 Streamlit Application

The project includes a Streamlit interface where users can:

1. Upload an image.
2. Preview the image.
3. Click the **Predict** button.
4. Get the predicted class.
5. View prediction confidence.

Example workflow:

```text
User
 │
 ▼
Upload Image
 │
 ▼
Image Preprocessing
 │
 ▼
CNN Model
 │
 ▼
Sigmoid Prediction
 │
 ▼
Cat / Dog
 │
 ▼
Confidence Score
```

---

## 📁 Project Structure

```text
Cat_Dog_Classifier/
│
├── app.py
├── cat_dog_cnn.keras
├── requirements.txt
├── README.md
│
└── notebooks/
    └── cat_dog_cnn.ipynb
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project

```bash
cd Cat_Dog_Classifier
```

### 3. Create a virtual environment

```bash
python -m venv tf_env
```

### 4. Activate the environment

Windows:

```bash
tf_env\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Prediction Logic

The model produces a sigmoid probability.

```python
prediction = model.predict(image)
probability = prediction[0][0]
```

The prediction is interpreted as:

```python
if probability >= 0.5:
    result = "Dog"
else:
    result = "Cat"
```

Confidence is calculated as:

```python
if probability >= 0.5:
    confidence = probability
else:
    confidence = 1 - probability
```

---

## 📈 Model Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Example:

```text
Accuracy
Precision
Recall
F1 Score
```

Training and validation curves can also be used to identify possible **overfitting or underfitting**.

---

## 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

- Deep Learning
- Convolutional Neural Networks
- Image Classification
- Image Preprocessing
- TensorFlow/Keras
- Model Training
- Model Evaluation
- Binary Classification
- Overfitting Prevention
- Model Serialization
- Streamlit Deployment
- Building an end-to-end ML application

---

## 🔮 Future Improvements

The project can be improved by:

- Using Transfer Learning.
- Experimenting with MobileNetV2, ResNet50, or EfficientNet.
- Applying data augmentation.
- Hyperparameter tuning.
- Increasing dataset size.
- Improving model accuracy.
- Adding multiple image prediction.
- Adding prediction history.
- Deploying the application to a cloud platform.

---

## 👨‍💻 Author

**Nithyanantham**

Data Science | Machine Learning | Deep Learning | Python

---

## ⭐ Project Highlights

```text
✔ CNN-based Image Classification
✔ TensorFlow + Keras
✔ Binary Classification
✔ 128 × 128 RGB Images
✔ Batch Normalization
✔ MaxPooling
✔ Dropout
✔ Sigmoid Activation
✔ Streamlit Web Application
✔ Real-time Image Prediction
```
