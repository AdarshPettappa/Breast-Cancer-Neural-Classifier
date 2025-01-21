# Breast-Cancer-Neural-Classifier

This project leverages machine learning and deep learning techniques to classify breast cancer as benign or malignant using a neural network model. The workflow includes data preprocessing, model building, evaluation, and a predictive system. Below are the key components:

---

## Technologies Used
- **Programming Language:** Python 3
- **Libraries and Frameworks:**
  - **NumPy:** Efficient numerical computations.
  - **Pandas:** Data manipulation and analysis.
  - **Matplotlib:** Data visualization for accuracy and loss plots.
  - **Scikit-learn:** Dataset handling, preprocessing (StandardScaler), and train-test splitting.
  - **TensorFlow/Keras:** Neural network construction, training, and evaluation.
- **Development Environment:** Google Colab for code execution and GPU acceleration.

---

## Data Collection and Preprocessing
- **Dataset:** Breast cancer dataset from the `sklearn.datasets` library.
- **Data Handling:**
  - Loaded data into a pandas DataFrame for analysis.
  - Added a 'label' column representing the target values (0 = Benign, 1 = Malignant).
  - Inspected data for missing values and calculated statistical measures.
  - Analyzed the target distribution and computed the mean of features grouped by target values.
- **Feature Extraction:**
  - Features (`X`) and labels (`Y`) were separated.
  - Data was split into training and testing sets (80%-20% split).
  - Standardized features using `StandardScaler` for optimal model performance.

---

## Neural Network Construction
- **Framework:** TensorFlow and Keras.
- **Model Architecture:**
  - Input layer: Flatten layer for input features (30 features).
  - Hidden layer: Dense layer with 20 neurons and ReLU activation.
  - Output layer: Dense layer with 2 neurons and sigmoid activation for binary classification.
- **Compilation:** Used Adam optimizer, sparse categorical cross-entropy loss, and accuracy metrics.
- **Training:** Trained the model for 10 epochs with 10% of training data reserved for validation.

---

## Model Evaluation and Visualization
- **Performance Metrics:**
  - Evaluated model accuracy on test data.
  - Visualized training and validation accuracy/loss across epochs to assess model performance.
- **Testing Results:** Achieved a measurable accuracy on the test dataset.

---

## Predictor System
- Developed a system to classify new data points as benign or malignant:
  - **Example Input:** A feature vector representing tumor characteristics.
  - **Steps:**
    1. Standardized input data using the trained scaler.
    2. Predicted probabilities using the trained model.
    3. Converted probabilities into class labels (0 or 1).
    4. Output the prediction (Benign or Malignant) based on the label.

---

This code provides a streamlined approach to build and evaluate a neural network for medical data classification, with clear steps for preprocessing, model development, and deploying a predictive system.

