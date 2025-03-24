# Churn Defender

This project aims to predict customer churn using a deep learning model built with TensorFlow and Keras. The dataset used is `Churn_Modelling.csv`.

## Steps Performed

### 1. Data Preprocessing
- **Dropped Irrelevant Columns**: Removed `RowNumber`, `CustomerId`, and `Surname`.
- **Encoded Categorical Variables**:
  - Used `LabelEncoder` for the `Gender` column.
  - Applied `OneHotEncoder` for the `Geography` column and merged the encoded columns back into the dataset.
- **Saved Encoders**: Stored the `LabelEncoder` and `OneHotEncoder` in `.pkl` files for future use.
- **Feature Scaling**: Standardized the features using `StandardScaler` and saved the scaler in a `.pkl` file.

### 2. Dataset Splitting
- Divided the dataset into independent features (`X`) and the target variable (`y`).
- Performed a train-test split with an 80-20 ratio.

### 3. Model Architecture
- Built a Sequential Neural Network with the following layers:
  - **Input Layer**: Connected to the first hidden layer.
  - **Hidden Layer 1**: 64 neurons, ReLU activation.
  - **Hidden Layer 2**: 32 neurons, ReLU activation.
  - **Output Layer**: 1 neuron, Sigmoid activation.

### 4. Model Compilation
- Optimizer: Adam with a learning rate of 0.01.
- Loss Function: Binary Crossentropy.
- Metric: Accuracy.

### 5. Training
- Used Early Stopping to monitor validation loss and stop training when no improvement was observed for 10 epochs.
- Logged training metrics using TensorBoard.

### 6. Model Saving
- Saved the trained model as `model.h5`.

### 7. TensorBoard
- Set up TensorBoard for visualizing training logs.

### 8. Prediction
- **Prediction Notebook**: The `prediction.ipynb` file demonstrates how to load the saved model and encoders to make predictions on new data.
- **Steps in Prediction**:
  - Load the saved model (`model.h5`).
  - Load the encoders (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`) and scaler (`scaler.pkl`).
  - Preprocess new data using the encoders and scaler.
  - Use the model to predict the probability of churn for each customer.

### 9. Streamlit Integration
- **Streamlit App**: The `app.py` file provides a web interface for interacting with the model.
- **Features**:
  - Users can input customer details through the web interface.
  - The app preprocesses the input using the saved encoders and scaler.
  - The model predicts the probability of churn and displays the result.
- **How to Run the App**:
  1. Install Streamlit:
     ```bash
     pip install streamlit
     ```
  2. Run the app:
     ```bash
     streamlit run app.py
     ```
  3. Open the provided URL in your browser to access the app.

## How to Run
1. Install the required libraries:
   ```bash
   pip install tensorflow pandas scikit-learn streamlit
   ```
2. Run the Jupyter Notebook to preprocess the data, train the model, and save the artifacts.
3. Use `prediction.ipynb` to make predictions on new data.
4. Run `app.py` to interact with the model through a web interface.
5. Start TensorBoard to visualize training logs:
   ```bash
   %tensorboard --logdir logs/fit
   ```

## Files
- `experiment.ipynb`: Contains the code for preprocessing, training, and saving the model.
- `prediction.ipynb`: Demonstrates how to load the model and make predictions on new data.
- `app.py`: Streamlit app for deploying the model with a web interface.
- `label_encoder_gender.pkl`: Saved `LabelEncoder` for the `Gender` column.
- `onehot_encoder_geo.pkl`: Saved `OneHotEncoder` for the `Geography` column.
- `scaler.pkl`: Saved `StandardScaler` for feature scaling.
- `model.h5`: Trained model file.

## Future Work
- Experiment with different architectures and hyperparameters.
- Add more advanced preprocessing techniques.
- Enhance the Streamlit app with additional features.
- Deploy the app to a cloud platform for real-time predictions.


