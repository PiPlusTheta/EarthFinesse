# EarthFinesse Terrain Recognition Engine

![Terrain Recognition](model_plot.png)

**EarthFinesse** is a state-of-the-art terrain recognition engine designed to classify images into four terrain categories: Grassy, Marshy, Rocky, and Sandy. This repository contains the code for training a deep learning model using TensorFlow and Keras to perform this classification task. The trained model can be used to recognize the terrain type in images with an exceptional accuracy of over 97%.

## Dataset

The EarthFinesse model was trained on an extensive dataset containing more than 45,100 images. Remarkably, each terrain category, including Grassy, Marshy, Rocky, and Sandy, was meticulously curated and comprised more than 10,000 images per class. This comprehensive dataset ensures that the model can generalize well and accurately classify a wide range of terrain types.

## Model

The EarthFinesse Terrain Recognition Engine leverages transfer learning with the MobileNetV2 architecture, a powerful pre-trained neural network. During training, the model fine-tuned the pre-trained weights while utilizing data augmentation techniques to achieve optimal performance. The model was trained for 10 epochs, resulting in an impressive accuracy of over 97%, setting a new benchmark in this domain.

## Files Needed for Training

To train the EarthFinesse model, several files and directories are required:

1. **Dataset Directory**: Organize your terrain image dataset into three directories: `train`, `val`, and `test`. Place the images for each terrain category in their respective subdirectories. The directory structure should resemble the following:

   ```
   - EarthFinesse
     - Split
       - train
         - Grassy
         - Marshy
         - Rocky
         - Sandy
       - val
         - Grassy
         - Marshy
         - Rocky
         - Sandy
       - test
         - Grassy
         - Marshy
         - Rocky
         - Sandy
   ```

   You can adapt the terrain categories to suit your specific use case.

2. **`terrain_recognition.py`**: This Python script is responsible for training the model, preprocessing data, and evaluating its performance. You can adjust the model configuration, batch size, and other hyperparameters as needed.

3. **Required Python Packages**: Ensure that you have the following Python packages installed:

   - Python 3.x
   - TensorFlow 2.x
   - NumPy
   - pandas
   - Matplotlib
   - scikit-learn

   You can install these packages using pip:

   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```

## Training the Model

The EarthFinesse Terrain Recognition Engine utilizes transfer learning with the MobileNetV2 architecture. You can tailor the training settings in the `terrain_recognition.py` script. To commence training, execute the following command:

```bash
python terrain_recognition.py
```

The script will automatically partition your data into training, validation, and test sets and carry out 10 training epochs by default. You are encouraged to fine-tune these settings to align with your specific requirements.

## Model Evaluation

Upon training completion, the script provides a comprehensive evaluation of the model's performance. It showcases the achieved accuracy and generates a classification report that encompasses precision, recall, and F1-score metrics for each terrain category. The model's remarkable accuracy of over 97% underscores its outstanding recognition capabilities.

## Model Saving

The trained model is automatically stored in the repository directory, bearing a filename that encapsulates the date and time of training along with the final accuracy. This facilitates the reuse of the model for terrain recognition tasks without necessitating additional training.

## Streamlit Interface

In addition to the training and evaluation features, EarthFinesse incorporates a user-friendly Streamlit-based interface for bulk image classification. The `app.py` script enables you to upload multiple reconnaissance images and obtain predictions for each image's terrain type and confidence score.

### How to Use the Streamlit Interface

1. **Install Required Packages**: First, make sure you've installed the necessary Python packages specified in the `app.py` script.

2. **Run the Streamlit App**: Execute the Streamlit app using the following command:

   ```bash
   streamlit run app.py
   ```

3. **Upload Reconnaissance Images**: You can upload one or more reconnaissance images.

4. **Configure Options**: Adjust the confidence threshold and choose whether to display prediction probabilities.

5. **View Predictions**: EarthFinesse will provide predictions for each uploaded image, including the terrain type and confidence score.

6. **Generate PDF Reports**: You can generate PDF reports summarizing the predictions, explanations, and images for your reconnaissance data.

![EarthFinesse Interface](earthfinesse_interface.png)

## Contributing

If you'd like to contribute to the EarthFinesse Terrain Recognition Engine or have suggestions for improvement, please follow these steps:

1. **Fork the Repository**: Fork this repository to your GitHub account.

2. **Create a New Branch**: Create a new branch for your feature or bug fix.

3. **Make Your Changes**: Implement your changes and thoroughly test them.

4. **Create a Pull Request**: Create a pull request with a clear description of your changes and their significance.

We welcome contributions and appreciate your help in advancing this terrain recognition engine.

## License

This project is licensed under the MIT License. Feel free to use and modify the code for your projects.
