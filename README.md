# EarthFinesse: Military Terrain Classifier

[![GitHub stars](https://img.shields.io/github/stars/PiPlusTheta/EarthFinesse)](https://github.com/PiPlusTheta/EarthFinesse/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PiPlusTheta/EarthFinesse)](https://github.com/PiPlusTheta/EarthFinesse/network/members)
[![GitHub license](https://img.shields.io/github/license/PiPlusTheta/EarthFinesse)](https://github.com/PiPlusTheta/EarthFinesse/blob/main/LICENSE)

EarthFinesse is a high-accuracy military terrain classifier powered by deep learning. It classifies terrain types such as Grassy, Marshy, Rocky, and Sandy with an accuracy of over 97.87%, setting a new benchmark in this domain. The model uses the MobileNetV2 architecture, optimized for efficient and accurate terrain classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Training Procedure](#training-procedure)
- [Training Results](#training-results)
- [Applications](#applications)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PiPlusTheta/EarthFinesse.git
   cd EarthFinesse
	```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit User Interface
![WhatsApp Image 2023-09-13 at 23 50 32](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/ca6f9098-4112-476f-b896-090410e9c439)

EarthFinesse comes with a user-friendly Streamlit interface for bulk image classification. Run the following command to start the application:

```bash
streamlit run app.py
```

Upload reconnaissance images, and the model will classify them into terrain types with confidence scores.

### Model Inference

To perform individual image classification using the trained model, use the following code snippet:

```python
# Load the model
from tensorflow.keras.models import load_model

model = load_model('terrain__2023_09_13__11_52_06___Accuracy_0.9787.h5')

# Load and preprocess the image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Perform inference
prediction = model.predict(img)
label_index = np.argmax(prediction)
terrain_label = {0: 'Grassy', 1: 'Marshy', 2: 'Rocky', 3: 'Sandy'}[label_index]
confidence = prediction[0, label_index]

print(f"Predicted Terrain: {terrain_label}")
print(f"Confidence: {confidence * 100:.2f}%")
```

## Model Training

### Dataset

The model was trained on a dataset consisting of 45.1k images, with more than 10k images for each terrain class (Grassy, Marshy, Rocky, Sandy).

![WhatsApp Image 2023-09-13 at 13 18 11](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/65ab6221-7657-4dca-99e2-87ed4eb9036f)

### Training Procedure

#### Data Augmentation

The training data is augmented using techniques like shear, zoom, and horizontal flip to increase diversity.

#### MobileNetV2 Base Model

<img width="252" alt="Screen_Shot_2020-06-06_at_10 37 14_PM" src="https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/e2766ee8-1442-4a9e-874b-cb9c772c621f">

The MobileNetV2 architecture, pre-trained on ImageNet, is used as the base model for feature extraction. All base model layers are frozen to retain pre-trained knowledge.

#### Custom Classification Head

A custom classification head is added to the base model. It includes a global average pooling layer, a dense layer with 1024 units and ReLU activation, and a final dense layer with softmax activation for the number of classes (4 in this case).

#### Compilation and Training

The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is then trained for 10 epochs.

Here's how the model was trained:

```python
# Data augmentation and generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# ... (similar setup for test and validation generators)

# MobileNetV2 base model
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ... (freeze base_model layers and add custom classification head)

# Compilation and training
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### Training Results

EarthFinesse achieved remarkable training results, setting a new benchmark in terrain classification:

#### Final Accuracy

The model achieved a stunning final accuracy of over 97.87%, showcasing its robust performance in classifying terrain types. This high accuracy can significantly enhance the effectiveness of military operations.

#### Confusion Matrix
![WhatsApp Image 2023-09-12 at 15 43 47](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/39b98fd1-ded6-4950-a06a-a76966865250)


#### Training History

| Epoch | Loss     | Accuracy | Validation Loss | Validation Accuracy |
|-------|----------|----------|-----------------|---------------------|
| 0     | 0.274514 | 0.897999 | 0.180294        | 0.934242            |
| 1     | 0.151308 | 0.945084 | 0.208954        | 0.924763            |
| 2     | 0.121970 | 0.956086 | 0.170471        | 0.941647            |
| 3     | 0.101868 | 0.962776 | 0.154959        | 0.947571            |
| 4     | 0.090680 | 0.967120 | 0.118927        | 0.961345            |
| 5     | 0.080031 | 0.970640 | 0.128688        | 0.959271            |
| 6     | 0.073431 | 0.974317 | 0.131562        | 0.957198            |
| 7     | 0.071057 | 0.974508 | 0.123268        | 0.961197            |
| 8     | 0.064471 | 0.977139 | 0.129367        | 0.958235            |
| 9     | 0.059202 | 0.978661 | 0.114494        | 0.966380            |

![WhatsApp Image 2023-09-13 at 13 03 33](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/d8d1f4fc-a2e2-4efd-84c4-1b03dc562955)


These training metrics illustrate the model's progression over the training epochs, with both training and validation accuracy steadily increasing.

## Applications

The EarthFinesse Military Terrain Classifier has versatile applications across various domains, including but not limited to:

#### 1. Military Operations

   - **Tactical Planning:** The classifier assists military strategists in understanding the terrain composition, helping them plan tactical maneuvers effectively.
   
   - **Mission Customization:** Military missions can be customized based on the terrain type, optimizing resource allocation and troop deployment.
   
   - **Camouflage Strategies:** Knowledge of terrain types aids in developing appropriate camouflage strategies to blend in with the surroundings.
   
#### 2. Environmental Monitoring

   - **Conservation Efforts:** Conservationists can utilize the classifier to monitor and protect specific ecosystems, such as marshlands and forests.
   
   - **Disaster Response:** During natural disasters, the classifier can identify affected terrain types, aiding in disaster response and recovery efforts.
   
   - **Ecological Research:** Researchers can employ the classifier for ecological studies to analyze terrain diversity and its impact on local ecosystems.
   
#### 3. Agriculture and Land Management

   - **Precision Agriculture:** Farmers can make data-driven decisions by assessing soil types and choosing optimal crops for specific terrains.
   
   - **Land Development:** Urban planners and land developers can benefit from understanding the terrain for sustainable land use.
   
#### 4. Autonomous Vehicles

   - **Navigation:** Autonomous vehicles, such as drones and self-driving cars, can use terrain classification for safe and efficient navigation.
   
   - **Obstacle Avoidance:** Identifying rough or impassable terrains helps autonomous vehicles avoid obstacles and hazards.
   
#### 5. Geographic Information Systems (GIS)

   - **Map Creation:** The classifier contributes to the creation of detailed maps by categorizing terrains accurately.
   
   - **Geospatial Analysis:** Geospatial analysts can integrate terrain data for comprehensive geospatial analysis.
   
These applications demonstrate the broad utility of the EarthFinesse Military Terrain Classifier in various fields, enhancing decision-making and resource optimization.

## Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please review our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
