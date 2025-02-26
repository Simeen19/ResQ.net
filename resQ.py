import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Load the Pretrained ResNet50 Model (using local weights or skipping it)
try:
    resnet_base = ResNet50(weights='/path/to/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))  # Use local weights
except Exception as e:
    print(f"Error loading weights: {e}, using weights=None")
    resnet_base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))  # Skip pretrained weights

# Step 2: Modify the Model
model = Sequential([
    resnet_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 weather classes: Sunny, Rainy, Cloudy
])

# Step 3: Compile the Model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Load and Preprocess the Image (for testing)
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Step 5: Test the Image with the Model
test_img_path = "/kaggle/input/riany1/rainy-landscape.webp"  
test_img = prepare_image(test_img_path)

# Predict weather condition
predictions = model.predict(test_img)
class_names = ['Sunny', 'Rainy', 'Cloudy']  # Modify according to your dataset's class names
predicted_class = class_names[np.argmax(predictions)]

# Step 6: Visualize the Result
plt.imshow(image.load_img(test_img_path))  # Show the input image
plt.title(f"Predicted Weather: {predicted_class}")
plt.show()

# Print the prediction result
print(f"Prediction: {predicted_class}")
