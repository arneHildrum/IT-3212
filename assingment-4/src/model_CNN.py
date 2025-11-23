import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# --------------------------------------------------------------------------------
# 1. Configuration & Constants
# --------------------------------------------------------------------------------
IMAGE_SIZE = 128 # Matching your preprocess_CNN function output
CHANNELS = 3     # RGB images
NUM_CLASSES = 8  # 8 facial expressions
BATCH_SIZE = 32
EPOCHS = 30 # A reasonable number for initial training

# --------------------------------------------------------------------------------
# 2. Setup: Create Dummy Data (Replace with your actual data loading)
# --------------------------------------------------------------------------------
# In a real scenario, you would load and process your 18*8 images here.
# The expected shape after loading and augmentation is (N_samples, 128, 128, 3).
N_SAMPLES = 144
X = np.random.rand(N_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, CHANNELS).astype('float32')
# Dummy labels for 8 expressions (0 to 7)
y_raw = np.random.randint(0, NUM_CLASSES, N_SAMPLES)

# Convert labels to one-hot encoding (required for multi-class classification)
lb = LabelBinarizer()
y_one_hot = lb.fit_transform(y_raw)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y_raw
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 3. Define the CNN Model Architecture
#    (Based on principles from the IT3212 Deep Learning slides)
# --------------------------------------------------------------------------------

def create_cnn_model(input_shape, num_classes):
    """
    Creates a Sequential CNN model for image classification.
    """
    model = Sequential([
        # --- Layer 1: Convolutional Block ---
        # Conv2D: Learns spatial features (edges, corners, etc.)
        # Activation: ReLU is a common choice for hidden layers (Slide 15)
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', name='conv_1'),
        # MaxPooling2D: Reduces spatial dimensions (downsampling) (Slide 17)
        MaxPooling2D((2, 2), name='pool_1'),
        
        # --- Layer 2: Convolutional Block ---
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
        MaxPooling2D((2, 2), name='pool_2'),
        
        # --- Layer 3: Convolutional Block ---
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'),
        MaxPooling2D((2, 2), name='pool_3'),

        # --- Transition to Fully Connected Layers ---
        # Flatten: Converts 3D feature maps into a 1D vector for the Dense layers
        Flatten(name='flatten'),
        
        # --- Fully Connected (Dense) Layers ---
        # Dense: Standard classification layers (Slide 20)
        Dense(512, activation='relu', name='dense_1'),
        # Dropout: Regularization technique to prevent overfitting (Slide 28)
        Dropout(0.5, name='dropout_1'),
        
        # --- Output Layer ---
        # Dense: Output layer must match the number of classes (8 expressions)
        # Activation: 'softmax' for multi-class classification (Slide 15)
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    return model

# Create and compile the model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES)

# Model Compilation
# Loss: 'categorical_crossentropy' for one-hot encoded multi-class labels
# Optimizer: Adam is a strong general-purpose choice
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model summary (showing layer types and parameter count)
print("\n--- Model Summary ---")
model.summary()

# --------------------------------------------------------------------------------
# 4. Model Training
# --------------------------------------------------------------------------------
print("\n--- Starting Model Training ---")

# Train the model using the training data
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Use 10% of training data for validation during training
    verbose=1
)

# --------------------------------------------------------------------------------
# 5. Model Evaluation
# --------------------------------------------------------------------------------
print("\n--- Evaluating Model on Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Example Prediction
sample_image = X_test[0:1]
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions[0])

# Note: The true label will be in one-hot format, e.g., [0, 0, 1, 0, 0, 0, 0, 0]
true_class = np.argmax(y_test[0])

print(f"Prediction for first test image: Class {predicted_class}")
print(f"True Label for first test image: Class {true_class}")