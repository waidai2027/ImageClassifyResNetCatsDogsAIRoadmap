from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load ResNet50 pre-trained on ImageNet without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers (optional - you can also unfreeze some layers later for fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier layers on top of ResNet
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),  # Additional layer
    Dropout(0.5),
    Dense(2, activation='sigmoid')
])


# Compile the model with an appropriate optimizer and loss function
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Example: Loading data using ImageDataGenerator for preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('data/validation', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Training the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Optional: Fine-tuning
# Unfreeze some layers of ResNet50 to fine-tune them (adjust learning rate to a smaller value)
for layer in base_model.layers[-10:]:  # Unfreeze the last 10 layers, for example
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_tune = model.fit(train_generator, epochs=5, validation_data=validation_generator)
