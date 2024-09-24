from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    rotation_range=20,  # Data augmentation: rotate images by 20 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.15,  # Shearing
    zoom_range=0.15,  # Zoom in/out
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in gaps created by transformations
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # Directory with training images
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,  # Number of images to pass through the network at a time
    class_mode='categorical'  # Categorical labels (for multi-class classification)
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',  # Directory with validation images
    target_size=(224, 224),  # Resize validation images
    batch_size=32,
    class_mode='categorical'
)
