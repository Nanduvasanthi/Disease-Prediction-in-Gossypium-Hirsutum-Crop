import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 16

VAL_DIR = "dataset/validation"
MODEL_PATH = "efficientnet_cotton_disease.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Validation generator (MATCH TRAINING PREPROCESSING)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predictions
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Metrics
print("\nClassification Report:\n")
print(classification_report(
    val_generator.classes,
    y_pred,
    target_names=list(val_generator.class_indices.keys())
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(val_generator.classes, y_pred))
