import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# here Load the trained model
model = load_model('face_detection_model.h5')

# here Load and preprocess the validation data

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'C:\\xampp\\htdocs\\finalcode\\images',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# here Make predictions on the validation data
y_val = validation_generator.classes  # True labels for the validation data

# Predict probabilities for each class
predictions = model.predict(validation_generator)

# Get predicted labels (class with the highest probability)
predicted_labels = np.argmax(predictions, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_val, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
