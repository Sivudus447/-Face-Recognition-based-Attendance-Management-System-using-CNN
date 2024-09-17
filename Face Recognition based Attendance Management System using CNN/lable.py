from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model.h5')

# Print the model summary
print(model.summary())
