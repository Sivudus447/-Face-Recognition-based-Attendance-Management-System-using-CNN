import os
import numpy as np
import cv2
import mysql.connector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import pickle

def connect_to_database():
    # Establish database connection
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Enter your database password here
        database="batch1"
    )
    return conn

def load_images_from_database():
    conn = connect_to_database()
    cursor = conn.cursor()

    cursor.execute("SELECT image_path, roll_number FROM registrations")
    rows = cursor.fetchall()

    images = []
    labels = []

    unique_labels = {}  # To store unique labels and their corresponding indices
    label_index = 0

    for row in rows:
        image_folder, label = row
        # Normalize image_folder path to handle slashes
        image_folder_path = os.path.normpath(image_folder)
        for i in range(300):
            # Use os.path.join for generating image_path
            image_path = os.path.join(image_folder_path, f"{i}.jpg")
            if os.path.exists(image_path):  # Check if the image file exists
                image = cv2.imread(image_path)
                if image is not None:  # Check if the image was successfully read
                    image = cv2.resize(image, (64, 64))  # Resize images to a consistent size
                    images.append(image)
                    if label not in unique_labels:
                        unique_labels[label] = label_index
                        label_index += 1
                    labels.append(unique_labels[label])

    cursor.close()
    conn.close()

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels, unique_labels

def create_cnn_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Softmax for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = create_cnn_model(num_classes=np.max(labels) + 1)  # Number of classes is determined by the maximum label
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return model

def save_model(model):
    model.save('trained_model.h5')
    print("Model saved successfully!")

def save_label_encoding(unique_labels, filename='label_encoding.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(unique_labels, f)
    print("Label encoding saved successfully!")

# Load images and labels from database
images, labels, unique_labels = load_images_from_database()

# Train CNN
model = train_cnn(images, labels)

# Save model upon achieving 100% accuracy
if model.history.history['accuracy'][-1] == 1.0:
    save_model(model)

# Save label encoding mapping
save_label_encoding(unique_labels)
