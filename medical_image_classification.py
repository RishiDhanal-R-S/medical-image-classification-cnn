import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

# IMG_SIZE = 64 ====> reduce the size 

def load_data(folder):
    data = []
    labels = []

# ===>Brest cancer 
# categories = os.listdir(folder) 

# for category in categories:
#     path = os.path.join(folder, category)
#     label = categories.index(category)
    
    for category in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(folder, category)
        label = 0 if category == "NORMAL" else 1
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)
    
    return np.array(data), np.array(labels)

train_data, train_labels = load_data(r"C:\Users\Altra Research\Documents\Rishi\MyProjects\CNN\test")
test_data, test_labels = load_data(r"C:\Users\Altra Research\Documents\Rishi\MyProjects\CNN\train")

train_data = train_data / 255.0
test_data = test_data / 255.0

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_data = test_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))

predictions = (model.predict(test_data) > 0.5).astype("int32")

print(classification_report(test_labels, predictions))

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training Graph")
plt.show()
