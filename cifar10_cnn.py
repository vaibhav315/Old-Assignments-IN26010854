# =====================================================
# CIFAR-10 IMAGE CLASSIFICATION USING CNN
# Complete Deep Learning Project
# =====================================================


# Import Libraries

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# =====================================================
# 1. Load CIFAR-10 Dataset
# =====================================================


print("Loading Dataset...")


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()



print("Training Images:", X_train.shape)

print("Testing Images:", X_test.shape)



# CIFAR-10 Classes

classes = [

'airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck'

]



# Convert labels

y_train = y_train.reshape(-1)

y_test = y_test.reshape(-1)




# =====================================================
# 2. Data Visualization
# =====================================================


plt.figure(figsize=(10,10))


for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(X_train[i])

    plt.title(classes[y_train[i]])

    plt.axis("off")



plt.show()



# =====================================================
# 3. Data Preprocessing
# =====================================================


# Normalize pixel values

X_train = X_train / 255.0

X_test = X_test / 255.0



print("Data Normalization Completed")





# =====================================================
# 4. CNN Model Architecture
# =====================================================



model = models.Sequential()



# First Convolution Layer

model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(32,32,3)
    )
)



model.add(
    layers.MaxPooling2D(
        (2,2)
    )
)




# Second Convolution Layer


model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)



model.add(
    layers.MaxPooling2D(
        (2,2)
    )
)




# Third Convolution Layer


model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)




# Flatten


model.add(
    layers.Flatten()
)




# Fully Connected Layers


model.add(
    layers.Dense(
        64,
        activation='relu'
    )
)



# Output Layer

model.add(
    layers.Dense(
        10,
        activation='softmax'
    )
)





# Model Summary


model.summary()





# =====================================================
# 5. Compile CNN Model
# =====================================================



model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)




# =====================================================
# 6. Train CNN Model
# =====================================================



print("\nTraining Started...")


history = model.fit(

    X_train,

    y_train,

    epochs=20,

    batch_size=64,

    validation_data=(X_test,y_test)

)



print("Training Completed")





# =====================================================
# 7. Training Accuracy Graph
# =====================================================



plt.figure(figsize=(8,5))


plt.plot(

history.history['accuracy'],

label="Training Accuracy"

)


plt.plot(

history.history['val_accuracy'],

label="Validation Accuracy"

)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.title("Training vs Validation Accuracy")


plt.legend()


plt.show()






# =====================================================
# 8. Loss Graph
# =====================================================



plt.figure(figsize=(8,5))


plt.plot(

history.history['loss'],

label="Training Loss"

)


plt.plot(

history.history['val_loss'],

label="Validation Loss"

)



plt.xlabel("Epoch")

plt.ylabel("Loss")


plt.title("Training vs Validation Loss")


plt.legend()


plt.show()






# =====================================================
# 9. Model Evaluation
# =====================================================



test_loss, test_accuracy = model.evaluate(

X_test,

y_test

)



print(
"\nTest Accuracy:",
test_accuracy*100,
"%"
)






# =====================================================
# 10. Predictions
# =====================================================



predictions = model.predict(

X_test

)



predicted_classes = np.argmax(

predictions,

axis=1

)




# =====================================================
# 11. Classification Report
# =====================================================



print(

classification_report(

y_test,

predicted_classes,

target_names=classes

)

)






# =====================================================
# 12. Confusion Matrix
# =====================================================



cm = confusion_matrix(

y_test,

predicted_classes

)




plt.figure(figsize=(10,8))


sns.heatmap(

cm,

annot=True,

fmt="d",

xticklabels=classes,

yticklabels=classes

)



plt.xlabel("Predicted")

plt.ylabel("Actual")


plt.title("Confusion Matrix")



plt.show()





# =====================================================
# 13. Test Single Image Prediction
# =====================================================



index = 10



image = X_test[index]



plt.imshow(image)

plt.title(
"Actual: " + classes[y_test[index]]
)

plt.show()




prediction = model.predict(

np.expand_dims(

image,

axis=0

)

)



result = np.argmax(

prediction

)



print(

"Predicted Class:",

classes[result]

)





# =====================================================
# 14. Save Model
# =====================================================



model.save(

"cifar10_cnn_model.h5"

)



print(

"\nModel Saved Successfully!"

)
