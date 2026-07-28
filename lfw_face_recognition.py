# ==========================================================
# FACE RECOGNITION USING CNN
# LFW (Labeled Faces in the Wild) Dataset
# ==========================================================


# Import Libraries


import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.datasets import fetch_lfw_people

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import tensorflow as tf

from tensorflow.keras import layers, models



# ==========================================================
# 1. Load LFW Dataset
# ==========================================================


print("Loading LFW Dataset...")


lfw = fetch_lfw_people(

    min_faces_per_person=50,

    resize=0.5,

    color=True

)



X = lfw.images

y = lfw.target



target_names = lfw.target_names



print("\nDataset Loaded")

print("Images:", X.shape)

print("Classes:", len(target_names))




# ==========================================================
# 2. Display Sample Faces
# ==========================================================



plt.figure(figsize=(10,8))


for i in range(20):

    plt.subplot(4,5,i+1)

    plt.imshow(X[i])

    plt.title(target_names[y[i]])

    plt.axis("off")


plt.show()





# ==========================================================
# 3. Image Preprocessing
# ==========================================================


# Normalize images

X = X / 255.0



# Convert labels


encoder = LabelEncoder()


y = encoder.fit_transform(y)



num_classes = len(np.unique(y))


print(
"\nNumber of classes:",
num_classes
)




# ==========================================================
# 4. Train Test Split
# ==========================================================


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



print("\nTraining Images:",X_train.shape)

print("Testing Images:",X_test.shape)





# ==========================================================
# 5. CNN Architecture
# ==========================================================


model = models.Sequential()



# Convolution Layer 1

model.add(

layers.Conv2D(

    32,

    (3,3),

    activation="relu",

    input_shape=X_train.shape[1:]

)

)



model.add(

layers.MaxPooling2D()

)




# Convolution Layer 2


model.add(

layers.Conv2D(

    64,

    (3,3),

    activation="relu"

)

)


model.add(

layers.MaxPooling2D()

)




# Convolution Layer 3


model.add(

layers.Conv2D(

    128,

    (3,3),

    activation="relu"

)

)



model.add(

layers.MaxPooling2D()

)




# Flatten


model.add(

layers.Flatten()

)




# Dense Layer


model.add(

layers.Dense(

    256,

    activation="relu"

)

)



model.add(

layers.Dropout(0.5)

)




# Output Layer


model.add(

layers.Dense(

    num_classes,

    activation="softmax"

)

)




model.summary()





# ==========================================================
# 6. Compile Model
# ==========================================================



model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)






# ==========================================================
# 7. Train Model
# ==========================================================


print("\nTraining Started...")


history = model.fit(

    X_train,

    y_train,

    epochs=30,

    batch_size=32,

    validation_data=(X_test,y_test)

)



print("Training Completed")






# ==========================================================
# 8. Accuracy Graph
# ==========================================================



plt.figure(figsize=(8,5))


plt.plot(

history.history["accuracy"],

label="Training Accuracy"

)


plt.plot(

history.history["val_accuracy"],

label="Validation Accuracy"

)



plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.title("CNN Training Accuracy")


plt.legend()


plt.show()





# ==========================================================
# 9. Model Evaluation
# ==========================================================



loss,accuracy=model.evaluate(

X_test,

y_test

)



print(

"\nTest Accuracy:",

accuracy*100,

"%"

)






# ==========================================================
# 10. Prediction
# ==========================================================



prediction=model.predict(

X_test

)



predicted_classes=np.argmax(

prediction,

axis=1

)





# ==========================================================
# 11. Classification Report
# ==========================================================


print(

classification_report(

y_test,

predicted_classes,

target_names=target_names

)

)







# ==========================================================
# 12. Confusion Matrix
# ==========================================================



cm=confusion_matrix(

y_test,

predicted_classes

)




plt.figure(figsize=(12,10))


sns.heatmap(

cm,

annot=True,

fmt="d"

)



plt.title(

"Face Recognition Confusion Matrix"

)


plt.xlabel(

"Predicted"

)


plt.ylabel(

"Actual"

)


plt.show()







# ==========================================================
# 13. Test Unknown Face
# ==========================================================



index=5



image=X_test[index]



plt.imshow(image)


plt.title(

"Actual: "

+

target_names[y_test[index]]

)


plt.axis("off")


plt.show()




image_prediction=model.predict(

np.expand_dims(

image,

axis=0

)

)



result=np.argmax(

image_prediction

)



print(

"Predicted Person:",

target_names[result]

)







# ==========================================================
# 14. Save Model
# ==========================================================


model.save(

"face_recognition_model.h5"

)



print(

"\nModel Saved Successfully!"

)
