# ==========================================================
# BRAIN CANCER DETECTION USING MRI IMAGES
# CNN Deep Learning Project
# ==========================================================


# Import Libraries


import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.preprocessing.image import ImageDataGenerator


import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# ==========================================================
# 1. Dataset Path
# ==========================================================


DATASET_PATH = "Brain_MRI_Dataset"



# Image Parameters


IMG_SIZE = 224

BATCH_SIZE = 32





# ==========================================================
# 2. Data Preprocessing
# ==========================================================



datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,


    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True

)




# Training Data


train_data = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training"

)





# Validation Data


validation_data = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False

)




print("\nClasses:")

print(train_data.class_indices)






# ==========================================================
# 3. Visualize MRI Images
# ==========================================================



images,labels = next(train_data)



plt.figure(figsize=(10,10))


for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    if labels[i]==1:

        plt.title("Tumor")

    else:

        plt.title("No Tumor")


    plt.axis("off")



plt.show()






# ==========================================================
# 4. CNN Model Architecture
# ==========================================================



model=models.Sequential()



# Conv Block 1


model.add(

layers.Conv2D(

32,

(3,3),

activation="relu",

input_shape=(IMG_SIZE,IMG_SIZE,3)

)

)


model.add(

layers.MaxPooling2D()

)





# Conv Block 2


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






# Conv Block 3


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






# Dense Layers


model.add(

layers.Dense(

256,

activation="relu"

)

)


model.add(

layers.Dropout(

0.5

)

)





# Output Layer


model.add(

layers.Dense(

1,

activation="sigmoid"

)

)




model.summary()







# ==========================================================
# 5. Compile Model
# ==========================================================



model.compile(

optimizer="adam",

loss="binary_crossentropy",

metrics=["accuracy"]

)







# ==========================================================
# 6. Train Model
# ==========================================================



print("\nTraining Started...")


history=model.fit(

train_data,

validation_data=validation_data,

epochs=25

)



print("\nTraining Completed")






# ==========================================================
# 7. Accuracy Graph
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



plt.title(

"Model Accuracy"

)


plt.xlabel(

"Epoch"

)


plt.ylabel(

"Accuracy"

)


plt.legend()


plt.show()





# ==========================================================
# 8. Loss Graph
# ==========================================================



plt.figure(figsize=(8,5))


plt.plot(

history.history["loss"],

label="Training Loss"

)


plt.plot(

history.history["val_loss"],

label="Validation Loss"

)



plt.title(

"Model Loss"

)


plt.xlabel(

"Epoch"

)


plt.ylabel(

"Loss"

)


plt.legend()


plt.show()







# ==========================================================
# 9. Model Evaluation
# ==========================================================



loss,accuracy=model.evaluate(

validation_data

)



print(

"\nValidation Accuracy:",

accuracy*100,

"%"

)








# ==========================================================
# 10. Predictions
# ==========================================================



validation_data.reset()



predictions=model.predict(

validation_data

)



predicted_classes=(predictions>0.5).astype(int)



true_classes=validation_data.classes







# ==========================================================
# 11. Classification Report
# ==========================================================



print(

classification_report(

true_classes,

predicted_classes

)

)







# ==========================================================
# 12. Confusion Matrix
# ==========================================================



cm=confusion_matrix(

true_classes,

predicted_classes

)



plt.figure(figsize=(6,5))


sns.heatmap(

cm,

annot=True,

fmt="d",

xticklabels=["No Tumor","Tumor"],

yticklabels=["No Tumor","Tumor"]

)



plt.xlabel(

"Predicted"

)


plt.ylabel(

"Actual"

)


plt.title(

"Confusion Matrix"

)


plt.show()







# ==========================================================
# 13. Predict New MRI Image
# ==========================================================


from tensorflow.keras.preprocessing import image



def predict_mri(path):


    img=image.load_img(

        path,

        target_size=(IMG_SIZE,IMG_SIZE)

    )


    img_array=image.img_to_array(img)


    img_array=img_array/255.0



    img_array=np.expand_dims(

        img_array,

        axis=0

    )



    prediction=model.predict(

        img_array

    )



    if prediction[0][0] > 0.5:


        print(

        "Result: Tumor Detected"

        )


    else:


        print(

        "Result: No Tumor"

        )






# Example:

# predict_mri("sample_mri.jpg")







# ==========================================================
# 14. Save Model
# ==========================================================



model.save(

"brain_tumor_model.h5"

)



print(

"\nModel Saved Successfully!"

)
