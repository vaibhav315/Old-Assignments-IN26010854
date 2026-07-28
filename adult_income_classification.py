# ==============================================
# Adult Census Income Classification
# Complete End-to-End Machine Learning Project
# ==============================================


# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib



# ==============================================
# 1. Load Dataset
# ==============================================


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]


# Download adult.data from UCI repository
# Rename it as adult.csv
# Keep it in the same folder as this python file


df = pd.read_csv(
    "adult.csv",
    names=columns,
    na_values=" ?"
)



print("\nDataset Loaded Successfully")
print(df.head())



print("\nDataset Shape:")
print(df.shape)



# ==============================================
# 2. Exploratory Data Analysis
# ==============================================


print("\nMissing Values:")
print(df.isnull().sum())



# Remove missing values

df.dropna(inplace=True)



# Income Distribution


plt.figure(figsize=(6,4))

sns.countplot(
    x="income",
    data=df
)

plt.title("Income Distribution")

plt.show()



# Age Distribution


plt.figure(figsize=(8,5))

sns.histplot(
    df["age"],
    kde=True
)

plt.title("Age Distribution")

plt.show()



# Correlation Heatmap


plt.figure(figsize=(10,8))


sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)


plt.title("Feature Correlation")

plt.show()



# ==============================================
# 3. Data Preprocessing
# ==============================================



# Convert target variable


df["income"] = df["income"].apply(
    lambda x: 1 if x == ">50K" else 0
)



# Separate Features and Target


X = df.drop(
    "income",
    axis=1
)


y = df["income"]




# Encode categorical columns


categorical_columns = X.select_dtypes(
    include="object"
).columns



encoder = LabelEncoder()



for column in categorical_columns:

    X[column] = encoder.fit_transform(
        X[column]
    )



print("\nData preprocessing completed")



# ==============================================
# 4. Train Test Split
# ==============================================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)



print("\nTraining Data:")
print(X_train.shape)


print("\nTesting Data:")
print(X_test.shape)




# ==============================================
# 5. Model Training
# ==============================================



models = {


"Logistic Regression":

LogisticRegression(
    max_iter=1000
),



"Decision Tree":

DecisionTreeClassifier(),



"Random Forest":

RandomForestClassifier(
    n_estimators=100
),



"XGBoost":

XGBClassifier()

}



results = {}

best_model = None

best_accuracy = 0




for name, model in models.items():


    print("\n==============================")

    print("Training:", name)

    print("==============================")


    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )



    accuracy = accuracy_score(
        y_test,
        predictions
    )



    results[name] = accuracy



    print(
        "Accuracy:",
        accuracy
    )


    print(
        classification_report(
            y_test,
            predictions
        )
    )



    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model





# ==============================================
# 6. Model Comparison
# ==============================================


print("\n\nMODEL PERFORMANCE")

for model,score in results.items():

    print(
        model,
        ":",
        round(score*100,2),
        "%"
    )




# ==============================================
# 7. Confusion Matrix
# ==============================================



final_prediction = best_model.predict(
    X_test
)



cm = confusion_matrix(
    y_test,
    final_prediction
)



plt.figure(figsize=(6,5))


sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")


plt.show()




# ==============================================
# 8. Save Best Model
# ==============================================



joblib.dump(
    best_model,
    "adult_income_model.pkl"
)



print("\nBest Model Saved Successfully")

print(
    "Best Accuracy:",
    round(best_accuracy*100,2),
    "%"
)





# ==============================================
# 9. Test Prediction
# ==============================================



loaded_model = joblib.load(
    "adult_income_model.pkl"
)



sample = X_test.iloc[0:1]



prediction = loaded_model.predict(
    sample
)



print("\nSample Prediction:")


if prediction[0] == 1:

    print(
        "Income is greater than 50K"
    )


else:

    print(
        "Income is less than or equal to 50K"
    )



print("\nProject Completed Successfully!")
