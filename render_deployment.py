# ============================================
# CAR PRICE PREDICTION MODEL TRAINING
# ============================================


import pandas as pd

import joblib


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_absolute_error, r2_score





# Load Dataset


df = pd.read_csv(
    "dataset/car_data.csv"
)



print(df.head())




# Drop unnecessary columns


df.drop(

"Car_Name",

axis=1,

inplace=True

)






# Encode categorical features


encoder = LabelEncoder()



categorical_columns=[

"Fuel_Type",

"Seller_Type",

"Transmission"

]



for col in categorical_columns:


    df[col]=encoder.fit_transform(

        df[col]

    )






# Split Data



X=df.drop(

"Selling_Price",

axis=1

)


y=df["Selling_Price"]






X_train,X_test,y_train,y_test=train_test_split(

X,

y,

test_size=0.2,

random_state=42

)







# Train Model


model=RandomForestRegressor(

n_estimators=200,

random_state=42

)



model.fit(

X_train,

y_train

)







# Evaluation



prediction=model.predict(

X_test

)



print(

"MAE:",

mean_absolute_error(

y_test,

prediction

)

)



print(

"R2 Score:",

r2_score(

y_test,

prediction

)

)








# Save Model



joblib.dump(

model,

"model/car_price_model.pkl"

)



print(

"Model Saved"

)
