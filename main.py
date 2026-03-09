from fastapi import FastAPI
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel

class PredictionInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


app = FastAPI()

model = None
preprocessor = None


@app.get("/")
def read_root():
    return {"message": "Student Performance Prediction API"}


@app.on_event("startup")
def load_model():
    global model, preprocessor

    df = pd.read_csv("data/stud.csv")

    X = df.drop(columns=["math_score"])
    y = df["math_score"]

    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )

    X = preprocessor.fit_transform(X)

    model = LinearRegression()
    model.fit(X, y)

    print("Model loaded successfully")


@app.post("/predict")
def predict(data: PredictionInput):

    input_df = pd.DataFrame([data.model_dump()])

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)

    return {
        "predicted_math_score": float(prediction[0])
    }