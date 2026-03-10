from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
import shap


class PredictionInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: int   # 0 = none, 1 = completed
    reading_score: int
    writing_score: int


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessor = None
explainer = None
feature_names = None


@app.get("/")
def read_root():
    return {"message": "Student Performance Prediction API"}


@app.on_event("startup")
def load_model():
    global model, preprocessor, explainer, feature_names

    df = pd.read_csv("data/stud.csv")

    # Convert test_preparation_course to binary
    df["test_preparation_course"] = df["test_preparation_course"].map({
        "none": 0,
        "completed": 1
    })

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

    X_transformed = preprocessor.fit_transform(X)

    model = LinearRegression()
    model.fit(X_transformed, y)

    explainer = shap.LinearExplainer(model, X_transformed)

    feature_names = preprocessor.get_feature_names_out()

    print("Model loaded successfully")

@app.post("/predict")
def predict(data: PredictionInput):

    input_data = data.model_dump()

    input_df = pd.DataFrame([input_data])

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)

    score = float(prediction[0])
    score = max(0, min(100, score))
    # Pass / Fail
    status = "Pass" if score >= 40 else "Fail"

    # Performance Level
    if score < 40:
        performance = "Very Weak"
    elif score < 50:
        performance = "Poor"
    elif score < 60:
        performance = "Average"
    elif score < 75:
        performance = "Good"
    else:
        performance = "Excellent"

    # SHAP Explanation
    shap_values = explainer(input_processed)

    contributions = {}

    for name, val in zip(feature_names, shap_values.values[0]):
        clean_name = name.split("__")[-1]
        contributions[clean_name] = float(val)

    # Important features
    filtered_contributions = {}

    other_contribution = 0

    for key, value in contributions.items():

        if "reading_score" in key:
            filtered_contributions["reading_score"] = value

        elif "writing_score" in key:
            filtered_contributions["writing_score"] = value

        elif "test_preparation_course" in key:
            filtered_contributions["test_preparation_course"] = value

        else:
            other_contribution += value

    # add remaining features
    filtered_contributions["other_factors"] = other_contribution

    return {
        "predicted_math_score": round(score, 2),
        "status": status,
        "performance_level": performance,
        "feature_contributions": filtered_contributions
    }

