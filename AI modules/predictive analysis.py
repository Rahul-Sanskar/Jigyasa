import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify

# Load dataset
data = pd.read_csv("student_data.csv")

# Preprocessing
# Handle missing values for numeric columns only
numeric_cols = data.select_dtypes(include=["number"]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode categorical variables
data = pd.get_dummies(data, columns=["gender", "socioeconomic_status"], drop_first=True)

# Split features and target
X = data.drop(columns=["dropout", "student_id"])
y = data["dropout"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize only numerical columns
numerical_cols = ["test_score", "attendance", "homework_completion"]
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train a model for progress monitoring
progress_model = RandomForestClassifier(n_estimators=100, random_state=42)
progress_model.fit(X_train, y_train)

# Evaluate progress model
y_pred = progress_model.predict(X_test)
print("Progress Model Classification Report:")
print(classification_report(y_test, y_pred))

# Train a model for dropout prediction
dropout_model = LogisticRegression()
dropout_model.fit(X_train, y_train)

# Evaluate dropout model
y_pred_dropout = dropout_model.predict(X_test)
print("Dropout Model Classification Report:")
print(classification_report(y_test, y_pred_dropout))

# Function to generate progress report
def generate_progress_report(student_id, model, scaler, data):
    student_data = data[data["student_id"] == student_id].drop(columns=["dropout", "student_id"])
    student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
    prediction = model.predict(student_data)
    areas_of_improvement = X.columns[model.feature_importances_ < model.feature_importances_.mean()]  # For RandomForest
    return {
        "student_id": student_id,
        "predicted_performance": prediction.tolist(),
        "areas_of_improvement": areas_of_improvement.tolist()
    }

# Example usage
report = generate_progress_report(1, progress_model, scaler, data)
print("Progress Report for Student ID 1:")
print(report)

# Function to identify at-risk students
def identify_at_risk_students(model, scaler, data, threshold=0.5):
    student_data = data.drop(columns=["dropout", "student_id"])
    student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
    probabilities = model.predict_proba(student_data)[:, 1]
    at_risk_students = data[probabilities > threshold]["student_id"].tolist()
    return at_risk_students

# Example usage
at_risk_students = identify_at_risk_students(dropout_model, scaler, data)
print("Students at risk of dropping out:", at_risk_students)

# Save models and scaler
joblib.dump(progress_model, "progress_model.pkl")
joblib.dump(dropout_model, "dropout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Flask API
app = Flask(__name__)

# Load models
progress_model = joblib.load("progress_model.pkl")
dropout_model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict_progress", methods=["POST"])
def predict_progress():
    data = request.json
    student_data = pd.DataFrame([data["features"]])
    student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
    prediction = progress_model.predict(student_data)
    return jsonify({"prediction": prediction.tolist()})

@app.route("/identify_at_risk", methods=["GET"])
def identify_at_risk():
    at_risk_students = identify_at_risk_students(dropout_model, scaler, data)
    return jsonify({"at_risk_students": at_risk_students})

if __name__ == "__main__":
    app.run(debug=True)
    
    import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ...

# Load dataset
logger.info('Loading dataset')
data = pd.read_csv("student_data.csv")

# Preprocessing
# Handle missing values for numeric columns only
logger.info('Handling missing values for numeric columns')
numeric_cols = data.select_dtypes(include=["number"]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode categorical variables
logger.info('Encoding categorical variables')
data = pd.get_dummies(data, columns=["gender", "socioeconomic_status"], drop_first=True)

# Split features and target
logger.info('Splitting features and target')
X = data.drop(columns=["dropout", "student_id"])
y = data["dropout"]

# Split into training and testing sets
logger.info('Splitting into training and testing sets')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize only numerical columns
logger.info('Normalizing numerical columns')
numerical_cols = ["test_score", "attendance", "homework_completion"]
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train a model for progress monitoring
logger.info('Training progress model')
progress_model = RandomForestClassifier(n_estimators=100, random_state=42)
progress_model.fit(X_train, y_train)

# Evaluate progress model
logger.info('Evaluating progress model')
y_pred = progress_model.predict(X_test)
print("Progress Model Classification Report:")
print(classification_report(y_test, y_pred))

# Train a model for dropout prediction
logger.info('Training dropout model')
dropout_model = LogisticRegression()
dropout_model.fit(X_train, y_train)

# Evaluate dropout model
logger.info('Evaluating dropout model')
y_pred_dropout = dropout_model.predict(X_test)
print("Dropout Model Classification Report:")
print(classification_report(y_test, y_pred_dropout))

# Function to generate progress report
def generate_progress_report(student_id, model, scaler, data):
    try:
        logger.info('Generating progress report for student ID %s', student_id)
        student_data = data[data["student_id"] == student_id].drop(columns=["dropout", "student_id"])
        student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
        prediction = model.predict(student_data)
        areas_of_improvement = X.columns[model.feature_importances_ < model.feature_importances_.mean()]  # For RandomForest
        return {
            "student_id": student_id,
            "predicted_performance": prediction.tolist(),
            "areas_of_improvement": areas_of_improvement.tolist()
        }
    except Exception as e:
        logger.error('Error generating progress report: %s', str(e))
        return None

# Example usage
report = generate_progress_report(1, progress_model, scaler, data)
print("Progress Report for Student ID 1:")
print(report)

# Function to identify at-risk students
def identify_at_risk_students(model, scaler, data, threshold=0.5):
    try:
        logger.info('Identifying at-risk students')
        student_data = data.drop(columns=["dropout", "student_id"])
        student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
        probabilities = model.predict_proba(student_data)[:, 1]
        at_risk_students = data[probabilities > threshold]["student_id"].tolist()
        return at_risk_students
    except Exception as e:
        logger.error('Error identifying at-risk students: %s', str(e))
        return None

# Example usage
at_risk_students = identify_at_risk_students(dropout_model, scaler, data)
print("Students at risk of dropping out:", at_risk_students)

# Save models and scaler
logger.info('Saving models and scaler')
joblib.dump(progress_model, "progress_model.pkl")
joblib.dump(dropout_model, "dropout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Flask API
app = Flask(__name__)

# Load models
logger.info('Loading models')
progress_model = joblib.load("progress_model.pkl")
dropout_model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict_progress", methods=["POST"])
def predict_progress():
    try:
        logger.info('Predicting progress')
        data = request.json
        student_data = pd.DataFrame([data["features"]])
        student_data[numerical_cols] = scaler.transform(student_data[numerical_cols])
        prediction = progress_model.predict(student_data)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        logger.error('Error predicting progress: %s', str(e))
        return jsonify({"error": str(e)})

@app.route("/identify_at_risk", methods=["GET"])
def identify_at_risk():
    try:
        logger.info('Identifying at-risk students')
        at_risk_students = identify_at_risk_students(dropout_model, scaler, data)
        return jsonify({"at_risk_students": at_risk_students})
    except Exception as e:
        logger.error('Error identifying at-risk students: %s', str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)