import pandas as pd

# Sample data
data = {
    "student_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "socioeconomic_status": ["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium", "High", "Low"],
    "test_score": [65, 80, 45, 90, 55, 75, 30, 85, 60, 95],
    "attendance": [0.85, 0.95, 0.70, 0.90, 0.65, 0.80, 0.50, 0.92, 0.75, 0.98],
    "homework_completion": [0.90, 0.80, 0.60, 0.95, 0.70, 0.85, 0.40, 0.88, 0.65, 0.97],
    "dropout": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("student_data.csv", index=False)