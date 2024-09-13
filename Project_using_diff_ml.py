import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import time

# Read the dataset
data = pd.read_csv('DataSet.csv')

encoder_dict = {}

categorical_columns = ['divisions', 'States']

for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoder_dict[col] = encoder

# Take user input for feature values
user_input = {
    'temperature': float(input("Enter temperature: ")),
    'humidity': float(input("Enter humidity: ")),
    'ph': float(input("Enter pH: ")),
    'rainfall': float(input("Enter rainfall: ")),
    'divisions': input("Enter DIVISIONS (e.g., cereals): "),
    'States': input("Enter States (e.g., UttarPradesh): ")
}

# Encode categorical columns in user input using label encoding with error handling
for col in categorical_columns:
    try:
        user_input[col] = encoder_dict[col].transform([user_input[col]])[0]
    except KeyError:
        print(f"Error: Unseen label '{user_input[col]}' encountered for column '{col}'. Please provide a valid label.")
        exit()

# Convert user input to a DataFrame
user_df = pd.DataFrame([user_input])

# Use the same encoder to transform the training data
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier()

# Train the Decision Tree model
start_time = time.time()
dt_model.fit(X_train, y_train)
end_time = time.time()

# Make predictions using Decision Tree
dt_predicted_label = dt_model.predict(user_df)[0]

# Evaluate Decision Tree model
dt_y_pred = dt_model.predict(X_test)
dt_f1 = f1_score(y_test, dt_y_pred, average='weighted')
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred, average='weighted')
dt_precision = precision_score(y_test, dt_y_pred, average='weighted')

print(f"Decision Tree - Predicted Crop Label: {dt_predicted_label}")
print(f"Decision Tree - F1 Score: {dt_f1}")
print(f"Decision Tree - Accuracy: {dt_accuracy}")
print(f"Decision Tree - Recall: {dt_recall}")
print(f"Decision Tree - Precision: {dt_precision}")
print(f"Decision Tree - Time taken: {end_time - start_time:.4f} seconds")

# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Train the Logistic Regression model
start_time = time.time()
lr_model.fit(X_train, y_train)
end_time = time.time()

# Make predictions using Logistic Regression
lr_predicted_label = lr_model.predict(user_df)[0]

# Evaluate Logistic Regression model
lr_y_pred = lr_model.predict(X_test)
lr_f1 = f1_score(y_test, lr_y_pred, average='weighted')
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred, average='weighted')
lr_precision = precision_score(y_test, lr_y_pred, average='weighted')

print(f"Logistic Regression - Predicted Crop Label: {lr_predicted_label}")
print(f"Logistic Regression - F1 Score: {lr_f1}")
print(f"Logistic Regression - Accuracy: {lr_accuracy}")
print(f"Logistic Regression - Recall: {lr_recall}")
print(f"Logistic Regression - Precision: {lr_precision}")
print(f"Logistic Regression - Time taken: {end_time - start_time:.4f} seconds")

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier()

# Train the Random Forest model
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()

# Make predictions using Random Forest
rf_predicted_label = rf_model.predict(user_df)[0]

# Evaluate Random Forest model
rf_y_pred = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, rf_y_pred, average='weighted')
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred, average='weighted')
rf_precision = precision_score(y_test, rf_y_pred, average='weighted')

print(f"Random Forest - Predicted Crop Label: {rf_predicted_label}")
print(f"Random Forest - F1 Score: {rf_f1}")
print(f"Random Forest - Accuracy: {rf_accuracy}")
print(f"Random Forest - Recall: {rf_recall}")
print(f"Random Forest - Precision: {rf_precision}")
print(f"Random Forest - Time taken: {end_time - start_time:.4f} seconds")

# Initialize the Support Vector Machine (SVM) model
svm_model = SVC()

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_input_scaled = scaler.transform(user_df)

# Train the SVM model
start_time = time.time()
svm_model.fit(X_train_scaled, y_train)
end_time = time.time()

# Make predictions using SVM
svm_predicted_label = svm_model.predict(user_input_scaled)[0]

# Evaluate SVM model
svm_y_pred = svm_model.predict(X_test_scaled)
svm_f1 = f1_score(y_test, svm_y_pred, average='weighted')
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_recall = recall_score(y_test, svm_y_pred, average='weighted')
svm_precision = precision_score(y_test, svm_y_pred, average='weighted')

print(f"SVM - Predicted Crop Label: {svm_predicted_label}")
print(f"SVM - F1 Score: {svm_f1}")
print(f"SVM - Accuracy: {svm_accuracy}")
print(f"SVM - Recall: {svm_recall}")
print(f"SVM - Precision: {svm_precision}")
print(f"SVM - Time taken: {end_time - start_time:.4f} seconds")
