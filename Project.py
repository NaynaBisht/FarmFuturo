import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('../Data set/DataSet.csv')


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
    'ph': float(input("Enter pH of the soil: ")),
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
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predicted_label = model.predict(user_df)[0]

print(f"Predicted Crop Label: {predicted_label}")
