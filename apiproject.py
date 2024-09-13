from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dataset
data = pd.read_csv('Data set/DataSet.csv')

# Initialize encoders dictionary
encoder_dict = {}
categorical_columns = ['divisions', 'States']

# Label encode categorical columns
for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoder_dict[col] = encoder

# Separate features and target
X = data.drop('label', axis=1)
y = data['label']

# Train/test split (using 20% of the data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return app.send_static_file('project.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    # Get user input from the request
    user_input = request.get_json()

    # Encode categorical columns in user input using label encoding
    for col in categorical_columns:
        try:
            user_input[col] = encoder_dict[col].transform([user_input[col]])[0]
        except KeyError:
            return jsonify({"error": f"Unseen label '{user_input[col]}' encountered for column '{col}'. Please provide a valid label."})

    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Make predictions using the pre-trained model
    predicted_label = model.predict(user_df)[0]

    return jsonify({"predicted_crop_label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
