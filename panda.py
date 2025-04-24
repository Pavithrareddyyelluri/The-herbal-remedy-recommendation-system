import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("herbal_remedies2.csv")

# Check the columns in the dataset
print("Columns in the dataset:", data.columns)

# Specify the columns to encode
required_columns = ["Condition", "Age_Group", "Dietary_Preferences"]

# Check for missing columns
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    # Encode categorical variables
    data_encoded = pd.get_dummies(data, columns=required_columns)

    # Define feature columns and target
    X = data_encoded.drop(["Remedy_Name", "Ingredients", "Recipe", "Dosage", "Cautions", "Contraindications"], axis=1)
    y = data_encoded["Remedy_Name"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Function to predict remedy based on user input
    def predict_remedy(condition, age_group, dietary_preferences):
        # Prepare input data in the same format as training data
        input_data = pd.DataFrame([[condition, age_group, dietary_preferences]], 
                                  columns=required_columns)
        input_data_encoded = pd.get_dummies(input_data)

        # Align with the model's features
        input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

        # Predict remedy
        predicted_remedy = model.predict(input_data_encoded)
        return predicted_remedy[0]

    # Example usage
    condition_input = "Fever"
    age_group_input = "Adults"
    # gender_input = "Any"
    dietary_preferences_input = "Vegan"

    predicted_remedy = predict_remedy(condition_input, age_group_input, dietary_preferences_input)
    print("Predicted Remedy for input condition:", predicted_remedy)
