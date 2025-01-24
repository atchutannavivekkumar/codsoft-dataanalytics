import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = '/content/Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)

# Drop unnecessary columns
titanic_data_cleaned = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
imputer_age = SimpleImputer(strategy="median")
titanic_data_cleaned["Age"] = imputer_age.fit_transform(titanic_data_cleaned[["Age"]])

 # Encode categorical variables
label_encoder_sex = LabelEncoder()
titanic_data_cleaned["Sex"] = label_encoder_sex.fit_transform(titanic_data_cleaned["Sex"])

label_encoder_embarked = LabelEncoder()
titanic_data_cleaned["Embarked"] = label_encoder_embarked.fit_transform(titanic_data_cleaned["Embarked"])


# Separate features and target variable
X = titanic_data_cleaned.drop(columns=["Survived"])
y = titanic_data_cleaned["Survived"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
