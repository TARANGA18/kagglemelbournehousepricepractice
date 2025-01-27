import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Step 1: Upload the model file
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
file_path = r"C:\Users\Acer\Documents\pythonProject\Melbourne_housing_FULL.csv"
data = pd.read_csv(file_path)

# Step 2: Data cleaning
# Inspect missing values in the dataset
print("Missing values per column before cleaning:")
print(data.isnull().sum())

# Drop rows with missing values in critical columns only
data_cleaned = data.dropna(subset=['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize'])

# Verify columns after cleaning
print("\nColumns after cleaning:")
print(data_cleaned.columns)

# Step 3: Feature selection
X = data_cleaned[['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize']]
y = data_cleaned['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on the testing data
y_pred = model.predict(X_test)

# Step 6: Measure accuracy
# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R² Score): {r2:.2f}")

# Step 7: Save the model to a file
output_file = 'melbournehouseprice.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {output_file}")
