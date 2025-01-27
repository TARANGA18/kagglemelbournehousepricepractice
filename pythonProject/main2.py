# Step 1: Load the model file
import pickle

# Specify the file name of the saved model
model_file = 'melbournehouseprice.pkl'

# Load the model from the file
with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

print(f"Model '{model_file}' successfully loaded.")

# Step 2: Take inputs from the user and prepare DataFrame
import pandas as pd

# User inputs for prediction
apartment_size = float(input("Enter the apartment size (in sq ft): "))
number_of_rooms = int(input("Enter the number of rooms: "))

# Prepare the input as a DataFrame with the same column names as the training data
user_input = pd.DataFrame([[apartment_size, number_of_rooms]], columns=['Apartment Size (sq ft)', 'Number of Rooms'])

# Step 3: Predict house rent using the loaded model
predicted_rent = loaded_model.predict(user_input)

# Display the predicted house rent
print(f"Predicted House Rent (Taka): {predicted_rent[0]:.2f}")
