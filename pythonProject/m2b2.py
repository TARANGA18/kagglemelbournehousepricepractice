import pickle
import pandas as pd

# Step 1: Load the model file
model_file = 'melbournehouseprice_randomforest.pkl'

# Load the model from the file
with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

print(f"Model '{model_file}' successfully loaded.")

# Step 2: Take inputs from the user
apartment_size = float(input("Enter the apartment size (in sq ft): "))
number_of_rooms = int(input("Enter the number of rooms: "))
distance = float(input("Enter the distance from the city center (in km): "))  # Added
bedroom2 = int(input("Enter the number of bedrooms (excluding the main bedroom): "))  # Added
bathroom = int(input("Enter the number of bathrooms: "))  # Added
car = int(input("Enter the number of car spaces: "))  # Added
landsize = float(input("Enter the land size (in sq meters): "))  # Added

# Step 3: Prepare the input as a DataFrame with the correct columns
user_input = pd.DataFrame([[number_of_rooms, distance, bedroom2, bathroom, car, landsize]],
                          columns=['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize'])

# Step 4: Predict house price using the loaded model
predicted_price = loaded_model.predict(user_input)

# Display the predicted house price
print(f"Predicted House Price: {predicted_price[0]:.2f}")
