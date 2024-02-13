import pandas as pd
from datetime import datetime

# Path to your CSV file
file_path = '/users/PAS0536/amsh/DSSRNN/DSSRNN-classification/dataset/ds68-class.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Function to change year from 2021 to 2020
def change_year(row):
    # Parse the date string to a datetime object
    date = datetime.strptime(row, '%Y-%m-%d %H:%M:%S')
    # Check if the year is 2021 and change it to 2020
    if date.year == 2021:
        date = date.replace(year=2020)
    # Return the modified date as a string in the same format
    return date.strftime('%Y-%m-%d %H:%M:%S')

# Apply the function to modify the dates in the specified range of rows
df.loc[3255:6052, 'date'] = df.loc[3255:6052, 'date'].apply(change_year)

# Save the modified DataFrame back to the CSV file
df.to_csv(file_path, index=False)
