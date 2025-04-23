import os
import pandas as pd
import numpy as np  # Import numpy for NaN values

# Define the paths
data_folder_path = 'data/nickel6'
numbers_of_interest_path = input('Enter the file with the paper\'s lines: ')  # Path to the CSV file with numbers of interest
output_file_path = numbers_of_interest_path

# Read the CSV file with numbers of interest and intensity
numbers_of_interest_df = pd.read_csv(numbers_of_interest_path)

# Initialize dictionaries to store the closest number, normalized value, count, intensity, and NIST for each number of interest
closest_numbers_dict = {num: None for num in numbers_of_interest_df['λobs']}
normalized_values_dict = {num: None for num in numbers_of_interest_df['λobs']}
count_dict = {num: None for num in numbers_of_interest_df['λobs']}
intensity_dict = {num: None for num in numbers_of_interest_df['λobs']}
nist_line_dict = {num: None for num in numbers_of_interest_df['λobs']}
filename_dict = {num: None for num in numbers_of_interest_df['λobs']}  # Dictionary to store the filename

# Define the distance threshold
distance_threshold = 0.05

# Iterate through the data folder and process each CSV file
for root, dirs, files in os.walk(data_folder_path):
    for file in files:
        if file.startswith('1master') and file.endswith('.csv'):
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Initialize dictionaries to store the closest number, normalized value, count, intensity, and NIST line for each number of interest in the current file
            file_closest_numbers_dict = {num: None for num in numbers_of_interest_df['λobs']}
            file_normalized_values_dict = {num: None for num in numbers_of_interest_df['λobs']}
            file_count_dict = {num: None for num in numbers_of_interest_df['λobs']}
            file_intensity_dict = {num: None for num in numbers_of_interest_df['λobs']}
            file_nist_line_dict = {num: None for num in numbers_of_interest_df['λobs']}
            file_filename_dict = {num: None for num in numbers_of_interest_df['λobs']}  # Dictionary to store the filename
            
            # Find the closest number for each number of interest in the current file and extract the normalized value, count, intensity, and NIST line
            for index, row in numbers_of_interest_df.iterrows():
                num_of_interest = row['λobs']
                closest_number_row = df.iloc[(df['Average Center'] - num_of_interest).abs().argsort()[0]]
                closest_num = closest_number_row['Average Center']
                norm_value = closest_number_row['Norm']  # Assuming the normalized value is in the 4th column
                count_value = closest_number_row['Count']  # Assuming the count value is in the "Count" column
                intensity_value = row['Intensity']  # Extract intensity value from numbers_of_interest_df
                nist_line_value = closest_number_row['NIST line'] if not pd.isnull(closest_number_row['NIST line']) else 'none'  # Handle missing NIST line
                
                # Check if the closest number is within the distance threshold
                if isinstance(closest_num, float) and abs(closest_num - num_of_interest) <= distance_threshold:
                    # Update the closest number, normalized value, count, intensity, and NIST line if they are closer than the previous ones in the current file
                    if file_closest_numbers_dict[num_of_interest] is None or (isinstance(file_closest_numbers_dict[num_of_interest], float) and abs(closest_num - num_of_interest) < abs(file_closest_numbers_dict[num_of_interest] - num_of_interest)):
                        file_closest_numbers_dict[num_of_interest] = closest_num
                        file_normalized_values_dict[num_of_interest] = norm_value
                        file_count_dict[num_of_interest] = count_value
                        file_intensity_dict[num_of_interest] = intensity_value
                        file_nist_line_dict[num_of_interest] = nist_line_value
                        file_filename_dict[num_of_interest] = file[-8:-4]  # Extract the last 4 characters of the filename
                else:
                    # Set the closest number, normalized value, count, intensity, and NIST line to 'none' if outside the distance threshold or not a float
                    file_closest_numbers_dict[num_of_interest] = np.inf  # Assign a large value
                    file_normalized_values_dict[num_of_interest] = 'none'
                    file_count_dict[num_of_interest] = 'none'
                    file_intensity_dict[num_of_interest] = 'none'
                    file_nist_line_dict[num_of_interest] = 'none'
                    file_filename_dict[num_of_interest] = 'none'
            
            # Update the overall closest number, normalized value, count, intensity, and NIST line if they are closer than the previous ones across all files
            for num_of_interest in numbers_of_interest_df['λobs']:
                if closest_numbers_dict[num_of_interest] is None or (isinstance(file_closest_numbers_dict[num_of_interest], float) and abs(file_closest_numbers_dict[num_of_interest] - num_of_interest) < abs(closest_numbers_dict[num_of_interest] - num_of_interest)):
                    closest_numbers_dict[num_of_interest] = file_closest_numbers_dict[num_of_interest]
                    normalized_values_dict[num_of_interest] = file_normalized_values_dict[num_of_interest]
                    count_dict[num_of_interest] = file_count_dict[num_of_interest]
                    intensity_dict[num_of_interest] = file_intensity_dict[num_of_interest]
                    nist_line_dict[num_of_interest] = file_nist_line_dict[num_of_interest]
                    filename_dict[num_of_interest] = file_filename_dict[num_of_interest]

# Create lists from the dictionaries to construct the DataFrame
closest_numbers = [closest_numbers_dict[num] for num in numbers_of_interest_df['λobs']]
normalized_values = [normalized_values_dict[num] for num in numbers_of_interest_df['λobs']]
count_values = [count_dict[num] for num in numbers_of_interest_df['λobs']]
intensity_values = [intensity_dict[num] for num in numbers_of_interest_df['λobs']]
nist_line_values = [nist_line_dict[num] for num in numbers_of_interest_df['λobs']]
filename_values = [filename_dict[num] for num in numbers_of_interest_df['λobs']]

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'λobs': numbers_of_interest_df['λobs'],
    'Intensity': intensity_values,  # Append the 'Intensity' column
    'Closest_Number': closest_numbers,
    'Normalized_Value': normalized_values,
    'Count_Value': count_values,
    'NIST line': nist_line_values,  # Append the 'NIST line' column
    'Filename': filename_values  # Append the 'Filename' column
})

# Replace np.inf with 'none' in the DataFrame
results_df.replace({np.inf: 'none'}, inplace=True)

# Save the combined results to a single output file
results_df.to_csv(output_file_path, index=False)
