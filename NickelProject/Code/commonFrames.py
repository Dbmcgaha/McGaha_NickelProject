import pandas as pd
import ast
import matplotlib.pyplot as plt

# Function to find overlapping entries
def find_overlapping_entries(dictionary, peak1, peak2):
    frames1 = set(dictionary[peak1]['Frames'])
    frames2 = set(dictionary[peak2]['Frames'])
    overlapping_frames = frames1 & frames2
    
    overlapping_intensities = {}
    for frame in overlapping_frames:
        index1 = dictionary[peak1]['Frames'].index(frame)
        index2 = dictionary[peak2]['Frames'].index(frame)
        intensity1 = dictionary[peak1]['Intensities'][index1]
        intensity2 = dictionary[peak2]['Intensities'][index2]
        overlapping_intensities[frame] = (intensity1, intensity2)
    
    return overlapping_frames, overlapping_intensities

file = str(input('Enter the last 4 digits of the file you would like to analyze: '))

# Load the CSV file into a DataFrame
df = pd.read_csv('data/nickel3/{}/2master{}.csv'.format(file, file))

# Set the maximum column width for display
pd.set_option('display.max_colwidth', None)  # Remove the maximum column width limit

# Display dataframe
print('\nPeaks list for {}:\n'.format(file))
print('Average Center', ' Frames')
print('_'*22)

# Concatenate two columns into a new DataFrame
concatenated_df = pd.concat([df['Average Center'].apply(lambda x: f'{x:.2f}'), df['Frames']], axis=1)

# Print the concatenated DataFrame with columns side by side and wrapping under itself
for index, row in concatenated_df.iterrows():
    avg_center = f"{row['Average Center']:<15}"  # Left-align the float value with padding
    frames_list = row['Frames']

    # Split the frames list into chunks for padding
    chunk_size = 100  # Number of elements per chunk
    chunks = [frames_list[i:i+chunk_size] for i in range(0, len(frames_list), chunk_size)]

    # Print each chunk with padding
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"{avg_center} {chunk}")
        else:
            print(f"{' ':<15} {chunk}")

print('\n', end='')

# Initialize an empty data dictionary
data = {}

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    peak = round(row['Average Center'], 2)
    frames = ast.literal_eval(row['Frames'])
    intensities = ast.literal_eval(row['Intensities'])
    
    # Check if the peak already exists in the data dictionary
    if peak in data:
        data[peak]['Frames'].extend(frames)
        data[peak]['Intensities'].extend(intensities)
    else:
        data[peak] = {'Frames': frames, 'Intensities': intensities}

# Prompt the user to enter two peaks from the table
peak1 = float(input("Enter the first peak from the table: "))
peak2 = float(input("Enter the second peak from the table: "))

# Check if the entered peaks are valid
if peak1 in data and peak2 in data:
    result_frames, result_intensities = find_overlapping_entries(data, peak1, peak2)
    
    # Check if there are common frames
    if result_frames:
        # Calculate the ratio of intensities
        ratios = [intensity[0] / intensity[1] for intensity in result_intensities.values()]
        
        print('Similar Frames: ', sorted(result_frames))

        # Create the plot
        plt.plot(sorted(result_frames), ratios, marker='o')
        plt.xlabel('Frames')
        plt.ylabel(f'Ratio ({peak1}/{peak2})')
        plt.title(f'Intensity Ratio ({peak1}/{peak2}) vs. Frames')
        plt.grid(True)
        plt.show()
    else:
        print("No common frames found.")
else:
    print("Invalid peak(s) entered.")
