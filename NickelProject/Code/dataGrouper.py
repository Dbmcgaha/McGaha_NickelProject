import numpy as np
import pandas as pd
import os
import sys
import json
from sklearn.cluster import DBSCAN

# Gets the average of our intensities and wavelength locations
def ave(nums):
    if len(nums) != 0:
        temp = 0
        for i in nums:
            temp += i
        temp = temp/len(nums)
        return temp
    else:
        return 0

# Checks how close a found line is to a NIST line
def checkProximity(x, NIST, NIST_names):
    for i in range(len(NIST)):
        for line in NIST[i]:
            a = 0.01
            if x + a >= float(line) and x - a <= float(line):
                return NIST_names[i]
        
    return 'None'

# Sorted function for dataframe
def sort_list(ls):
    return sorted(ls)

# Get NIST line paths
with open('NISTlines.json', 'r') as lines:
    lines_data = json.load(lines)

# Extract the file paths and descriptions from the JSON data
lines_paths = lines_data['lines']

NIST = [] # GOLD DATA, Ni I, Ni II, H, Ar, Hg, Ne, 'C'
NIST_names = ('GOLD', 'Ni I', 'Ni II', 'H', 'Ar', 'Hg', 'Ne', 'C')
for i in range(len(lines_paths)):
    file = pd.read_csv(lines_paths[i]['path'])
    NIST.append(file.values)

# Get the directory path of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

element = input('Pick either nickel3, nickel6, or gold for analysis: ')

# Append the 'data' directory to the script directory
data_dir = os.path.join(script_dir, 'data/{}'.format(element))

# Check if the directory exists
if os.path.exists(data_dir):
    print(f"The directory '{data_dir}' exists.")
else:
    print(f"The directory '{data_dir}' does not exist.")
    sys.exit()

# Display the directories
print("Directories in the data folder:")
for i in range(len(os.listdir(data_dir))):
    print('{} - {}'.format(i, os.listdir(data_dir)[i]))

# Directory containing the CSV files
csv_directory = ['data/{}/{}'.format(element, item) for item in os.listdir(data_dir)]
print('\n', end='')

usrInput = input('Enter a directory you would like to analyze or type ALL to run through all: ')

# Initialize an empty dictionary to store aggregated data
aggregated_data = {'Peak': [], 'Average Center': [], 'Difference': [], 'Centers': [], 'Intensities': [], 'Frames': [], 'Std Devs': []}

# Iterate over each CSV file in the directory
for directory in csv_directory:
    if (usrInput == 'ALL' or directory == csv_directory[int(usrInput)]) and not directory.endswith('.csv'):
        for filename in os.listdir(directory):
            if filename.endswith('.csv') and not filename.endswith(f'{directory[-4:]}.csv'):
                filepath = os.path.join(directory, filename)

                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)
                
                
                # Group the data by the peak value and aggregate centers, intensities, and std devs
                grouped = df.groupby('Peaks').agg({'centers': list, 'intensities': list, 'std devs': list, 'frames': list, 'noisy': list}).reset_index()

                # Update the aggregated data dictionary
                for _, row in grouped.iterrows():
                    if not row['noisy'][0]: # Removes data that goes past 65565 intensity
                        peak = row['Peaks']
                        frms = row['frames']
                        centers = row['centers']
                        intensities = row['intensities']
                        std_devs = row['std devs']
                        
                        if peak not in aggregated_data['Peak']:
                            aggregated_data['Peak'].append(peak)
                            aggregated_data['Centers'].append([])
                            aggregated_data['Intensities'].append([])
                            aggregated_data['Std Devs'].append([])
                            aggregated_data['Average Center'].append(0)
                            aggregated_data['Difference'].append(0)
                            aggregated_data['Frames'].append([])
                            
                        idx = aggregated_data['Peak'].index(peak)
                        aggregated_data['Centers'][idx].extend(centers)
                        aggregated_data['Intensities'][idx].extend(intensities)
                        aggregated_data['Std Devs'][idx].extend(std_devs)
                        aggregated_data['Average Center'][idx] = ave(aggregated_data['Centers'][idx])
                        aggregated_data['Difference'][idx] = abs(aggregated_data['Average Center'][idx] - aggregated_data['Peak'][idx])
                        aggregated_data['Frames'][idx].extend(frms)


# Convert the aggregated data dictionary to a DataFrame
master_df = pd.DataFrame(aggregated_data)
master_df = master_df.sort_values(by='Peak', ascending=True)

# Write the master DataFrame to a CSV file
if usrInput == 'ALL':
    master_df.to_csv('data/{}/0masterALL.csv'.format(element), index=False)
else:
    master_df.to_csv('data/{}/{}/0master{}.csv'.format(element, os.listdir(data_dir)[int(usrInput)], os.listdir(data_dir)[int(usrInput)]), index=False)

df = master_df

# Perform DBSCAN clustering on 'Average Center' column
X = np.array(df['Average Center']).reshape(-1, 1)
dbscan = DBSCAN(eps=0.05, min_samples=1)
df['Cluster'] = dbscan.fit_predict(X)

# Get unique non-negative cluster labels
non_neg_labels = np.arange(len(np.unique(df['Cluster'])))

# Replace -1 labels with unique non-negative labels
df.loc[df['Cluster'] == -1, 'Cluster'] = non_neg_labels.max() + 1

# Group data by clusters
grouped_data = df.groupby('Cluster').agg({
    'Centers': lambda x: sum(x, []),
    'Intensities': lambda x: sum(x, []),
    'Frames': lambda x: sum(x, [])
}).reset_index()

# Find average center of these clusters
# Find mins and maxes of each cluster
# Find the range between min and max
avgCenters = []
avgInt = []
mins = []
maxes = []
rnge = []
lines = []
app = []
for item in grouped_data['Centers']:
    m = max(item)
    n = min(item)
    average = ave(item)
    avgCenters.append(average)
    lines.append(checkProximity(average, NIST, NIST_names))
    app.append(len(item))
    mins.append(n)
    maxes.append(m)
    rnge.append(m - n)

for item in grouped_data['Intensities']:
    avgInt.append(ave(item))

norm = max(avgInt)
lsNorm = []
for i in avgInt:
    lsNorm.append((i/norm)*100)

grouped_data.insert(0, 'Max', maxes)
grouped_data.insert(0, 'Min', mins)
grouped_data.insert(2, 'Range', rnge)
grouped_data.insert(0, 'Average Center', avgCenters)
grouped_data.insert(1, 'Average Intensity', avgInt)
grouped_data.insert(2, 'Norm', lsNorm)
grouped_data.insert(0, 'NIST line', lines)
grouped_data.insert(1, 'Count', app)

# Write the master DataFrame to a CSV file
if usrInput == 'ALL':
    grouped_data.to_csv('data/{}/1masterALL.csv'.format(element), index=False)
else:
    grouped_data.to_csv('data/{}/{}/1master{}.csv'.format(element, os.listdir(data_dir)[int(usrInput)], os.listdir(data_dir)[int(usrInput)]), index=False)

# Removes GOLD DATA lines and lines that appear only once
filtered_data = grouped_data[~((grouped_data['NIST line'] == 'GOLD') | (grouped_data['Count'] <= 2))]  

# Renormalize
filtered_data.drop('Norm', axis=1, inplace=True, errors='ignore')

avgInt = []
for item in filtered_data['Intensities']:
    avgInt.append(ave(item))

norm = max(avgInt)
lsNorm = []
for i in avgInt:
    lsNorm.append((i/norm)*100)

filtered_data.insert(4, 'Norm', lsNorm)

if usrInput == 'ALL':
    filtered_data.to_csv('data/{}/2masterALL.csv'.format(element), index=False)
else:
    filtered_data.to_csv('data/{}/{}/2master{}.csv'.format(element, os.listdir(data_dir)[int(usrInput)], os.listdir(data_dir)[int(usrInput)]), index=False)


