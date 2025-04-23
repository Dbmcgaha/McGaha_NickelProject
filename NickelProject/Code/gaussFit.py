import os
import json
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import peakutils
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
import warnings
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings("ignore", message="overflow encountered in exp")

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

# Converts wavelength to corresponding index
def number_to_index(number, wavelengths):
    for i in range(len(wavelengths)):
        if number<wavelengths[i]+0.02 and number>wavelengths[i]-0.02:
            return i

# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# Creates list with params to feed into curve fitter
def getParams(peaksInd, wavelengths, intensity):
    
    initial_params = []
    
    for i in range(len(peaksInd)):
        initial_params.append(intensity[peaksInd[i]])
        initial_params.append(wavelengths[peaksInd[i]])
        initial_params.append(0.06)

    return initial_params

# Multi-gaussian defintion
def multiple_gaussians(x, *params):
    result = np.zeros_like(x)
    num_gaussians = len(params) // 3  # Each Gaussian has three parameters: amplitude, mean, standard deviation
    
    # Precompute squared standard deviations outside the loop
    std_devs_sq = np.array(params[2::3])**2
    
    # Vectorized computation
    for i in range(num_gaussians):
        start_idx = i * 3
        amplitude, mean = params[start_idx:start_idx+2]
        result += amplitude * np.exp(-(x - mean)**2 / (2 * std_devs_sq[i]))
    
    return result

# Noise reduction function
def baseline_arPLS(y, ratio=1e-6, lam=200, niter=20, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  

        count += 1

        if count > niter:
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

# Function to find peaks
def find_peaks_in_intensity(wavelengths, intensity, threshold):
        
    sigma = 0.5
    smoothedInt = gaussian_filter1d(intensity, sigma)
    peaks, _ = find_peaks(smoothedInt, height=threshold, distance=4, prominence=10)
    lb=0
    ub=len(peaks)

    # Remove edge peaks
    if len(peaks) == 0:
        return None
    
    for i in range(len(peaks)):
        if abs(wavelengths[peaks[i]] - wavelengths[0]) <= 0.1:
            lb += 1
        if abs(wavelengths[peaks[i]] - wavelengths[-1]) <= 0.1:
            ub -= 1
            
    if len(peaks[lb:ub]) == 0:
        return None

    return peaks[lb:ub]
    
def shrinkNIST(NIST, wavelengths):
    specNIST = []
    for value in NIST:
        if float(value) > wavelengths[0] and float(value) < wavelengths[-1]:
            specNIST.append(float(value))
            
    return specNIST

def remove_duplicates(data, decimal_places=3):
    rounded_data = {}
    for value in data:
        rounded_value = round(value, decimal_places)
        rounded_data[rounded_value] = value
    
    return list(rounded_data.values())

# Finds and returns A multigaussian to fit over a found set of peaks
def gaussianFitter(wavelengths, intensity, frame, usrInput, thresh, file_paths_with_descriptions, element):

    localTable = {}
    # noise = peakutils.baseline(intensity, deg=3)
    noise = baseline_arPLS(intensity)
    intensity -= noise

    # Sets negative values to 0
    for i in range(len(intensity)):
        if intensity[i] < 0:
            intensity[i] = 0

    # Peaks List via function or generated average list
    peaksList = find_peaks_in_intensity(wavelengths, intensity, thresh)
    #peaksList = 

    if peaksList is None:
        dt = datetime.now()
        print(f'No peaks.... Frame: {frame} Time: {dt.minute}.{dt.second}')

        # Plot for reference
        plt.plot(wavelengths, intensity, linewidth=0.2, label='intensity - noise')
        plt.plot(wavelengths, intensity + noise, linewidth=0.2, linestyle='dotted', alpha=1, label='intensity raw')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.1)
        plt.legend()
        plt.title('Frame: {}    Range: {}'.format(frame, file_paths_with_descriptions[int(usrInput)]['description']))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        # Check if the directory exists
        if not os.path.exists('images/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4])):
            # If it doesn't exist, create it
            os.makedirs('images/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4]))
            plt.savefig('images/{}/{}/{}.{}.{}.svg'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), dpi=400)
        else:
            plt.savefig('images/{}/{}/{}.{}.{}.svg'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), dpi=400)

        # Clear figure
        plt.clf()
        plt.close()

        return localTable  # Return an empty dictionary if no peaks are found

    initial_params = getParams(peaksList, wavelengths, intensity)

    try:
        # Fit the data to the multiple Gaussians model
        lowerBound = [item for i in range(len(peaksList)) for item in [intensity[peaksList[i]] - 20, wavelengths[peaksList[i]] - 0.05, 0.01]]
        upperBound = [item for i in range(len(peaksList)) for item in [intensity[peaksList[i]] + 20, wavelengths[peaksList[i]] + 0.05, 0.1]]
        bounds = (lowerBound, upperBound)

        dt = datetime.now()
        print(f'Fitting Gaussians.... Frame: {frame} Time: {dt.minute}.{dt.second} Peaks: {len(peaksList)}')    

        fit_params, _ = curve_fit(multiple_gaussians, wavelengths, intensity, p0=initial_params, bounds=bounds)
        
        dt = datetime.now()
        print(f'Completed.... Frame: {frame} Time: {dt.minute}.{dt.second}')

        # Create figure and axes
        fig, ax1 = plt.subplots()

        # Plot the line graph
        ax1.plot(wavelengths, intensity, linewidth=0.2, label='Intensity - noise')
        ax1.plot(wavelengths, intensity + noise, linewidth=0.2, linestyle='dotted', alpha=1, label='Intensity Raw')
        ax1.plot(wavelengths, multiple_gaussians(wavelengths, *fit_params), linewidth=0.2, label='Gaussian Fit')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.2)
        ax1.legend()
        ax1.set_title('Frame: {}    Range: {}'.format(frame, file_paths_with_descriptions[int(usrInput)]['description']))
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity')

        # Plot the mirrored bar graph
        bar_width = 0.1  # Adjust as needed
        ax1.bar(np.array(wavelengths)[peaksList], -1*np.array(intensity)[peaksList], width=bar_width, color='black', alpha=1, label='Peaks')

        # Set limits for y-axis to share the same scale
        max_y = max(max(intensity), max(intensity + noise), max(np.array(intensity)[peaksList]))
        min_y = -1*max(max(intensity), max(intensity + noise), max(np.array(intensity)[peaksList]))
        ax1.set_ylim(min_y, max_y)

        # Optionally, add legend outside the entire plot
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Check if the directory exists
        if not os.path.exists('images/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4])):
            # If it doesn't exist, create it
            os.makedirs('images/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4]))
            plt.savefig('images/{}/{}/{}.{}.{}.svg'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), dpi=400)
        else:
            plt.savefig('images/{}/{}/{}.{}.{}.svg'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), dpi=400)

        flag = 0

        # Iterate over each newly found gaussians
        for j in range(len(peaksList)):
            if wavelengths[peaksList[j]] not in localTable:
                localTable[wavelengths[peaksList[j]]] = {'centers': 0, 'intensities': 0, 'frames': 0, 'std devs': 0, 'noisy': False}

            localTable[wavelengths[peaksList[j]]]['intensities'] = fit_params[3*j]
            localTable[wavelengths[peaksList[j]]]['centers'] = fit_params[1 + 3*j]
            localTable[wavelengths[peaksList[j]]]['frames'] = frame
            localTable[wavelengths[peaksList[j]]]['std devs'] = fit_params[2 + 3*j]
            average = ave(intensity)
            intOG = intensity + noise
            if average < 2 or (intOG[peaksList[j]] == 65535):
                localTable[wavelengths[peaksList[j]]]['noisy'] = True
                flag += 1

        if flag >= 30 or frame < 50:
            for item in localTable:
                localTable[item]['noisy'] = True


        dfTable = pd.DataFrame(localTable)
        dfTable = dfTable.T
        dfTable.insert(0, 'Peaks', wavelengths[peaksList])
        # Check if the directory exists
        if not os.path.exists('data/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4])):
            # If it doesn't exist, create it
            os.makedirs('data/{}/{}'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4]))
            dfTable.to_csv('data/{}/{}/{}.{}.{}.csv'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), index=False)
        else:
            dfTable.to_csv('data/{}/{}/{}.{}.{}.csv'.format(element, file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], file_paths_with_descriptions[int(usrInput)]['path'][-8:-4], thresh, frame), index=False)

        # Clear figure
        plt.clf()
        plt.close()

        return localTable

    except RuntimeError:
        print('Failed to fit gaussians')

# Define a function to process each frame
def process_frame(args):
    wavelengths, intensity, frame, usrInput, thresh, file_paths_with_descriptions, element = args
    localTable = gaussianFitter(wavelengths, intensity, frame, usrInput, thresh, file_paths_with_descriptions, element)
    return localTable


if __name__ == '__main__':

    # Read the paths JSON file
    with open('paths.json', 'r') as json_file:
        file_paths_data = json.load(json_file)

    with open('NISTlines.json', 'r') as lines:
        lines_data = json.load(lines)

    # Extract the file paths and descriptions from the JSON data
    element = input('Pick either nickel3, nickel6, or gold: ')
    file_paths_with_descriptions = file_paths_data[element]

    # Make intro table
    print('_'*8 + '_'*25)
    print(' '*7 + '|   File   |' + '     Range  ' + '  |')
    print(' '*7 + '|'*27)
    for i in range(len(file_paths_with_descriptions)):
        file_path = file_paths_with_descriptions[i]['path']
        description = file_paths_with_descriptions[i]['description']
        print(f"{i:2}  -  | {file_path[-12:-4]} | {description} |")
    print('_'*7 + '/'*27)
    print('\n', end='')

    # Import Data
    usrInput = input(f'Choose a file by entering a number between 0 and {len(file_paths_with_descriptions) - 1}: ')
    selected_file_name = file_paths_with_descriptions[int(usrInput)]['path']
    df = pd.read_csv(selected_file_name)

    # Full range of file
    wavelengths = list(df.columns)
    wavelengths = [float(wavelength) for wavelength in wavelengths]
    wavelengths = np.array(wavelengths)
    intensity = df.values
    
    lb = int(input(f'Enter the frame range lower bound between 0 and {len(intensity) - 1}: '))
    ub = int(input(f'Enter the frame range upper bound between 0 and {len(intensity) - 1}: '))
    thresh = int(input(f'Enter a threshold for minimum peak height: '))
    print('\n', end='')

    # Define the range of frames
    frameRange = list(range(lb, ub+1))

    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Create a pool of worker processes
    pool = Pool(processes=num_processes)

    # Define the inputs as tuples
    inputs = [(wavelengths, intensity[i], i, usrInput, thresh, file_paths_with_descriptions, element) for i in frameRange]

    # Map the function to the inputs in parallel
    pool.map(process_frame, inputs)

   