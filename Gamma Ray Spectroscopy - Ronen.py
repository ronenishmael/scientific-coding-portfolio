#!/usr/bin/env python
# coding: utf-8

# ## Ronen Ishmayelov
# ## PHY424 - Experiment 1
# ## Compton Scattering


"""
--------------------------------------------------------------
Gamma Ray Spectroscopy Data Analysis
--------------------------------------------------------------

This Python script is designed for the analysis of gamma ray spectroscopy data, specifically focusing on 
the calibration and measurement ofs photo peak and Compton edge energies for isotopes such as Co-60, 
Cs-137, and Bi-207. The code employs various mathematical models and fitting techniques to extract 
important physical parameters from experimental data.

Key functionalities include:

1. **Data Calibration**: The script reads experimental data from CSV files and calibrates the channel 
   counts to energy values using known photo peak energies for each isotope.

2. **Statistical Analysis**: The average and standard error of the mean (SEM) are computed for energy 
   measurements to assess the precision of the data.

3. **Intensity Modeling**: The script models the intensity of gamma rays as a function of absorber 
   thickness using an exponential decay function. It estimates the absorption coefficients for materials 
   (e.g., aluminum) based on the fitted data.

4. **Curve Fitting**: Gaussian fitting is applied to the energy spectra to identify photo peaks and 
   Compton edges. Linear fitting is also performed on the logarithm of intensity ratios to extract 
   absorption coefficients.

5. **Visualization**: The script generates plots to visually represent the energy spectra, fitted 
   functions, and the relationship between thickness and intensity.

6. **Output**: Final results, including fitted parameters and estimated absorption coefficients, are printed 
   for each isotope analyzed.
sssss

--------------------------------------------------------------
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ## CO-60 Data Processing

# Load CO-60 data files
data_CO60_32 = np.genfromtxt("7-BI207-AL38.15-1200-64-2.00.csv", delimiter=',', skip_header=22, filling_values=np.nan)
data2_CO60 = np.genfromtxt("2-CO60-AL12.78-1200-64-2.00.csv", delimiter=',', skip_header=22, filling_values=np.nan)
data3_CO60 = np.genfromtxt("3-CO60-AL6.25-1200-64-2.00.csv", delimiter=',', skip_header=22, filling_values=np.nan)

# Extract columns from data
a1, b1, c1 = data_CO60_32.T
a2, b2, c2 = data2_CO60.T
a3, b3, c3 = data3_CO60.T

# Plot CO-60 data
fig1, ax1 = plt.subplots(figsize=(20, 10))
ax1.scatter(a1, c1, color="red", s=5, label="CO60-38")
ax1.set(xlim=(0, 1000), ylim=(0, 100), autoscale_on=False, title='CO-60: Click to zoom')
plt.legend()
plt.show()

fig2, ax2 = plt.subplots(figsize=(20, 10))
ax2.scatter(a2, c2, color="red", s=2, label="CO60-12")
ax2.set(xlim=(0, 1000), ylim=(0, 500), autoscale_on=False, title='CO-60: Click to zoom')
plt.legend()
plt.show()

fig3, ax3 = plt.subplots(figsize=(20, 10))
ax3.scatter(a3, c3, color="red", s=2, label="CO60-6")
ax3.set(xlim=(0, 1000), ylim=(0, 500), autoscale_on=False, title='CO-60: Click to zoom')
plt.legend()
plt.show()

# ## Fitting a Gaussian to data

# Function to define a Gaussian
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Initial guess for Gaussian fitting parameters
p0 = [275000, 490, 10]

# Fit Gaussian to the CO-60 data
popt, _ = curve_fit(gaussian, a1, c1, p0)

# Extract fitted Gaussian parameters
fitted_counts = gaussian(a1, *popt)

# Plot Gaussian fit
plt.bar(a1[450:525], c1[450:525], align='center', alpha=0.7, label='Data')
plt.plot(a1[450:525], fitted_counts[450:525], 'r-', label='Fit')
plt.legend()
plt.xlabel('Channels')
plt.ylabel('Photon Counts')
plt.title('CO-60 Histogram with Gaussian Fit')
plt.show()

# ## Processing CS-137 Data

# Load CS-137 data
data_CS137 = np.genfromtxt("3-CS173-AL-38.15(5).csv", delimiter=',', skip_header=22, filling_values=np.nan)

# Extract columns from data
channel, _, counts = data_CS137.T

# Plot CS-137 data
fig_cs, ax_cs = plt.subplots(figsize=(5, 5))
ax_cs.scatter(channel, counts, color="blue", s=3)
ax_cs.set(xlim=(0, 30), ylim=(0, 140), autoscale_on=False, title='Click to zoom')
plt.show()

# ## Fitting Gaussian to CS-137 Data

# Function to define Gaussian for CS-137 data
def gaussian1(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Initial guess for Gaussian fit
p0_cs = [110, 16, 5]

# Fit Gaussian to CS-137 data
popt_cs, _ = curve_fit(gaussian1, channel[:30], counts[:30], p0_cs)

# Extract fitted Gaussian parameters
fitted_counts_cs = gaussian1(channel, *popt_cs)

# Plot CS-137 data with Gaussian fit
plt.bar(channel[:30], counts[:30], align='center', alpha=0.7, label='Data')
plt.plot(channel[:30], fitted_counts_cs[:30], 'r-', label='Fit')
plt.legend()
plt.xlabel('Channels')
plt.ylabel('Photon Counts')
plt.title('CS-137 Histogram with Gaussian Fit')
plt.show()

# ## Analyzing Multiple Data Files

# Dictionary of data files and observed values
file_data = {
    "1-CO60-AL38.15-1200-64-2.00.csv": {'photo': [55000, 700], 'edge': [100, 125]},
    "2-CO60-AL12.78-1200-64-2.00.csv": {'photo': [67000, 700], 'edge': [130, 125]},
    "3-CO60-AL6.25-1200-64-2.00.csv": {'photo': [82000, 700], 'edge': [90, 120]},
    "4-CS137-AL38.15-1200-64-2.00.csv": {'photo': [42000, 700], 'edge': [100, 120]},
    "5-CS137-AL12.78-1200-64-2.00.csv": {'photo': [62000, 700], 'edge': [300, 120]},
    "6-CS137-AL6.25-1200-64-2.00.csv": {'photo': [60000, 700], 'edge': [12, 120]},
    "7-BI207-AL38.15-1200-64-2.00.csv": {'photo': [32000, 700], 'edge': [12, 120]},
    "8-BI207-AL12.78-1200-64-2.00.csv": {'photo': [41000, 700], 'edge': [20, 125]},
    "9-BI207-AL6.25-1200-64-2.00.csv": {'photo': [50000, 700], 'edge': [30, 125]}
}

# Calibration values for CS-137 and CO-60
calibration_values = {
    'CS137': {'photo_peak': 662, 'compton_edge': 480},
    'CO60': {'photo_peak': 1173.2, 'compton_edge': 963.4}
}

# Loop through files and fit Gaussian for photo peak and Compton edge
for file, values in file_data.items():
    data = np.genfromtxt(file, delimiter=',', skip_header=22, filling_values=np.nan)
    channel, _, counts = data.T
    
    # Fit Gaussian to the photo peak
    p0_photo = [values['photo'][0], values['photo'][1], 10]
    popt_photo, _ = curve_fit(gaussian, channel, counts, p0_photo)
    fitted_counts_photo = gaussian(channel, *popt_photo)
    
    # Fit Gaussian to the Compton edge
    p0_compton = [values['edge'][0], values['edge'][1], 10]
    popt_compton, _ = curve_fit(gaussian, channel, counts, p0_compton)
    fitted_counts_compton = gaussian(channel, *popt_compton)
    
    # Plot the fits
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(channel, counts, color="blue", s=3)
    ax.plot(channel, fitted_counts_photo, 'r-', label='Photo Peak Fit')
    ax.plot(channel, fitted_counts_compton, 'g-', label='Compton Edge Fit')
    ax.legend()
    plt.show()
    
    # Print peak information
    print(f"File: {file}")
    print(f"Photo Peak: {popt_photo[0]:.2f}, SD: {popt_photo[2]:.2f}")
    print(f"Compton Edge Peak: {popt_compton[0]:.2f}, SD: {popt_compton[2]:.2f}")


# In[75]:

co60_values = {
    'photo_peak': [989.19, 994.29, 994.40],
    'photo_error': [14.46, 14.22, 11.77],
    'edge_peak': [219.81, 236.79, 214.46],
    'edge_error': [23.69, 32.63, 21.59]
}

cs137_values = {
    'photo_peak': [563.41, 564.13, 561.48],
    'photo_error': [7.91, 8.18, 5.00],
    'edge_peak': [130.99, 136.78, 113.20],
    'edge_error': [17.08, 19.99, 7.45]
}

def compute_average_and_SEM(values, errors):
    n = len(values)
    average = sum(values) / n
    SEM = (sum([error**2 for error in errors]) / n) ** 0.5
    return average, SEM

print("CO60 Averages:")
co60_photo_avg, co60_photo_SEM = compute_average_and_SEM(co60_values['photo_peak'], co60_values['photo_error'])
print(f"Photo Peak Average Energy: {co60_photo_avg:.2f} KeV ± {co60_photo_SEM:.2f} KeV")

co60_edge_avg, co60_edge_SEM = compute_average_and_SEM(co60_values['edge_peak'], co60_values['edge_error'])
print(f"Compton Edge Average Energy: {co60_edge_avg:.2f} KeV ± {co60_edge_SEM:.2f} KeV")

print("\nCS137 Averages:")
cs137_photo_avg, cs137_photo_SEM = compute_average_and_SEM(cs137_values['photo_peak'], cs137_values['photo_error'])
print(f"Photo Peak Average Energy: {cs137_photo_avg:.2f} KeV ± {cs137_photo_SEM:.2f} KeV")

cs137_edge_avg, cs137_edge_SEM = compute_average_and_SEM(cs137_values['edge_peak'], cs137_values['edge_error'])
print(f"Compton Edge Average Energy: {cs137_edge_avg:.2f} KeV ± {cs137_edge_SEM:.2f} KeV")


# In[76]: Intensity function

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def intensity(x, I0, n_sigma_total):
    return I0 * np.exp(-n_sigma_total * x)

# Experimental data (example for CS137)
x_values = np.array([6.25, 12.78, 38.15])  # Thickness of aluminum
I_values_CS = np.array([500, 400, 300])  # Measured intensities for CS137

# Fit the experimental data
params_CS, _ = curve_fit(intensity, x_values, I_values_CS, p0=[max(I_values_CS), 0.1])
I0_CS, n_sigma_total_CS = params_CS

# Plot data and fit
plt.scatter(x_values, I_values_CS, label='Data for CS', color='red')
plt.plot(x_values, intensity(x_values, *params_CS), label='Fit for CS', color='red', linestyle='--')
plt.xlabel("Thickness of Al (cm)")
plt.ylabel("Intensity")
plt.legend()
plt.title("CS137: Intensity vs. Thickness of Aluminum")
plt.show()

# Display the absorption coefficient
print(f"Absorption coefficient for CS: {n_sigma_total_CS:.2e} m^2")

# In[94]: Fit for aluminum absorption

x_values = np.array([6.25, 12.78, 38.15])
I_values = np.array([500, 400, 250])  # Intensity values for another set of experiments

params, covariance = curve_fit(intensity, x_values, I_values, p0=[max(I_values), 0.1])
I0_fit, n_sigma_total_fit = params

n_aluminium = 6.022e28  # Number density of aluminum
sigma_total = n_sigma_total_fit / n_aluminium  # Total cross section

# Plot data and fit
plt.scatter(x_values, I_values, label='Experimental Data', color='red')
plt.plot(x_values, intensity(x_values, *params), label='Fit', linestyle='--')
plt.xlabel('Thickness of Al (cm)')
plt.ylabel('Intensity')
plt.legend()
plt.title('Intensity vs. Thickness of Aluminum')
plt.show()

# Display the fitted values and calculated sigma_total
print(f"Fitted I(0): {I0_fit:.2f}")
print(f"Fitted n*sigma_total: {n_sigma_total_fit:.2e} m^2")
print(f"Calculated sigma_total for Aluminum: {sigma_total:.2e} m^2/atom")


# In[99]: Photopeak amplitudes vs. aluminum thickness

photopeak_amplitudes = [60000, 40000, 10000]
thicknesses = [6.25, 12.78, 38.15]

params, covariance = curve_fit(intensity, thicknesses, photopeak_amplitudes, p0=[max(photopeak_amplitudes), 0.1])
I0_fit, n_sigma_total_fit = params

# Plot data and fit
plt.scatter(thicknesses, photopeak_amplitudes, label='Photopeak Amplitudes', color='red')
plt.plot(thicknesses, intensity(thicknesses, *params), label='Fit', linestyle='--')
plt.xlabel('Thickness of Al (cm)')
plt.ylabel('Photopeak Amplitude')
plt.legend()
plt.title('Photopeak Amplitude vs. Thickness of Aluminum')
plt.show()

# Display fitted values
print(f"Fitted I(0): {I0_fit:.2f}")
print(f"Fitted n*sigma_total: {n_sigma_total_fit:.2e} m^2")


# In[106]: Analyzing log(I/I0) vs. thickness

def linear(x, m, c):
    return m * x + c

# Experimental data
log_ratios = []
thickness_vals = []

for file, values in energy_values.items():
    isotope = None
    for key in ['CS137', 'CO60', 'BI207']:
        if key in file:
            isotope = key
            break
    
    I0 = I0_values[isotope]
    peak_energy = values['photo_peak']
    log_ratio = np.log(peak_energy / I0)
    
    if not np.isnan(log_ratio):  # Avoid NaN entries
        log_ratios.append(log_ratio)
        
        for t in thicknesses:
            if str(t) in file:
                thickness_vals.append(t)
                break

params, _ = curve_fit(linear, thickness_vals, log_ratios)

# Plot data and linear fit
plt.figure(figsize=(10,6))
plt.scatter(thickness_vals, log_ratios, color='red', label="Data")
plt.plot(thickness_vals, linear(np.array(thickness_vals), *params), '--', label='Fit')
plt.xlabel("Thickness (cm)")
plt.ylabel("log(I/I0)")
plt.legend()
plt.title("Thickness vs. log(I/I0)")
plt.show()

n_sigma_total = -params[0]  # The slope gives us n*sigma_total
print(f"Estimated n*sigma_total: {n_sigma_total:.2e} m^2")

