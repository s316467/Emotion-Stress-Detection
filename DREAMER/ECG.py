import scipy.io as sio  # For loading MATLAB .mat files
import neurokit2 as nk  # For processing ECG signals
import pandas as pd  # For data manipulation
from sklearn import preprocessing as pre  # For data preprocessing (scaling)
import warnings  # For suppressing warnings

# Ignore warnings to reduce clutter in the output
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    total = 0  # Counter to track progress
    path = u'DREAMER.mat'  # Path to the MATLAB file containing the data
    data = sio.loadmat(path)  # Load the MATLAB file

    print("ECG signals are being feature extracted...")  # Notify the user that processing has started

    ECG = {}  # Initialize an empty dictionary to store processed data

    # Loop through all subjects (23 in total) and stimuli (18 in total)
    for k in range(0, 23):  # Loop through 23 subjects
        for j in range(0, 18):  # Loop through 18 stimuli per subject

            # Extract baseline and stimuli ECG signals for both left and right channels
            basl_l = data['DREAMER'][0, 0]['Data'][0, k]['ECG'][0, 0]['baseline'][0, 0][j, 0][:, 0]
            stim_l = data['DREAMER'][0, 0]['Data'][0, k]['ECG'][0, 0]['stimuli'][0, 0][j, 0][:, 0]
            basl_r = data['DREAMER'][0, 0]['Data'][0, k]['ECG'][0, 0]['baseline'][0, 0][j, 0][:, 1]
            stim_r = data['DREAMER'][0, 0]['Data'][0, k]['ECG'][0, 0]['stimuli'][0, 0][j, 0][:, 1]

            # Process ECG signals using neurokit2 for left and right channels
            ecg_signals_b_l, info_b_l = nk.ecg_process(basl_l, sampling_rate=256)  # Baseline left channel
            ecg_signals_s_l, info_s_l = nk.ecg_process(stim_l, sampling_rate=256)  # Stimuli left channel
            ecg_signals_b_r, info_b_r = nk.ecg_process(basl_r, sampling_rate=256)  # Baseline right channel
            ecg_signals_s_r, info_s_r = nk.ecg_process(stim_r, sampling_rate=256)  # Stimuli right channel

            # Calculate interval-related features for left and right channels
            # Ratio of stimuli features to baseline features
            processed_ecg_l = nk.ecg_intervalrelated(ecg_signals_s_l) / nk.ecg_intervalrelated(ecg_signals_b_l)
            processed_ecg_r = nk.ecg_intervalrelated(ecg_signals_s_r) / nk.ecg_intervalrelated(ecg_signals_b_r)

            # Average the left and right channel features to create a combined feature set
            processed_ecg = (processed_ecg_l + processed_ecg_r) / 2

            # Append the processed features to the ECG dictionary or DataFrame
            if not len(ECG):  # If ECG is empty, initialize it
                ECG = processed_ecg
            else:  # If ECG already has data, concatenate the new data
                ECG = pd.concat([ECG, processed_ecg], ignore_index=True)

            # Update progress counter
            total += 1
            print("\rprogress: %d%%" % (total / (23 * 18) * 100), end="")  # Display progress percentage

    # Optional preprocessing (currently commented out):
    # - Standardize the features using sklearn's StandardScaler
    # col = ECG.columns.values
    # scaler = pre.StandardScaler()
    # for i in range(len(col)):
    #     ECG[col[i][:-3]] = scaler.fit_transform(ECG[[col[i]]])
    # ECG.drop(col, axis=1, inplace=True)
    # ECG.columns = col

    # Save the processed ECG data to a CSV file
    ECG.to_csv("ECG.csv")  # Save as "ECG.csv" in the current directory
