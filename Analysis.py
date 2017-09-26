import numpy as np
from os import listdir
from collections import defaultdict
import scipy.signal as signal
import pandas as pd

"""
Scales a 1D array (timeseries) to range (minval, max observed value in a) 
"""
def scale(a, minval):
    return (a - np.min(a)) * (np.max(a) - minval) / (np.max(a) - np.min(a)) + minval

"""
Loads data and fills a dictionary SAMPLE -> SERIES_TYPE -> SERIES
"""
def load_data():
    # SAMPLE -> SERIES_TYPE -> SERIES
    data = defaultdict(dict)

    for filename in listdir('./New CSVs/'):
        if filename.endswith('.csv'):
            file = open('./New CSVs/' + filename, 'r')
            file.readline()
            elapsed_time = []
            abs_time = []
            temp = []
            hr = []
            t = 0
            for line in file:
                line = line.strip().rsplit(',', 2)
                line[0] = line[0].replace('"', '').replace(',', '')
                try:
                    abs_time.append(line[0])
                    temp.append(float(line[1]))
                    hr.append(float(line[2]))
                    elapsed_time.append(t)
                except ValueError:
                    print(line)
                t += 4
            data[filename[:len(filename) - 4]]['time'] = np.array(elapsed_time[:])
            data[filename[:len(filename) - 4]]['abs-time'] = np.array(abs_time[:])
            data[filename[:len(filename) - 4]]['temp'] = np.array(temp[:])
            data[filename[:len(filename) - 4]]['hr'] = np.array(hr[:])
    return data

"""
Loads data, applies LP filter on HR and scaling on Tb
"""
def load_and_filter():
    data = load_data()
    # Loop over all available samples
    for sample in data:
        # First-order low-pass filter with critical freq 0.03 half-cycles / sample
        b, a = signal.butter(N=1, Wn=0.03)
        # Apply the filter to hr, forwards and backwards.
        hrlp = signal.filtfilt(b, a, data[sample]['hr'])
        # Store filtered data in dictionary
        data[sample]['hr-lp'] = hrlp
        # Scale and store Tb in dictionary
        data[sample]['temp-scaled'] = scale(data[sample]['temp'], 5.0)
    return data

"""
Create summary for time at which a HR-LP threshold is reached
"""
def hrlp_thresholds_summary(data, sample, thresh):
    thresh_time = 0
    diff_hr_filt = np.diff(data[sample]['hr-lp'])
    max_hr_filt = np.max(data[sample]['hr-lp'])
    for t in range(len(data[sample]['hr-lp']) - 1):
        if diff_hr_filt[t] < -0.1 and diff_hr_filt[t + 1] < -0.1 and data[sample]['hr-lp'][t] < (thresh * max_hr_filt):
            thresh_time = t
            break
    thresh_abs_time = data[sample]['abs-time'][thresh_time]
    Tb = data[sample]['temp'][thresh_time]
    Tb_scaled = data[sample]['temp-scaled'][thresh_time]
    HR = data[sample]['hr'][thresh_time]
    return '%.0f%% Entrance Criteria,%s,%s,%s,%s,%s\n' % ((thresh*100), Tb, Tb_scaled, HR, data[sample]['hr-lp'][thresh_time], thresh_abs_time)


"""
Create summary for time according to early entrance criteria, default 30.0 C
"""
def early_entrance_summary(data, sample, temp=30.0):
    thresh_time = 0
    diff_temp = np.diff(data[sample]['temp'])
    for t in range(len(data[sample]['hr-lp']) - 1):
        if diff_temp[t] < -0.1 and diff_temp[t + 1] < -0.1 and data[sample]['temp'][t] <= temp:
            thresh_time = t
            break
    thresh_abs_time = data[sample]['abs-time'][thresh_time]
    Tb = data[sample]['temp'][thresh_time]
    Tb_scaled = data[sample]['temp-scaled'][thresh_time]
    HR = data[sample]['hr'][thresh_time]
    return 'Early (%sC) Entrance,%s,%s,%s,%s,%s\n' % (str(temp), Tb, Tb_scaled, HR, data[sample]['hr-lp'][thresh_time], thresh_abs_time)

"""
Computes time at which signal first increases above 7C
"""
def get_arousal_time_7degrees(data, sample):
    hr_filt = data[sample]['hr-lp']
    temp = data[sample]['temp']
    diff_temp = np.diff(temp)
    for t in range(len(hr_filt) - 1):
        if diff_temp[t] > 0.1 and diff_temp[t + 1] > 0.1 and temp[t] > 7.0:
            return t

"""
Computes arousal time based on raw HR criteria (increasing, above 5C)
"""
def get_arousal_time_hr_criteria(data, sample):
    hr_raw = data[sample]['hr']
    diff_hr = np.diff(hr_raw)
    for t in range(len(hr_raw) - 1):
        if diff_hr[t] > 0.1 and diff_hr[t + 1] > 0.1 and hr_raw[t] > 5:
            return t


"""
Computes entrance time given a HR-LP threshold
"""
def get_entrance_time(data, sample, thresh=0.65):
    diff_hr_filt = np.diff(data[sample]['hr-lp'])
    max_hr_filt = np.max(data[sample]['hr-lp'])
    for t in range(len(data[sample]['hr-lp']) - 1):
        if diff_hr_filt[t] < -0.1 and diff_hr_filt[t + 1] < -0.1 \
                and data[sample]['hr-lp'][t] < (thresh * max_hr_filt):
            return t


"""
Create summary for mid-IBE, computed by finding half-way
point between arousal and entrance.
"""
def mid_IBE_summary(data, sample):
    arousal_time = get_arousal_time_7degrees(data, sample)
    entrance_time = get_entrance_time(data, sample)
    mid_IBE_time = (arousal_time + entrance_time) // 2
    mid_IBE_abs_time = data[sample]['abs-time'][mid_IBE_time]
    Tb = data[sample]['temp'][mid_IBE_time]
    Tb_scaled = data[sample]['temp-scaled'][mid_IBE_time]
    HR = data[sample]['hr'][mid_IBE_time]
    return 'Mid IBE,%s,%s,%s,%s,%s\n'%(Tb, Tb_scaled, HR, data[sample]['hr-lp'][mid_IBE_time], mid_IBE_abs_time)


"""
Create summary for torpor, computed by taking
the initial time point in the data.
"""
def mid_torpor_summary(data, sample):
    mid_torpor_time = 0
    mid_torpor_abs_time = data[sample]['abs-time'][mid_torpor_time]
    Tb = data[sample]['temp'][mid_torpor_time]
    Tb_scaled = data[sample]['temp-scaled'][mid_torpor_time]
    HR = data[sample]['hr'][mid_torpor_time]
    return 'Mid Torpor,%s,%s,%s,%s,%s\n' % (Tb, Tb_scaled, HR, data[sample]['hr-lp'][mid_torpor_time], mid_torpor_abs_time)

"""
Create summary for arousal based on raw HR criteria
"""
def arousal_criteria_summary(data, sample):
    arousal_time = get_arousal_time_hr_criteria(data, sample)
    arousal_abs_time = data[sample]['abs-time'][arousal_time]
    Tb = data[sample]['temp'][arousal_time]
    Tb_scaled = data[sample]['temp-scaled'][arousal_time]
    HR = data[sample]['hr'][arousal_time]
    return 'Arousal HR Criteria,%s,%s,%s,%s,%s\n' % (Tb, Tb_scaled, HR, data[sample]['hr-lp'][arousal_time], arousal_abs_time)


"""
Create summary for arousal based on conventional
7C criteria.
"""
def early_arousal_summary(data, sample):
    early_arousal_time = 0
    for t in range(len(data[sample]['temp'])):
        if data[sample]['temp'][t] >= 7.0:
            early_arousal_time = t
            break
    early_arousal_abs_time = data[sample]['abs-time'][early_arousal_time]
    Tb = data[sample]['temp'][early_arousal_time]
    Tb_scaled = data[sample]['temp-scaled'][early_arousal_time]
    HR = data[sample]['hr'][early_arousal_time]
    return 'Early (7C) Arousal,%s,%s,%s,%s,%s\n' % (Tb, Tb_scaled, HR, data[sample]['hr-lp'][early_arousal_time], early_arousal_abs_time)


"""
Create summary at max observed raw HR.
"""
def max_hr_summary(data, sample):
    max_hr_time = np.argmax(data[sample]['hr'])
    max_hr_abs_time = data[sample]['abs-time'][max_hr_time]
    Tb = data[sample]['temp'][max_hr_time]
    Tb_scaled = data[sample]['temp-scaled'][max_hr_time]
    HR = data[sample]['hr'][max_hr_time]
    return 'Max HR,%s,%s,%s,%s,%s\n' % (
    Tb, Tb_scaled, HR, data[sample]['hr-lp'][max_hr_time], max_hr_abs_time)


"""
Main script
"""
data = load_and_filter()

# Create and store summaries as CSV files
for sample in data.keys():
    file = open('./For Jim/Analysis/Summary-%s.csv' % sample, 'w')
    file.write(',Tb,Tb Transformed,Raw HR,HR-LP,Abs. Time\n')
    file.write(mid_torpor_summary(data, sample))
    file.write(arousal_criteria_summary(data, sample))
    file.write(early_arousal_summary(data, sample))
    file.write(max_hr_summary(data, sample))
    file.write(mid_IBE_summary(data, sample))
    file.write(hrlp_thresholds_summary(data, sample, 0.7))
    file.write(hrlp_thresholds_summary(data, sample, 0.65))
    file.write(early_entrance_summary(data, sample))
    file.close()

    frame = pd.DataFrame(np.vstack((data[sample]['abs-time'], data[sample]['hr'], data[sample]['hr-lp'], data[sample]['temp'], data[sample]['temp-scaled'])).T)
    frame.columns = ['Time', 'HR', 'HR-LP', 'Tb', 'Tb Transformed']
    frame.to_csv('./Analysis/Data-%s.csv' % sample)

print('Finished!')