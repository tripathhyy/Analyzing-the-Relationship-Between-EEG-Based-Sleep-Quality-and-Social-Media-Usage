import os
import urllib.request
import pandas as pd
import numpy as np
import mne
from mne.time_frequency import psd_array_welch

base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

subject_ids = [
    'SC4001E0', 'SC4002E0', 'SC4031E0', 'SC4012E0',
    'SC4022E0', 'SC4032E0', 'SC4041E0', 'SC4042E0',
    'SC4051E0', 'SC4052E0', 'SC4061E0', 'SC4062E0',
    'SC4071E0', 'SC4072E0'
]

suffixes = ['C', 'J', 'P', 'H']
all_eeg_features = []

for subject_id in subject_ids:
    print(f"\n🔄 Checking files for subject: {subject_id}")

    psg_file = f"eeg_dataset/{subject_id}-PSG.edf"
    
    try:
        if not os.path.exists(psg_file):
            print(f"⬇️  Downloading {psg_file}...")
            urllib.request.urlretrieve(base_url + psg_file, psg_file)
        else:
            print(f"✅ {psg_file} already exists.")
    except Exception as e:
        print(f"❌ Error downloading {psg_file}: {e}")
        continue

    # Try all suffixes to find the hypnogram
    hypnogram_file = None
    for suffix in suffixes:
        candidate = f"eeg_dataset/{subject_id[:-1]}{suffix}-Hypnogram.edf"
        try:
            if not os.path.exists(candidate):
                print(f"🔍 Trying {candidate}...")
                urllib.request.urlretrieve(base_url + candidate, candidate)
                hypnogram_file = candidate
                print(f"✅ Found and downloaded hypnogram: {candidate}")
                break
            else:
                hypnogram_file = candidate
                print(f"✅ Hypnogram file already exists: {candidate}")
                break
        except Exception:
            continue

    if not hypnogram_file:
        print(f"❌ Could not find hypnogram file for {subject_id}. Skipping...")
        continue

    try:
        # Load EEG and annotations
        psg = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        annotations = mne.read_annotations(hypnogram_file)
        psg.set_annotations(annotations)

        # Check the subject info in the raw EDF file
        info = psg.info
        print(f"\n🔍 Subject Info for {subject_id}: {info}")

        # Extract demographic info from the subject_info field (if available)
        subject_info = info.get('subject_info', {})
        
        # Print subject_info dictionary to see all available fields
        print(f"🔍 Subject Info Dict: {subject_info}")
        
        # Extract age from 'last_name' if it's in the format 'XXyr'
        age_str = subject_info.get('last_name', '')
        age = np.nan
        if 'yr' in age_str:
            try:
                age = int(age_str.split('yr')[0].strip())  # Extract the numeric part
            except ValueError:
                age = np.nan

        print(f"Age: {age}")

        # Proceed with EEG processing
        eeg = psg.copy().pick_types(eeg=True)
        eeg.filter(0.3, 35., fir_design='firwin')

        events, _ = mne.events_from_annotations(eeg)

        event_id = {
            'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3': 4,
            'Sleep stage 4': 4,
            'Sleep stage R': 5
        }

        epochs = mne.Epochs(
            eeg,
            events=events,
            event_id=event_id,
            tmin=0,
            tmax=30,
            baseline=None,
            preload=True,
            verbose=False
        )

        data_uv = epochs.get_data() * 1e6  # Convert to microvolts

        psd_band, freqs = psd_array_welch(
            data_uv,
            sfreq=epochs.info['sfreq'],
            fmin=0.3,
            fmax=35,
            n_fft=2048
        )

        freq_bands = {
            'Delta': (0.3, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }

        band_powers = {}
        for band, (fmin, fmax) in freq_bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            band_psd = psd_band[:, :, mask]
            band_avg_power = band_psd.mean(axis=(0, 1))
            band_powers[band] = band_avg_power

        min_len = min(len(p) for p in band_powers.values())
        for band in band_powers:
            band_powers[band] = band_powers[band][:min_len]

        # Add demographic info (age) to the EEG data
        df = pd.DataFrame(band_powers)
        df['User_ID'] = subject_id
        df['Age'] = age

        all_eeg_features.append(df)

    except Exception as e:
        print(f"❌ Error processing EEG data for {subject_id}: {e}")
        continue

if all_eeg_features:
    combined_eeg_df = pd.concat(all_eeg_features, ignore_index=True)
    combined_eeg_df.to_csv("eeg_features_combined_with_demographics.csv", index=False)
    print("✅ EEG features with demographics saved to 'eeg_features_combined_with_demographics.csv'")
else:
    print("⚠️ No EEG features extracted. Check for errors above.")
