import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import joblib  # For loading saved models


def load_audio_files(directory, sample_rate=22050, max_duration=None):
    audio_data = []
    file_names = []
    print(f"Loading RHD audio files from directory: {directory}")
    for file in tqdm(os.listdir(directory)):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(directory, file)
            try:
                # Load audio file
                if max_duration:
                    y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=max_duration)
                else:
                    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

                # Trim silence
                y, _ = librosa.effects.trim(y)

                # Normalize audio to have maximum amplitude of 1
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))

                audio_data.append(y)
                file_names.append(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    print(f"Loaded {len(audio_data)} RHD audio files successfully.")
    return audio_data, file_names


def pad_audio_data(audio_data, max_length):
    """
    Pad all audio signals to the specified maximum length.

    Parameters:
        audio_data (list of np.ndarray): List of audio time series.
        max_length (int): Length to pad/truncate each audio signal to.

    Returns:
        np.ndarray: 2D array where each row is a padded audio signal.
    """
    print(f"Padding/truncating audio data to {max_length} samples.")
    padded_data = np.array([
        y[:max_length] if len(y) >= max_length else np.pad(y, (0, max_length - len(y)), 'constant') for y in audio_data
    ])
    print(f"Padded RHD audio data shape: {padded_data.shape}")
    return padded_data


def transform_features_fastica(padded_data, scaler, ica):
    """
    Transform the padded audio data using the pre-fitted scaler and FastICA models.

    Parameters:
        padded_data (np.ndarray): 2D array where each row is a padded audio signal.
        scaler (StandardScaler): Fitted scaler object.
        ica (FastICA): Fitted FastICA model.

    Returns:
        np.ndarray: Feature matrix where each row corresponds to an audio file.
    """
    print("Standardizing the RHD data using the pre-fitted scaler...")
    padded_data_scaled = scaler.transform(padded_data)
    print("Data standardized.")

    print("Transforming data using the pre-fitted FastICA model...")
    ica_components = ica.transform(padded_data_scaled)
    print("FastICA feature extraction for RHD data completed.")
    print(f"Feature matrix shape: {ica_components.shape}")
    return ica_components


def create_feature_dataframe(features, file_names, label='RHD'):
    feature_columns = [f'IC_{i + 1}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_columns)
    df['file_name'] = file_names
    df['label'] = label
    print("RHD Feature DataFrame created.")
    return df


def save_features_to_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"RHD features saved to {output_path}")


def main():
    # Configuration
    audio_directory = '/Users/visar/Desktop/rhd/minga-rhd-afflicted'  # Replace with your actual RHD directory
    output_csv = 'fastica_features_rhd.csv'  # Output CSV file name for RHD features
    sample_rate = 22050  # Target sampling rate
    max_duration = None  # Set to None to load entire files or specify in seconds
    n_components = 20  # Number of FastICA components (should match non-RHD)
    label = 'RHD'  # Label for these files

    # Paths to the saved scaler and FastICA models from non-RHD processing
    scaler_path = 'scaler.pkl'
    ica_path = 'fastica_model.pkl'

    # Step 1: Load and Preprocess RHD Audio Files
    audio_data, file_names = load_audio_files(
        directory=audio_directory,
        sample_rate=sample_rate,
        max_duration=max_duration
    )

    if not audio_data:
        print("No RHD audio data loaded. Exiting.")
        return

    # Step 2: Determine the maximum length used in non-RHD padding
    # Load the padded data from non-RHD to get the max_length
    # Alternatively, ensure that RHD and non-RHD have similar lengths or handle discrepancies
    # For simplicity, we'll assume the non-RHD padding used the maximum length from non-RHD
    # You may need to adjust this based on your specific setup

    # Load the scaler and FastICA models
    if not os.path.exists(scaler_path) or not os.path.exists(ica_path):
        print(f"Scaler or FastICA model not found. Please ensure '{scaler_path}' and '{ica_path}' exist.")
        return

    scaler = joblib.load(scaler_path)
    ica = joblib.load(ica_path)
    print("Loaded pre-fitted StandardScaler and FastICA models.")

    # Determine the max_length used in non-RHD padding
    # Load the non-RHD padded data shape from the scaler or FastICA model
    # Since we saved the models separately, we need to ensure that RHD data is padded to the same length
    # One approach is to store the max_length separately during non-RHD processing
    # For simplicity, let's assume the max_length is consistent and known
    # Alternatively, you can modify the non-RHD script to save the max_length

    # Here, we'll load the max_length from the padded non-RHD data
    # Assuming you have saved the max_length or can retrieve it
    # For this example, we'll hardcode it or prompt the user

    # Option 1: If you have saved the max_length during non-RHD processing, load it
    # For example, save it to a file 'max_length.txt' in non-RHD script
    # Here, we'll skip and infer from the scaler's mean_

    # Since StandardScaler was fitted on non-RHD padded data, the number of features equals max_length
    # Thus, we can infer max_length from scaler.mean_
    max_length = scaler.mean_.shape[0]
    print(f"Using max_length of {max_length} samples for padding RHD data.")

    # Step 3: Pad RHD Audio Data to match non-RHD max_length
    padded_data = pad_audio_data(audio_data, max_length=max_length)

    # Step 4: Extract FastICA Features using the pre-fitted models
    features = transform_features_fastica(padded_data, scaler, ica)

    # Step 5: Create Feature DataFrame
    df_features = create_feature_dataframe(features, file_names, label=label)

    # Step 6: Save Features to CSV
    save_features_to_csv(df_features, output_csv)

    print("RHD feature extraction pipeline completed successfully.")


if __name__ == "__main__":
    main()
