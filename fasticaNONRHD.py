import os
import numpy as np
import librosa
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import joblib  # For saving models


def load_audio_files(directory, sample_rate=22050, max_duration=None):
    audio_data = []
    file_names = []
    print(f"Loading audio files from directory: {directory}")
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
    print(f"Loaded {len(audio_data)} audio files successfully.")
    return audio_data, file_names


def pad_audio_data(audio_data):
    max_length = max(len(y) for y in audio_data)
    print(f"Maximum audio length: {max_length} samples.")
    padded_data = np.array([
        np.pad(y, (0, max_length - len(y)), 'constant') for y in audio_data
    ])
    print(f"Padded audio data shape: {padded_data.shape}")
    return padded_data


def extract_features_fastica(padded_data, n_components=20):
    print("Standardizing the data...")
    scaler = StandardScaler()
    padded_data_scaled = scaler.fit_transform(padded_data)
    print("Data standardized.")

    print(f"Applying FastICA with {n_components} components...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    ica_components = ica.fit_transform(padded_data_scaled)
    print("FastICA feature extraction completed.")
    print(f"Feature matrix shape: {ica_components.shape}")
    return ica_components, scaler, ica


def create_feature_dataframe(features, file_names, label='non-RHD'):
    feature_columns = [f'IC_{i + 1}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_columns)
    df['file_name'] = file_names
    df['label'] = label
    print("Feature DataFrame created.")
    return df


def save_features_to_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


def save_models(scaler, ica, scaler_path='scaler.pkl', ica_path='fastica_model.pkl'):
    joblib.dump(scaler, scaler_path)
    joblib.dump(ica, ica_path)
    print(f"Scaler saved to {scaler_path}")
    print(f"FastICA model saved to {ica_path}")


def main():
    # Configuration
    audio_directory = '/Users/visar/Desktop/rhd/minga/wav'  # Replace with your actual non-RHD directory
    output_csv = 'fastica_features_non_rhd.csv'  # Output CSV file name
    sample_rate = 22050  # Target sampling rate
    max_duration = None  # Set to None to load entire files or specify in seconds
    n_components = 20  # Number of FastICA components
    label = 'non-RHD'  # Label for these files

    # Step 1: Load and Preprocess Audio Files
    audio_data, file_names = load_audio_files(
        directory=audio_directory,
        sample_rate=sample_rate,
        max_duration=max_duration
    )

    if not audio_data:
        print("No audio data loaded. Exiting.")
        return

    # Step 2: Pad Audio Data
    padded_data = pad_audio_data(audio_data)

    # Step 3: Extract FastICA Features
    features, scaler, ica = extract_features_fastica(padded_data, n_components=n_components)

    # Step 4: Create Feature DataFrame
    df_features = create_feature_dataframe(features, file_names, label=label)

    # Step 5: Save Features to CSV
    save_features_to_csv(df_features, output_csv)

    # Step 6: Save the scaler and FastICA models for future use
    save_models(scaler, ica, scaler_path='scaler.pkl', ica_path='fastica_model.pkl')

    print("Non-RHD feature extraction pipeline completed successfully.")


if __name__ == "__main__":
    main()