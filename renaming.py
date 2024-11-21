import os
from pydub import AudioSegment

# Define paths
input_folder = '/Users/visar/Desktop/rhd/wav'  # Folder with .mp3 files
output_folder = '/Users/visar/Desktop/rhd/wav/wav'  # Folder where .wav files will be saved

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize file counter for numbering
file_counter = 1  # Start numbering at 696

# Loop through and process files in the input folder
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".wav"):  # Process only .mp3 files
        # Full path to the input file
        input_path = os.path.join(input_folder, filename)

        # Load the .mp3 file
        audio = AudioSegment.from_file(input_path)

        # Construct the new filename (e.g., "696.y.wav")
        new_name = f"{file_counter}.n.wav"
        output_path = os.path.join(output_folder, new_name)

        # Export the audio file to .wav format
        audio.export(output_path, format="wav")
        print(f"Converted {filename} to {new_name}")

        # Increment the file counter
        file_counter += 1

print("All files converted and renamed!")
