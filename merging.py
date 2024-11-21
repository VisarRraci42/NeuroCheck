import pandas as pd

def merge_csv_files(non_rhd_csv, rhd_csv, output_csv):
    # Load non-RHD features
    df_non_rhd = pd.read_csv(non_rhd_csv)
    print(f"Loaded non-RHD features: {df_non_rhd.shape[0]} samples.")

    # Load RHD features
    df_rhd = pd.read_csv(rhd_csv)
    print(f"Loaded RHD features: {df_rhd.shape[0]} samples.")

    # Combine the DataFrames
    df_combined = pd.concat([df_non_rhd, df_rhd], ignore_index=True)
    print(f"Combined dataset shape: {df_combined.shape}")

    # Shuffle the combined dataset (optional but recommended)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Shuffled the combined dataset.")

    # Save to a new CSV file
    df_combined.to_csv(output_csv, index=False)
    print(f"Combined features saved to {output_csv}")

def main():
    # Configuration
    non_rhd_csv = 'fastica_features_non_rhd.csv'  # Path to non-RHD features CSV
    rhd_csv = 'fastica_features_rhd.csv'          # Path to RHD features CSV
    output_csv = 'fastica_features_combined.csv'  # Output CSV file name

    merge_csv_files(non_rhd_csv, rhd_csv, output_csv)

if __name__ == "__main__":
    main()
