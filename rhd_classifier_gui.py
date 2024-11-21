import PySimpleGUI as sg
import joblib
import librosa
import numpy as np
import os
import traceback

# Optional: For better visuals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Function to load models
def load_models(model_dir='.'):
    try:
        rf_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        fastica_path = os.path.join(model_dir, 'fastica_model.pkl')

        rf_model = joblib.load(rf_model_path)
        label_encoder = joblib.load(label_encoder_path)
        scaler = joblib.load(scaler_path)
        fastica = joblib.load(fastica_path)

        print("Models loaded successfully.")
        return rf_model, label_encoder, scaler, fastica
    except Exception as e:
        sg.popup_error(f"Error loading models: {e}")
        traceback.print_exc()
        return None, None, None, None


# Function to preprocess and extract features from audio
def preprocess_audio(file_path, scaler, fastica, max_length):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050, mono=True)

        # Trim silence
        y, _ = librosa.effects.trim(y)

        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Pad or truncate
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)), 'constant')
        else:
            y = y[:max_length]

        # Reshape for scaler
        y = y.reshape(1, -1)

        # Scale
        y_scaled = scaler.transform(y)

        # FastICA
        features = fastica.transform(y_scaled)

        return features
    except Exception as e:
        sg.popup_error(f"Error processing audio file: {e}")
        traceback.print_exc()
        return None


# Function to draw matplotlib figures in PySimpleGUI
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas.TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main():
    # Load models
    model_dir = '.'  # Change if models are in a different directory
    rf_model, label_encoder, scaler, fastica = load_models(model_dir)

    if not all([rf_model, label_encoder, scaler, fastica]):
        sg.popup_error("One or more models failed to load. Exiting.")
        return

    # Infer max_length from scaler
    max_length = scaler.mean_.shape[0]
    print(f"Inferred max_length: {max_length} samples.")

    # Define the window layout
    sg.theme('LightBlue2')

    layout = [
        [sg.Text('RHD Detection AI Model', size=(40, 1), justification='center', font=('Helvetica', 20))],
        [sg.Text('Drag and Drop an MP3 or WAV file below:', font=('Helvetica', 14))],
        [sg.Input(enable_events=True, key='-FILE-', visible=False, do_not_clear=True),
         sg.FilesBrowse('Browse', file_types=(("Audio Files", "*.wav *.mp3"),))],
        [sg.Text('', size=(60, 1), key='-FILENAME-', font=('Helvetica', 12))],
        [sg.Button('Exit', size=(10, 1)), sg.Button('Clear', size=(10, 1))],
        [sg.Frame('Result', [
            [sg.Text('Classification Result:', font=('Helvetica', 12, 'bold')),
             sg.Text('', key='-RESULT-', font=('Helvetica', 12))],
            [sg.Text('Confidence Scores:', font=('Helvetica', 12, 'bold'))],
            [sg.Text('RHD:', size=(10, 1)), sg.Text('', key='-CONF_RHD-', font=('Helvetica', 12))],
            [sg.Text('Non-RHD:', size=(10, 1)), sg.Text('', key='-CONF_NON_RHD-', font=('Helvetica', 12))]
        ])],
        [sg.Canvas(key='-CANVAS-')]
    ]

    window = sg.Window('RHD Detection AI Model', layout, finalize=True)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        if event == 'Clear':
            window['-FILENAME-'].update('')
            window['-RESULT-'].update('')
            window['-CONF_RHD-'].update('')
            window['-CONF_NON_RHD-'].update('')
            # Clear the canvas
            window['-CANVAS-'].TKCanvas.delete("all")
            continue

        if event == '-FILE-':
            file_paths = values['-FILE-']
            if file_paths:
                # Support multiple files by splitting
                files = file_paths.split(';')
                for file_path in files:
                    if not os.path.isfile(file_path):
                        sg.popup_error(f"File not found: {file_path}")
                        continue

                    # Update filename display
                    window['-FILENAME-'].update(os.path.basename(file_path))

                    # Preprocess and extract features
                    features = preprocess_audio(file_path, scaler, fastica, max_length)
                    if features is None:
                        continue

                    # Predict
                    prediction = rf_model.predict(features)
                    prediction_proba = rf_model.predict_proba(features)

                    # Decode label
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    confidence_non_rhd = prediction_proba[0][label_encoder.transform(['non-RHD'])[0]]
                    confidence_rhd = prediction_proba[0][label_encoder.transform(['RHD'])[0]]

                    # Update result display
                    window['-RESULT-'].update(predicted_label)
                    window['-CONF_NON_RHD-'].update(f"{confidence_non_rhd * 100:.2f}%")
                    window['-CONF_RHD-'].update(f"{confidence_rhd * 100:.2f}%")

                    # Optionally, plot feature importances or other visuals
                    # Here, we'll plot a simple bar chart of confidence scores
                    fig, ax = plt.subplots(figsize=(4, 3))
                    categories = ['Non-RHD', 'RHD']
                    scores = [confidence_non_rhd, confidence_rhd]
                    sns.barplot(x=categories, y=scores, palette='viridis', ax=ax)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Confidence')
                    ax.set_title('Confidence Scores')
                    for i, v in enumerate(scores):
                        ax.text(i, v + 0.02, f"{v * 100:.2f}%", ha='center')

                    # Clear previous figure
                    window['-CANVAS-'].TKCanvas.delete("all")
                    # Draw new figure
                    draw_figure(window['-CANVAS-'], fig)
                    plt.close(fig)

    window.close()


if __name__ == "__main__":
    main()
