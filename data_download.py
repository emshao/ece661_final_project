# if needed:
# pip install librosa matplotlib numpy requests tqdm

# only need to run once

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm
import zipfile

# Download GTZAN dataset
def download_gtzan_dataset(url, save_path):
    if not os.path.exists(save_path):
        print("Downloading GTZAN dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=save_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                f.write(data)
        print("Download complete!")
    else:
        print("GTZAN dataset already downloaded.")

    # Extract the dataset
    extract_dir = save_path.rsplit('.', 1)[0]
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
    return extract_dir

# Convert audio to spectrogram
def audio_to_spectrogram(audio_path, output_dir, n_fft=2048, hop_length=512):
    os.makedirs(output_dir, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    spectrogram_file = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '.png'))
    plt.savefig(spectrogram_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    return spectrogram_file

# Process GTZAN dataset
def process_gtzan_dataset(dataset_dir, output_dir):
    audio_dir = os.path.join(dataset_dir, "genres")
    for genre in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre)
        if os.path.isdir(genre_path):
            print(f"Processing genre: {genre}")
            genre_output_dir = os.path.join(output_dir, genre)
            os.makedirs(genre_output_dir, exist_ok=True)
            for audio_file in os.listdir(genre_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(genre_path, audio_file)
                    audio_to_spectrogram(audio_path, genre_output_dir)
                    print(f"Processed {audio_file}")

if __name__ == "__main__":
    # Set URLs and paths
    GTZAN_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    DOWNLOAD_PATH = "gtzan_dataset.zip"
    DATASET_DIR = "gtzan_dataset"
    OUTPUT_DIR = "gtzan_spectrograms"

    # Step 1: Download and extract dataset
    dataset_path = download_gtzan_dataset(GTZAN_URL, DOWNLOAD_PATH)

    # Step 2: Convert audio files to spectrograms
    process_gtzan_dataset(dataset_path, OUTPUT_DIR)
    print(f"Spectrograms saved in: {OUTPUT_DIR}")
