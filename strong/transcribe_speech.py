#- Script to generate phoneme transcription and text references for audio files

import os
import torch
import Levenshtein

import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

from datasets import load_dataset, set_caching_enabled
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.cluster import DBSCAN

def load_wav_files(wav_dir):
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    data = []
    for file in files:
        file_path = os.path.join(wav_dir, file)
        speech, _ = sf.read(file_path)
        data.append((file, speech))
    return data

def transcribe_speech(data, processor, model):
    inputs = processor(data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda')).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def cluster_transcriptions(df: pd.DataFrame):
    data = df['transcription'].to_list()
    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])     # extract indices
        return Levenshtein.distance(data[i], data[j])/max(len(data[i]), len(data[j]))

    X = np.arange(len(data)).reshape(-1, 1)
    result = DBSCAN(eps=0.3, metric=lev_metric,n_jobs=20,min_samples=3).fit(X)
    df['cluster'] = result.labels_
    text_medians = df.groupby('cluster').apply(lambda x:Levenshtein.median(x['transcription'].to_list()))
    medians = []
    for idx, row in df.iterrows():
        if row['cluster'] == -1:
            medians.append(row['transcription'])
        else:
            medians.append(text_medians[row['cluster']])
    df['reference'] = medians
    return df

if __name__ == '__main__':
    set_caching_enabled(False)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model.to('cuda')

    # Directory containing your .wav files
    wav_dir = "/home/fernanda/ufrn/mos-pred/dataset"

    audio_data = load_wav_files(wav_dir)

    # DataFrame to store results
    all_df = pd.DataFrame(columns=['wav_name', 'transcription'])

    for file, speech in tqdm(audio_data):
        transcription = transcribe_speech(speech, processor, model)
        new_row = pd.DataFrame([{'wav_name': file, 'transcription': transcription}])
        all_df = pd.concat([all_df, new_row], ignore_index=True)

    clustered_df = cluster_transcriptions(all_df)

    clustered_df.to_csv('transcriptions_clustered.csv', index=False)

