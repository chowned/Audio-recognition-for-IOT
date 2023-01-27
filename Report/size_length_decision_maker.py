import os
import pandas as pd
from pydub import AudioSegment

def scan_folder(folder):
    duration_count = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(file_path)
                duration = len(audio)
                if duration in duration_count:
                    duration_count[duration] += 1
                else:
                    duration_count[duration] = 1
    return duration_count

def create_dataframe(duration_count):
    data = {"Duration of audio file": list(duration_count.keys()), 
            "Number of audio files with that duration": list(duration_count.values())}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Number of audio files with that duration', ascending=False)
    return df


# find the percentage. The duration returned in second is the size that include 1-percentage inside

def find_duration(folder_path, percentage_files=0.9):
    duration_count = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(file_path)
                duration = len(audio) / 1000 #convert from ms to sec
                if duration in duration_count:
                    duration_count[duration] += 1
                else:
                    duration_count[duration] = 1
    total_files = sum(duration_count.values())
    target_files = total_files * percentage_files
    current_count = 0
    for duration, count in sorted(duration_count.items()):
        current_count += count
        if current_count >= target_files:
            duration = round(duration)
            print(f"Duration of audio that makes {percentage_files*100}% of the files have that duration is: {duration} seconds")
            return
    


folder_path = '../../datasets/dsl_data/audio'
find_duration(folder_path)
