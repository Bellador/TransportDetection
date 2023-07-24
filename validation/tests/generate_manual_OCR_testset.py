import os

import numpy as np
import pandas as pd

PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_w_quartiers.csv"

sample_size = 50
seed = 41

df = pd.read_csv(PATH, delimiter=';')
df_total_unique = df['location_name'].unique()

# some statistics
print(f'[+] total unique locations: {len(df_total_unique)}')
# get count of unique location per origin
OCR_location_names = df.loc[df['origin'] == 'OCR', 'location_name'].unique()
print(f'[+] unique OCR locations: {len(OCR_location_names)}')
TEXT_location_names = df.loc[df['origin'] == 'TEXT', 'location_name'].unique()
print(f'[+] unique TEXT locations: {len(TEXT_location_names)}')
# calc overlap between the two unique location sets
overlap = sum([1 for location_name in OCR_location_names if location_name in TEXT_location_names])
overlap_location_names = [location_name for location_name in OCR_location_names if location_name in TEXT_location_names]
print(f'[+] overlap OCR and TEXT: {overlap}')
only_OCR = sum([1 for location_name in OCR_location_names if location_name not in TEXT_location_names])
print(f'[+] locations only detected by OCR: {only_OCR}')
only_TEXT = sum([1 for location_name in TEXT_location_names if location_name not in OCR_location_names])
print(f'[+] locations only detected by TEXT: {only_TEXT}')
print('-' * 30)
# get random sample of OCR for manual verification
df_OCR_unique = df.loc[df['origin'] == 'OCR', ['origin', 'frame_nr', 'fps', 'videoID', 'location_name']].drop_duplicates(subset=['location_name'])
df_OCR_unique_sample = df_OCR_unique.sample(n=sample_size, random_state=seed)
# calc sec (minus offset) into video where OCR detection should be found
df_OCR_unique_sample['sec'] = df_OCR_unique_sample.apply(lambda row: int(row.frame_nr / row.fps) - 2, axis=1)
# add direct video link
df_OCR_unique_sample['primer'] = df_OCR_unique_sample.apply(lambda row: 'https://www.youtube.com/watch?v=' + str(row.videoID) + '&t=' + str(row.sec), axis=1)
# export as csv
export_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\manual_video_analysis\OCR_location_test"
# df_OCR_unique_sample.to_csv(os.path.join(export_PATH, f'{sample_size}_OCR_manual_testset.csv'), sep=';')
print(f'[+] export manual test sample of {sample_size}:\n{export_PATH}')
print('-' * 30)
