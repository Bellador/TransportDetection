import pandas as pd


PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\output\Paris_walk\location_statistics_45s_buffer.csv"

df = pd.read_csv(PATH, sep=';', encoding='utf-8')

# filter unique locations

df_new = df.loc[:, ['location_name', 'origin']]

OCR_len = df_new.loc[df_new['origin'] == 'OCR',:]
TXT_len = df_new.loc[df_new['origin'] == 'TEXT',:]

# drop duplicates
OCR_len = OCR_len.drop_duplicates(['location_name'])
TXT_len = TXT_len.drop_duplicates(['location_name'])

# exclusives
ocr_l = list(OCR_len['location_name'])
txt_l = list(TXT_len['location_name'])
OCR_ex = 0
TXT_ex = 0

for location in ocr_l:
    if location not in txt_l:
        OCR_ex += 1

for location in txt_l:
    if location not in ocr_l:
        TXT_ex += 1

# overlap
overlap = 0

for location in ocr_l:
    if location in txt_l:
        overlap += 1

pass


