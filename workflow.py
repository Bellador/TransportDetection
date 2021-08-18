'''
WORKFLOW WRAPPER THAT COMBINES ALL PARTS OF EXTRACTING TRANSPORTATION MODES FROM VIDEOS WHICH ALSO GET GEOREFERENCED

'''
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import subprocess
# import other python objects from helper scripts
from analyse_tracklog import total_objects_per_class
from analyse_tracklog import total_transportation_modes
from geoparsing import str_prefiltering
from geoparsing import geoparsing


# 0. load input videos from directory and iterate over them
VIDEOS_PATH = r"./input_videos"
for video_index, video in enumerate(os.listdir(VIDEOS_PATH), 1):
    print('***' * 30)
    print(f'------------------- PROCESSING VIDEO {video_index}: {video} ----------------------------------------------')
    VIDEO_PATH = os.path.join(VIDEOS_PATH, video)
    # 1.1 execute object_tracker.py for object detection and object tracking
    command = [sys.executable, 'object_tracker.py',
               '--source', f'{VIDEO_PATH}',
               '--yolo_weights', 'yolov5/weights/yolov5s.pt',
               '--img-size', '640',
               '--save-txt',
               '--save-vid']
    # print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None: # output == '' and  / process.poll == None means process is still alive!
            break
        if output:
            print(f'\r{output.strip()}', end='')
    # 1.2 the tracking and OCR log filenames are stored in ./inference/tmp_output/tmp_log_filenames.txt
    tmp_log_filenames_path = r'./inference/tmp_output/tmp_log_filenames.txt'
    files = []
    with open(tmp_log_filenames_path, 'rt', encoding='utf-8') as f:
        for line in f:
            files.append(line)
    TRACKLOG_PATH = files[0][:-1] # cut off the linebreak
    OCR_LOG_PATH = files[1][:-1]
    print(f'[*] tracking log: {TRACKLOG_PATH}')
    print(f'[*] OCR log: {OCR_LOG_PATH}')
    # 2. perfrom the transportation mode detection on the tracking log
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog']
    source_str = VIDEO_PATH.split('\\')[-1][:-4]
    total_objects_per_class(TRACKLOG_PATH, classnames_to_consider, source_str)
    # highest_ocr_confidence(OCR_LOG_PATH)
    total_transportation_modes(TRACKLOG_PATH, source_str)
    # 3. perform geoparsing on the ocr tracking log
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    df = geoparsing(frame_dict, TRACKLOG_PATH, source_str)
    print('[*] All videos processed. Done.')