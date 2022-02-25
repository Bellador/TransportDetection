'''
WORKFLOW WRAPPER THAT COMBINES ALL PARTS OF EXTRACTING TRANSPORTATION MODES FROM VIDEOS WHICH ALSO GET GEOREFERENCED

'''
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import subprocess
# import other python objects from helper scripts
from output_analysis_wrapper import perform_analysis


# 0. load input videos from directory and iterate over them
VIDEOS_PATH = r"./input_videos"
SAVE_PATH = './output'
OCR_SKIP_NTH_FRAME = 5   # only every nth frame is processed via OCR (29.97 frames / second)
for video_index, video in enumerate(os.listdir(VIDEOS_PATH), 1):
    start = time.time()
    # define tracklog and ocrlog filenames for that video
    video_name = video[:-4] + '_' + time.strftime("%H%M%S")
    OUTPUT_VIDEO_FOLDER_PATH = os.path.join(SAVE_PATH, video_name)
    # create a folder with the video name in the output folder that stores all related files
    os.mkdir(OUTPUT_VIDEO_FOLDER_PATH)
    tracklog_filename = f'{time.strftime("%Y%m%d-%H%M%S")}_{video_name}_tracklog.csv'
    ocr_log_filename = f'{time.strftime("%Y%m%d-%H%M%S")}_{video_name}_ocrlog.csv'
    # NEW create new CSV log_file with input video name and ocrlog.csv ending
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_FOLDER_PATH, video)
    TRACKLOG_PATH = os.path.join(OUTPUT_VIDEO_FOLDER_PATH, tracklog_filename)
    OCR_LOG_PATH = os.path.join(OUTPUT_VIDEO_FOLDER_PATH, ocr_log_filename)
    # write header of log fileS
    with open(TRACKLOG_PATH, 'wt', encoding='utf-8') as f:
        f.write('frame_idx;identity;class_name;bbox_top;bbox_left;bbox_w;bbox_h\n')
    with open(OCR_LOG_PATH, 'wt', encoding='utf-8') as f:
        f.write('frame_num;text;certainty;bbox_xmin;bbox_ymin;bbox_xmax;bbox_ymax\n')
    print('***' * 30)
    print(f'------------------- PROCESSING VIDEO {video_index}: {video} ----------------------------------------------')
    INPUT_VIDEO_PATH = os.path.join(VIDEOS_PATH, video)
    # 1.1 execute object_tracker.py for object detection and object tracking
    command = [sys.executable, 'object_tracker.py',
               '--source', f'{INPUT_VIDEO_PATH}',
               '--yolo_weights', 'yolov5/weights/yolov5s.pt',
               '--img-size', '1020',
               '--save-txt',
               '--save-vid',
               '--ocr-skip-nth-frame', f'{OCR_SKIP_NTH_FRAME}',
               '--output-video-path', f'{OUTPUT_VIDEO_PATH}',
               '--track-log', f'{TRACKLOG_PATH}',
               '--ocr-log', f'{OCR_LOG_PATH}'
               ]
    # print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None: # output == '' and  / process.poll == None means process is still alive!
            break
        if output:
            print(f'\r{output.strip()}', end='')
    # 1.2 the tracking and OCR log filenames are stored in ./inference/tmp_output/tmp_log_filenames.txt
    print(f'[*] tracking log: {TRACKLOG_PATH}')
    print(f'[*] OCR log: {OCR_LOG_PATH}')
    # # start complete output analysis
    # perform_analysis(TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video[:-4])
    end = time.time()
    duration_in_hours = round((end - start) / 3600, 2)
    print(f'---------- PROCESSING TIME OF {video_index}: {duration_in_hours} hrs or {round(end-start, 2)} secs -------')
    print('***' * 30)
print('[*] All videos processed. Done.')