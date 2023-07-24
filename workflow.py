'''
WORKFLOW WRAPPER THAT COMBINES ALL PARTS OF EXTRACTING TRANSPORTATION MODES FROM VIDEOS WHICH ALSO GET GEOREFERENCED

'''
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import torch
from pathlib import Path
# import other python objects from helper scripts
from output_analysis_wrapper import start_analysis
# responsible for fetching relevant video IDs and their metadata for further downloaded and processing
sys.path.insert(0, "./YoutubeAPI-Query")
from ytquery_wrapper import ytapi_main
# responsible for downloading relevant youtube videos based on a query
sys.path.insert(0, "./PyTube")
from yt_downloader import ytvideo_download
# import core detection module
from object_tracker import detect

# ---------------------- ADAPT PARAMETERS HERE -------------------------------------------------------------------------
# 1. query YT API for videos and their metadata, text process that data to extract timestamps with potential location names
yt_search_query = 'Paris walk' # query string for the YouTube API based on which videos are searched and downloaded
VIDEO_RESOLUTION = 720 # desired video resolution
SEC_BUFFER = 45 # transport attribution sbuffer, for timestamps its 2x after, for OCR 1x before and after detected location
OCR_PER_SEC = 3  # defines the amount of frames processed with OCR per second, based on the videos respective frame rate
NUMBER_OF_VIDEOS = 'all'  # amount of videos to be downloaded, str 'all' to fetch everything
# can specify a concrete list of videoIDs, if supplied only these videos will be considered
with open('./specific_videoids_to_process.txt') as file:
    SPECIFIC_VIDEOIDS_TO_PROCESS = [line.rstrip() for line in file]
# SPECIFIC_VIDEOIDS_TO_PROCESS = ['WAmYjDIZ9gQ', 'zpEPbl-4DaY']
# only download and process YouTube videos in which manually added timestamps were detected
PROCESS_VIDEOS_WITH_TIMESTAMPS = True
# define Project name base on search query
PROJECT_NAME = yt_search_query.replace(' ', '_')
ROOT_PATH = './validation/workflow_performance/'
# ROOT_PATH = '/mnt' # directory that will contain the final output of the pipeline
# ----------------------------------------------------------------------------------------------------------------------
complete_start = time.time()

OUTPUT_PATH = os.path.join(ROOT_PATH, 'output', PROJECT_NAME)
# # # returns video IDs and video metadata json
videoIDs_to_process, YTQUERY_FILEPATH_JSON = ytapi_main(OUTPUT_PATH,
                                                        query=yt_search_query,
                                                        NUMBER_OF_VIDEOS=NUMBER_OF_VIDEOS,
                                                        PROCESS_VIDEOS_WITH_TIMESTAMPS=PROCESS_VIDEOS_WITH_TIMESTAMPS,
                                                        PROCESS_SPECIFIC_IDS=SPECIFIC_VIDEOIDS_TO_PROCESS)

if NUMBER_OF_VIDEOS != 'all':
    videoIDs_to_process = videoIDs_to_process[:NUMBER_OF_VIDEOS]
    print(f'[*] will fetch first {NUMBER_OF_VIDEOS} videos')

# 1. download needed YT videos
if not SPECIFIC_VIDEOIDS_TO_PROCESS:
    yt_video_links = [f'https://www.youtube.com/watch?v={videoID}' for videoID in videoIDs_to_process]
else:
    yt_video_links = [f'https://www.youtube.com/watch?v={videoID}' for videoID in SPECIFIC_VIDEOIDS_TO_PROCESS]

INPUT_VIDEOS_PATH = ytvideo_download(yt_video_links, PROJECT_NAME, ROOT_PATH,
                                     MAX_RESOLUTION=f'{VIDEO_RESOLUTION}p',
                                     MIME_TYPE='video/mp4',
                                     ONLY_VIDEO=False)
# INPUT_VIDEOS_PATH = os.path.join(ROOT_PATH, 'input_videos', PROJECT_NAME)
# 1.1 create dir structure to store the output from the video analysis
VIDEODATA_OUTPUT_FOLDERS = os.path.join(OUTPUT_PATH, 'yt_videos')
Path(VIDEODATA_OUTPUT_FOLDERS).mkdir(parents=True, exist_ok=True)
# ----------------------------------------------------------------------------------------------------------------------

# 2. load input videos from directory and iterate over them
# create file that stores video specific metadata for all videos
video_metadata_log_filename = f'{time.strftime("%Y%m%d-%H%M%S")}_videos_metadata.csv'
VIDEOS_METADATA_PATH = os.path.join(OUTPUT_PATH, video_metadata_log_filename)
with open(VIDEOS_METADATA_PATH, 'wt', encoding='utf-8') as f:
    f.write(f'video_name;fps;width;height;ocr_nth_frame\n')
for video_index, video in enumerate(os.listdir(INPUT_VIDEOS_PATH), 1):
    start = time.time()
    # define tracklog and OCR log filenames for that video
    video_name = video[:-4] # + '_' + time.strftime("%H%M%S")
    OUTPUT_VIDEO_FOLDER_PATH = os.path.join(VIDEODATA_OUTPUT_FOLDERS, video_name)
    # create a folder with the video name in the output folder that stores all related files
    Path(OUTPUT_VIDEO_FOLDER_PATH).mkdir(parents=False, exist_ok=True)
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
    print(f'[*] ------------------- PROCESSING VIDEO {video_index}: {video} ------------------------------------------')
    INPUT_VIDEO_PATH = os.path.join(INPUT_VIDEOS_PATH, video)
    # 3.1 execute object_tracker.py for object detection and object tracking
    with torch.no_grad():
        detect(
            yolo_weights='yolov5/weights/yolov5s.pt',
            deep_sort_weights='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
            config_deepsort='deep_sort_pytorch/configs/deep_sort.yaml',
            conf_thres=0.4,
            iou_thres=0.5,
            fourcc='mp4v',
            device=0,
            augment=False,
            video_name=video_name,
            source=INPUT_VIDEO_PATH,
            img_size=VIDEO_RESOLUTION,
            classes=None, # filter by class: --class 0, or --class 16 17; class 0 is person, 1 is bicycle, 2 is car... 79 is oven
            save_txt=True,
            save_vid=False,
            show_vid=False,
            agnostic_nms=False,
            ocr_per_sec=OCR_PER_SEC,
            output_video_path=OUTPUT_VIDEO_PATH,
            track_log=TRACKLOG_PATH,
            ocr_log=OCR_LOG_PATH,
            videos_metadata_log=VIDEOS_METADATA_PATH
        )
    # 3.2 the tracking and OCR log filenames are stored in ./inference/tmp_output/tmp_log_filenames.txt
    print(f'[*] tracking log: {TRACKLOG_PATH}')
    print(f'[*] OCR log: {OCR_LOG_PATH}')
    end = time.time()
    duration_in_hours = round((end - start) / 3600, 2)
    print(f'---------- PROCESSING TIME OF {video_index}: {video} {duration_in_hours} hrs or {round(end-start, 2)} secs -------')
    print('***' * 30)
# 4. after videos are processed perform the output analysis inc. figures and plots
start_analysis(OUTPUT_PATH, VIDEODATA_OUTPUT_FOLDERS, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON, SEC_BUFFER)
print('[*] All videos processed. Done.')
complete_end = time.time()
complete_duration_in_hours = round((complete_end - complete_start) / 3600, 2)
print('---' * 30)
print(f'---------- COMPLETE PROCESSING TIME: {complete_duration_in_hours} hrs or {round(complete_end-complete_start, 2)} secs -------')
print('---' * 30)