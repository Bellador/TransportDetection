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

videoID = 'w7yTArIaI_g'
VIDEO_RESOLUTION = 1280
INPUT_VIDEO_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance\w7yTArIaI_g\validation_w7yTArIaI_g.mp4"
OUTPUT_VIDEO_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance\w7yTArIaI_g\w7yTArIaI_g_objectdet.mp4"
TRACKLOG_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance\w7yTArIaI_g\validation_w7yTArIaI_g_tracklog_test.csv"
VIDEOS_METADATA_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\20221118-134355_videos_metadata.csv"
# 3.1 execute object_tracker.py for object detection and object tracking
with torch.no_grad():
    detect(
        yolo_weights='yolov5/weights/yolov5s.pt',
        deep_sort_weights='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
        config_deepsort='deep_sort_pytorch/configs/deep_sort.yaml',
        conf_thres=0.4,
        iou_thres=0.5,
        fourcc='mp4v',
        device='cpu',
        augment=False,
        video_name='video_name',
        source=INPUT_VIDEO_PATH,
        img_size=VIDEO_RESOLUTION,
        classes=2, # filter by class: --class 0, or --class 16 17; class 0 is person, 1 is bicycle, 2 is car... 79 is oven
        save_txt=True,
        save_vid=True,
        show_vid=False,
        agnostic_nms=False,
        ocr_per_sec=1,
        output_video_path=OUTPUT_VIDEO_PATH,
        track_log=TRACKLOG_PATH,
        ocr_log=None,
        videos_metadata_log=VIDEOS_METADATA_PATH
    )