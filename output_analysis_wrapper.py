# import other python objects from helper scripts
from analyse_tracklog import total_objects_per_class
from analyse_tracklog import total_transportation_modes
from geoparsing import str_prefiltering
from geoparsing import geoparsing
from modes_per_frame_location_plot import plotting


def perform_analysis(TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video_name):
    # 1. perform the transportation mode detection on the tracking log
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog']
    total_objects_dict = total_objects_per_class(TRACKLOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, classnames_to_consider)
    total_modes_dict = total_transportation_modes(TRACKLOG_PATH, OUTPUT_VIDEO_FOLDER_PATH)
    # 3. perform geoparsing on the ocr tracking log
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    location_names_df = geoparsing(frame_dict, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH)
    # location_names_df = None
    # 4. generate plot encompassing all video frames and the detected objects, transportation modes and location names
    plotting(total_objects_dict, total_modes_dict, location_names_df, OUTPUT_VIDEO_FOLDER_PATH, video_name)


# for isolated testing of output analysis
if __name__ == '__main__':
    TRACKLOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\20220223-102146_Walking_in_GENEVA_4K_Switzerland_102146_tracklog.csv"
    OCR_LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\20220223-102146_Walking_in_GENEVA_4K_Switzerland_102146_ocrlog.csv"
    OUTPUT_VIDEO_FOLDER_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\analysis_output"
    video_name = 'Walking_in_GENEVA_4K_Switzerland'
    perform_analysis(TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video_name)
