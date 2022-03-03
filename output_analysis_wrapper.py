# import other python objects from helper scripts
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from analyse_tracklog import total_objects_per_class
from analyse_tracklog import total_transportation_modes
from geoparsing import str_prefiltering
from geoparsing import geoparsing
from output_plotting import map_plotting
from output_plotting import figure_plotting
from output_plotting import output_location_statistics
from output_plotting import all_locations_plot


def analysis_core(TRACKLOG_PATH, OCR_LOG_PATH, VIDEOS_STORE_PATH, VIDEO_PATH, video_name):
    # 1. perform the transportation mode detection on the tracking log
    total_objects_dict = total_objects_per_class(TRACKLOG_PATH, VIDEO_PATH)
    total_modes_dict = total_transportation_modes(TRACKLOG_PATH, VIDEO_PATH)
    # 2. perform geoparsing on the ocr tracking log
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    location_names_df = geoparsing(frame_dict, VIDEO_PATH, video_name)
    # location_names_df = None
    # 3. generate plot encompassing all video frames and the detected objects, transportation modes and location names
    figure_plotting(total_objects_dict, total_modes_dict, location_names_df, VIDEO_PATH, video_name)
    map_plotting(total_objects_dict, total_modes_dict, location_names_df, VIDEO_PATH, video_name)
    CSV_LOCATION_STATISTCS_OUTPUT_PATH = output_location_statistics(total_objects_dict, total_modes_dict, location_names_df, VIDEOS_STORE_PATH)
    return CSV_LOCATION_STATISTCS_OUTPUT_PATH


def perfrom_analysis_from_pickle(PICKLE_PATH, TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video_name):
    # 1. perform the transportation mode detection on the tracking log
    total_objects_dict = total_objects_per_class(TRACKLOG_PATH, OUTPUT_VIDEO_FOLDER_PATH)
    total_modes_dict = total_transportation_modes(TRACKLOG_PATH, OUTPUT_VIDEO_FOLDER_PATH)
    # 3. perform geoparsing on the ocr tracking log
    location_names_df = pd.read_pickle(PICKLE_PATH)
    # location_names_df = None
    # 4. generate plot encompassing all video frames and the detected objects, transportation modes and location names
    figure_plotting(total_objects_dict, total_modes_dict, location_names_df, OUTPUT_VIDEO_FOLDER_PATH, video_name)
    map_plotting(total_objects_dict, total_modes_dict, location_names_df, OUTPUT_VIDEO_FOLDER_PATH, video_name)
    output_location_statistics(total_objects_dict, total_modes_dict, location_names_df, OUTPUT_VIDEO_FOLDER_PATH)
    # for isolated testing of output analysis

def start_analysis(VIDEOS_STORE_PATH):
    for video_index, video_name in enumerate(os.listdir(VIDEOS_STORE_PATH), 1):
        # assign all necessary file paths for the following functions
        print(f'[*] {video_index} - CREATING OUTPUT PLOTS FOR {video_name}')
        TRACKLOG_PATH = None
        OCR_LOG_PATH = None
        VIDEO_PATH = os.path.join(VIDEOS_STORE_PATH, video_name)
        for file in os.listdir(VIDEO_PATH):
            if file.endswith('ocrlog.csv'):
                OCR_LOG_PATH = os.path.join(VIDEO_PATH, file)
            elif file.endswith('tracklog.csv'):
                TRACKLOG_PATH = os.path.join(VIDEO_PATH, file)
        CSV_LOCATION_STATISTCS_OUTPUT_PATH = analysis_core(TRACKLOG_PATH, OCR_LOG_PATH, VIDEOS_STORE_PATH,
                                                              VIDEO_PATH, video_name)
        # create detected object statistics over all location in an interactive graph
        all_locations_plot(CSV_LOCATION_STATISTCS_OUTPUT_PATH, VIDEOS_STORE_PATH)
        print('--' * 30)

if __name__ == '__main__':
    VIDEOS_STORE_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\all_videos"
    all_locations_plot(r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\all_videos\location_statistics_5000_buffer.csv", VIDEOS_STORE_PATH)
    # for video_index, video_name in enumerate(os.listdir(VIDEOS_STORE_PATH), 1):
    #     # assign all necessary file paths for the following functions
    #     print(f'[*] {video_index} - CREATING OUTPUT PLOTS FOR {video_name}')
    #     TRACKLOG_PATH = None
    #     OCR_LOG_PATH = None
    #     VIDEO_PATH = os.path.join(VIDEOS_STORE_PATH, video_name)
    #     for file in os.listdir(VIDEO_PATH):
    #         if file.endswith('ocrlog.csv'):
    #             OCR_LOG_PATH = os.path.join(VIDEO_PATH, file)
    #         elif file.endswith('tracklog.csv'):
    #             TRACKLOG_PATH = os.path.join(VIDEO_PATH, file)
    #     CSV_LOCATION_STATISTCS_OUTPUT_PATH = analysis_core(TRACKLOG_PATH, OCR_LOG_PATH, VIDEOS_STORE_PATH, VIDEO_PATH, video_name)
    #     # create detected object statistics over all location in an interactive graph
    #     all_locations_plot(CSV_LOCATION_STATISTCS_OUTPUT_PATH, VIDEOS_STORE_PATH)
    #     print('--' * 30)


    # TRACKLOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\20220223-102146_Walking_in_GENEVA_4K_Switzerland_102146_tracklog.csv"
    # OCR_LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\20220223-102146_Walking_in_GENEVA_4K_Switzerland_102146_ocrlog.csv"
    # OUTPUT_VIDEO_FOLDER_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\analysis_output"
    # video_name = 'Walking_in_GENEVA_4K_Switzerland'
    # df_pickle_filepath = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\Walking_in_GENEVA_4K_Switzerland_102146\analysis_output\20220228_122846_Walking_in_GENEVA_4K_Switzerland_pickle.pkl"
    # # perform_analysis(TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video_name)
    # perfrom_analysis_from_pickle(df_pickle_filepath, TRACKLOG_PATH, OCR_LOG_PATH, OUTPUT_VIDEO_FOLDER_PATH, video_name)
