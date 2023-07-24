import os
import json
import time
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
pd.options.mode.chained_assignment = None  # default='warn'
# import other python objects from helper scripts
from analyse_tracklog import total_objects_per_class, total_transportation_modes
from geoparsing import geoparsing, str_prefiltering
from output_plotting import map_plotting, output_location_statistics, all_locations_plot, all_locations_map, location_names_by_frequency_plot, temporal_analysis_per_location


def analysis_core(OUTPUT_PATH, TRACKLOG_PATH, OCR_LOG_PATH, VIDEODATA_OUTPUT_FOLDERS, videodata_output_folder, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON, SEC_BUFFER):
    VIDEODATA_OUTPUT_FOLDER_PATH = os.path.join(VIDEODATA_OUTPUT_FOLDERS, videodata_output_folder)
    # 1. perform the transportation mode detection on the tracking log
    total_objects_dict = total_objects_per_class(TRACKLOG_PATH, VIDEODATA_OUTPUT_FOLDER_PATH)
    total_modes_dict = total_transportation_modes(TRACKLOG_PATH, VIDEODATA_OUTPUT_FOLDER_PATH)
    # 2. perform geoparsing on the OCR tracking log
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    location_names_df = geoparsing(frame_dict, videodata_output_folder, YTQUERY_FILEPATH_JSON, video_metadata_dict=get_video_metadata(videodata_output_folder, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON))
    # define output statistic CSV filename
    statistics_filename = f'location_statistics_{SEC_BUFFER}s_buffer.csv'
    CSV_LOCATION_STATISTCS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, statistics_filename)
    # 2.1 check if valid result was returned i.e df not empty
    if location_names_df is not None:
        # 3. find relevant locations based on duplicates and their time of appearance
        relevant_location_names_df = create_relevant_location_names_df(location_names_df, secs_diff=2*SEC_BUFFER)
        # 4. aggregate transportation modes per location through a given timely buffer
        output_location_statistics(total_objects_dict, total_modes_dict, relevant_location_names_df, CSV_LOCATION_STATISTCS_OUTPUT_PATH, SEC_BUFFER=SEC_BUFFER)
        # 5. generate plot encompassing all video frames and the detected objects, transportation modes and location names
        try:
            map_plotting(CSV_LOCATION_STATISTCS_OUTPUT_PATH, VIDEODATA_OUTPUT_FOLDER_PATH, videodata_output_folder)
        except Exception as e:
            print(f'[!] map plotting error: {e}. Skipping.')
        # figure_plotting(total_objects_dict, total_modes_dict, relevant_location_names_df, VIDEODATA_OUTPUT_FOLDER_PATH, videodata_output_folder)
    return CSV_LOCATION_STATISTCS_OUTPUT_PATH


def get_video_metadata(video_data_output_folder, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON, data_to_extract=['fps', 'publishedAt', 'extractedDates', 'extractedTimes', 'goodWeather', 'badWeather']):
    '''
    function that retrieves video relevant information.
    Data like FPS, HEIGHT and WIDTH is extracted during the object_tracker.py and stored to <datetime>_videos_metadata.csv
    Other relevant information like publication date, and extracted timestamps, dates have to be retrieved from the youtube API
    file <YoutbeAPIQuery>_videoInfoEExtracted.json!
    :param video_data_output_folder:
    :param VIDEOS_METADATA_PATH:
    :param data_to_extract:
    :return:
    '''
    # storage dict that will be returned
    metadata_dict = {}
    # load video metadata to extract fps to calculate frame difference requirement between locations of the same name
    videos_metadata_csv_df = pd.read_csv(VIDEOS_METADATA_PATH, delimiter=';')
    # load video relevant metadata retrieved from Youtube API by loading json
    with open(YTQUERY_FILEPATH_JSON, 'rt', encoding='utf-8') as f:
        videos_metadata_API_dict = json.load(f)
        video_metadata_API_data = videos_metadata_API_dict[video_data_output_folder]
    # extracted requested data of corresponding video
    for data in data_to_extract:
        try:
            # 1st: check if data is in metadata.csv holding fps, height, width etc.
            if data in videos_metadata_csv_df.columns:
                metadata_dict[data] = videos_metadata_csv_df.loc[videos_metadata_csv_df['video_name'] == video_data_output_folder, [data]].iloc[0, 0]
            # 2nd: check <YoutbeAPIQuery>_videoInfoExtracted.json
            else:
                value = video_metadata_API_data[data]
                # check if weather relevant data and reduce to binary value (1 present, 0 absent)
                if data == 'goodWeather' or data == 'badWeather':
                    # if list is not empty, meaning there is a weather related term
                    if value:
                        metadata_dict[data] = 1
                    else:
                        metadata_dict[data] = 0
                else:
                    # check if type is list to cast to str otherwise issues with df since type: 'list' is unhashable
                    if isinstance(value, list):
                        #check if not empty
                        if value:
                            metadata_dict[data] = ';'.join(value)
                        else:
                            metadata_dict[data] = None
                    else:
                        metadata_dict[data] = value
        except Exception as e:
            print(f'[!] Metadata extraction of key: {data} -- {e}')
            continue
    return metadata_dict


def create_relevant_location_names_df(location_names_df, secs_diff=60):
    """
    all detected locations from geoparsing.py are parsed. 'Unique locations' are considered to be locations (even once of the same name)
    if they were detected with at least a given time or frame nr. between each other.
    :return:
    """

    # iterate over df and find locations that fulfil requirement
    processed_locations = {}
    index_todrop = []
    for location_index, row in location_names_df.iterrows():
        frame = row['frame_nr']
        location_name = row['location_name']
        # extract fps, should be the same for all locations of the same video, but is queried individually nevertheless
        fps = row['fps']
        frames_diff = fps * secs_diff
        if location_name in processed_locations.keys():
           previous_frame = processed_locations[location_name]
           current_frame = frame
           # decides if already existing location will be dropped
           if not (current_frame - previous_frame) > frames_diff:
               index_todrop.append(location_index)
           processed_locations[location_name] = current_frame

        else:
            processed_locations[location_name] = frame

    # create new df through dropping of found location index
    relevant_locations_df = location_names_df.drop(index_todrop)
    return relevant_locations_df

def filter_for_spatial_outliers(CSV_LOCATION_STATISTICS_OUTPUT_PATH, forms_of_transportation, geo_filter = Point([260835, 6255558]).buffer(1000000)):
    '''
    - filter out geometries outside the target area
    - add public, active and motorised mobility counts to the df

    :param CSV_LOCATION_STATISTICS_OUTPUT_PATH:
    :param geo_filter:
    :return:
    '''
    # load df and convert to gdf
    locations_df = pd.read_csv(CSV_LOCATION_STATISTICS_OUTPUT_PATH, sep=';')
    # drop invalid geom rows before conversion to geodataframe
    locations_df = locations_df.replace(to_replace='None', value=np.nan)
    locations_df = locations_df[locations_df['geo'].notna()]
    locations_df['geo'] = locations_df['geo'].apply(wkt.loads)
    locations_gdf = gpd.GeoDataFrame(locations_df, crs='epsg:3857', geometry='geo')
    # set spatial extend of axis by excluding all geoms outside target area
    paris_geom = geo_filter
    locations_gdf = locations_gdf[locations_gdf['geo'].map(lambda x: x.within(paris_geom))]
    locations_gdf['year'] = locations_gdf['publishedAt'].apply(lambda x: time.strptime(x, '%Y-%m-%dT%H:%M:%SZ').tm_year)
    # create new class name column and fill it with default value (to ensure dtype int64 instead of float64)
    for transport_form in forms_of_transportation.keys():
        if transport_form not in locations_gdf.columns:
            locations_gdf[transport_form] = 0
    # add public, motorised and active transportation counts
    for index, row in locations_gdf.iterrows():
        for transport_form, class_names in forms_of_transportation.items():
            transport_form_counter = 0
            for class_name in class_names:
                transport_form_counter += row[class_name]
            locations_gdf.loc[index, transport_form] = transport_form_counter
    # new output file
    FILTERED_CSV_STATISTICS_PATH = CSV_LOCATION_STATISTICS_OUTPUT_PATH[:-4] + '_FILTERED.csv'
    locations_gdf.to_csv(FILTERED_CSV_STATISTICS_PATH, sep=';', encoding='utf-8')
    return FILTERED_CSV_STATISTICS_PATH


def start_analysis(OUTPUT_PATH, VIDEODATA_OUTPUT_FOLDERS, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON, SEC_BUFFER):
    forms_of_transportation = {
        "active": ["pedestrian", "bicycle", "dogwalker"],
        "motorised": ["car", "motorcycle", "truck"],
        "public": ["bus", "train", "boat"]
    }
    # create folder for generated plots
    PLOTS_PATH = os.path.join(OUTPUT_PATH, 'plots')
    if not os.path.isdir(PLOTS_PATH):
        print(f'[+] Created folder to store plots')
        os.mkdir(PLOTS_PATH)

    for video_index, videodata_output_folder in enumerate(os.listdir(VIDEODATA_OUTPUT_FOLDERS), 1):
        # assign all necessary file paths for the following functions
        print('---' * 30)
        print(f'[*] {video_index} - CREATING OUTPUT PLOTS FOR {videodata_output_folder}')
        TRACKLOG_PATH = None
        OCR_LOG_PATH = None
        VIDEO_DATA_OUTPUT_PATH = os.path.join(VIDEODATA_OUTPUT_FOLDERS, videodata_output_folder)
        # exclude metadata file
        if not videodata_output_folder.endswith('.csv'):
            for file in os.listdir(VIDEO_DATA_OUTPUT_PATH):
                if file.endswith('ocrlog.csv'):
                    OCR_LOG_PATH = os.path.join(VIDEO_DATA_OUTPUT_PATH, file)
                elif file.endswith('tracklog.csv'):
                    TRACKLOG_PATH = os.path.join(VIDEO_DATA_OUTPUT_PATH, file)
        else:
            continue
        CSV_LOCATION_STATISTICS_OUTPUT_PATH = analysis_core(OUTPUT_PATH, TRACKLOG_PATH, OCR_LOG_PATH, VIDEODATA_OUTPUT_FOLDERS,
                                                            videodata_output_folder, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON, SEC_BUFFER)
    # check if location statistics csv was created
    if os.path.isfile(CSV_LOCATION_STATISTICS_OUTPUT_PATH):
        # pre-defined filter for region of interest geom to eliminate clear spatial outliers
        FILTERED_CSV_LOCATION_STATISTICS_PATH = filter_for_spatial_outliers(CSV_LOCATION_STATISTICS_OUTPUT_PATH, forms_of_transportation)
        # create detected object statistics over all location in an interactive graph
        all_locations_plot(FILTERED_CSV_LOCATION_STATISTICS_PATH, PLOTS_PATH, forms_of_transportation)
        # mosaic_plot_locations(CSV_LOCATION_STATISTICS_OUTPUT_PATH, OUTPUT_PATH, forms_of_transportation)
        # map all locations on a map (with a predefined filter geom to eliminate clear spatial outliers)
        all_locations_map(FILTERED_CSV_LOCATION_STATISTICS_PATH)
        location_names_by_frequency_plot(FILTERED_CSV_LOCATION_STATISTICS_PATH, PLOTS_PATH, min_frequency=3)
        temporal_analysis_per_location(FILTERED_CSV_LOCATION_STATISTICS_PATH, forms_of_transportation, PLOTS_PATH, min_frequency=6)

    print('--' * 30)

if __name__ == '__main__':
    yt_search_query = 'Paris walk'
    ROOT_PATH = '/mnt'
    PROJECT_NAME = yt_search_query.replace(' ', '_')
    OUTPUT_PATH = os.path.join(ROOT_PATH, 'output', PROJECT_NAME)
    VIDEODATA_OUTPUT_FOLDERS = os.path.join(OUTPUT_PATH, 'yt_videos')
    VIDEOS_METADATA_PATH = "/mnt/output/Paris_walk/20221118-134355_videos_metadata.csv"
    YTQUERY_FILEPATH_JSON = "/mnt/output/Paris_walk/ytapi_data/Paris_walk_videoInfoExtracted.json"
    # OUTPUT_PATH = ""

    VIDEODATA_OUTPUT_FOLDERS = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\yt_videos"
    videodata_output_folder = "AWMpvQhFKzA"
    # VIDEOS_METADATA_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\20221118-134355_videos_metadata.csv"
    # YTQUERY_FILEPATH_JSON = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\ytapi_data\Paris_walk_videoInfoExtracted.json"
    forms_of_transportation = {
        "active": ["pedestrian", "bicycle", "dogwalker"],
        "motorised": ["car", "motorcycle", "truck"],
        "public": ["bus", "train", "boat"]
    }
    SEC_BUFFER = 45

    # TRACKLOG_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance\w7yTArIaI_g\validation_w7yTArIaI_g_tracklog.csv"D
    TRACKLOG_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance\w7yTArIaI_g\validation_w7yTArIaI_g_tracklog.csv"
    VIDEODATA_OUTPUT_FOLDER_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\validation\workflow_performance"
    total_objects_dict = total_objects_per_class(TRACKLOG_PATH, VIDEODATA_OUTPUT_FOLDER_PATH)
    total_modes_dict = total_transportation_modes(TRACKLOG_PATH, VIDEODATA_OUTPUT_FOLDER_PATH)
    pass
