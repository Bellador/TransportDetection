import re
import regex as reg
import csv
import json
import time
import hdbscan
import datetime
import pandas as pd
import geopandas as gpd
from itertools import product
from db_querier import DbQuerier
from collections import defaultdict
from shapely.geometry import MultiLineString, LineString, Point, Polygon

# csv.field_size_limit(sys.maxsize)

# pyproj fix 'no database context specified'
# os.environ["PROJ_LIB"] = r"C:\Users\mhartman\Anaconda3\envs\transportation_Yolov5_env\Library\share\proj"

def str_polishing(text, special_chars):
    # remove curly brackets at this step
    polished_text = ''
    for char in text:
        if char not in special_chars:
            polished_text = polished_text + char
    # remove leading and trailing white space
    polished_text = polished_text.strip()
    return polished_text

def str_prefiltering(OCR_LOG_PATH, confidence_th = 0.0, min_str_length = 10, max_nr_special_chars=1):
    # special characters that are a sign of artifacts during the ocr
    frame_dict = defaultdict(lambda: {'string_list': [], 'bbox_xmin': [], 'bbox_ymin': [], 'bbox_xmax': [], 'bbox_ymax': []})
    special_chars = "\"][()|{}_~€$!?%&+,;:~><*@¦=#^£\/´`'\''"
    # open and iterate over tracking log
    with open(OCR_LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE) #, quotechar='|'
        # skip header
        next(reader)
        for index, line in enumerate(reader):
            frame_num = line[0]
            text = line[1]
            bbox_xmin = line[3]
            bbox_ymin = line[4]
            bbox_xmax = line[5]
            bbox_ymax = line[6]
            try:
                confidence = round(float(line[2]), 2)
            except Exception as e:
                pass
                # print(f'[!] Geoparsing Error: {e}')
            # filter steps on pure string
            if confidence >= confidence_th:
                count_special_chars = sum([1 for c in text if c in special_chars])
                if count_special_chars <= max_nr_special_chars:
                    polished_text = str_polishing(text, special_chars)
                    # check if only numbers
                    if not polished_text.isdigit():
                        try:
                            # print(f'{frame_num} - {confidence}: {polished_text}')
                            frame_dict[frame_num]['string_list'].append(polished_text)
                            frame_dict[frame_num]['bbox_xmin'].append(round(float(bbox_xmin)))
                            frame_dict[frame_num]['bbox_ymin'].append(round(float(bbox_ymin)))
                            frame_dict[frame_num]['bbox_xmax'].append(round(float(bbox_xmax)))
                            frame_dict[frame_num]['bbox_ymax'].append(round(float(bbox_ymax)))
                        except Exception as e:
                            print(f'[!] Geoparsing Error - Digit: {e}')
    # create independent copy of frame_dict
    frame_dict_complete = frame_dict
    # initialise HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    # keep single frame strings as well as a combination of the strings from the same frame
    # based on spatial closeness of strings (use center point of bbox for reference)
    for frame_nr, element in frame_dict.items():
        string_list = element['string_list']
        # build df based on bboxes in for the same video frame
        df = pd.DataFrame(list(zip(element['bbox_xmin'], element['bbox_ymin'], element['bbox_xmax'], element['bbox_ymax'])), columns=['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax'])
        # check if lenght of df exceeds 1
        if len(df.index) > 1:
            # apply HDBSCAN to bbox coordinates to find spatially clustered strings detected by the OCR
            clusterer.fit(df)
            cluster_labels = clusterer.labels_
            # determine number of clusters
            unique_labels = set(cluster_labels)
            nr_clusters = len(unique_labels)
            if -1 in unique_labels:
                nr_clusters = nr_clusters - 1
            # add list to store concatenated strings to be added at the end
        else:
            cluster_labels = [0]
            nr_clusters = 1
        conc_strings = []
        # iterate over the labels and concatinate clustered strings based on their index in the labels list and their cluster nr
        for cluster_nr in range(nr_clusters):
            cluster_indexes = [index for index, label in enumerate(cluster_labels) if label == cluster_nr and label != -1]
            # get OCR strings in same cluster
            strings_in_cluster_original = [string_list[index] for index in cluster_indexes]
            strings_in_cluster = strings_in_cluster_original.copy()
            # add all possible combinations of strings inside the cluster
            for e1, e2 in product(strings_in_cluster, strings_in_cluster):
                if e1 != e2:
                    string_mutation = e1 + ' ' + e2
                    if string_mutation not in strings_in_cluster:
                        strings_in_cluster.append(e1 + ' ' + e2)
            # concatenate all strings in same cluster
            combined_string = ' '.join(strings_in_cluster_original)
            if combined_string not in strings_in_cluster:
                strings_in_cluster.append(combined_string)
            conc_strings += strings_in_cluster
        # additionally, add a combination of all strings from the same frame
        conc_strings.append(' '.join(string_list))
        string_list += conc_strings
        frame_dict_complete[frame_nr]['string_list'] = string_list
    # check minimum length of all final strings
    for frame_nr, element in frame_dict_complete.items():
        filtered_string_list = [string for string in element['string_list'] if len(string) >= min_str_length]
        frame_dict_complete[frame_nr]['string_list'] = filtered_string_list

    return frame_dict_complete

def trans_to_stringobj(str_line):
    '''
    transform postgres geometry output which is a literal string into a stringified input for LineString or MultiLineString object
    e.g. "LINESTRING(6.1467406 46.2011719,6.1465602 46.2010601)" -> LineString([[0, 0], [1, 0], [1, 1]])
    e.g. A sequence of line-like coordinate sequences or objects that provide the numpy array interface, including instances of LineString.
    :return:
    '''
    obj_type = None
    try:
        if re.search(r'^MULTILINESTRING', str_line):
            eval_str = str_line[15:].replace(' ', ',')
            obj_type = 'MULTILINESTRING'
        elif re.search(r'^LINESTRING', str_line):
            eval_str = str_line[10:].replace(' ', ',')
            obj_type = 'LINESTRING'
        elif re.search(r'^POINT', str_line):
            eval_str = str_line[5:].replace(' ', ',')
            obj_type = 'POINT'
        elif re.search(r'^POLYGON', str_line):
            eval_str = str_line[7:].replace(' ', ',')
            obj_type = 'POLYGON'
        else:
            print(f'[!] Unknown geometry object: {str_line.split("(")[0]}.\n[!] Returning Point object.')
            return None
        geom_obj = eval(eval_str)
    except Exception as e:
        print(f'[!] Error: {e}.\n[!] Exiting...')
        exit()


    try:
        if obj_type == 'MULTILINESTRING':
            storage = []
            for linestring_obj in geom_obj:
                linestring = []
                for point_pair in zip(linestring_obj[::2], linestring_obj[1::2]):
                    point_x = point_pair[0]
                    point_y = point_pair[1]
                    linestring.append([float(point_x), float(point_y)])
                storage.append(LineString(linestring))
            geom_str = MultiLineString(storage)
        elif obj_type == 'LINESTRING':
            linestring = []
            for point_pair in zip(geom_obj[::2], geom_obj[1::2]):
                point_x = point_pair[0]
                point_y = point_pair[1]
                linestring.append([float(point_x), float(point_y)])
            geom_str = LineString(linestring)
        elif obj_type == 'POINT':
            point_x = geom_obj[0]
            point_y = geom_obj[1]
            geom_str = Point([float(point_x), float(point_y)])
        elif obj_type == 'POLYGON':
            polygonstring = []
            for point_pair in zip(geom_obj[::2], geom_obj[1::2]):
                point_x = point_pair[0]
                point_y = point_pair[1]
                polygonstring.append([float(point_x), float(point_y)])
            geom_str = Polygon(polygonstring)
        else:
            print(f'[!] Unknown object type: {obj_type}.\n[!] Exiting...')
            exit()

    except Exception as e:
        print(f'[!] trans_to_linestring error: {e}')
        print(f'[!] Point string causing error: {str_line}')
        print(f'[!] Returning Point object.')
        geom_str = None
    return geom_str


def get_seeds(stamp):
    '''
    based on a text (stamp) return string subsets
    (1) start and end with a Capitalised Word (e.g.  turn "Courtyard of the Hotel de Ville" into ["Courtyard of the Hotel", "Hotel de Ville"])
    (2) split string based on delimiters such as '/', ',' '|'
    :param stamp:
    :return: seeds
    '''

    seeds = [stamp]
    separators = '/|,'

    for sep in separators:
        for seed in seeds:
            split = seed.split(sep)
            if len(split) > 1:
                seeds = seeds + split
    # check if the stamp is not all upper or lower letters
    if not stamp.islower() or not stamp.isupper():
        pattern = r"[[:upper:]].+?[[:upper:]][a-z]+"
        seeds + reg.findall(pattern, stamp, overlapped=True)
    # only keep stamps without any sep
    pure_seeds = []
    for seed in set(seeds):
        if not any(sep in seed for sep in separators):
            pure_seeds.append(seed)
    # strip stamps of whitespace
    seeds = [seed.strip() for seed in pure_seeds if seed != '']

    return seeds


def geoparsing(frame_dict, videodata_output_folder, YTQUERY_FILEPATH_JSON, video_metadata_dict={}):
    '''
    This function geoparses location found within the OCR data and the TEXTual descriptions of a video (e.g. timestamps)
    In a first step, OCR is processed, followed by the text geoparsing. The results are merged in one dataframe

    Levenshtein Distance (word similarity) [https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a]
    between the OCR detected words above a certain threshold and a gazetteer.
    '''

    # dict that stores for each geolocated string a subordinate video metadata dict
    storage_dict = defaultdict(lambda: {
                                    'videoID': None,
                                    'frame_nr': None,
                                    'location_name': None,
                                    'geo': None,
                                    'origin': None,
                                    'fps': None,
                                    'publishedAt': None,
                                    'extractedDates': None,
                                    'extractedTimes': None,
                                    'goodWeather': 0,
                                    'badWeather': 0,
                                    'lat': None,
                                    'lng': None
                                    })
    # start db connection
    db_querier = DbQuerier()
    # count geolocations
    geolocations_found = 0

    # -----------------------------------------------------------------------------------------------------------------
    # find the Youtube API data for the given videoID/video_name
    # 1. geoparsing OCR
    MODE = 'OCR'
    for index, (frame_nr, element) in enumerate(frame_dict.items()):
        geoparsing_str_list = element['string_list']
        for index, geoparsing_str in enumerate(geoparsing_str_list, 1):
            result = db_querier.spatial_db_query(geoparsing_str, mode=MODE)
            # check result type
            if result:
                for item in result:
                    geolocations_found += 1
                    # retrieve lat lng coordinates and name if there is a match
                    name = item[0]
                    geo = item[1]
                    # set unique dict key for this match
                    dict_key = geoparsing_str + str(geolocations_found)
                    storage_dict[dict_key]['videoID'] = videodata_output_folder
                    storage_dict[dict_key]['frame_nr'] = int(frame_nr)
                    storage_dict[dict_key]['location_name'] = name
                    storage_dict[dict_key]['geo'] = trans_to_stringobj(geo)
                    storage_dict[dict_key]['origin'] = MODE
                    # add data extracted from the video's metadata
                    for k, v in video_metadata_dict.items():
                        storage_dict[dict_key][k] = v


    # -----------------------------------------------------------------------------------------------------------------

    ## 2. geoparsing video TEXT
    MODE = 'TEXT'
    # load input file
    with open(YTQUERY_FILEPATH_JSON, 'rt', encoding='utf-8') as f:
        data_dict = json.load(f)
    data = data_dict[videodata_output_folder]
    # get extractedVideoTimestamps key
    extractedVideoTimestamps = data['extractedVideoTimestamps']
    if extractedVideoTimestamps:
        for tuple_ in extractedVideoTimestamps:
            time_str, stamp = tuple_
            # convert time string to time obj
            colon_count = len(time_str.split(':')) - 1
            if colon_count == 1:
                try:
                    time_obj = time.strptime(time_str, '%M:%S')
                except Exception:
                    print(f'[!] invalid timestr: {time_str}, skipping..')
                    continue
            elif colon_count == 2:
                try:
                    time_obj = time.strptime(time_str, '%H:%M:%S')
                except Exception:
                    print(f'[!] invalid timestr: {time_str}, skipping..')
                    continue
            else:
                print(f'[!] strange time_obj: {time_obj}, skipping..')
                continue
            # check for irrelevant string intro often used in video timestamps
            irrelevant_stamps = ['intro', 'preview', 'highlight', 'overview', 'starting point']
            if stamp.lower() not in irrelevant_stamps:
                seeds = get_seeds(stamp)
                # perform database query based on each substring of the timestamp
                for seed in seeds:
                    result = db_querier.spatial_db_query(seed, mode=MODE)
                    if result:
                        for item in result:
                            geolocations_found += 1
                            # retrieve lat lng coordinates and name if there is a match
                            name = item[0]
                            geo = item[1]
                            # set unique dict key for this match
                            dict_key = seed + str(geolocations_found)
                            # add data extracted from the video's metadata
                            for k, v in video_metadata_dict.items():
                                storage_dict[dict_key][k] = v
                            # get total seconds from time object
                            total_seconds = datetime.timedelta(hours=time_obj.tm_hour, minutes=time_obj.tm_min, seconds=time_obj.tm_sec).total_seconds()
                            frame = int(total_seconds * video_metadata_dict['fps'])
                            # add all other elements to dict
                            storage_dict[dict_key]['videoID'] = videodata_output_folder
                            storage_dict[dict_key]['frame_nr'] = frame
                            storage_dict[dict_key]['location_name'] = name
                            storage_dict[dict_key]['geo'] = trans_to_stringobj(geo)
                            storage_dict[dict_key]['origin'] = MODE

    # -----------------------------------------------------------------------------------------------------------------

    # check if there was anything returned
    if storage_dict.keys():
        location_names_df = gpd.GeoDataFrame(storage_dict.values(), geometry='geo', crs={'init': 'epsg:4326'}) # , crs="EPSG:4326"
        # reproject
        location_names_df.to_crs("EPSG:3857", inplace=True)
        location_names_df.drop_duplicates(inplace=True)
        return location_names_df
    else:
        print(f'[!] {videodata_output_folder}: geoparsing did not find any matches.')
        return None

if __name__ == '__main__':
    from output_analysis_wrapper import create_relevant_location_names_df, get_video_metadata
    OCR_LOG_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\yqOlY5uBBbo\20221118-153542_yqOlY5uBBbo_ocrlog.csv"
    VIDEO_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\yqOlY5uBBbo"
    YTQUERY_FILEPATH_JSON = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\ytapi_data\Paris_walk_videoInfoExtracted.json"
    videodata_output_folder = 'yqOlY5uBBbo'
    VIDEOS_METADATA_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\20221118-134355_videos_metadata.csv"
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    df = geoparsing(frame_dict, videodata_output_folder, YTQUERY_FILEPATH_JSON, video_metadata_dict=get_video_metadata(videodata_output_folder, VIDEOS_METADATA_PATH, YTQUERY_FILEPATH_JSON))
    rel_df = create_relevant_location_names_df(df)
    print()
