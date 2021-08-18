import csv
import time
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from db_querier import DbQuerier
from collections import defaultdict
from shapely.geometry import LineString, Point

import os
# pyproj fix 'no database context specified'
# os.environ["PROJ_LIB"] = r"C:\Users\mhartman\Anaconda3\envs\transportation_Yolov5_env\Library\share\proj"

def str_polishing(text, special_chars=r'][()|{}_~€$£\/'):
    # remove curly brackets at this step
    polished_text = ''
    for char in text:
        if char not in special_chars:
            polished_text = polished_text + char
    return polished_text

def str_prefiltering(OCR_LOG_PATH, confidence_th = 0.5, min_str_length = 4, max_nr_special_chars=1):
    # special characters that are a sign of artificats during the ocr
    special_chars = r'][()|{}_~€$£\/'
    frame_dict = defaultdict(lambda: [])
    # open and iterate over tracking log
    with open(OCR_LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quotechar='|')
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
                # print(confidence)
                pass
            # filter steps on pure string
            if confidence >= confidence_th:
                len_text = len(text)
                if len_text >= min_str_length:
                    count_special_chars = sum([1 for c in text if c in special_chars])
                    if count_special_chars <= max_nr_special_chars:
                        text = str_polishing(text)
                        # check if only numbers
                        if not text.isdigit():
                            print(f'{frame_num} - {confidence}: {text}')
                            frame_dict[frame_num].append(text)

    # keep single frame strings as well as a combination of the strings from the same frame
    frame_dict_complete = frame_dict
    for frame_nr, string_list in frame_dict.items():
        # frame_nr string combination
        combined_string = ' '.join(string_list)
        string_list.append(combined_string)
        frame_dict_complete[frame_nr] = string_list

    return frame_dict_complete

def trans_to_linestring(str_line):
    '''
    transform postgres geometry output which is a literal string into a stringly input for LineString object
    e.g. "LINESTRING(6.1467406 46.2011719,6.1465602 46.2010601)" -> LineString([[0, 0], [1, 0], [1, 1]])
    :return:
    '''
    point_string = str_line[11:-1].split(',')
    linestring = []
    for point_pair in point_string:
        point_x = point_pair.split(' ')[0]
        point_y = point_pair.split(' ')[1]
        linestring.append([float(point_x), float(point_y)])
    line_string = LineString(linestring)
    return line_string


def geoparsing(frame_dict, LOG_PATH, source_str):
    '''
    calculate Levenshtein Distance (word similarity) [https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a]
    between the OCR detected words above a certain threshold and a gazetteer.
    This gazetteer can be narrowed down to a manually entered location e.g. Geneva
    '''
    # stores the df rows so that they can be added in one go -> better performance than df.append()
    df_row_storage = []
    # initialise dict that stores already geocoded and processed strings
    processed_dict = defaultdict(lambda: {'name': None, 'geo': None})
    # start db connection
    db_querier = DbQuerier()
    # count geolocations found
    geolocations_found = 0
    for frame_nr, geoparsing_str_list in frame_dict.items():
        # print(f'[*] processing frame_nr {frame_nr}...')
        len_geoparsing_str_list = len(geoparsing_str_list)
        for index, geoparsing_str in enumerate(geoparsing_str_list, 1):
            # print(f'\r[*] geoparsing {index} of {len_geoparsing_str_list} strings....', end='')
            # check if string was geoparsed before, if yes retrieve previous result or skip
            if geoparsing_str in processed_dict.keys():
                # append to pandas result df
                geolocations_found += 1
                df_row_storage.append(processed_dict[geoparsing_str])
            result = db_querier.levenshtein_dist_query(geoparsing_str)
            # check result type
            if result is None:
                # add to processed with default values since no result
                processed_dict[geoparsing_str]
                print(f'\r[-] no matching geolocation found', end='')
                continue
            else:
                for item in result:
                    geolocations_found += 1
                    # retrieve lat lng coordinates and name if there is a match
                    name = item[0]
                    geo = item[1]
                    processed_dict[geoparsing_str]['name'] = name
                    processed_dict[geoparsing_str]['geo'] = trans_to_linestring(geo)
                    # append to result rows
                    df_row_storage.append(processed_dict[geoparsing_str])
                    print(f'location: {name}, geo: {geo}')
    # check if there was anything returned
    if len(df_row_storage) != 0:
        # load geneva shapefile
        geneva_shp_df = gpd.read_postgis('SELECT * FROM geneva_poly', db_querier.conn, geom_col='st_polygonize')
        geneva_shp_df.to_crs("EPSG:3857", inplace=True)
        geneva_shp_df.rename_geometry('geometry', inplace=True)
        # add row_storage to df in one go
        df = gpd.GeoDataFrame(df_row_storage, geometry='geo', crs={'init': 'epsg:4326'}) # , crs="EPSG:4326"
        # reproject
        df.to_crs("EPSG:3857", inplace=True)
        # join both dfs
        # df = gpd.overlay(matchted_geoms_df, geneva_shp_df, how='union', keep_geom_type=False)
        df_len_with_duplicates = df.shape[0]
        df.drop_duplicates(inplace=True)
        df_len_without_duplicates = df.shape[0]
        # save geolocations to output file
        GEOLOCATIONS_OUTPUT_PATH = './output/geoparsing/'
        GEOLOCATIONS_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_source_{source_str}_geolocations.csv'
        with open(os.path.join(GEOLOCATIONS_OUTPUT_PATH, GEOLOCATIONS_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('name;geo\n')
            for i_index, line in df.iterrows():
                name = line[0]
                geo = line[1]
                f.write(f'{name};{geo}\n')
        print(f'\n[*] unique geolocation found: {df_len_without_duplicates}; (dublicates: {df_len_with_duplicates - df_len_without_duplicates})')
        ax = df.plot(figsize=(20, 20), linewidth=50, color='red')
        # label each location with its name
        df.apply(lambda x: ax.annotate(text=x['name'], xy=x['geo'].centroid.coords[0], ha='center'), axis=1)
        # set spatial extent of axis based on geneva shapefile bounds
        geneva_bounds = geneva_shp_df.geometry.total_bounds
        xlim = ([geneva_bounds[0], geneva_bounds[2]])
        ylim = ([geneva_bounds[1], geneva_bounds[3]])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # add basemap
        ctx.add_basemap(ax, url=ctx.providers.OpenStreetMap.Mapnik)
        plt.show()
        # save figure
        fig_filename = f'./output/geoparsing/maps/{time.strftime("%Y%m%d-%H%M%S")}_{LOG_PATH.split("/")[-1][:-4]}_map.png'
        plt.savefig(fig_filename)
        # pretty print df
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        return df
    else:
        print(f'[!] geoparsing did not find any matches.')

if __name__ == '__main__':
    LOG_PATH = r"./output/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_tracklog.csv"
    # LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\tmp_output\20210812-151959_source_0000_0030_City_center_walk_in_Geneva_Winter_720p_1628694652801_tracklog.csv"
    OCR_LOG_PATH = r"./output/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_ocrlog.csv"
    # OCR_LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\tmp_output\20210812-151959_source_0000_0030_City_center_walk_in_Geneva_Winter_720p_1628694652801_ocrlog.csv"
    frame_dict = str_prefiltering(OCR_LOG_PATH)
    df = geoparsing(frame_dict, LOG_PATH)
