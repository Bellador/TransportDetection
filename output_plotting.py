import os
import time
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
from collections import Counter
warnings.simplefilter(action='ignore', category=FutureWarning)
# figure plotting
from shapely import wkt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
# map plotting
import contextily as ctx
# from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.graphics.mosaicplot import mosaic

def all_locations_map(CSV_LOCATION_STATISTCS_OUTPUT_PATH):
    # load df and convert to gdf
    locations_df = pd.read_csv(CSV_LOCATION_STATISTCS_OUTPUT_PATH, sep=';')
    # drop invalid geom rows before conversion to geodataframe
    locations_df = locations_df.replace(to_replace='None', value=np.nan)
    locations_df = locations_df[locations_df['geo'].notna()]
    locations_df['geo'] = locations_df['geo'].apply(wkt.loads)
    locations_gdf = gpd.GeoDataFrame(locations_df, crs='epsg:3857', geometry='geo')
    # initialise figure
    fig, ax = plt.subplots(figsize=(15, 14))
    locations_gdf.plot(linewidth=2, color='red', ax=ax) #figsize=(20, 20),
    # set spatial extend of axis based on detected features
    feature_bounds = locations_gdf.geometry.total_bounds
    x_span = feature_bounds[2] - feature_bounds[0]
    y_span = feature_bounds[3] - feature_bounds[1]
    xlim = ([(feature_bounds[0] - (x_span * 0.5)), (feature_bounds[2] + (x_span * 0.5))])
    ylim = ([(feature_bounds[1] - (y_span * 0.5)), (feature_bounds[3] + (y_span * 0.5))])
    # add map boundaries
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()
    # add basemap
    ctx.add_basemap(ax, url=ctx.providers.OpenStreetMap.Mapnik)
    plt.show()

def mosaic_plot_locations(CSV_LOCATION_STATISTICS_OUTPUT_PATH, OUTPUT_PATH, forms_of_transportation, plot_locations='all'):
    '''
    relative plot of transportation forms across all locations, easily comparable

    plot_locations: If equals string 'all' all locations are plotted, otherwise it defines how many locations are plotted in total.
    Included locations will be chosen after the dataframe was ordered and chosen in the given sequence
    '''

    storage_ = {}
    location_df = pd.read_csv(CSV_LOCATION_STATISTICS_OUTPUT_PATH, delimiter=';')

    for transport_form, class_names in forms_of_transportation.items():
        for location_name in location_df.location_name.unique():
            transport_form_counter = 0
            for class_name in class_names:
                transport_form_counter += location_df.loc[location_df.location_name == location_name, class_name].mean()
            storage_[(location_name, transport_form)] = transport_form_counter

    active_share_list = []
    for location_name in location_df.location_name.unique():
        active_share = storage_[(location_name, 'active')] / (storage_[(location_name, 'active')] + storage_[(location_name, 'public')] + storage_[(location_name, 'motorised')])
        active_share_list.append((location_name, round(active_share, 2)))
    # sort dictionary by highest share of active transportation
    active_share_list_ordered = sorted(active_share_list, key=lambda x: x[1], reverse=True)
    ordered_locations = [item[0] for item in active_share_list_ordered]
    # sort storage based on highest active share locations
    storage_sorted = {}
    for location in ordered_locations:
        for transport_form in ['active', 'motorised', 'public']:
            storage_sorted[(location, transport_form)] = storage_[(location, transport_form)]

    def props(key):
        return {'color': '#AB63FA' if 'active' in key else ('#19D3F3' if 'public' in key else '#FFA15A')}

    def labelizer(key):
        # return data[key]
        return None

    fig, ax = plt.subplots()
    fig.set_figheight(100)
    fig.set_figwidth(100)
    mosaic(storage_sorted, labelizer=labelizer, properties=props, gap=0.015, label_rotation=[90, 0], ax=ax) #, labelizer=labelizer
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=20)
    # plt.show()
    print('[+] saving mosaic plot')
    OUTPUT_FILEPATH = os.path.join(OUTPUT_PATH, 'location_mosaic_plot.png')
    fig.savefig(OUTPUT_FILEPATH)

def all_locations_plot(CSV_LOCATION_STATISTICS_OUTPUT_PATH, PLOTS_PATH, forms_of_transportation):
    '''
    creates a bar plot of all locations while aggregating the averaged, detected transportation modes by transportation form:
    active, public and motorsied
    '''

    storage_ = defaultdict(dict)
    location_df = pd.read_csv(CSV_LOCATION_STATISTICS_OUTPUT_PATH, delimiter=';')
    # initialise figure
    fig = go.Figure()
    colors = {
        "active": "#AB63FA",
        "motorised": "#FFA15A",
        "public": "#19D3F3"
    }

    for transport_form in forms_of_transportation.keys():
        for location_name in location_df.location_name.unique():
            transport_form_counter = round(location_df.loc[location_df.location_name == location_name, transport_form].mean(), 2)
            storage_[location_name][transport_form] = transport_form_counter
        # add bar element to figure
        fig.add_trace(go.Bar(x=list(storage_.keys()),
                             y=[v[transport_form] for k, v in storage_.items()],
                             name=f"{transport_form} transport",
                             marker_color=colors[transport_form]))

    # adapt layout to a stacked bar chart
    fig.update_layout(
        xaxis_title="location names",
        yaxis_title="count",
        barmode='stack',
        xaxis={'categoryorder': 'total descending'}
    )
    # fig.show()
    print('[+] saving all locations figure as interactive HTML plot')
    PLOTS_FILEPATH = os.path.join(PLOTS_PATH, 'location_statistics_plot.html')
    fig.write_html(PLOTS_FILEPATH)

def output_location_statistics(total_objects_dict, total_modes_dict, relevant_location_names_df, CSV_LOCATION_STATISTCS_OUTPUT_PATH, SEC_BUFFER=30):
    '''
    create location statistics in CSV output across all videos, will be called for each video iteratively
    creates new output file if not existent, and appends otherwise.
    file includes the found objects, and transportation modes across videos around detected locations
    based on a defined buffer around the detected location.
    Thereby, between two location types is differentiated which have their individual buffers (fps dependent!):
    (1) Locations extracted from OCR detected street names -> sec_buffer +/- the location of detection
    (2) Locations extracted from video timestamps  -> 2x sec_buffer after the location (since video authors highlight the start of location)
    '''
    merged_dict = total_objects_dict | total_modes_dict
    # classnames_to_output = "airplane;backpack;bench;bicycle;bird;boat;boat_passenger;bus;bus_passenger;car;car_passenger;chair;clock;cyclist;dogwalker;handbag;motorcycle;motorcyclist;pedestrian;person;potted plant;refrigerator;skateboard;suitcase;traffic light;train;train_passenger;truck;truck_passenger".split(';')

    # add lat, lng to dataframe as separate columns
    relevant_location_names_df['lat'] = relevant_location_names_df.apply(lambda x: x['geo'].centroid.coords[0][1] if x['geo'] is not None else None, axis=1)
    relevant_location_names_df['lng'] = relevant_location_names_df.apply(lambda x: x['geo'].centroid.coords[0][0] if x['geo'] is not None else None, axis=1)
    # iterate over all input, add traces for each object, mode and add locations to x-axis
    for row_index, row in relevant_location_names_df.iterrows():
        location_frame_nr = int(row['frame_nr'])
        fps = row['fps']
        origin = row['origin'] # origin of the location, either 'TEXT' or 'OCR'
        FRAME_BUFFER = int(SEC_BUFFER * fps)
        # location from detected street name
        if origin == 'OCR':
            # defining frame buffer boundaries, in which modes and objects are counted and attributed to the location
            upper_frame_limit = location_frame_nr + FRAME_BUFFER
            lower_frame_limit = location_frame_nr - FRAME_BUFFER
        # location from video timestamp
        elif origin == 'TEXT':
            upper_frame_limit = 2 * FRAME_BUFFER + location_frame_nr
            lower_frame_limit = location_frame_nr
        else:
            print(f'[!] origin value "{origin}" not valid. Exiting')
            exit()
        # to ensure consistency in which the classes are processed
        for class_name, value in merged_dict.items():
            # aggregate counts across frames based on the defined bins_per_video
            counts_in_frame_buffer = 0
            processed_frames = []
            for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                frame = int(frame)
                if frame >= lower_frame_limit and frame <= upper_frame_limit and frame not in processed_frames:
                    counts_in_frame_buffer += value_['count']
                    processed_frames.append(frame)
            # create new class name column and fill it with default value (to ensure dtype int64 instead of float64
            # first check if column already exists, if not create
            if class_name not in relevant_location_names_df.columns:
                relevant_location_names_df[class_name] = 0
            # add the class name count to the df under the new column and the current row index
            relevant_location_names_df.loc[row_index, [class_name]] = counts_in_frame_buffer

    # write/append to CSV
    delimiter = ';'
    # check if file was already create from previous video analysis
    if not os.path.isfile(CSV_LOCATION_STATISTCS_OUTPUT_PATH):
        print(f'[+] locations statistics file does NOT exist. Creating.')

        relevant_location_names_df.to_csv(CSV_LOCATION_STATISTCS_OUTPUT_PATH,
                                                                          sep=delimiter,
                                                                          index=False,
                                                                          encoding='utf-8')
    else:
        print(f'[+] locations statistics file EXISTS, appending {len(relevant_location_names_df.index)} new locations.')
        # append to existing csv in 'a' mode and neglect header and index
        relevant_location_names_df.to_csv(CSV_LOCATION_STATISTCS_OUTPUT_PATH,
                                                                          sep=delimiter,
                                                                          mode='a',
                                                                          header=False,
                                                                          index=False,
                                                                          encoding='utf-8')


# function to create inset axes and plot bar chart on it
# this is good for 3 items bar chart
def build_bar(mapx, mapy, ax, width, xvals=['a','b','c'], yvals=[1,4,2], fcolors=['r','y','b']):
    ax_h = inset_axes(ax, width=width,
                    height=width,
                    loc=3,
                    bbox_to_anchor=(mapx, mapy),
                    bbox_transform=ax.transData,
                    borderpad=0,
                    axes_kwargs={'alpha': 0.35, 'visible': True})
    for x,y,c in zip(xvals, yvals, fcolors):
        ax_h.bar(x, y, label=str(x), fc=c)
    #ax.xticks(range(len(xvals)), xvals, fontsize=10, rotation=30)
    ax_h.axis('off')
    return ax_h

def map_plotting(CSV_LOCATION_STATISTCS_OUTPUT_PATH, VIDEODATA_OUTPUT_FOLDER_PATH, videodata_output_folder):
    '''
    map the locations and visualise their respective, detected transportation modes
    uses the location_statistics.csv as input
    :param total_objects_dict:
    :param total_modes_dict:
    :param relevant_location_names_df:
    :param VIDEODATA_OUTPUT_FOLDER_PATH:
    :param videodata_output_folder:
    :param frame_buffer:
    :return:
    '''
    # classnames considered for the map for function map_plotting
    classnames_to_map = ['pedestrian', 'bicycle', 'car']
    # load df and convert to gdf
    locations_df = pd.read_csv(CSV_LOCATION_STATISTCS_OUTPUT_PATH, sep=';')
    # drop invalid geom rows before conversion to geodataframe
    locations_df = locations_df.replace(to_replace='None', value=np.nan)
    locations_df = locations_df[locations_df['geo'].notna()]
    locations_df['geo'] = locations_df['geo'].apply(wkt.loads)
    locations_gdf = gpd.GeoDataFrame(locations_df, crs='epsg:3857', geometry='geo')
    # filter the df for the current video ID to only include its locations
    locations_gdf = locations_gdf[locations_gdf['videoID'] == videodata_output_folder]
    # set spatial extend of axis based on detected features
    feature_bounds = locations_gdf.geometry.total_bounds
    x_span = feature_bounds[2] - feature_bounds[0]
    y_span = feature_bounds[3] - feature_bounds[1]
    xlim = ([(feature_bounds[0] - (x_span * 0.5)), (feature_bounds[2] + (x_span * 0.5))])
    ylim = ([(feature_bounds[1] - (y_span * 0.5)), (feature_bounds[3] + (y_span * 0.5))])

    # initialise figure
    fig, ax = plt.subplots(figsize=(15, 14))
    locations_gdf.plot(linewidth=4, color='red', ax=ax) #figsize=(20, 20),
    # add x y as separate columns
    y_offset = y_span * 0.05
    x_offset = x_span * 0.05
    locations_gdf['x'] = locations_gdf.centroid.map(lambda p: p.x) - x_offset
    locations_gdf['y'] = locations_gdf.centroid.map(lambda p: p.y) + y_offset

    bar_width = 0.5
    colors = ['green', 'orange', 'blue']

    storage_ = defaultdict(dict)
    for location_name in locations_gdf.location_name.unique():
        for class_name in classnames_to_map:
            storage_[location_name][class_name] = locations_gdf[class_name].mean()
        # average all count of the same location

        x_cords = locations_gdf.loc[locations_gdf.location_name == location_name, 'x'].copy()
        x_cord = x_cords.iloc[0]
        y_cords = locations_gdf.loc[locations_gdf.location_name == location_name, 'y'].copy()
        y_cord = y_cords.iloc[0]
        y_vals = [round(storage_[location_name][classname], 2) for classname in classnames_to_map]
        try:
            build_bar(x_cord, y_cord, ax, bar_width, xvals=['a', 'b', 'c'],
                            yvals=y_vals,
                            fcolors=colors)
        except Exception as e:
            print(f'[!] map plotting error: {e}')
            print(f'[!] params:\nx_cord:{x_cord}\ny_cord:{y_cord}\nax:{ax}\nbar_width:{bar_width}\ny_vals:{y_vals}')

    # label each location with its name
    locations_gdf.apply(lambda x: ax.annotate(text=x['location_name'],
                                                           xy=x['geo'].centroid.coords[0],
                                                           ha='center'),
                                     axis=1)
    # create legend (of the 3 classes)
    legend_patch = []
    for index, classname in enumerate(classnames_to_map):
        patch = mpatches.Patch(color=colors[index], label=classnames_to_map[index])
        legend_patch.append(patch)
    ax.legend(handles=legend_patch, loc=1)
    try:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axis_off()
    except:
        print(f'[!] Skip setting axis limits.')
    # add basemap
    ctx.add_basemap(ax, url=ctx.providers.OpenStreetMap.Mapnik)
    # save figure
    fig_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{videodata_output_folder}_map.png"
    fig_filepath = os.path.join(VIDEODATA_OUTPUT_FOLDER_PATH, fig_filename)
    plt.savefig(fig_filepath)
    # plt.show()

def location_names_by_frequency_plot(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH, min_frequency=10):
    '''
    plot detected location names by decreasing observation frequency
    :param FILTERED_CSV_STATISTICS_PATH:
    :return:
    '''
    # load df and convert to gdf
    locations_df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, sep=';')
    # drop invalid geom rows before conversion to geo df
    locations_df = locations_df.replace(to_replace='None', value=np.nan)
    locations_df = locations_df[locations_df['geo'].notna()]
    locations_df['geo'] = locations_df['geo'].apply(wkt.loads)
    locations_gdf = gpd.GeoDataFrame(locations_df, crs='epsg:3857', geometry='geo')
    # share of TEXT and OCR detected locations
    origin_c = Counter(locations_gdf['origin'])
    print(f'[+] Origin analysis - TEXT: {origin_c["TEXT"]}, OCR: {origin_c["OCR"]}')
    # frequency of unique locations
    locations_c = Counter(locations_gdf['location_name'])
    print(f'[+] Overall unique locations: {len(locations_c.keys())}')
    # filter for min. frequency and compile bar chart data
    x_vals = []
    y_vals = []
    for k, v in locations_c.items():
        if v >= min_frequency:
            x_vals.append(k)
            y_vals.append(v)
    print(f'[+] unique locations with min. frequency {min_frequency}: {len(x_vals)}')
    # initialise figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals,
                         y=y_vals, width=0.8))
    # adapt layout to a stacked bar chart
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
        font=dict(size=35),
        xaxis_title="Top 8 unique locations",
        yaxis_title="Number of detections",
        barmode='stack',
        xaxis={'categoryorder': 'total ascending'}
    )
    fig.update_annotations(font_size=35)
    fig.show()
    print('[+] saving data exploration as interactive HTML plot')
    PLOTS_FILEPATH = os.path.join(PLOTS_PATH, 'data_exploration.html')
    # fig.write_html(PLOTS_FILEPATH)

def frequency_of_frequencies_plot(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH):
    '''
    plot detected location names by decreasing observation frequency
    :param FILTERED_CSV_STATISTICS_PATH:
    :return:
    '''
    # load df and convert to gdf
    locations_df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, sep=';')
    # drop invalid geom rows before conversion to geo df
    locations_df = locations_df.replace(to_replace='None', value=np.nan)
    locations_df = locations_df[locations_df['geo'].notna()]
    locations_df['geo'] = locations_df['geo'].apply(wkt.loads)
    locations_gdf = gpd.GeoDataFrame(locations_df, crs='epsg:3857', geometry='geo')
    # share of TEXT and OCR detected locations
    origin_c = Counter(locations_gdf['origin'])
    print(f'[+] Origin analysis - TEXT: {origin_c["TEXT"]}, OCR: {origin_c["OCR"]}')
    # frequency of unique locations
    locations_c = Counter(locations_gdf['location_name'])
    print(f'[+] Overall unique locations: {len(locations_c.keys())}')
    # filter for min. frequency and compile bar chart data
    x_vals = []
    y_vals = []
    counter = defaultdict(lambda: 0)
    for k, v in locations_c.items():
        if v >= 10:
            counter[10] += 1
        else:
            counter[v] += 1

    for k, v in counter.items():
        x_vals.append(k)
        y_vals.append(v)

    # initialise figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals,
                         y=y_vals))

    # adapt layout to a stacked bar chart
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
        font=dict(size=35),
        # title="Frequency distribution of unique locations",
        xaxis_title="Number of detections",
        yaxis_title="Unique locations",
        barmode='stack',
        xaxis=dict(categoryorder='total descending',
                   tickmode='array',
                   tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   ticktext=['1', '2', '3', '4', '5', '6', '7', '8', '9', '+10'])
    )
    fig.update_annotations(font_size=35)
    fig.show()
    print('[+] saving frequency distribution of locations as interactive HTML plot')
    PLOTS_FILEPATH = os.path.join(PLOTS_PATH, 'frequency_plot', 'frequency_of_locations.html')
    fig.write_html(PLOTS_FILEPATH)

def temporal_analysis_per_location(FILTERED_CSV_STATISTICS_PATH, forms_of_transportation, PLOTS_PATH, min_frequency=7):
    TEMP_PLOTS_PATH = os.path.join(PLOTS_PATH, f'temp_analysis_w_people_car_bikes_freq_{min_frequency}')
    if not os.path.isdir(TEMP_PLOTS_PATH):
        os.mkdir(TEMP_PLOTS_PATH)

    storage_ = defaultdict(lambda: defaultdict(dict))
    location_df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, delimiter=';')
    colors = {
        "active": "#AB63FA",
        "motorised": "#FFA15A",
        "public": "#19D3F3"
    }
    # add years to locations
    # location_df['year'] = location_df.publishedAt.apply(lambda x: time.strptime(x, '%Y-%m-%dT%H:%M:%SZ').tm_year)
    for location_name in location_df.location_name.unique():
        # get all location records
        location_records_df = location_df[location_df.location_name == location_name]
        # check min frequency
        if len(location_records_df.values) >= min_frequency:
            # initialise figure
            fig = go.Figure()
            for transport_form in forms_of_transportation.keys():
                # iterative over years
                for year in location_records_df['year'].unique():
                    nr_records_per_year = len(location_records_df[location_records_df.year == year].values)
                    transport_form_counter = round(location_records_df.loc[((location_records_df.location_name == location_name) & (location_records_df.year == year)), transport_form].mean(), 2)
                    storage_[location_name][f'{year}, {nr_records_per_year}'][transport_form] = transport_form_counter
                # sort based on year (due to records in x tick label)
                x_vals = []
                y_vals = []
                for item in sorted(list(storage_[location_name].keys()), key=lambda x: int(x.split(',')[0])):
                    x_vals.append(item)
                    y_vals.append(storage_[location_name][item][transport_form])
                # add bar element to figure
                fig.add_trace(go.Bar(x=x_vals,
                                     y=y_vals,
                                     name=f"{transport_form} transport",
                                     marker_color=colors[transport_form]))
            # adapt layout to a stacked bar chart
            fig.update_layout(
                xaxis_title="year, records",
                yaxis_title="count",
                barmode='stack',
                title=f'Location: {location_name}'
            )
            fig.show()
            TEMP_PLOTS_FILEPATH = os.path.join(TEMP_PLOTS_PATH, f'{location_name.replace(" ","_")}_temp_analysis.html')
            fig.write_html(TEMP_PLOTS_FILEPATH)

def small_multiples_by_quartier():
    '''
    within QGIS the location_statistic was joined with the quartiers (districts) of Paris for a level of
    spatial aggregation. This allows transport analysis not only on the street level but also on a district level.

    This function produces a figure of small multiples which includes the 5 most data dominant quartiers and visualises
    the contribution of transport modes across the years

    :return:
    '''

    FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_w_quartiers.csv"

    # load data
    location_df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS, delimiter=';')

    # find top X quartiers with most data
    nr_top_quartiers = 5
    quartiers = location_df.groupby(['l_qu'])['l_qu'].count().sort_values(ascending=False)[:nr_top_quartiers].index

    transport_modes = [
        'active',
        'motorised',
        'public'
    ]

    colors = [
        "#AB63FA",
        "#FFA15A",
        "#19D3F3"
    ]

    years = range(2017, 2022, 1)

    fig = make_subplots(rows=5, cols=5,
                        vertical_spacing=0.025, horizontal_spacing=0.005,
                        shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=['temp' for i in range(1, 26)]) # column_widths=[0.7, 0.3]
    # populate subplots
    subplot_counter = 0
    for row, quartier in enumerate(quartiers, 1):
        for column, year in enumerate(years, 1):
            transport_values_df = location_df.loc[(location_df.l_qu == quartier) & (location_df.year == year), transport_modes]
            data_points = len(transport_values_df.index)
            if not transport_values_df.empty:
                # average the transport values per location and year across all measurements
                mean_transport_values = transport_values_df.mean(axis=0)
                # convert to relative
                sum_values = mean_transport_values.sum()
                rel_transport_values = [round(val/sum_values, 2) for val in mean_transport_values]
            else:
                rel_transport_values = [0, 0, 0]
            fig.add_trace(go.Bar(x=transport_modes, y=rel_transport_values, marker=dict(color=colors)), row=row, col=column)
            # update axes if conditions are met
            if column == 1:
                # make quartier name more legible (if contains saint-germain)
                if '-' in quartier:
                    quartier_label = 'St.-Ger. ' + quartier[14:]
                else:
                    quartier_label = quartier
                fig.update_yaxes(title_text=quartier_label, row=row, col=column)
            if row == 5:
                fig.update_xaxes(title_text=year, row=row, col=column)

            fig.update_yaxes(range=[0, 1], row=row, col=column)
            # add amount of data points a subplot title
            fig.layout.annotations[subplot_counter]['text'] = 'data points: ' + str(data_points)
            subplot_counter += 1


    fig.update_annotations(font_size=10)
    fig.update_layout(showlegend=False)
    fig.show()

    fig.write_html(r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\plots\small_multiples\quartiers_years.html")

def boxplots_COVID_by_quartier():
    '''
    build 2 sets of boxplots for the period before (2015-2019) and after COVID (2020-2021) including subplots for pedestrians, cyclists, motorised

    :return:
    '''

    FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_w_quartiers.csv"

    # load data
    location_df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS, delimiter=';')
    # add relative values for all target modes
    location_df['pedestrian_rel'] = location_df.apply(lambda x: round(x['pedestrian'] / (x['active'] + x['motorised']), 3) if x['pedestrian'] != 0 else 0, axis=1)
    location_df['cyclist_rel'] = location_df.apply(lambda x: round(x['cyclist'] / (x['active'] + x['motorised']), 3) if x['cyclist'] != 0 else 0, axis=1)
    location_df['motorised_rel'] = location_df.apply(lambda x: round(x['motorised'] / (x['active'] + x['motorised']), 3) if x['motorised'] != 0 else 0, axis=1)


    # find top X quartiers with most data
    nr_top_quartiers = 5
    quartiers = location_df.groupby(['l_qu'])['l_qu'].count().sort_values(ascending=False)[:nr_top_quartiers].index

    transport_modes = [
        'pedestrian_rel',
        'cyclist_rel',
        'motorised_rel'
    ]

    transport_modes_labels = [
        'pedestrians',
        'cyclists',
        'motorised'
    ]

    colors = [
        "#3182bd",
        "#dd1c77",
        "#FFA15A"
    ]

    years = range(2015, 2022, 1)
    year_covid_boundary = 2020

    fig = make_subplots(rows=5, cols=3,
                        vertical_spacing=0.03, horizontal_spacing=0.005,
                        shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=['Pedestrians', 'Cyclists', 'Motorised'],
                        x_title='2015 - 2019 (Baseline) to 2020 - 2021 (Covid-19)')

    # create df to store figure data for later statistical tests
    df_dict = {'time': [], 'mode': [], 'value': [], 'quartier': []}
    # define quartier order manually
    quartier_names = ['Clignancourt', 'Halles', 'Saint-Germain-des-Prés', "Saint-Germain-l'Auxerrois", 'Sorbonne']
    for row, quartier in enumerate(quartier_names, 1):

    # populate subplots
    # for row, quartier in enumerate(quartiers, 1):
        pre_covid_df = location_df.loc[(location_df.l_qu == quartier) & (location_df.year < year_covid_boundary), transport_modes]
        post_covid_df = location_df.loc[(location_df.l_qu == quartier) & (location_df.year >= year_covid_boundary), transport_modes]
        for column, mode in enumerate(transport_modes, 1):
            # update axes if conditions are met
            if column == 1:
                # make quartier name more legible (if contains saint-germain)
                if quartier == 'Saint-Germain-des-Prés':
                    quartier_label = 'SGdP'
                elif quartier == "Saint-Germain-l'Auxerrois":
                    quartier_label = 'SGlA'
                else:
                    quartier_label = quartier
                fig.update_yaxes(title_text=quartier_label, row=row, col=column)
            if row == 5:
                fig.update_xaxes(row=row, col=column)
            # average transport values across columns (pre/post covid)
            # transport_mean_vals_column_average = [sum(l) / len(l) for l in zip(*time_box)]
            # generate box plot for each mode in each subfigure
            for index, df in enumerate([pre_covid_df, post_covid_df]):
                vals_ = df[mode]
                if index == 0:
                    name = 'Base'
                else:
                    name = 'Covid'
                fig.add_trace(go.Box(y=vals_, name=name, boxpoints='all', line=dict(color=colors[column-1])), row=row, col=column)
                # populate test dataframe
                for val_ in vals_:
                    if column == 1:
                        df_dict['time'].append('pre_covid')
                    else:
                        df_dict['time'].append('post_covid')
                    df_dict['mode'].append(transport_modes_labels[index])
                    df_dict['value'].append(val_)
                    df_dict['quartier'].append(quartier)

            fig.update_yaxes(range=[0, 1], row=row, col=column)

    # build df for statistical test and export as csv to verification dir
    df_test = pd.DataFrame(df_dict)
    verification_dir = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\verification\tests\COVID_quartier_data.csv"
    df_test.to_csv(verification_dir, sep=';', encoding='utf-8')
    print(f'[+] export figure data for further statistical significance tests...\n[*] Path: {verification_dir}')

    fig.update_annotations(font_size=30)
    fig.update_layout(showlegend=False,
                      # plot_bgcolor="#ffffff",
                      # paper_bgcolor="#ffffff",
                      colorway=px.colors.qualitative.G10,
                      font=dict(size=27)
                    )
    fig.show()
    fig.write_html(r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\plots\small_multiples\quartiers_pre_post_covid.html")


def weather_boxplots(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH):

    df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, delimiter=';')
    bad_weather_df = df.loc[df["badWeather"] == 1, ["badWeather", "active", "motorised", "public"]]
    neutral_good_weather_df = df.loc[df["badWeather"] == 0, ["goodWeather", "active", "motorised", "public"]]


    # Create the boxplot figure
    fig = make_subplots(rows=1, cols=3,
                        vertical_spacing=0.025, horizontal_spacing=0.005,
                        shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=['active', 'motorised', 'public'],
                        y_title='absolute counts')
    # first subplot with active transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["active"], name='bad weather', showlegend=False, line=dict(color="#AB63FA")), row=1, col=1)
    fig.add_trace(go.Box(y=neutral_good_weather_df["active"], name='good weather', showlegend=False, line=dict(color="#5e06bc")), row=1, col=1)
    # second subplot with motorised transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["motorised"], name='bad weather', showlegend=False, line=dict(color="#FFA15A")), row=1, col=2)
    fig.add_trace(go.Box(y=neutral_good_weather_df["motorised"], name='good weather', showlegend=False, line=dict(color="#c25400")), row=1, col=2)
    # third subplot with public transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["public"], name='bad weather', showlegend=False, line=dict(color="#19D3F3")), row=1, col=3)
    fig.add_trace(go.Box(y=neutral_good_weather_df["public"], name='good weather', showlegend=False, line=dict(color="#0889a0")), row=1, col=3)
    fig.update_layout(
        font=dict(size=28)
    )
    fig.update_annotations(font_size=28)
    fig.write_html(os.path.join(PLOTS_PATH, 'weather_boxplot.html'))
    fig.show()

def weather_boxplots_pedest_cyclist(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH):

    df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, delimiter=';')

    bad_weather_df = df.loc[df["badWeather"] == 1, ["badWeather", "pedestrian", "cyclist", "motorised"]]
    neutral_good_weather_df = df.loc[df["badWeather"] == 0, ["goodWeather", "pedestrian", "cyclist", "motorised"]]


    # Create the boxplot figure
    fig = make_subplots(rows=1, cols=3,
                        vertical_spacing=0.025, horizontal_spacing=0.01,
                        shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=['Pedestrians', 'Cyclists', 'Motorised'],
                        y_title='Count')
    # first subplot with active transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["pedestrian"], name='Bad Weather', showlegend=False, line=dict(color="#3182bd")), row=1, col=1)
    fig.add_trace(go.Box(y=neutral_good_weather_df["pedestrian"], name='Good Weather', showlegend=False, line=dict(color="#276896")), row=1, col=1)
    # second subplot with motorised transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["cyclist"], name='Bad Weather', showlegend=False, line=dict(color="#dd1c77")), row=1, col=2)
    fig.add_trace(go.Box(y=neutral_good_weather_df["cyclist"], name='Good Weather', showlegend=False, line=dict(color="#a81559")), row=1, col=2)
    # third subplot with public transport boxplot
    fig.add_trace(go.Box(y=bad_weather_df["motorised"], name='Bad Weather', showlegend=False, line=dict(color="#FFA15A")), row=1, col=3)
    fig.add_trace(go.Box(y=neutral_good_weather_df["motorised"], name='Good Weather', showlegend=False, line=dict(color="#c25400")), row=1, col=3)
    fig.update_layout(
        font=dict(size=35),
        margin_pad=0,
        yaxis=dict(tickprefix="     ")
    )
    fig.update_annotations(font_size=40, borderpad=25)
    fig.write_html(os.path.join(PLOTS_PATH, 'weather_w_pedestrian_cyclist_boxplot.html'))
    fig.show()

def quartiers_per_year(PLOTS_PATH):
    FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_w_quartiers.csv"

    df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH_W_QUARTIERS, delimiter=';')

    unique_quartiers_per_year = df.groupby(by=['year'])

    # data about the amount of videos (with timestamps) per year
    data_per_year = {
                    2015: 1,
                    2016: 1,
                    2017: 5,
                    2018: 7,
                    2019: 19,
                    2020: 7,
                    2021: 36
    }

    total_quartiers = 80

    x_vals_w_data = []
    y_vals_w_data = []
    x_vals_without_data = []
    y_vals_without_data = []

    for name, group in unique_quartiers_per_year:
        year = int(name)
        unique_quartiers = len(group['l_qu'].unique())

        x_vals_w_data.append(year)
        x_vals_without_data.append(year)
        y_vals_w_data.append(unique_quartiers)
        y_vals_without_data.append(total_quartiers - unique_quartiers)

    # initialise figure
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals_w_data,
                         y=y_vals_w_data,
                         width=0.5,
                         offset=-0.3,
                         name="Quartiers\nwith data   "))
    fig.add_trace(go.Bar(x=x_vals_without_data,
                         y=y_vals_without_data,
                         width=0.5,
                         offset=-0.3,
                         name="Quartiers\nno data"))
    # data per year
    x_vals_datayear = [k for k in data_per_year.keys()]
    y_vals_datayear = [v for v in data_per_year.values()]

    fig.add_trace(go.Bar(x=x_vals_datayear,
                         y=y_vals_datayear,
                         base=0,
                         width=0.2,
                         offset=+0.225,
                         name="Analysed videos"))

    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
        font=dict(size=35),
        xaxis_title="Year",
        yaxis_title="Count",
        barmode='stack',
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.10,
            font=dict(size=32)
        )
    )

    fig.show()
    # fig.write_image(os.path.join(PLOTS_PATH, 'quartiers_per_year.eps'), width=1920, height=1080)
    fig.write_html(os.path.join(PLOTS_PATH, 'quartiers_per_year.html'))

def heatmap_matrix_plot_quartiles(classes=10):
    '''
    Visualise the share of a transport modes per quartier in quantiles relative to all quartiers (x-axis) and years (y-axis)

    :return:
    '''

    INPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer.csv"
    OUTPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\plots\heatmap"

    location_df = pd.read_csv(INPUT_PATH, delimiter=';', encoding='utf-8')
    # iterate over target transport modes
    modes = ['pedestrian', 'cyclist', 'motorised']
    # get complete list of quarties to check if quartier is missing and to keep order
    all_quartiers = location_df['l_qu'].unique()
    for mode in modes:
        # store year specific values per quartier, absolute count later used for sorting of quartiers
        data_quartier_dict = defaultdict(lambda: dict(relative=[], absolute_count=0))
        data_year_dict = defaultdict(lambda: dict(relative=[], absolute_count=0, quantile=[]))
        data_total_dict = dict(absolute_counts=[])
        # iterate over years and calc relative share of mode X on the total number of mode X across entire Paris i.e. all quartiers
        years = range(2017, 2022, 1)
        for year in years:
            year_df = location_df.loc[location_df['year'] == year]
            # count total amount per mode per year (for relative measurement)
            total_count_mode_year = year_df[mode].sum()
            # add total per year (as separate column for the heatmap)
            data_total_dict['absolute_counts'].append(total_count_mode_year)
            # iterate over quartiers
            for quartier_name in all_quartiers:
                # get subdf
                quartier_df = year_df.loc[year_df['l_qu'] == quartier_name]
                if quartier_df.empty:
                    data_quartier_dict[quartier_name]['relative'].append(0.0)
                    data_year_dict[year]['relative'].append(0.0)
                else:
                    data_quartier_dict[quartier_name]['absolute_count'] += quartier_df[mode].sum()
                    data_quartier_dict[quartier_name]['relative'].append(quartier_df[mode].sum() / total_count_mode_year)
                    data_year_dict[year]['relative'].append(quartier_df[mode].sum() / total_count_mode_year)

        # sort data based on absolute values in decreasing order
        data_quartier_dict = dict(sorted(data_quartier_dict.items(), key=lambda item: item[1]['absolute_count'], reverse=True))

        district_values = [dict_['relative'] for dict_ in data_quartier_dict.values()]
        yearly_values = [val_ for val_ in zip(*district_values)]
        # convert yearly values into quartiles
        quartile_values = []
        for l_ in yearly_values:
            bucket = []

            # sort l to find index of given value to determine its quantile
            l_sorted = sorted(l_)
            for val in l_:
                classes_d = {}
                for i, n in enumerate(range(classes-1), 1):
                    class_border = len(l_) * (i / (classes))
                    classes_d[class_border] = i
                classes_d[len(l_)] = classes
                val_index_sorted = l_sorted.index(val)
                for k, v in classes_d.items():
                    if val_index_sorted <= k:
                        bucket.append(v)
                        break

            quartile_values.append(bucket)

        # display quartier mode counts over the years
        fig = go.Figure(go.Heatmap(
            z=quartile_values,
            x=list(data_quartier_dict.keys()),
            y=list(years),
            colorscale='RdBu',
            reversescale=True
            ))

        fig.update_layout(title=f"Mode: {mode[0].upper() + mode[1:]}",
                          xaxis_title="Quartier",
                          yaxis_title="Year",
                          font=dict(size=28)
                           )
        fig.update_annotations(font_size=28)
        fig.show()
        fig.write_html(os.path.join(OUTPUT_PATH, f'heatmap_{mode.upper()}_{classes}_classes.html'))



def heatmap_matrix_plot():
    '''
    Visualise the share of a transport modes per quartier in quantiles relative to all quartiers (x-axis) and years (y-axis)

    :return:
    '''

    INPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer.csv"
    OUTPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\plots\heatmap"

    location_df = pd.read_csv(INPUT_PATH, delimiter=';', encoding='utf-8')
    # iterate over target transport modes
    modes = ['pedestrian', 'cyclist', 'motorised']
    # get complete list of quarties to check if quartier is missing and to keep order
    all_quartiers = location_df['l_qu'].unique()
    for mode in modes:
        # store year specific values per quartier, absolute count later used for sorting of quartiers
        data_quartier_dict = defaultdict(lambda: dict(relative=[], absolute_count=0))
        data_year_dict = defaultdict(lambda: dict(relative=[], absolute_count=0, quantile=[]))
        data_total_dict = dict(absolute_counts=[])
        # iterate over years and calc relative share of mode X on the total number of mode X across entire Paris i.e. all quartiers
        years = range(2017, 2022, 1)
        for year in years:
            year_df = location_df.loc[location_df['year'] == year]
            # count total amount per mode per year (for relative measurement)
            total_count_mode_year = year_df[mode].sum()
            # add total per year (as separate column for the heatmap)
            data_total_dict['absolute_counts'].append(total_count_mode_year)
            # iterate over quartiers
            for quartier_name in all_quartiers:
                # get subdf
                quartier_df = year_df.loc[year_df['l_qu'] == quartier_name]
                if quartier_df.empty:
                    data_quartier_dict[quartier_name]['relative'].append(0.0)
                    data_year_dict[year]['relative'].append(0.0)
                else:
                    data_quartier_dict[quartier_name]['absolute_count'] += quartier_df[mode].sum()
                    data_quartier_dict[quartier_name]['relative'].append(quartier_df[mode].sum() / total_count_mode_year)
                    data_year_dict[year]['relative'].append(quartier_df[mode].sum() / total_count_mode_year)

        # sort data based on absolute values in decreasing order
        data_quartier_dict = dict(sorted(data_quartier_dict.items(), key=lambda item: item[1]['absolute_count'], reverse=True))

        district_values = [dict_['relative'] for dict_ in data_quartier_dict.values()]
        yearly_values = [val_ for val_ in zip(*district_values)]
        # # convert yearly values into quartiles
        # quartile_values = []
        # for l_ in yearly_values:
        #     bucket = []
        #     low_q = (len(l_) + 1) * 0.25
        #     mid_q = (len(l_) + 1) * 0.5
        #     up_q = (len(l_) + 1) * 0.75
        #     # sort l to find index of given value to determine its quantile
        #     l_sorted = sorted(l_)
        #     for val in l_:
        #         val_index_sorted = l_sorted.index(val)
        #         if val_index_sorted <= low_q:
        #             qu_v = 1
        #             bucket.append(qu_v)
        #         elif val_index_sorted <= mid_q:
        #             qu_v = 2
        #             bucket.append(qu_v)
        #         elif val_index_sorted <= up_q:
        #             qu_v = 3
        #             bucket.append(qu_v)
        #         else:
        #             qu_v = 4
        #             bucket.append(qu_v)
        #     quartile_values.append(bucket)


        # display quartier mode counts over the years
        fig = go.Figure(go.Heatmap(
            z=yearly_values,
            x=list(data_quartier_dict.keys()),
            y=list(years),
            colorscale='RdBu',
            reversescale=True
            ))

        fig.update_layout(title=f"Mode: {mode[0].upper() + mode[1:]}",
                          xaxis_title="Quartier",
                          yaxis_title="Year",
                          font=dict(size=28)
                           )
        fig.update_annotations(font_size=28)
        fig.show()
        fig.write_html(os.path.join(OUTPUT_PATH, f'heatmap_{mode.upper()}.html'))



if __name__ == '__main__':
    FILTERED_LOCATION_STATISTICS_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_FINISH_FILTERED.csv"
    VIDEODATA_OUTPUT_FOLDER_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster"
    videodata_output_folder = 'yqOlY5uBBbo'

    forms_of_transportation = {
        "active": ["pedestrian", "bicycle", "dogwalker"],
        "motorised": ["car", "motorcycle", "truck"],
        "public": ["bus", "train", "boat"]
    }

    OUTPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster"
    PLOTS_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\plots"
    FILTERED_CSV_STATISTICS_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_FINISH_FILTERED.csv"
    # all_locations_map(FILTERED_LOCATION_STATISTICS_PATH)
    # all_locations_plot(FILTERED_CSV_STATISTICS_PATH, OUTPUT_PATH, forms_of_transportation)
    # data_exploration(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH, min_frequency=2)
    # temporal_analysis_per_location(FILTERED_CSV_STATISTICS_PATH, forms_of_transportation, PLOTS_PATH, min_frequency=9)
    # small_multiples_by_quartier()
    # frequency_of_frequencies_plot(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH)
    # weather_boxplots(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH)
    quartiers_per_year(PLOTS_PATH)
    # boxplots_COVID_by_quartier()
    # weather_boxplots_pedest_cyclist(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH)
    # location_names_by_frequency_plot(FILTERED_CSV_STATISTICS_PATH, PLOTS_PATH)
    # heatmap_matrix_plot()
    # heatmap_matrix_plot_quartiles(classes=20)