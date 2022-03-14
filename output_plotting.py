import os
import time
import warnings
import pandas as pd
from itertools import product
warnings.simplefilter(action='ignore', category=FutureWarning)
# figure plotting
import plotly.graph_objs as go
from collections import defaultdict
# map plotting
import contextily as ctx
# from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.graphics.mosaicplot import mosaic

def mosaic_plot_locations(CSV_PATH, OUTPUT_PATH, forms_of_transportation):
    '''
    relative plot of transportation forms across all locations, easily comperable
    '''
    location_df = pd.read_csv(CSV_PATH, delimiter=';')
    # unique names frop by location is only one way to deal with the data - analysing difference instances of the same location
    # could also be extremely interesting from an analysis perspective
    location_df.drop_duplicates(subset=['location_name'], inplace=True, ignore_index=True)
    # aggregate forms of transport: active transport, motorised transport, public transport
    # and add to bar graph
    # stores the forms of transportation per location
    data = {}
    plot_data = {}

    for form_, value_ in forms_of_transportation.items():
        # holds the final bin values for a given transportation form
        form_counts_per_location = []
        # holds the bin values for the individual modes per form
        mode_bin_counts = []
        for transportation_mode in value_:
            transportation_mode_values = location_df[transportation_mode].values
            mode_bin_counts.append(transportation_mode_values)
        # calculate sum over all modes per form
        all_data_per_form = []
        for index, row in location_df.iterrows():
            added_bin_count = 0
            for list_ in mode_bin_counts:
                added_bin_count += list_[index]
            form_counts_per_location.append(added_bin_count)

            all_data_per_form.append(added_bin_count)
            data[(row['location_name'], form_)] = added_bin_count
        # add to plot_data
        plot_data[form_] = all_data_per_form

    def props(key):
        return {'color': '#AB63FA' if 'active' in key else ('#19D3F3' if 'public' in key else '#FFA15A')}

    def labelizer(key):
        # return data[key]
        return None

    # create df from plot_data
    plot_df = pd.DataFrame.from_dict(plot_data, orient='index', columns=location_df['location_name'])
    plot_df = plot_df.T
    # # sort data by highest active transportation values
    # def sort_func(x):
    #     form_ = x[0][1]
    #     if form_ == 'active':
    #         count = x[1]
    #         return count
    #     else:
    #         return 0
    # sort dictionary by highest share of active transportation
    plot_df['active_share'] = plot_df.apply(lambda x: x['active'] / (x['active'] + x['motorised'] + x['public']), axis=1)
    plot_df.sort_values(by=['active_share'], inplace=True)
    ordered_locations = plot_df.index.values.tolist()
    ordered_data = {}
    for location in reversed(ordered_locations):
        for form_ in ['active', 'motorised', 'public']:
            for key, value in data.items():
                unordered_location = key[0]
                unordered_form = key[1]
                if unordered_location == location and unordered_form == form_:
                    ordered_data[key] = value
    # data = {k: v for k, v in sorted(data.items(), key=lambda x: sort_func(x), reverse=True)}
    # data = {k: v for k, v in sorted(data.items(), key=lambda x: x[0][0])}
    # mosaic(plot_df_transposed, index=['active', 'motorised', 'public']) #, labelizer=labelizer
    mosaic(ordered_data, labelizer=labelizer, properties=props, gap=0.015, label_rotation=[90, 0]) #, labelizer=labelizer
    plt.show()
    print('[+] saving mosaic plot')
    OUTPUT_HTML = os.path.join(OUTPUT_PATH, 'location_statistics_plot.html')
    # fig.write_html(OUTPUT_HTML)

def all_locations_plot(CSV_PATH, OUTPUT_PATH, forms_of_transportation):
    '''
    think about how multiple location entries shall be treated

    '''
    location_df = pd.read_csv(CSV_PATH, delimiter=';')
    # unique names frop by location is only one way to deal with the data - analysing difference instances of the same location
    # could also be extremely interesting from an analysis perspective
    location_df.drop_duplicates(subset=['location_name'], inplace=True)
    # initialise figure
    fig = go.Figure()
    # # add location data
    # for classname in classnames_for_map:
    #     fig.add_trace(go.Bar(x=location_df['location_name'], y=location_df[classname], name=classname))
    # aggregate forms of transport: active transport, motorised transport, public transport
    # and add to bar graph

    colors = {
        "active": "#AB63FA",
        "motorised": "#FFA15A",
        "public": "#19D3F3"
    }
    # forms_of_transportation = {
    #     "active": ["pedestrians", "cyclist", "dogwalker"],
    #     "motorised": ["car_driver", "motorcyclist", "truck_driver"],
    #     "public": ["bus_driver", "train_driver"]
    # }
    for form_, value_ in forms_of_transportation.items():
        # holds the final bin values for a given transportation form
        form_counts_per_location = []
        # holds the bin values for the individual modes per form
        mode_bin_counts = []
        for transportation_mode in value_:
            transportation_mode_values = location_df[transportation_mode].values
            mode_bin_counts.append(transportation_mode_values)
        # calculate sum over all modes per form
        for index in range(len(location_df['location_name'])):
            added_bin_count = 0
            for list_ in mode_bin_counts:
                added_bin_count += list_[index]
            form_counts_per_location.append(added_bin_count)
        # add bar element to figure
        fig.add_trace(go.Bar(x=location_df['location_name'], y=form_counts_per_location, name=f"{form_} transport", marker_color=colors[form_]))

    # adapt layout to a stacked bar chart
    fig.update_layout(
        xaxis_title="location names",
        yaxis_title="count",
        barmode='stack',
        xaxis={'categoryorder': 'total descending'}
    )
    fig.show()
    print('[+] saving all locations figure as interactive HTML plot')
    OUTPUT_HTML = os.path.join(OUTPUT_PATH, 'location_statistics_plot.html')
    fig.write_html(OUTPUT_HTML)

def output_location_statistics(total_objects_dict, total_modes_dict, unique_location_names_df, VIDEOS_STORE_PATH, frame_buffer=2500):
    '''
    create location statistics in CSV output across all videos
    this includes the found objects, and transportation modes across videos around detected locations
    based on a defined frame_buffer +/- of the frame where the location was found.
    '''
    dicts_ = [total_objects_dict, total_modes_dict]
    classnames_to_output = "airplane;backpack;bench;bicycle;bird;boat;boat_driver;bus;bus_driver;car;car_driver;chair;clock;cyclist;dogwalker;handbag;motorcycle;motorcyclist;pedestrians;person;potted plant;refrigerator;skateboard;suitcase;traffic light;train;train_driver;truck;truck_driver".split(';')
    data_dict = defaultdict(lambda: {'frame_nr': None, 'lat': None, 'lng': None, 'classnames': defaultdict(dict)})
    # add lat, lng to dataframe as separate columns
    unique_location_names_df['lat'] = unique_location_names_df.apply(lambda x: x['geo'].centroid.coords[0][1], axis=1)
    unique_location_names_df['lng'] = unique_location_names_df.apply(lambda x: x['geo'].centroid.coords[0][0], axis=1)
    # iterate over all input, add traces for each object, mode and add locations to x-axis
    for row_index, row in unique_location_names_df.iterrows():
        location_name = row['location_name']
        location_frame_nr = int(row['frame_nr'])
        location_lng = row['lng']
        location_lat = row['lat']
        # defining frame buffer boundaries, in which modes and objects are counted and attributed to the location
        upper_frame_limit = location_frame_nr + int((frame_buffer / 2))
        lower_frame_limit = location_frame_nr - int((frame_buffer / 2))
        for dict_ in dicts_:
            for classname, value in dict_.items():
                # aggregate counts across frames based on the defined bins_per_video
                counts_in_frame_buffer = 0
                processed_frames = []
                for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                    frame = int(frame)
                    if frame >= lower_frame_limit and frame <= upper_frame_limit and frame not in processed_frames:
                        counts_in_frame_buffer += value_['count']
                        processed_frames.append(frame)
                data_dict[location_name]['classnames'][classname] = counts_in_frame_buffer
        data_dict[location_name]['frame_nr'] = location_frame_nr
        data_dict[location_name]['lat'] = location_lat
        data_dict[location_name]['lng'] = location_lng

    # write/append to CSV
    delimiter = ';'
    ## find index of path separator
    # slash_index = [i for i, ltr in enumerate(OUTPUT_VIDEO_FOLDER_PATH) if ltr == '\\'][-1]
    filename = f'location_statistics_{frame_buffer}_buffer.csv'
    # CSV_LOCATION_STATISTCS_OUTPUT_PATH = os.path.join(OUTPUT_VIDEO_FOLDER_PATH[:slash_index], filename)
    CSV_LOCATION_STATISTCS_OUTPUT_PATH = os.path.join(VIDEOS_STORE_PATH, filename)
    # check if file was already create from previous video analysis
    if not os.path.isfile(CSV_LOCATION_STATISTCS_OUTPUT_PATH):
        print(f'[!] locations statistics file NOT there')
        # create header for CSV
        base_string = f'{delimiter}'.join([classname for classname in classnames_to_output])
        header = f'location_name{delimiter}lng{delimiter}lat{delimiter}frame_nr{delimiter}' + base_string
        # newly create CSV and write header
        with open(CSV_LOCATION_STATISTCS_OUTPUT_PATH, 'wt', encoding='utf-8') as f:
            f.write(f'{header}\n')

    print(f'[!] locations statistics file  EXISTS')
    # append the location statistics from the current video to the output CSV
    with open(CSV_LOCATION_STATISTCS_OUTPUT_PATH, 'at', encoding='utf-8') as f:
        for location_name, value in data_dict.items():
            csv_location_line = f'{location_name}{delimiter}{value["lng"]}{delimiter}{value["lat"]}{delimiter}{value["frame_nr"]}'
            # to make sure it is the right order and missing key values are denoted with 0
            for classname in classnames_to_output:
                if classname in value['classnames']:
                    count_value = value['classnames'][classname]
                else:
                    count_value = 0
                csv_location_line += f'{delimiter}{count_value}'
            # write line for location
            f.write(f'{csv_location_line}\n')
    return CSV_LOCATION_STATISTCS_OUTPUT_PATH


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

def map_plotting(total_objects_dict, total_modes_dict, location_names_df, VIDEO_FOLDER_PATH, video_name, frame_buffer=2500):
    # count transportation modes and objects in a given buffer around detected locations, for location specific statistics
    # classnames considered for the map for function map_plotting
    classnames_for_map = ['pedestrians', 'cyclist', 'car_driver']
    dicts_ = [total_objects_dict, total_modes_dict]
    map_plotting_data = defaultdict(dict)
    # create unique location names df
    unique_location_names_df = location_names_df.drop_duplicates(subset=['location_name'])
    # check validity of object geometry to avoid parsing errors later on
    indexes_to_drop = []
    for index, row in unique_location_names_df.iterrows():
        try:
            row['geo'].centroid.coords[0]
        except:
            indexes_to_drop.append(index)
    unique_location_names_df.drop(indexes_to_drop, inplace=True)

    # set spatial extend of axis based on detected features
    feature_bounds = unique_location_names_df.geometry.total_bounds
    x_span = feature_bounds[2] - feature_bounds[0]
    y_span = feature_bounds[3] - feature_bounds[1]
    xlim = ([(feature_bounds[0] - (x_span * 0.5)), (feature_bounds[2] + (x_span * 0.5))])
    ylim = ([(feature_bounds[1] - (y_span * 0.5)), (feature_bounds[3] + (y_span * 0.5))])
    # iterate over all input, add traces for each object, mode and add locations to x-axis
    for row_index, row in unique_location_names_df.iterrows():
        location_name = row['location_name']
        location_frame_nr = int(row['frame_nr'])
        # defining frame buffer boundaries, in which modes and objects are counted and attributed to the location
        upper_frame_limit = location_frame_nr + int((frame_buffer / 2))
        lower_frame_limit = location_frame_nr - int((frame_buffer / 2))
        for dict_ in dicts_:
            for classname, value in dict_.items():
                if classname in classnames_for_map:
                    # aggregate counts across frames based on the defined bins_per_video
                    counts_in_frame_buffer = 0
                    processed_frames = []
                    for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                        frame = int(frame)
                        if frame >= lower_frame_limit and frame <= upper_frame_limit and frame not in processed_frames:
                            counts_in_frame_buffer += value_['count']
                            processed_frames.append(frame)
                    map_plotting_data[location_name][classname] = counts_in_frame_buffer

    # initialise figure
    fig, ax = plt.subplots(figsize=(15, 14))
    unique_location_names_df.plot(linewidth=4, color='red', ax=ax) #figsize=(20, 20),
    # add x y as separate columns
    y_offset = y_span * 0.05
    x_offset = x_span * 0.05
    unique_location_names_df['x'] = unique_location_names_df.centroid.map(lambda p: p.x) - x_offset
    unique_location_names_df['y'] = unique_location_names_df.centroid.map(lambda p: p.y) + y_offset

    bar_width = 0.5
    colors = ['green', 'orange', 'blue']

    for location_name, value in map_plotting_data.items():
        x_cord = unique_location_names_df.loc[unique_location_names_df.location_name == location_name, 'x'].copy()
        y_cord = unique_location_names_df.loc[unique_location_names_df.location_name == location_name, 'y'].copy()
        y_vals = [map_plotting_data[location_name][classname] for classname in classnames_for_map]
        # print(f'bar chart - location: {location_name}, y_vals: {y_vals}, classnames: {classnames_for_map}')
        bax = build_bar(x_cord, y_cord, ax, bar_width, xvals=['a', 'b', 'c'],
                        yvals=y_vals,
                        fcolors=colors)
    # label each location with its name
    unique_location_names_df.apply(lambda x: ax.annotate(text=x['location_name'],
                                                         xy=x['geo'].centroid.coords[0],
                                                         ha='center'),
                                                         axis=1)
    # create legend (of the 3 classes)
    legend_patch = []
    for index, classname in enumerate(classnames_for_map):
        patch = mpatches.Patch(color=colors[index], label=classnames_for_map[index])
        legend_patch.append(patch)
    ax.legend(handles=legend_patch, loc=1)
    # ylim = ([feature_bounds[1], feature_bounds[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()
    # add basemap
    ctx.add_basemap(ax, url=ctx.providers.OpenStreetMap.Mapnik)
    # save figure
    fig_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{video_name}_map.png"
    fig_filepath = os.path.join(VIDEO_FOLDER_PATH, fig_filename)
    plt.savefig(fig_filepath)
    # plt.show()
    return unique_location_names_df

def figure_plotting(total_objects_dict, total_modes_dict, location_names_df, video_folder_path, video_name, bins_per_video = 20):
    # classnames considered for this plot
    classnames_for_figure = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'pedestrians',
                              'dogwalker', 'cyclist', 'motorcyclist', 'car_driver', 'truck_driver', 'bus_driver', 'train_driver']
    # store counts per classname and bin to later aggregate different forms of transport (active, motorised, public)
    classnames_count_dict = defaultdict(lambda: {'counts_per_bin': []})
    # initialise figure
    fig = go.Figure()
    dicts_ = [total_objects_dict, total_modes_dict]
    # find highest frame with a present object - a plot x range will be defined from 0 to that max frame
    max_frame = 0
    for dict_ in dicts_:
        for key, value in dict_.items():
            frames = sorted(list(value['count_per_frame'].keys()), key=lambda x: int(x))
            if frames:
                if int(frames[-1]) > max_frame:
                    max_frame = int(frames[-1])
    # get all frames for the given object or transportation mode
    frames_per_bin = round(max_frame / bins_per_video)
    # delimiters between bins and add max frame at the end
    real_bins = [frames_per_bin * bin_ for bin_ in range(1, bins_per_video)]
    real_bins.append(max_frame)
    # position where plotted on x-axis
    x_bins = [(frames_per_bin / 2) + (frames_per_bin * bin_) for bin_ in range(bins_per_video)]
    # iterate over all input, add traces for each object, mode and add locations to x-axis
    for dict_ in dicts_:
        for key_, value in dict_.items():
            if key_ in classnames_for_figure:
                # aggregate counts across frames based on the defined bins_per_video
                counts_per_bin = []
                processed_frames = []
                for bin_index, (x_bin, real_bin) in enumerate(zip(x_bins, real_bins)):
                    count_aggregate = 0
                    for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                        frame = int(frame)
                        if frame not in processed_frames and frame < real_bin:
                            count_aggregate += value_['count']
                            processed_frames.append(frame)
                    counts_per_bin.append(count_aggregate)
                # add to result dict
                classnames_count_dict[key_]['counts_per_bin'] = counts_per_bin
                # add trace to figure
                if max(counts_per_bin) > 0:
                    fig.add_trace(go.Bar(x=x_bins, y=counts_per_bin, name=key_, width=frames_per_bin))
                else:
                    # print(f'[*] class {key_} not in plot, count: {count_aggregate}')
                    pass
    # aggregate forms of transport: active transport, motorised transport, public transport
    # and add to bar graph
    forms_of_transportation = {
        "active": ["pedestrians", "cyclist", "dogwalker"],
        "motorised": ["car_driver", "motorcyclist", "truck_driver"],
        "public": ["bus_driver", "train_driver"]
    }
    for form_, value_ in forms_of_transportation.items():
        # holds the final bin values for a given transportation form
        form_counts_per_bin = []
        # holds the bin values for the individual modes per form
        mode_bin_counts = []
        for transportation_mode in value_:
            mode_bin_counts.append(classnames_count_dict[transportation_mode]['counts_per_bin'])
        # calculate sum over all modes per form
        for index in range(len(x_bins)):
            added_bin_count = 0
            for list_ in mode_bin_counts:
                added_bin_count += list_[index]
            form_counts_per_bin.append(added_bin_count)
        # add bar element to figure
        fig.add_trace(go.Bar(x=x_bins, y=form_counts_per_bin, name=f"{form_} transport", width=frames_per_bin))


    # sort df by frame_nr and location name
    location_names_df.sort_values(by=['frame_nr', 'location_name'], inplace=True)
    # keep track of processed location names and their frame_nr, only include multiple same location if their frames are far apart
    plotted_locations = {}
    # if previously processed frames are to close to one another, they might overlap
    previously_processed_frame = {'frame_nr': 0, 'switch': True}
    allowed_frame_gap = 6000 # if more than 6000 frames are between the same location name they are plotted multiple times
    unique_location_names_df = location_names_df.drop_duplicates(subset=['location_name'])
    for index, row in unique_location_names_df.iterrows():
        location_name = row['location_name']
        frame_nr = int(row['frame_nr'])
        # add transportation mode counts also to dict for map plotting,
        # 1. find out in which bin the location_name is present
        location_bin_index = [index_ for index_, real_bin in enumerate(real_bins) if frame_nr < real_bin][0]

    # add location_names at the frames where they were detected
    if location_names_df is not None:
        for index, row in location_names_df.iterrows():
            to_plot = False
            location_name = row['location_name']
            frame_nr = int(row['frame_nr'])
            # process location names for figure annotation
            if location_name in plotted_locations:
                existing_frame_nr = plotted_locations[location_name]
                if (existing_frame_nr + allowed_frame_gap) <= frame_nr:
                    to_plot = True
                    # update frame of existing dict entry
                    plotted_locations[location_name] = frame_nr
                else:
                    continue
            else:
                to_plot = True
                # add to dict
                plotted_locations[location_name] = frame_nr
            if to_plot:
                # add offsets to avoid annotation overlap
                y_axis_offset = -30
                x_axis_offset = 20
                # check potential overlap between annotations
                if (previously_processed_frame['frame_nr'] + allowed_frame_gap) > frame_nr:
                    if previously_processed_frame['switch']:
                        y_axis_offset -= 30
                        x_axis_offset += 40
                        previously_processed_frame['switch'] = False
                    else:
                        previously_processed_frame['switch'] = True

                previously_processed_frame['frame_nr'] = frame_nr
                fig.add_annotation(
                    x=frame_nr,
                    y=0,
                    xref="x",
                    yref="y",
                    text=location_name,
                    showarrow=True,
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=x_axis_offset,
                    ay=y_axis_offset,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=1
                )
    fig.update_layout(
        xaxis_title="frames",
        yaxis_title="count",
        barmode='stack'
    )
    # fig.show()
    print('[+] saving figure as interactive HTML plot')
    OUTPUT_HTML = os.path.join(video_folder_path, f'{video_name}_plot.html')
    fig.write_html(OUTPUT_HTML)