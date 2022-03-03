import os
import time
# figure plotting
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
# map plotting
import contextily as ctx

def map_plotting(location_names_df, VIDEO_FOLDER_PATH, video_name):
    ax = location_names_df.plot(figsize=(20, 20), linewidth=50, color='red')
    # label each location with its name
    location_names_df.apply(lambda x: ax.annotate(text='frame ' + x['frame_nr'] + ' - ' + x['location_name'],
                                   xy=x['geo'].centroid.coords[0], ha='center'), axis=1)
    # set spatial extend of axis based on detected freatures
    feature_bounds = location_names_df.geometry.total_bounds
    xlim = ([feature_bounds[0], feature_bounds[2]])
    ylim = ([feature_bounds[1], feature_bounds[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # add basemap
    ctx.add_basemap(ax, url=ctx.providers.OpenStreetMap.Mapnik)
    # save figure
    fig_filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{video_name}_map.png"
    fig_filepath = os.path.join(VIDEO_FOLDER_PATH, fig_filename)
    plt.savefig(fig_filepath)
    plt.show()

def figure_plotting(total_objects_dict, total_modes_dict, location_names_df, video_folder_path, video_name, bins_per_video = 20):
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'pedestrians',
                              'dogwalker', 'cyclist', 'motorcyclist', 'car_driver', 'truck_driver', 'bus_driver', 'train_driver']

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
    # delimiters between bins
    real_bins = [frames_per_bin * bin_ for bin_ in range(1, bins_per_video)]
    # position where plotted on x-axis
    x_bins = [(frames_per_bin / 2) + (frames_per_bin * bin_) for bin_ in range(bins_per_video)]
    # iterate over all input, add traces for each object, mode and add locations to x-axis
    for dict_ in dicts_:
        for key_, value in dict_.items():
            if key_ in classnames_to_consider:
                # aggregate counts across frames based on the defined bins_per_video
                counts_per_bin = []
                processed_frames = []
                for x_bin, real_bin in zip(x_bins, real_bins):
                    count_aggregate = 0
                    for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                        frame = int(frame)
                        if frame not in processed_frames and frame < real_bin:
                            count_aggregate += value_['count']
                            processed_frames.append(frame)
                    counts_per_bin.append(count_aggregate)
                # add trace to figure
                if max(counts_per_bin) > 0:
                    fig.add_trace(go.Bar(x=x_bins, y=counts_per_bin, name=key_, width=frames_per_bin))
                else:
                    print(f'[*] class {key_} not in plot, count: {count_aggregate}')

    # sort df by frame_nr and location name
    location_names_df.sort_values(by=['frame_nr', 'location_name'], inplace=True)
    # keep track of processed location names and their frame_nr, only include multiple same location if their frames are far apart
    plotted_locations = {}
    # if previously processed frames are to close to one another, they might overlap
    previously_processed_frame = {'frame_nr': 0, 'switch': True}
    allowed_frame_gap = 6000 # if more than 6000 frames are between the same location name they are plotted multiple times
    # add location_names at the frames where they were detected
    if location_names_df is not None:
        for index, row in location_names_df.iterrows():
            to_plot = False
            location_name = row['location_name']
            frame_nr = int(row['frame_nr'])
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

                # font = dict(
                #     size=16,
                #     color="#000000"
                # ),

    fig.update_layout(
        title=f"Video: {video_name}",
        xaxis_title="frames",
        yaxis_title="count",
        legend_title="Legend Title",
        barmode='stack'
    )
    # fig.show()
    # print('[+] saving figure as interactive HTML plot')
    # OUTPUT_HTML = os.path.join(video_folder_path, f'{video_name}_plot.html')
    # fig.write_html(OUTPUT_HTML)