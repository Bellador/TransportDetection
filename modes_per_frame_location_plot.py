import os
# plotting
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def plotting(total_objects_dict, total_modes_dict, location_names_df, video_folder_path, video_name, bins_per_video = 20):
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
        for key, value in dict_.items():
            if key in classnames_to_consider:
                # aggregate counts across frames based on the defined bins_per_video
                counts_per_bin = []
                processed_frames = []
                for real_bin in real_bins:
                    count_aggregate = 0
                    for index, (frame, value_) in enumerate(value['count_per_frame'].items()):
                        frame = int(frame)
                        if frame not in processed_frames and frame < real_bin:
                            count_aggregate += value_['count']
                            processed_frames.append(frame)
                    counts_per_bin.append(count_aggregate)
                # debugging
                if key == 'cyclist':
                    print()
                    pass
                # add trace to figure
                if max(counts_per_bin) > 0:
                    fig.add_trace(go.Bar(x=x_bins, y=counts_per_bin, name=key, width=frames_per_bin))
                else:
                    print(f'[*] class {key} not in plot, count: {count_aggregate}')

    # add location_names at the frames where they were detected
    if location_names_df is not None:
        for index, row in location_names_df.iterrows():
            # fig.add_trace(go.Bar(x=location_names_df['frame_nr'], y=, text=location_names_df['location_name'], name='locations'))
            fig.add_annotation(
                x=int(row['frame_nr']),
                y=0,
                xref="x",
                yref="y",
                text=row['location_name'],
                showarrow=True,
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=20,
                ay=-30,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
            )

    fig.update_layout(
        title=f"Video: {video_name}",
        xaxis_title="frames",
        yaxis_title="count",
        legend_title="Legend Title"
    )
    fig.show()
    print('[+] saving figure as interactive HTML plot')
    OUTPUT_HTML = os.path.join(video_folder_path, f'{video_name}_plot.html')
    fig.write_html(OUTPUT_HTML)