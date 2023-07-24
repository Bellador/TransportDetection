# Transport Mode Detection Pipeline in Videos

## Into

This repository holds a full fledged pipeline to analyse collections of videos to obtain spatio-temporal transport mode statistics. Videos of interest for this project contain user-generated street view imagery. Our project in the field of Urban Analytics aims to explore mobility patterns and estimate the sustainability of the transport mix in a city over space and time by detecting varying forms of active (pedestrians, cyclists), motorised (cars, trucks) and public (trams, trains, busses) transportation. By using alternative data sources we try to complement existing e.g stationary count station data or Google Street View imagery to fill in existing data gaps. 


## Workflow

![workflow](workflow.png)

The workflow is separated into two distinct parts.
The first part deals with the transport relevant objects. Therein the objects (e.g. people and bikes) are detected ([YOLOv5](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/)) and tracked ([DeepSort](https://github.com/nwojke/deep_sort)) across frames. Subsequently, transport modes are derived based on the spatial inter-frame relation of said objects.

:bicyclist: = :walking: + :bike:

The second part deals with context information around the detected modes of transport. Attributing a location to certain video segments - so called geolocation - allows spatial linkage. This was achieved twofold. Firstly, the video frames were scanned for street names through Optical Character Recognition (OCR) ([EasyOCR](https://pypi.org/project/easyocr/)) and secondly, manually added video timestamps by the author were scanned for mentions of specific locations. The textual video-metadata (title and description) are besides the timestamps also scanned for mentions about weather conditions (e.g. sunny/rainy) and concrete time references of the recording.

For the pipeline to run autonomosly, it encompasses a Youtube Video Downloader ([PyTube](https://github.com/pytube/pytube) with a custom wrapper) and Youtube API module (YoutubeAPI-Query) to obtain all necessary data to run. 


## Output

The workflow generates a project folder for each Youtube video query, that contains:

- Location summary CSV that holds all necessary information for each detected location - aggregated over all processed videos in a given project. Each location holds information on the detected transport modes, weather and date
- Metadata video CSV that holds information on Frames Per Second (FPS), frame width and height for all videos
- Folder `plots` that stores visualisations
- Folder `yt_videos` that holds an individual folder for each processed video

Each video folder contains:

- Track log CSV of all object IDs and their respective object class across frames
- OCR log CSV of all text strings extracted from all frames
- Classnames CSV, a statistical summary of all detected classnames
- Transportation mode CSV, a statistical summary of all detected transportation modes
- Video MP4, containing visual bounding boxes and object ids of detections (only if `save_vid = True`)

### Plots and figures
all visualisation that are created by the workflow are stored in a dedicated folder `plots` found within the project folder.
These include so far:
- all locations ploted by min. frequency (stored in subfolder `temp_analysis_freq_n`) 
- all locations ploted by active, motorised and public transportation counts
- map with all locations


## Setup

To run the `workflow.py` complete the following steps:

1. Create the necessary virtual environment (preferably Anaconda)
- `conda env create --file ENV.yml`
- `conda activate <ENV>`

3. Change parameters under `<ADAPT PARAMETERS HERE>` in `workflow.py`
4. Run `nohup python -u workflow.py > mylogfile.log &` 



## External resources

The workflow is based on external libraries for (1) object detection and (2) object tracking as well as for (3) Optical Character Recognition (OCR) and (4) video downloading. TThese resources were taken from:

- (1) [YOLOv5](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/)
- (2) [DeepSort](https://github.com/nwojke/deep_sort)
- (3) [EasyOCR](https://pypi.org/project/easyocr/)
- (4) [PyTube](https://github.com/pytube/pytube)