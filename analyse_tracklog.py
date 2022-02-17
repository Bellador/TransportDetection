'''
Anlayse the video object tracking .csv log produced by `object_tracker.py` which are stored under
`./inference/output
- determine the total amount of objects per class throughout the video
- determine hotspots (frame sections) of objects in the video
'''
import os
import csv
import time
from collections import defaultdict

#csv.field_size_limit(sys.maxsize)

'''
TO DO:
- plot the temporal occurence of objects classes across frames
'''

def total_transportation_modes(LOG_PATH, source_str=None):
    '''
    - detect transportation modes through the position of the detected objects in relation to one another
        (e.g. person + bicycle = cyclist)
    - process all objects of the same frame together
    :param LOG_PATH:
    :return:
    '''
    def get_bbox(bbox_top, bbox_left, bbox_w, bbox_h):
        '''
        :param bbox_top:
        :param bbox_left:
        :param bbox_w:
        :param bbox_h:
        :return: bbox tuple bottom left, top right
        '''
        return (bbox_left, bbox_top - bbox_h, bbox_left + bbox_w, bbox_top)

    def process_frame_objects(frame_objects):
        '''
        detect transportation modes based on the detected objects in the same frame and their
        spatial relation to another
        :param frame_objects:
        :return:
        '''
        def calc_stacked_bbox(bbox):
            '''
            returns the vertically stacked bbox by lowering down the bottom left corner. The top right corner remains
            pretty much at the same place. Resultingly the output bbox is double in vertical hight - downwards.
            To identify e.g. cyclists or motorcyclists
            :return:
            '''
            # xb = bbox[0]
            # yb = bbox[1]
            # xt = bbox[2]
            # yt = bbox[3]
            # n_xb = xb - (1.5 * (xt - xb))
            # n_yb = 0.95 * yb
            # n_xt = 1.05 * xt
            # n_yt = 1.05 * yt

            xb = bbox[0]
            yb = bbox[1]
            xt = bbox[2]
            yt = bbox[3]
            n_xb = xb - abs(0.05 * xb)  # xb - (1.5 * (xt - xb))
            n_yb = yb - abs(1.5 * (yt - yb))  # 0.95 * yb
            n_xt = xt + abs(0.05 * xt)
            n_yt = yt + abs(0.05 * yt)
            return (n_xb, n_yb, n_xt, n_yt)

        def calc_bloated_bbox(bbox, factor=2):
            '''
            retournes volume increased bbox based upon the input bbox and a bloating factor. To identify e.g. dogwalkers
            The larger the factor the lower the volume increase!
            :return:
            '''
            # (bbox_top - bbox_h, bbox_left, bbox_top, bbox_left + bbox_w)
            xb = bbox[0]
            yb = bbox[1]
            xt = bbox[2]
            yt = bbox[3]
            substract_with = abs(xt - xb) / factor
            substract_height = abs(yt - yb) / factor
            return (xb - substract_with, yb - substract_height, xt + substract_with, yt + substract_height)

        def calc_containing(outside_bbox, inside_bbox, mode='stacked_bbox', bloating_factor=2):
            '''
            calaculates if 
            The mode defines the way the bboxes are modified before overlapping
            :param outside_bbox:
            :param inside_bbox:
            :param mode: stacked_bbox (person bbox ontop of other bbox), bloated_bbox (person bbox inside of other bbox)
            :param bloating_factor: 2 (for calc_bloated_bbox - the higher the smaller the volume increase)
            :return:
            '''
            if mode == 'stacked_bbox':
                outside_bbox = calc_stacked_bbox(outside_bbox)
            elif mode == 'bloated_bbox':
                outside_bbox = calc_bloated_bbox(outside_bbox, bloating_factor)
            if inside_bbox[0] >= outside_bbox[0] and inside_bbox[1] >= outside_bbox[1] and inside_bbox[2] <= outside_bbox[2] and inside_bbox[3] <= outside_bbox[3]:
                return True
            else:
                return False

        # track detected transportation modes
        # pedestrians = 0   # person - NEW will be defined at the end according to all people not assigned to other transportation modes

        detected_transportation_modes_dict = {
            'dogwalker': 0,  # dog
            'cyclist': 0,  # bicycle
            'motorcyclist': 0,  # motorcycle
            'car_driver': 0,  # moving car
            'truck_driver': 0,  # moving truck
            'bus_driver': 0,  # moving bus
            'train_driver': 0,  # moving train
            'boat_driver': 0,  # moving boat
            'not_assigned_persons': []  # basically pedestrians since not attriubted to any other transportation relevant object
        }

        # store transportation mode relevant objects to check their spatial relation
        transportation_rel_obj_dict = {
            'person': [],
            'dog': [],
            'bicycle': [],
            'motorcycle': [],
            'car': [],
            'truck': [],
            'buse': [],
            'train': [],
            'boat': []
        }
        # frame_object: [frame_id, identity, class_id, class_name, bbox_tuple]
        for object_ in frame_objects:
            class_name = object_[3]
            if class_name in transportation_rel_obj_dict.keys():
                transportation_rel_obj_dict[class_name].append(object_)

        # track which people were already assigned to a transportation to avoid duplicate assignment
        assigned_ids = []
        for person in transportation_rel_obj_dict['person']:
            person_id = person[1]
            person_bbox = person[4]
            assigned = False
            # check dogwalkers
            if not assigned:
                for dog in transportation_rel_obj_dict['dog']:
                    dog_id = dog[1]
                    dog_bbox = dog[4]
                    # check for overlap
                    if calc_containing(person_bbox, dog_bbox, mode='bloated_bbox', bloating_factor=2):
                        detected_transportation_modes_dict['dogwalker'] += 1
                        assigned_ids = assigned_ids + [person_id, dog_id]
                        assigned = True
            # check cyclists
            if not assigned:
                for bicycle in transportation_rel_obj_dict['bicycle']:
                    bicycle_id = bicycle[1]
                    bicycle_bbox = bicycle[4]
                    # check for overlap
                    if calc_containing(person_bbox, bicycle_bbox, mode='stacked_bbox'):
                        detected_transportation_modes_dict['cyclist'] += 1
                        assigned_ids = assigned_ids + [person_id, bicycle_id]
                        assigned = True
            # check motorcyclists
            if not assigned:
                for motorcycle in transportation_rel_obj_dict['motorcycle']:
                    motorcycle_id = motorcycle[1]
                    motorcycle_bbox = motorcycle[4]
                    # check for overlap
                    if calc_containing(person_bbox, motorcycle_bbox, mode='stacked_bbox'):
                        detected_transportation_modes_dict['motorcyclist'] += 1
                        assigned_ids = assigned_ids + [person_id, motorcycle_id]
                        assigned = True
            # check cars
            if not assigned:
                for car in transportation_rel_obj_dict['car']:
                    car_id = car[1]
                    car_bbox = car[4]
                    # check for overlap
                    if calc_containing(car_bbox, person_bbox, mode='bloated_bbox', bloating_factor=4):
                        detected_transportation_modes_dict['car_driver'] += 1
                        assigned_ids = assigned_ids + [person_id, car_id]
                        assigned = True
            # check trucks
            if not assigned:
                for truck in transportation_rel_obj_dict['truck']:
                    truck_id = truck[1]
                    truck_bbox = truck[4]
                    # check for overlap
                    if calc_containing(truck_bbox, person_bbox, mode='bloated_bbox', bloating_factor=5):
                        detected_transportation_modes_dict['truck_driver'] += 1
                        assigned_ids = assigned_ids + [person_id, truck_id]
                        assigned = True
            # check buses
            if not assigned:
                for bus in transportation_rel_obj_dict['buse']:
                    bus_id = bus[1]
                    bus_bbox = bus[4]
                    # check for overlap
                    if calc_containing(bus_bbox, person_bbox, mode='bloated_bbox', bloating_factor=5):
                        detected_transportation_modes_dict['bus_driver'] += 1
                        assigned_ids = assigned_ids + [person_id, bus_id]
                        assigned = True
            # check trains
            if not assigned:
                for train in transportation_rel_obj_dict['train']:
                    train_id = train[1]
                    train_bbox = train[4]
                    # check for overlap
                    if calc_containing(train_bbox, person_bbox, mode='bloated_bbox', bloating_factor=6):
                        detected_transportation_modes_dict['train_driver'] += 1
                        assigned_ids = assigned_ids + [person_id, train_id]
                        assigned = True
            # check boats
            if not assigned:
                for boat in transportation_rel_obj_dict['boat']:
                    boat_id = boat[1]
                    boat_bbox = boat[4]
                    # check for overlap
                    if calc_containing(boat_bbox, person_bbox, mode='bloated_bbox', bloating_factor=4):
                        detected_transportation_modes_dict['boat_driver'] += 1
                        assigned_ids = assigned_ids + [person_id, boat_id]
                        assigned = True
            # at the end, if still not assigned, add person to list that can be matched later on
            if not assigned:
                detected_transportation_modes_dict['not_assigned_persons'].append(person_id)

        return detected_transportation_modes_dict, assigned_ids

    # coco_class_names
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
    # transportation modes
    total_detected_transportation_modes_dict = {
        'dogwalker': 0,  # dog
        'cyclist': 0,  # bicycle
        'motorcyclist': 0,  # motorcycle
        'car_driver': 0,  # moving car
        'truck_driver': 0,  # moving truck
        'bus_driver': 0,  # moving bus
        'train_driver': 0,  # moving train
        'boat_driver': 0,  # moving boat
        'not_assigned_persons': [] # keep track of people not assigned yet to a transportation mode but maybe later on (due to detection errors)
        # basically pedestrians since not attributed to any other transportation relevant object
    }
    # open and iterate over tracking log
    with open(LOG_PATH, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # keep track of the object ids that were already accounted for
        assigned_ids = []
        # store object information of the same video frame
        frame_objects = []
        start_frame = 0
        current_frame = 0
        # skip header row
        next(reader)
        for index, line in enumerate(reader):
            frame_id = int(line[0])
            if index == 0:
                start_frame = frame_id
            identity = int(line[1])
            # it's actually the corresponding class integer according to COCO mapping
            class_id = int(line[2])
            class_name = names[class_id]
            bbox_tuple = get_bbox(int(line[3]), int(line[4]), int(line[5]), int(line[6]))
            # if identity == 736 or identity == 691:
            #     print(f'frame: {frame_id}, id: {identity}, {bbox_tuple}, {int(line[3]), int(line[4]), int(line[5]), int(line[6])}')
            # check if identity was already assigned as one specific transportation mode
            if identity not in assigned_ids:
                if frame_id == start_frame:
                    # add object to frame_objects list
                    frame_objects.append([frame_id, identity, class_id, class_name, bbox_tuple])
                    current_frame = frame_id
                elif frame_id == current_frame:
                    # add object to frame_objects list
                    frame_objects.append([frame_id, identity, class_id, class_name, bbox_tuple])
                else:
                    # process all objects if all objects from the same frame were gathered
                    detected_transportation_modes_dict, add_to_assigned_ids = process_frame_objects(frame_objects)
                    # update total detected transportation mode counters
                    for key, value in detected_transportation_modes_dict.items():
                        total_detected_transportation_modes_dict[key] += value
                    # update assigned_ids with ids that were assigned to transportation modes
                    assigned_ids += add_to_assigned_ids
                    # remove assigned ids from not_assigned_persons
                    # total_detected_transportation_modes_dict['not_assigned_persons'] = set(total_detected_transportation_modes_dict['not_assigned_persons'] + detected_transportation_modes_dict['not_assigned_persons'])
                    # clear frame_objects and append current object to a new frame
                    frame_objects = []
                    frame_objects.append([frame_id, identity, class_id, class_name, bbox_tuple])
                    # change current frame
                    current_frame = frame_id
                # processed_ids.append(identity)
            # identity already processed, skipping
            else:
                continue
        # define amount of pedestrians according to the still not assigned and detected people
        total_detected_transportation_modes_dict["pedestrians"] = len(set(total_detected_transportation_modes_dict['not_assigned_persons']))
        # write to output file
        TRANS_MODES_OUTPUT_PATH = './output/'
        TRANS_MODES_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_source_{source_str}_transportation_modes.csv'
        print_string = ""
        with open(os.path.join(TRANS_MODES_OUTPUT_PATH, TRANS_MODES_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('transportation_mode;count\n')
            # write detected transportation modes
            for key, value in total_detected_transportation_modes_dict.items():
                if key != 'not_assigned_persons':
                    _string = f'{key}: {value}\n'
                    f.write(_string)
                    print_string += _string

        print('[+] detected active transportation modes:')
        print(print_string)


def total_objects_per_class(LOG_PATH, classnames_to_consider, source_str=None):
    # instantiate default dict that stores detected object information
    d = defaultdict(lambda: dict({'count': 0, 'frames': [], 'coco_id': None}))
    # coco_class_names
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
    # open and iterate over tracking log
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # keep track of the object ids that were already accounted for
        assigned_ids = []
        # skip header row
        next(reader)
        for index, line in enumerate(reader):
            frame_id = line[0]
            identity = line[1]
            # it's actually the corresponding class integer according to COCO mapping
            class_id = int(line[2])
            class_name = names[class_id]
            # check if id was already seen
            if identity not in assigned_ids:
                d[class_name]['coco_id'] = class_id
                d[class_name]['count'] += 1
                # frame where the object was first recorded
                d[class_name]['frames'].append(frame_id)
                assigned_ids.append(identity)

        OBJECT_CLASS_OUTPUT_PATH = './output/'
        OBJECT_CLASS_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_source_{source_str}_class_names.csv'
        with open(os.path.join(OBJECT_CLASS_OUTPUT_PATH, OBJECT_CLASS_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('class_name;count\n')
            # print output
            print('[+] detected mobility related objects:')
            for key, value in d.items():
                if key in classnames_to_consider:
                    print(f'{key}: {value["count"]}')
                    # add to output file
                    f.write(f'{key};{value["count"]}\n')

def highest_ocr_confidence(OCR_LOG_PATH, confidence_th = 0.5, min_str_length = 3, max_nr_special_chars=1):
    # special characters that are a sign of artificats during the ocr
    special_chars = '][()|_~€$£'
    # open and iterate over tracking log
    with open(OCR_LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar=';')
        # skip header
        next(reader)
        for index, line in enumerate(reader):
            frame_num = line[0]
            text = line[1]

            try:
                confidence = round(float(line[2]), 2)
            except Exception as e:
                # print(confidence)
                pass

            bbox_xmin = line[3]
            bbox_ymin = line[4]
            bbox_xmax = line[5]
            bbox_ymax = line[6]

            len_text = len(text)
            # filter steps

            if confidence > confidence_th:
                if len_text >= min_str_length:
                    count_special_chars = sum([1 for c in text if c in special_chars])
                    if count_special_chars <= max_nr_special_chars:
                        print(f'{frame_num} - {confidence}: {text}')


if __name__ == '__main__':
    LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\20220216-152057_source_5min_excerpt_Old_Town_walk_in_Geneva_Switzerland_Autumn_2020_tracklog.csv"
    OCR_LOG_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\from_cluster\20220216-152057_source_5min_excerpt_Old_Town_walk_in_Geneva_Switzerland_Autumn_2020_ocrlog.csv"
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog']
    # highest_ocr_confidence(OCR_LOG_PATH)
    total_objects_per_class(LOG_PATH, classnames_to_consider, source_str='')
    total_transportation_modes(LOG_PATH, source_str='')
