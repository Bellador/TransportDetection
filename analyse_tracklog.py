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


def process_frame_objects(frame_objects, detected_transportation_modes_dict, all_unassigned_people_dict, all_assigned_object_dict):
    '''
    detect transportation modes based on the detected objects in the same frame and their
    spatial relation to another
    :param frame_objects: detected transport relevant objects of an entrie frame
    :param all_unassigned_people: dict of detected people (potential pedestrians), not yet assigned to a transportation mode
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
        if inside_bbox[0] >= outside_bbox[0] and inside_bbox[1] >= outside_bbox[1] and inside_bbox[2] <= outside_bbox[
            2] and inside_bbox[3] <= outside_bbox[3]:
            return True
        else:
            return False

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
    # people object ids and frame of unassigned people to keep track of, to at the end count appearances of pedestrians
    # add unassigned ids to all_unassigned_people_dict, and assigned_ids are removed when passed back to mother function

    # assigned ids that later must be removed from the all_unassigned_people_dict
    # frame_object: [frame_id, identity, class_id, class_name, bbox_tuple]
    for object_ in frame_objects:
        class_name = object_[3]
        if class_name in transportation_rel_obj_dict.keys():
            transportation_rel_obj_dict[class_name].append(object_)
    # to remove newly assigned person_ids from all_unassigned_people_dict in one go, they are stored in list
    newly_assigned_person_ids = []

    # all object ids passed for frame processing have already been checked for assignment to a transportation mode!
    for person in transportation_rel_obj_dict['person']:
        # person/object_ = [frame_id, identity, class_id, class_name, bbox_tuple]
        frame_nr = person[0]
        person_bbox = person[4]
        person_id = person[1]
        assigned = False
        # check dogwalkers
        if not assigned:
            for dog in transportation_rel_obj_dict['dog']:
                dog_id = dog[1]
                dog_bbox = dog[4]
                # check for overlap
                if calc_containing(person_bbox, dog_bbox, mode='bloated_bbox', bloating_factor=2):
                    detected_transportation_modes_dict['dogwalker']['total_count'] += 1
                    detected_transportation_modes_dict['dogwalker']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['dog'].append(dog_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check cyclists
        if not assigned:
            for bicycle in transportation_rel_obj_dict['bicycle']:
                bicycle_id = bicycle[1]
                bicycle_bbox = bicycle[4]
                # check for overlap
                if calc_containing(person_bbox, bicycle_bbox, mode='stacked_bbox'):
                    detected_transportation_modes_dict['cyclist']['total_count'] += 1
                    detected_transportation_modes_dict['cyclist']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['bicycle'].append(bicycle_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check motorcyclists
        if not assigned:
            for motorcycle in transportation_rel_obj_dict['motorcycle']:
                motorcycle_id = motorcycle[1]
                motorcycle_bbox = motorcycle[4]
                # check for overlap
                if calc_containing(person_bbox, motorcycle_bbox, mode='stacked_bbox'):
                    detected_transportation_modes_dict['motorcyclist']['total_count'] += 1
                    detected_transportation_modes_dict['motorcyclist']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['motorcycle'].append(motorcycle_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check cars
        if not assigned:
            for car in transportation_rel_obj_dict['car']:
                car_id = car[1]
                car_bbox = car[4]
                # check for overlap
                if calc_containing(car_bbox, person_bbox, mode='bloated_bbox', bloating_factor=4):
                    detected_transportation_modes_dict['car_driver']['total_count'] += 1
                    detected_transportation_modes_dict['car_driver']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['car'].append(car_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check trucks
        if not assigned:
            for truck in transportation_rel_obj_dict['truck']:
                truck_id = truck[1]
                truck_bbox = truck[4]
                # check for overlap
                if calc_containing(truck_bbox, person_bbox, mode='bloated_bbox', bloating_factor=5):
                    detected_transportation_modes_dict['truck_driver']['total_count'] += 1
                    detected_transportation_modes_dict['truck_driver']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['truck'].append(truck_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check buses
        if not assigned:
            for bus in transportation_rel_obj_dict['buse']:
                bus_id = bus[1]
                bus_bbox = bus[4]
                # check for overlap
                if calc_containing(bus_bbox, person_bbox, mode='bloated_bbox', bloating_factor=5):
                    detected_transportation_modes_dict['bus_driver']['total_count'] += 1
                    detected_transportation_modes_dict['bus_driver']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['bus'].append(bus_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check trains
        if not assigned:
            for train in transportation_rel_obj_dict['train']:
                train_id = train[1]
                train_bbox = train[4]
                # check for overlap
                if calc_containing(train_bbox, person_bbox, mode='bloated_bbox', bloating_factor=6):
                    detected_transportation_modes_dict['train_driver']['total_count'] += 1
                    detected_transportation_modes_dict['train_driver']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['train'].append(train_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # check boats
        if not assigned:
            for boat in transportation_rel_obj_dict['boat']:
                boat_id = boat[1]
                boat_bbox = boat[4]
                # check for overlap
                if calc_containing(boat_bbox, person_bbox, mode='bloated_bbox', bloating_factor=4):
                    detected_transportation_modes_dict['boat_driver']['total_count'] += 1
                    detected_transportation_modes_dict['boat_driver']['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict['boat'].append(boat_id)
                    all_assigned_object_dict['person'].append(person_id)
                    newly_assigned_person_ids.append(person_id)
                    assigned = True
        # at the end, if still not assigned, the person is a pedestrian
        if not assigned:
            # detected_transportation_modes_dict['pedestrian']['total_count'] += 1
            # detected_transportation_modes_dict['pedestrian']['count_per_frame'][frame_id]['count'] += 1
            if person_id not in all_unassigned_people_dict.keys():
                all_unassigned_people_dict[person_id]['frame_nr'] = frame_nr

    # remove all newly assigned person ids from the unassigned dict if present
    for person_id in newly_assigned_person_ids:
        all_unassigned_people_dict.pop(person_id, None)

    return detected_transportation_modes_dict, all_unassigned_people_dict, all_assigned_object_dict


def total_transportation_modes(LOG_PATH, OUTPUT_PATH):
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

    # coco_class_names
    names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
            ]

    # people object ids (potential pedestrians) not yet assigned to a transportation mode
    # key: person object id, value: frame nr of first occurence
    all_unassigned_people_dict = defaultdict(lambda: dict({'frame_nr': 0}))
    # store all assigned objects ids ordered by class id
    # key: class_name, value: list of object ids
    all_assigned_object_dict = defaultdict(lambda: [])
    # transportation modes
    detected_transportation_modes_dict = {
        'pedestrians': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # people walking
        'dogwalker': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # dog
        'cyclist': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # bicycle
        'motorcyclist': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # motorcycle
        'car_driver': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving car
        'truck_driver': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving truck
        'bus_driver': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving bus
        'train_driver': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving train
        'boat_driver': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))}  # moving boat
    }
    # open and iterate over tracking log
    with open(LOG_PATH, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
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
            object_id = int(line[1])
            # it's actually the corresponding class integer according to COCO mapping
            class_id = int(line[2])
            class_name = names[class_id]
            bbox_tuple = get_bbox(int(line[3]), int(line[4]), int(line[5]), int(line[6]))
            # check if object_id was already assigned to transportation mode
            if object_id not in all_assigned_object_dict[class_name]:
                if frame_id == start_frame:
                    # add object to frame_objects list
                    frame_objects.append([frame_id, object_id, class_id, class_name, bbox_tuple])
                    current_frame = frame_id
                elif frame_id == current_frame:
                    # add object to frame_objects list
                    frame_objects.append([frame_id, object_id, class_id, class_name, bbox_tuple])
                else:
                    # process all objects if all objects from the same frame were gathered
                    detected_transportation_modes_dict, all_unassigned_people_dict, all_assigned_object_dict = process_frame_objects(frame_objects, detected_transportation_modes_dict, all_unassigned_people_dict, all_assigned_object_dict)
                    # clear frame_objects and append current object to a new frame
                    frame_objects = []
                    frame_objects.append([frame_id, object_id, class_id, class_name, bbox_tuple])
                    # change current frame
                    current_frame = frame_id
                # processed_ids.append(object_id)
            # object_id already processed, skipping
            else:
                continue

        # at the end, all still unassigned person_ids will be regarded as pedestrians
        for people_id, value in all_unassigned_people_dict.items():
            frame_nr = value['frame_nr']
            detected_transportation_modes_dict['pedestrians']['total_count'] += 1
            detected_transportation_modes_dict['pedestrians']['count_per_frame'][frame_nr]['count'] += 1

        # write to output file
        TRANS_MODES_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_transportation_modes.csv'
        print_string = ""
        with open(os.path.join(OUTPUT_PATH, TRANS_MODES_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('transportation_mode;count\n')
            # write detected transportation modes
            for key, value in detected_transportation_modes_dict.items():
                _string = f"{key}: {value['total_count']}\n"
                f.write(_string)
                print_string += _string

        print('[+] detected active transportation modes:')
        print(print_string)
        return detected_transportation_modes_dict


def total_objects_per_class(LOG_PATH, OUTPUT_PATH, classnames_to_consider):
    # instantiate default dict that stores detected object information
    d = defaultdict(lambda: dict({'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))}))
    # coco_class_names
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
    # open and iterate over tracking log
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # keep track of the object ids that were already accounted for
        assigned_ids = []
        # skip header row
        next(reader)
        for index, line in enumerate(reader):
            frame_id = line[0]
            object_id = line[1]
            # it's actually the corresponding class integer according to COCO mapping
            class_id = int(line[2])
            class_name = names[class_id]
            # check if id was already seen
            if object_id not in assigned_ids:
                d[class_name]['total_count'] += 1
                # frame where the object was first recorded
                d[class_name]['count_per_frame'][frame_id]['count'] += 1
                assigned_ids.append(object_id)

        OBJECT_CLASS_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_class_names.csv'
        with open(os.path.join(OUTPUT_PATH, OBJECT_CLASS_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('class_name;count\n')
            # print output
            print('[+] detected mobility related objects:')
            for key, value in d.items():
                if key in classnames_to_consider:
                    print(f'{key}: {value["total_count"]}')
                    # add to output file
                    f.write(f'{key};{value["total_count"]}\n')
    return d

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
    total_objects = total_objects_per_class(LOG_PATH, OUTPUT_PATH, classnames_to_consider)
    total_modes = total_transportation_modes(LOG_PATH)
