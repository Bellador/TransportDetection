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
- plot the temporal occurrence of objects classes across frames
'''


def process_frame_objects(frame_objects, det_transport_modes_dict, all_unassigned_people_dict, all_assigned_object_dict):
    '''
    detect transportation modes based on the detected objects in the same frame and their
    spatial relation to another
    :param frame_objects: detected transport relevant objects of an entrie frame
    :param all_unassigned_people: dict of detected people (potential pedestrian), not yet assigned to a transportation mode
    :return:
    '''

    def calc_stacked_bbox(bbox):
        '''
        returns the vertically stacked bbox by lowering down the bottom left corner. The top right corner remains
        pretty much at the same place. The resulting output bbox is double in vertical height - downwards.
        To identify e.g. cyclists or motorcyclists
        :return:
        '''

        xb = bbox[0]
        yb = bbox[1]
        xt = bbox[2]
        yt = bbox[3]
        n_xb = xb - abs(0.05 * xb)  # xb - (1.5 * (xt - xb))
        n_yb = yb - abs(0.6 * (yt - yb))  # yb - abs(1.5 * (yt - yb))
        n_xt = xt + abs(0.05 * xt)
        n_yt = yt + abs(0.05 * yt)
        return (n_xb, n_yb, n_xt, n_yt)

    def calc_bloated_bbox(bbox, bloating_factor=1):
        '''
        returns volume increased bbox based upon the input bbox and a bloating factor. To identify e.g. dogwalkers
        The larger the factor the lower the volume increase!
        :return:
        '''
        if bloating_factor == 0:
            return bbox
        else:
            # (bbox_top - bbox_h, bbox_left, bbox_top, bbox_left + bbox_w)
            xb = bbox[0]
            yb = bbox[1]
            xt = bbox[2]
            yt = bbox[3]
            diff_width = ((abs(xt - xb) * bloating_factor) - abs(xt - xb)) / 2
            diff_height = ((abs(yt - yb) * bloating_factor) - abs(yt - yb)) / 2
            return (xb - diff_width, yb - diff_height, xt + diff_width, yt + diff_height)

    def calc_containing(outside_bbox, inside_bbox, mode='stacked_bbox', bloating_factor=None):
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
            outside_bbox = calc_bloated_bbox(outside_bbox, bloating_factor=bloating_factor)
        if inside_bbox[0] >= outside_bbox[0] and inside_bbox[1] >= outside_bbox[1] and inside_bbox[2] <= outside_bbox[2] and inside_bbox[3] <= outside_bbox[3]:
            return True
        else:
            return False

    def assign_transportation_mode(class_name, person_bbox, transport_rel_obj_dict, det_transport_modes_dict, assigned):
        '''

        :param class_name:
        :param person_bbox:
        :param transport_rel_obj_dict:
        :param det_transport_modes_dict:
        :param assigned:
        :return:
        '''
        mode = transport_rel_obj_dict[class_name]['mode']
        bloating_factor = transport_rel_obj_dict[class_name]['bloating_factor']
        for obj in transport_rel_obj_dict[class_name]['obj_list']:
            obj_id = obj[1]
            obj_bbox = obj[4]
            link_mode = transport_rel_obj_dict[class_name]['link']
            if not assigned and obj_id not in all_assigned_object_dict[class_name]:
                # if the link mode contains 'passenger' than the order person_bbox and obj_bbox has to be switched!
                bbox_order = [person_bbox, obj_bbox]
                if 'passenger' in link_mode:
                    bbox_order = [obj_bbox, person_bbox]
                # check for overlap
                if calc_containing(bbox_order[0], bbox_order[1], mode=mode, bloating_factor=bloating_factor):
                    det_transport_modes_dict[link_mode]['total_count'] += 1
                    det_transport_modes_dict[link_mode]['count_per_frame'][frame_nr]['count'] += 1
                    all_assigned_object_dict[class_name].append(obj_id)
                    all_assigned_object_dict['person'].append(person_id)
                    assigned = True
            else:
                break

        return transport_rel_obj_dict, det_transport_modes_dict, assigned

    # ------------------------------------------------------------------------------------------------------------------

    # store transportation mode relevant objects to check their spatial relation
    transport_rel_obj_dict = {
        'person': {'obj_list': [], 'link': 'pedestrian', 'mode': None, 'bloating_factor': None},
        'dog': {'obj_list': [], 'link': 'dogwalker', 'mode': 'bloated_bbox', 'bloating_factor': 1},
        'bicycle': {'obj_list': [], 'link': 'cyclist', 'mode': 'stacked_bbox', 'bloating_factor': None},
        'motorcycle': {'obj_list': [], 'link': 'motorcyclist', 'mode': 'stacked_bbox', 'bloating_factor': None},
        'car': {'obj_list': [], 'link': 'car_passenger', 'mode': 'bloated_bbox', 'bloating_factor': 0},
        'truck': {'obj_list': [], 'link': 'truck_passenger', 'mode': 'bloated_bbox', 'bloating_factor': 0},
        'bus': {'obj_list': [], 'link': 'bus_passenger', 'mode': 'bloated_bbox', 'bloating_factor': 0},
        'train': {'obj_list': [], 'link': 'train_passenger', 'mode': 'bloated_bbox', 'bloating_factor': 0},
        'boat': {'obj_list': [], 'link': 'boat_passenger', 'mode': 'bloated_bbox', 'bloating_factor': 0}
    }

    # assigned ids that later must be removed from the all_unassigned_people_dict
    # frame_object: [frame_id, identity, class_id, class_name, bbox_tuple]
    for object_ in frame_objects:
        class_name = object_[3]
        if class_name in transport_rel_obj_dict:
            transport_rel_obj_dict[class_name]['obj_list'].append(object_)

    # to remove newly assigned person_ids from all_unassigned_people_dict in one go, they are stored in list
    newly_assigned_person_ids = []

    # all object ids passed for frame processing have already been checked for assignment to a transportation mode!
    for person in transport_rel_obj_dict['person']['obj_list']:
        # person/object_ = [frame_id, identity, class_id, class_name, bbox_tuple]
        frame_nr = person[0]
        person_id = person[1]
        person_bbox = person[4]
        # make sure person is not assigned multiple times within the same frame
        if person_id not in newly_assigned_person_ids:
            assigned = False
            for class_name, v in transport_rel_obj_dict.items():
                if class_name != 'person' and transport_rel_obj_dict[class_name]['obj_list'] and not assigned:
                    transport_rel_obj_dict, det_transport_modes_dict, assigned = assign_transportation_mode(class_name,
                                                                                    person_bbox,
                                                                                    transport_rel_obj_dict,
                                                                                    det_transport_modes_dict,
                                                                                    assigned)
            # at the end, if still not assigned, the person is noted as 'unassigned'
            if not assigned:
                if person_id not in all_unassigned_people_dict.keys():
                    all_unassigned_people_dict[person_id]['frame_nr'] = frame_nr
            else:
                newly_assigned_person_ids.append(person_id)

    # remove all newly assigned person ids from the unassigned dict if present
    for person_id in newly_assigned_person_ids:
        all_unassigned_people_dict.pop(person_id, None)

    return det_transport_modes_dict, all_unassigned_people_dict, all_assigned_object_dict


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

    # people object ids (potential pedestrian) not yet assigned to a transportation mode
    # key: person object id, value: frame nr of first occurrence
    all_unassigned_people_dict = defaultdict(lambda: dict({'frame_nr': 0}))
    # store all assigned objects ids ordered by class id
    # key: class_name, value: list of object ids
    all_assigned_object_dict = defaultdict(lambda: [])
    # transportation modes
    det_transport_modes_dict = {
        'pedestrian': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # people walking
        'dogwalker': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # dog
        'cyclist': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # bicycle
        'motorcyclist': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # motorcycle
        'car_passenger': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving car
        'truck_passenger': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving truck
        'bus_passenger': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving bus
        'train_passenger': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))},  # moving train
        'boat_passenger': {'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))}  # moving boat
    }
    # open and iterate over tracking log
    with open(LOG_PATH, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # store object information of the same video frame
        frame_objects = []
        # skip header row
        next(reader)
        for index, line in enumerate(reader):
            frame_idx = int(line[0])
            if index == 0:
                start_frame_idx = frame_idx
            object_id = int(line[1])
            # it's actually the corresponding class integer according to COCO mapping
            class_id = int(line[2])
            class_name = names[class_id]
            bbox_tuple = get_bbox(int(line[3]), int(line[4]), int(line[5]), int(line[6]))
            # check if object_id was already assigned to transportation mode
            if object_id not in all_assigned_object_dict[class_name]:

                if frame_idx == start_frame_idx:
                    # add object to frame_objects list
                    frame_objects.append([frame_idx, object_id, class_id, class_name, bbox_tuple])
                    current_frame_idx = frame_idx
                elif frame_idx == current_frame_idx:
                    # add object to frame_objects list
                    frame_objects.append([frame_idx, object_id, class_id, class_name, bbox_tuple])
                else:
                    # process all objects if all objects from the same frame were gathered
                    det_transport_modes_dict, all_unassigned_people_dict, all_assigned_object_dict = process_frame_objects(frame_objects,
                                                                                                       det_transport_modes_dict,
                                                                                                       all_unassigned_people_dict,
                                                                                                       all_assigned_object_dict)
                    # clear frame_objects and append current object to a new frame
                    frame_objects = []
                    if object_id not in all_assigned_object_dict[class_name]:
                        frame_objects.append([frame_idx, object_id, class_id, class_name, bbox_tuple])
                    # change current frame
                    current_frame_idx = frame_idx

            # object_id already processed, skipping
            else:
                continue
        # at the end process the remaining frame_obj
        det_transport_modes_dict, all_unassigned_people_dict, all_assigned_object_dict = process_frame_objects(frame_objects,
                                                                                                                det_transport_modes_dict,
                                                                                                                all_unassigned_people_dict,
                                                                                                                all_assigned_object_dict)
        # at the end, all still unassigned person_ids will be regarded as pedestrian
        for people_id, value in all_unassigned_people_dict.items():
            frame_nr = value['frame_nr']
            det_transport_modes_dict['pedestrian']['total_count'] += 1
            det_transport_modes_dict['pedestrian']['count_per_frame'][frame_nr]['count'] += 1

        # write to output file
        TRANS_MODES_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_transportation_modes.csv'
        print_string = ""
        # with open(os.path.join(OUTPUT_PATH, TRANS_MODES_FILENAME), 'wt', encoding='utf-8') as f:
        #     # header
        #     f.write('transportation_mode;count\n')
        #     # write detected transportation modes
        #     for key, value in det_transport_modes_dict.items():
        #         _string = f"{key};{value['total_count']}\n"
        #         f.write(_string)
        #         print_string += _string

    print('[+] detecting active transportation modes. Done.')
    return det_transport_modes_dict


def total_objects_per_class(LOG_PATH, OUTPUT_PATH):
    # instantiate default dict that stores detected object information
    d = defaultdict(lambda: dict({'total_count': 0, 'count_per_frame': defaultdict(lambda: dict({'count': 0}))}))
    # classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog']
    # coco_class_names
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
    # add keys to defaultdict
    [d[key] for key in classnames_to_consider]
    # open and iterate over tracking log
    with open(LOG_PATH, 'rt', encoding='utf-8') as f:
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
            class_name = classnames_to_consider[class_id]
            # check if id was already seen
            if object_id not in assigned_ids:
                d[class_name]['total_count'] += 1
                # frame where the object was first recorded
                d[class_name]['count_per_frame'][frame_id]['count'] += 1
                assigned_ids.append(object_id)

        OBJECT_CLASS_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_class_names.csv'
        # with open(os.path.join(OUTPUT_PATH, OBJECT_CLASS_FILENAME), 'wt', encoding='utf-8') as f:
        #     # header
        #     f.write('class_name;count\n')
        #     for key, value in d.items():
        #         if key in classnames_to_consider:
        #             f.write(f'{key};{value["total_count"]}\n')
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
    LOG_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\Paris_walk\yt_videos\zpEPbl-4DaY\20221111-202029_zpEPbl-4DaY_tracklog.csv"
    OUTPUT_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster"
    total_objects = total_objects_per_class(LOG_PATH, OUTPUT_PATH)
    total_modes = total_transportation_modes(LOG_PATH, OUTPUT_PATH)
    print()
    pass
