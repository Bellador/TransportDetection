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

'''
TO DO:
- plot the temporal occurence of objects classes across frames
'''

def total_transportation_modes(LOG_PATH, source_str):
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
        detect transportation modes based on the detected objects in the same frame as well as their
        spatial relation to another
        :param frame_objects:
        :return:
        '''
        def calc_stacked_bbox(bbox):
            '''
            returns the vertically stacked lower left and top right corner of an input bbox.
            To identify e.g. cyclists or motorcyclists
            :return:
            '''
            # (bbox_top - bbox_h, bbox_left, bbox_top, bbox_left + bbox_w)
            xb = bbox[0]
            yb = bbox[1]
            xt = bbox[2]
            yt = bbox[3]
            n_xb = xb - (1.5 * (xt - xb))
            n_yb = 0.95 * yb
            n_xt = 1.05 * xt
            n_yt = 1.05 * yt
            return (n_xb, n_yb, n_xt, n_yt)

        def calc_bloated_bbox(bbox):
            '''
            returns double volume bbox of an input bbox. To identify e.g. dogwalkers
            :return:
            '''
            # (bbox_top - bbox_h, bbox_left, bbox_top, bbox_left + bbox_w)
            xb = bbox[0]
            yb = bbox[1]
            xt = bbox[2]
            yt = bbox[3]
            half_width = (xt - xb) / 2
            half_height = (yt - yb) / 2
            return (xb - half_width, yb - half_height, xt + half_width, yt + half_height)

        def calc_overlap(person_bbox, other_bbox, mode='stacked_bbox'):
            '''
            calaculates the overlap between bbox of different objects.
            The mode defines the way the bboxes are modified before overlappign
            :param person_bbox:
            :param other_bbox:
            :param mode: stacked_bbox, bloated_bbox
            :return:
            '''
            if mode == 'stacked_bbox':
                person_bbox = calc_stacked_bbox(person_bbox)
            elif mode == 'bloated_bbox':
                person_bbox = calc_bloated_bbox(person_bbox)
            if other_bbox[0] >= person_bbox[0] and other_bbox[1] >= person_bbox[1] and other_bbox[2] <= person_bbox[2] and other_bbox[3] <= person_bbox[3]:
                return True
            else:
                return False

        # track detected transportation modes
        # pedestrians = 0   # person - NEW will be defined at the end according to all people not assigned to other transportation modes
        dogwalkers = 0    # dog
        cyclists = 0      # bicycle
        motorcyclists = 0  # motorcycle
        # store transportation mode relevant objects to check their spatial relation
        persons = []
        dogs = []
        bicycles = []
        motorcycles = []
        # frame_object: [frame_id, identity, class_id, class_name, bbox_tuple]
        for object in frame_objects:
            class_name = object[3]
            if class_name == 'person':
                persons.append(object)
            elif class_name == 'dog':
                dogs.append(object)
            elif class_name == 'bicycle':
                bicycles.append(object)
            elif class_name == 'motorcycles':
                motorcycles.append(object)
        # track which people were already assigned to a transportation to avoid duplicate assignment
        not_assigned_persons = []
        assigned_ids = []
        for person in persons:
            person_id = person[1]
            person_bbox = person[4]
            assigned = False
            # check cyclists
            if not assigned:
                for bicycle in bicycles:
                    bicycle_id = bicycle[1]
                    bicycle_bbox = bicycle[4]
                    # if bicycle_id == 67:
                        ## test for known bicycle rider
                        # print('here')
                    # check for overlap
                    if calc_overlap(person_bbox, bicycle_bbox, mode='stacked_bbox'):
                        cyclists += 1
                        assigned_ids = assigned_ids + [person_id, bicycle_id]
                        assigned = True
            if not assigned:
                # check dogwalkers
                for dog in dogs:
                    dog_id = dog[1]
                    dog_bbox = dog[4]
                    # check for overlap
                    if calc_overlap(person_bbox, dog_bbox, mode='bloated_bbox'):
                        dogwalkers += 1
                        assigned_ids = assigned_ids + [person_id, dog_id]
                        assigned = True
            if not assigned:
                # check motorcyclists
                for motorcycle in motorcycles:
                    motorcycle_id = motorcycle[1]
                    motorcycle_bbox = motorcycle[4]
                    # check for overlap
                    if calc_overlap(person_bbox, motorcycle_bbox, mode='stacked_bbox'):
                        motorcyclists += 1
                        assigned_ids = assigned_ids + [person_id, motorcycle_id]
                        assigned = True
            # at the end, if still not assigned, add person to list that can be matched later on
            if not assigned:
                not_assigned_persons.append(person_id)

        return dogwalkers, cyclists, motorcyclists, assigned_ids, not_assigned_persons

    ## func: total_transportation_modes
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
    dogwalkers = 0
    cyclists = 0
    motorcyclists = 0
    # open and iterate over tracking log
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # keep track of the object ids that were already accounted for
        assigned_ids = []
        # keep track of people not assigned yet to a transportation mode but maybe later on (due to detection errors)
        not_assigned_persons = []
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
                    add_to_dogwalkers, \
                    add_to_cyclists, \
                    add_to_motorcyclists, \
                    add_to_assigned_ids, \
                    add_to_not_assigned_persons = process_frame_objects(frame_objects)

                    dogwalkers += add_to_dogwalkers
                    cyclists += add_to_cyclists
                    motorcyclists += add_to_motorcyclists
                    # update assigned_ids with ids that were assigned to transportation mode
                    assigned_ids += add_to_assigned_ids
                    not_assigned_persons += add_to_not_assigned_persons
                    # remove assigned ids from not_assigned_persons
                    not_assigned_persons = [id_ for id_ in not_assigned_persons if id_ not in add_to_assigned_ids]
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
        pedestrians = len(set(not_assigned_persons))
        # write to output file
        TRANS_MODES_OUTPUT_PATH = './output/transportation/'
        TRANS_MODES_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_source_{source_str}_transportation_modes.csv'
        with open(os.path.join(TRANS_MODES_OUTPUT_PATH, TRANS_MODES_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('transportation_mode;count\n')
            # detected transportation modes
            f.write(f'dogwalkers;{dogwalkers}\n')
            f.write(f'cyclists;{cyclists}\n')
            f.write(f'motorcyclists;{motorcyclists}\n')
            f.write(f'pedestrians;{pedestrians}\n')
        print('[+] detected active transportation modes:')
        print(f'dogwalkers: {dogwalkers}\n cyclists: {cyclists}\n motorcyclists: {motorcyclists}\n pedestrians: {pedestrians}')


def total_objects_per_class(LOG_PATH, classnames_to_consider, source_str):
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
            else:
                continue
        # save geolocations to output file
        OBJECT_CLASS_OUTPUT_PATH = './output/transportation/'
        OBJECT_CLASS_FILENAME = f'{time.strftime("%Y%m%d-%H%M%S")}_source_{source_str}_class_names.csv'
        with open(os.path.join(OBJECT_CLASS_OUTPUT_PATH, OBJECT_CLASS_FILENAME), 'wt', encoding='utf-8') as f:
            # header
            f.write('class_name;count\n')
            # print output
            print('[+] detected mobility related objects:')
            for key, value in d.items():
                if key in classnames_to_consider:
                    print(f'{key}s: {value["count"]}')
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

            '''
            calculate Levenshtein Distance (word similarity) [https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a]
            between the OCR detected words above a certain threshold and a gazetteer.
            This gazetteer can be narrowed down to a manually entered location e.g. Geneva
            '''


if __name__ == '__main__':
    # LOG_PATH = r"./output_BACKUP/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_tracklog.csv"
    LOG_PATH = r"./output_BACKUP/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_tracklog.csv"
    # OCR_LOG_PATH = r"./output_BACKUP/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_ocrlog.csv"
    OCR_LOG_PATH = r"./output_BACKUP/complete_OCR_log/20210701-213838_source_virtual_walk_geneva_ocrlog.csv"
    classnames_to_consider = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog']
    total_objects_per_class(LOG_PATH, classnames_to_consider)
    # highest_ocr_confidence(OCR_LOG_PATH)
    total_transportation_modes(LOG_PATH)
