"""
Interface for the PIE dataset:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

MIT License

Copyright (c) 2019 Amir Rasouli, Iuliia Kotseruba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import pickle
import cv2
import sys

import xml.etree.ElementTree as ElementTree

from os.path import join, abspath, isfile, isdir
from os import makedirs, listdir


class PIE(object):
    def __init__(self, regen_database=False, data_path=''):
        """
        Class constructor
        :param regen_database: Whether generate the database or not
        :param data_path: The path to wh
        """
        self._year = '2023'
        self._name = 'pie'
        self._image_ext = '.png'
        self._regen_database = regen_database

        # Paths
        self._pie_path = data_path
        assert isdir(self._pie_path), \
            'pie path does not exist: {}'.format(self._pie_path)

        self._annotation_path = join(self._pie_path, 'annotations')
        self._annotation_attributes_path = join(self._pie_path, 'annotations_attributes')
        self._annotation_vehicle_path = join(self._pie_path, 'annotations_vehicle')

        self._clips_path = join(self._pie_path, 'PIE_clips')
        self._images_path = join(self._pie_path, 'images')

    # Path generators
    @property
    def cache_path(self):
        """
        Generates a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._pie_path, 'data_cache'))
        if not isdir(cache_path):
            makedirs(cache_path)
        return cache_path

    @staticmethod
    def _get_image_set_ids(image_set):
        """
        Returns default image set ids
        :param image_set: Image set split
        :return: Set ids of the image set
        """
        image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03',
                                  'set04', 'set05', 'set06']}
        return image_set_nums[image_set]

    def _get_image_path(self, sid, vid, fid):
        """
        Generates and returns the image path given ids
        :param sid: Set id
        :param vid: Video id
        :param fid: Frame id
        :return: Return the path to the given image
        """
        return join(self._images_path, sid, vid,
                    '{:05d}.png'.format(fid))

    # Visual helpers
    @staticmethod
    def update_progress(progress):
        """
        progress bar
        :param progress: The progress thus far
        """
        barLength = 30
        status = ""
        if isinstance(progress, int):
            progress = float(progress)

        block = int(round(barLength * progress))
        text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def _print_dict(dic):
        """
        Prints a dictionary, one key-value pair per line
        :param dic: Dictionary
        """
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))

    @staticmethod
    def _get_dim():
        """
        :return: Image dimensions
        """
        return 1920, 1080

    # Image processing helpers
    def get_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and  frames for each video in the give set
        :param set_id: set_id to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<frame_id_0>,... <frame_id_n>]}
        """
        print("Generating frame numbers for", set_id)
        frame_ids = {v.split('_annt.xml')[0]: [] for v in sorted(listdir(join(self._annotation_path,
                                                                              set_id))) if
                     v.endswith("annt.xml")}
        for vid, frames in sorted(frame_ids.items()):
            path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
            tree = ElementTree.parse(path_to_file)
            num_frames = int(tree.find("./meta/task/size").text)
            frames.extend([i for i in range(num_frames)])
            frames.insert(0, num_frames)
        return frame_ids

    def extract_and_save_images(self, setName, videoName):
        """
        Extracts images from clips and saves on hard drive
        """
        print('Extracting frames from', setName)
        set_folder_path = join(self._clips_path, setName)
        extract_frames = self.get_frame_numbers(setName)

        set_images_path = join(self._pie_path, "images", setName)
        for vid, frames in sorted(extract_frames.items()):
            if videoName == vid:
                print(vid)
                video_images_path = join(set_images_path, vid)
                num_frames = frames[0]
                frames_list = frames[1:]
                if not isdir(video_images_path):
                    makedirs(video_images_path)
                videoCapture = cv2.VideoCapture(join(set_folder_path, vid + '.mp4'))
                success, image = videoCapture.read()
                frame_num = 0
                img_count = 0
                if not success:
                    print('Failed to open the video {}'.format(vid))
                while success:
                    if frame_num in frames_list:
                        self.update_progress(img_count / num_frames)
                        img_count += 1
                        if not isfile(join(video_images_path, "%05.f.png") % frame_num):
                            cv2.imwrite(join(video_images_path, "%05.f.png") % frame_num, image)
                    success, image = videoCapture.read()
                    frame_num += 1
                if num_frames != img_count:
                    print('num images don\'t match {}/{}'.format(num_frames, img_count))
                print('\n')

    # Annotation processing helpers
    @staticmethod
    def _map_text_to_scalar(label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
                   'action': {'standing': 0, 'walking': 1},
                   'look': {'not-looking': 0, 'looking': 1},
                   'gesture': {'__undefined__': 0, 'hand_ack': 1, 'hand_yield': 2,
                               'hand_rightofway': 3, 'nod': 4, 'other': 5},
                   'cross': {'not-crossing': 0, 'crossing': 1, 'crossing-irrelevant': -1},
                   'crossing': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
                   'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
                   'designated': {'ND': 0, 'D': 1},
                   'gender': {'n/a': 0, 'female': 1, 'male': 2},
                   'intersection': {'midblock': 0, 'T': 1, 'T-left': 2, 'T-right': 3, 'four-way': 4},
                   'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
                   'traffic_direction': {'OW': 0, 'TW': 1},
                   'signalized': {'n/a': 0, 'C': 1, 'S': 2, 'CS': 3},
                   'vehicle': {'car': 0, 'truck': 1, 'bus': 2, 'train': 3, 'bicycle': 4, 'bike': 5},
                   'sign': {'ped_blue': 0, 'ped_yellow': 1, 'ped_white': 2, 'ped_text': 3, 'stop_sign': 4,
                            'bus_stop': 5, 'train_stop': 6, 'construction': 7, 'other': 8},
                   'traffic_light': {'regular': 0, 'transit': 1, 'pedestrian': 2},
                   'state': {'__undefined__': 0, 'red': 1, 'yellow': 2, 'green': 3}}

        return map_dic[label_type][value]

    @staticmethod
    def _map_scalar_to_text(label_type, value):
        """
        Maps a scalar value to a text label
        :param label_type: The label type
        :param value: The scalar to be mapped
        :return: The text label
        """
        map_dic = {'occlusion': {0: 'none', 1: 'part', 2: 'full'},
                   'action': {0: 'standing', 1: 'walking'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'hand_gesture': {0: '__undefined__', 1: 'hand_ack',
                                    2: 'hand_yield', 3: 'hand_rightofway',
                                    4: 'nod', 5: 'other'},
                   'cross': {0: 'not-crossing', 1: 'crossing', -1: 'crossing-irrelevant'},
                   'crossing': {0: 'not-crossing', 1: 'crossing', -1: 'irrelevant'},
                   'age': {0: 'child', 1: 'young', 2: 'adult', 3: 'senior'},
                   'designated': {0: 'ND', 1: 'D'},
                   'gender': {0: 'n/a', 1: 'female', 2: 'male'},
                   'intersection': {0: 'midblock', 1: 'T', 2: 'T-left', 3: 'T-right', 4: 'four-way'},
                   'motion_direction': {0: 'n/a', 1: 'LAT', 2: 'LONG'},
                   'traffic_direction': {0: 'OW', 1: 'TW'},
                   'signalized': {0: 'n/a', 1: 'C', 2: 'S', 3: 'CS'},
                   'vehicle': {0: 'car', 1: 'truck', 2: 'bus', 3: 'train', 4: 'bicycle', 5: 'bike'},
                   'sign': {0: 'ped_blue', 1: 'ped_yellow', 2: 'ped_white', 3: 'ped_text', 4: 'stop_sign',
                            5: 'bus_stop', 6: 'train_stop', 7: 'construction', 8: 'other'},
                   'traffic_light': {0: 'regular', 1: 'transit', 2: 'pedestrian'},
                   'state': {0: '__undefined__', 1: 'red', 2: 'yellow', 3: 'green'}}

        return map_dic[label_type][value]

    def _get_annotations(self, set_id, vid):
        """
        Generates a dictionary of annotations by parsing the video XML file
        :param set_id: The set id
        :param vid: The video id
        :return: A dictionary of annotations
        """
        path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
        print(path_to_file)

        tree = ElementTree.parse(path_to_file)
        ped_annt = 'ped_annotations'
        traffic_annt = 'traffic_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}
        annotations[traffic_annt] = {}
        tracks = tree.findall('./track')
        for t in tracks:
            boxes = t.findall('./box')
            obj_label = t.get('label')
            obj_id = boxes[0].find('./attribute[@name=\"id\"]').text

            if obj_label == 'pedestrian':
                annotations[ped_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': []}
                annotations[ped_annt][obj_id]['behavior'] = {'gesture': [], 'look': [], 'action': [], 'cross': []}
                for b in boxes:
                    # Exclude the annotations that are outside the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[ped_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    occ = self._map_text_to_scalar('occlusion', b.find('./attribute[@name=\"occlusion\"]').text)
                    annotations[ped_annt][obj_id]['occlusion'].append(occ)
                    annotations[ped_annt][obj_id]['frames'].append(int(b.get('frame')))
                    for beh in annotations['ped_annotations'][obj_id]['behavior']:
                        # Read behavior tags for each frame and add to the database
                        annotations[ped_annt][obj_id]['behavior'][beh].append(
                            self._map_text_to_scalar(beh, b.find('./attribute[@name=\"' + beh + '\"]').text))

            else:
                obj_type = boxes[0].find('./attribute[@name=\"type\"]')
                if obj_type is not None:
                    obj_type = self._map_text_to_scalar(obj_label,
                                                        boxes[0].find('./attribute[@name=\"type\"]').text)

                annotations[traffic_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': [],
                                                     'obj_class': obj_label,
                                                     'obj_type': obj_type,
                                                     'state': []}

                for b in boxes:
                    # Exclude the annotations that are outside the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[traffic_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    annotations[traffic_annt][obj_id]['occlusion'].append(int(b.get('occluded')))
                    annotations[traffic_annt][obj_id]['frames'].append(int(b.get('frame')))
                    if obj_label == 'traffic_light':
                        annotations[traffic_annt][obj_id]['state'].\
                            append(self._map_text_to_scalar('state', b.find('./attribute[@name=\"state\"]').text))
        return annotations

    def _get_ped_attributes(self, set_id, vid):
        """
        Generates a dictionary of attributes by parsing the video XML file
        :param set_id: The set id
        :param vid: The video id
        :return: A dictionary of attributes
        """
        path_to_file = join(self._annotation_attributes_path, set_id, vid + '_attributes.xml')
        tree = ElementTree.parse(path_to_file)

        attributes = {}
        pedestrians = tree.findall("./pedestrian")
        for p in pedestrians:
            ped_id = p.get('id')
            attributes[ped_id] = {}
            for k, v in p.items():
                if 'id' in k:
                    continue
                try:
                    if k == 'intention_prob':
                        attributes[ped_id][k] = float(v)
                    else:
                        attributes[ped_id][k] = int(v)
                except ValueError:
                    attributes[ped_id][k] = self._map_text_to_scalar(k, v)

        return attributes

    def _get_vehicle_attributes(self, set_id, vid):
        """
        Generates a dictionary of vehicle attributes by parsing the video XML file
        :param set_id: The set id
        :param vid: The video id
        :return: A dictionary of vehicle attributes (obd sensor recording)
        """
        path_to_file = join(self._annotation_vehicle_path, set_id, vid + '_obd.xml')
        tree = ElementTree.parse(path_to_file)

        veh_attributes = {}
        frames = tree.findall("./frame")

        for f in frames:
            dict_vals = {k: float(v) for k, v in f.attrib.items() if k != 'id'}
            veh_attributes[int(f.get('id'))] = dict_vals

        return veh_attributes

    def generate_database(self):
        """
        Generates and saves a database of the pie dataset by integrating all annotations
        Dictionary structure:
        'set_id'(str): {
            'vid_id'(str): {
                'num_frames': int
                'width': int
                'height': int
                'traffic_annotations'(str): {
                    'obj_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'obj_class': str,
                        'obj_type': str,    # only for traffic lights, vehicles, signs
                        'state': list(int)  # only for traffic lights
                'ped_annotations'(str): {
                    'ped_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'behavior'(str): {
                            'action': list(int)
                            'gesture': list(int)
                            'cross': list(int)
                            'look': list(int)
                        'attributes'(str): {
                             'age': int
                             'id': str
                             'num_lanes': int
                             'crossing': int
                             'gender': int
                             'crossing_point': int
                             'critical_point': int
                             'exp_start_point': int
                             'intersection': int
                             'designated': int
                             'signalized': int
                             'traffic_direction': int
                             'group_size': int
                             'motion_direction': int
                'vehicle_annotations'(str){
                    'frame_id'(int){'longitude': float
                          'yaw': float
                          'pitch': float
                          'roll': float
                          'OBD_speed': float
                          'GPS_speed': float
                          'latitude': float
                          'longitude': float
                          'heading_angle': float
                          'accX': float
                          'accY': float
                          'accZ: float
                          'gyroX': float
                          'gyroY': float
                          'gyroZ': float

        :return: A database dictionary
        """

        print('---------------------------------------------------------')
        print("Generating database for pie")

        cache_file = join(self.cache_path, 'pie_database.pkl')
        if isfile(cache_file) and not self._regen_database:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('pie annotations loaded from {}'.format(cache_file))
            return database

        # Path to the folder annotations
        set_ids = [f for f in sorted(listdir(self._annotation_path))]

        # Read the content of set folders
        database = {}
        for set_id in set_ids:
            video_ids = [v.split('_annt.xml')[0] for v in sorted(listdir(join(self._annotation_path,
                                                                              set_id))) if v.endswith("annt.xml")]
            database[set_id] = {}
            for vid in video_ids:
                print('Getting annotations for %s, %s' % (set_id, vid))
                database[set_id][vid] = self._get_annotations(set_id, vid)
                vid_attributes = self._get_ped_attributes(set_id, vid)
                database[set_id][vid]['vehicle_annotations'] = self._get_vehicle_attributes(set_id, vid)
                for ped in database[set_id][vid]['ped_annotations']:
                    database[set_id][vid]['ped_annotations'][ped]['attributes'] = vid_attributes[ped]

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database

    def get_data_stats(self):
        """
        Generates statistics for the dataset
        """
        annotations = self.generate_database()

        set_count = len(annotations.keys())

        ped_count = 0
        ped_box_count = 0
        video_count = 0
        total_frames = 0
        age = {'child': 0, 'adult': 0, 'senior': 0}
        gender = {'male': 0, 'female': 0}
        signalized = {'n/a': 0, 'C': 0, 'S': 0, 'CS': 0}
        traffic_direction = {'OW': 0, 'TW': 0}
        intersection = {'midblock': 0, 'T': 0, 'T-right': 0, 'T-left': 0, 'four-way': 0}
        crossing = {'crossing': 0, 'not-crossing': 0, 'irrelevant': 0}

        traffic_obj_types = {'vehicle': {'car': 0, 'truck': 0, 'bus': 0, 'train': 0, 'bicycle': 0, 'bike': 0},
                             'sign': {'ped_blue': 0, 'ped_yellow': 0, 'ped_white': 0, 'ped_text': 0, 'stop_sign': 0,
                                      'bus_stop': 0, 'train_stop': 0, 'construction': 0, 'other': 0},
                             'traffic_light': {'regular': 0, 'transit': 0, 'pedestrian': 0},
                             'crosswalk': 0,
                             'transit_station': 0}
        traffic_box_count = {'vehicle': 0, 'traffic_light': 0, 'sign': 0, 'crosswalk': 0, 'transit_station': 0}
        for sid, videos in annotations.items():
            video_count += len(videos)
            for vid, annots in videos.items():
                total_frames += annots['num_frames']
                for trf_ids, trf_annots in annots['traffic_annotations'].items():
                    obj_class = trf_annots['obj_class']
                    traffic_box_count[obj_class] += len(trf_annots['frames'])
                    if obj_class in ['traffic_light', 'vehicle', 'sign']:
                        obj_type = trf_annots['obj_type']
                        traffic_obj_types[obj_class][self._map_scalar_to_text(obj_class, obj_type)] += 1
                    else:
                        traffic_obj_types[obj_class] += 1
                for ped_ids, ped_annots in annots['ped_annotations'].items():
                    ped_count += 1
                    ped_box_count += len(ped_annots['frames'])
                    age[self._map_scalar_to_text('age', ped_annots['attributes']['age'])] += 1
                    if self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing']) == 'crossing':
                        crossing[self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing'])] += 1
                    else:
                        if ped_annots['attributes']['intention_prob'] > 0.5:
                            crossing['not-crossing'] += 1
                        else:
                            crossing['irrelevant'] += 1                    
                    intersection[
                        self._map_scalar_to_text('intersection', ped_annots['attributes']['intersection'])] += 1
                    traffic_direction[self._map_scalar_to_text('traffic_direction',
                                                               ped_annots['attributes']['traffic_direction'])] += 1
                    signalized[self._map_scalar_to_text('signalized', ped_annots['attributes']['signalized'])] += 1
                    gender[self._map_scalar_to_text('gender', ped_annots['attributes']['gender'])] += 1

        print('---------------------------------------------------------')
        print("Number of sets: %d" % set_count)
        print("Number of videos: %d" % video_count)
        print("Number of annotated frames: %d" % total_frames)
        print("Number of pedestrians %d" % ped_count)
        print("age:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(age.items())))
        print("gender:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(gender.items())))
        print("signal:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(signalized.items())))
        print("traffic direction:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_direction.items())))
        print("crossing:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crossing.items())))
        print("intersection:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(intersection.items())))
        print("Number of pedestrian bounding boxes: %d" % ped_box_count)
        print("Number of traffic objects")
        for trf_obj, values in sorted(traffic_obj_types.items()):
            if isinstance(values, dict):
                print(trf_obj + ':\n', '\n '.join('{}: {}'.format(k, v) for k, v in sorted(values.items())),
                      '\n total: ', sum(values.values()))
            else:
                print(trf_obj + ': %d' % values)
        print("Number of pedestrian bounding boxes:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_box_count.items())),
              '\n total: ', sum(traffic_box_count.values()))

    @staticmethod
    def _squarify(bbox, ratio, img_width):
        """
        Changes the ratio of bounding boxes to a fixed ratio
        :param bbox: Bounding box
        :param ratio: Ratio to be changed to
        :param img_width: Image width
        :return: Squarified bounding box
        """
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * ratio - width

        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2

        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    @staticmethod
    def _height_check(height_rng, frame_ids, boxes, images, occlusion):
        """
        Checks whether the bounding boxes are within a given height limit. If not, it
        will adjust the length of bounding boxes in data sequences accordingly
        :param height_rng: Height limit [lower, higher]
        :param frame_ids: List of frame ids
        :param boxes: List of bounding boxes
        :param images: List of images
        :param occlusion: List of occlusions
        :return: The adjusted data sequences
        """
        imageList, box, frames, occ = [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imageList.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusion[i])
        return imageList, box, frames, occ

    @staticmethod
    def _get_center(box):
        """
        Calculates the center coordinate of a bounding box
        :param box: coordinates of box
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def generate_data_trajectory_sequence(self, image_set, min_track_size, setName="", videoName=""):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param min_track_size: minimum track size
        :param setName: name of the set
        :param videoName: name of the video
        :return: Sequence data
        """
        params = {'fstride': 1,
                  'sample_type': 'all',
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'min_track_size': min_track_size}

        print('---------------------------------------------------------')
        print("Generating trajectory sequence data")
        self._print_dict(params)
        annot_database = self.generate_database()
        sequence_data = self._get_crossing(image_set, annot_database, setName, videoName, **params)
        return sequence_data

    def _get_crossing(self, image_set, annotations, setName="", videoName="", **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param setName: name of the set
        :param videoName: name of the video
        :param params: Parameters to generate data (see generate_database)
        :return: A dictionary of trajectories
        """

        print('---------------------------------------------------------')
        print("Generating crossing data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq = [], [], [], [], []
        activities = []

        set_ids = self._get_image_set_ids(image_set)
        if setName != "":
            set_ids = [setName]
        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                if videoName == "" or vid == videoName:
                    img_width = annotations[sid][vid]['width']
                    pid_annots = annotations[sid][vid]['ped_annotations']
                    vid_annots = annotations[sid][vid]['vehicle_annotations']
                    for pid in sorted(pid_annots):
                        num_pedestrians += 1

                        frame_ids = pid_annots[pid]['frames']
                        event_frame = pid_annots[pid]['attributes']['crossing_point']

                        end_idx = frame_ids.index(event_frame)
                        boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                        frame_ids = frame_ids[: end_idx + 1]
                        images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                        occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                        if height_rng[0] > 0 or height_rng[1] < float('inf'):
                            images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                        frame_ids, boxes,
                                                                                        images, occlusions)

                        if len(boxes) / seq_stride < params['min_track_size']:
                            continue

                        if sq_ratio:
                            boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                        image_seq.append(images[::seq_stride])
                        box_seq.append(boxes[::seq_stride])
                        center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                        occ_seq.append(occlusions[::seq_stride])

                        ped_ids = [[pid]] * len(boxes)
                        pids_seq.append(ped_ids[::seq_stride])

                        intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                        intent_seq.append(intent[::seq_stride])

                        acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
                        activities.append(acts[::seq_stride])

                        gpsc_seq.append([[(vid_annots[i]['latitude'], vid_annots[i]['longitude'])]
                                            for i in frame_ids][::seq_stride])
                        obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                        gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids][::seq_stride])
                        head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids][::seq_stride])
                        yrp_seq.append([[(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])]
                                        for i in frame_ids][::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'intention_prob': intent_seq,
                'activities': activities,
                'image_dimension': self._get_dim()}
