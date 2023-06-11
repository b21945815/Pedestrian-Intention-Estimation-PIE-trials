import sys
import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img


def get_pose(img_sequences,
             ped_ids, file_path,
             data_type='train'):
    """
    Reads the pie poses from saved .pkl files
    Args:
        img_sequences: Sequences of image names
        ped_ids: Sequences of pedestrian ids
        file_path: Path to where poses are saved
        data_type: Whether it is for training or testing
    Return:
         Sequences of poses
    """

    print('\n#####################################')
    print('Getting poses %s' % data_type)
    print('#####################################')
    poses_all = []
    set_poses_list = [x for x in os.listdir(file_path) if x.endswith('.pkl')]
    set_poses = {}
    for s in set_poses_list:
        with open(os.path.join(file_path, s), 'rb') as fid:
            try:
                p = pickle.load(fid)
            except:
                p = pickle.load(fid, encoding='bytes')
        set_poses[s.split('.pkl')[0].split('_')[-1]] = p
    i = -1
    for seq, pid in zip(img_sequences, ped_ids):
        i += 1
        update_progress(i / len(img_sequences))
        pose = []
        for imp, p in zip(seq, pid):

            set_id = imp.split('\\')[-3]
            
            vid_id = imp.split('\\')[-2]
            img_name = imp.split('\\')[-1].split('.')[0]
            k = img_name + '_' + p[0]
            if k in set_poses[set_id][vid_id].keys():
                pose.append(set_poses[set_id][vid_id][k])
            else:
                pose.append([0] * 36)
        poses_all.append(pose)
    poses_all = np.array(poses_all)
    return poses_all


def jitter_bbox(img_path, bbox, ratio):
    """
    Jitters the position or dimensions of the bounding box.
    mode: increases the size of bounding box based on the given ratio
    Args:
        img_path: to the image
        bbox: The bounding box to be jittered
        ratio: The ratio of change relative to the size of the bounding box.
           For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Return:
        Jittered bounding boxes
    """

    img = load_img(img_path)
    jitter_ratio = abs(ratio)

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        b[0] = b[0] - width_change // 2
        b[1] = b[1] - height_change // 2

        b[2] = b[2] + width_change // 2
        b[3] = b[3] + height_change // 2

        # Checks to make sure the bbox is not exiting the image boundaries
        b = bbox_sanity_check(img.size, b)
        jit_boxes.append(b)

    return jit_boxes


def squarify(bbox, img_width):
    """
    Changes the dimensions of a bounding box to a fixed ratio
    Args:
        bbox: Bounding box
        img_width: Image width
    Return:
        Squarified bounding boxes
    """
    print(bbox)
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0

    # check whether the new bounding box goes beyond image borders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def update_progress(progress):
    """
    Shows the progress
    Args:
        progress: Progress thus far
    """
    barLength = 30
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength * progress))
    text = "\r[{}] {:0.2f}%".format("#" * block + "-" * (barLength - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()


def img_pad(img, mode='warp', size=224):
    """
    Pads and/or resizes a given image
    Args:
        img: The image to be coropped and/or padded
        mode: The type of padding or resizing. Options are,
            warp: crops the bounding box and resize to the output size
            same: only crops the image
            pad_same: maintains the original size of the cropped box  and pads with zeros
            pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
                        the desired output size in that direction while maintaining the aspect ratio. The rest
                        of the image is	padded with zeros
            pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
                    in which case it scales the image down, and then pads it
        size: Target size of image
    Return:
        Padded image
    """
    assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = np.copy(img)
    if mode == 'warp':
        warped_image = cv2.resize(img, (size, size))
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
        img_size = image.shape[:2][::-1] # original size is in (height, width)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or \
                (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = cv2.resize(image, img_size)
        padded_image = np.zeros((size, size)+(image.shape[-1],), dtype=img.dtype)
        w_off = (size-img_size[0])//2
        h_off = (size-img_size[1])//2
        padded_image[h_off:h_off + img_size[1], w_off:w_off+ img_size[0],:] = image
        return padded_image


def bbox_sanity_check(img_size, bbox):
    """
    Checks whether  bounding boxes are within image boundaries.
    If this is not the case, modifications are applied.
    Args:
        img_size: The size of the image
        bbox: The bounding box coordinates
    Return:
        The modified/original bbox
    """
    img_width, img_heigth = img_size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox


def get_path(file_name='',
             sub_folder='',
             save_folder='models',
             dataset='pie',
             save_root_folder='data/'):
    """
    Generates paths for saving model and config data.
    Args:
        file_name: The actual save file name , e.g. 'model.h5'
        sub_folder: If another folder to be created within the root folder
        save_folder: The name of folder containing the saved files
        dataset: The name of the dataset used
        save_root_folder: The root folder
    Return:
        The full path and the path to save folder
    """
    save_path = os.path.join(save_root_folder, dataset, save_folder, sub_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return os.path.join(save_path, file_name), save_path
