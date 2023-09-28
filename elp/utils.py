
import cv2
import numpy as np

def string_to_numpy(string, shape):
    string = string.replace('[', '').replace(']', '').replace('\n', '')
    array = np.fromstring(string, sep=' ')
    array = array.reshape(shape)
    return array

def gaze_container_from_text(gaze_log_dict):

    shape_map = {
        'pitch': (-1),
        'yaw': (-1),
        'bboxes': (-1,4),
        'landmarks': (-1,5,2),
        'scores': (-1)
    }

    for k in gaze_log_dict:
        gaze_log_dict[k] = string_to_numpy(gaze_log_dict[k], shape_map[k])

    return gaze_log_dict


def ensure_rgb(img):

    # Convert all grey images to rgb images
    if len(img.shape) == 2:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        rgb_img = img
    else:
        raise RuntimeError(f"Unexpected number of channels: {img.shape}")

    return rgb_img


def combine_frames(img1:np.ndarray, img2:np.ndarray):

    # Ensure the input images have the same number of channels
    safe_img1 = ensure_rgb(img1)
    safe_img2 = ensure_rgb(img2)

    h1, w1 = safe_img1.shape[:2]
    h2, w2 = safe_img2.shape[:2]

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # combine 2 images
    vis[:h1, :w1, :3] = safe_img1
    vis[:h2, w1 : w1 + w2, :3] = safe_img2

    return vis
