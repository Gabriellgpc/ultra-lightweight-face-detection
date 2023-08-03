# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-07-29 15:41:01
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-08-03 18:59:32
import numpy as np
import cv2

def image_resize(image, width=None, height=None):
    assert width is not None or height is not None, 'width or height must be specified.'

    h, w = image.shape[:2]
    hw_ratio = h/w

    if None in [width, height]:
        if width == None:
            width = height * (1.0 / hw_ratio)
        else:
            height = width * hw_ratio
    
    dsize = (int(width), int(height))
    return cv2.resize(image, dsize=dsize)

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the boxes by
    # their bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the bounding box and
        # other bounding boxes
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap
        # greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick], pick

def put_text_on_image(image, text, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (10, 50)  # top left corner position

    # Add the text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

    # Convert the image back to BGR if it was originally color
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def draw_boxes_with_scores(image, boxes, scores):
    if len(boxes) == 0:
        return image

    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        box = boxes[i]
        score = scores[i][0]

        # Convert the box coordinates to integers
        box = box.astype(int)

        # Draw the box on the image
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Define the text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)
        thickness = 1

        # Create the text string
        text = '{:.2f}'.format(score)

        # Determine the text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Define the text position relative to the box
        text_x = box[0]
        text_y = box[1] - text_size[1]

        # Draw the text background rectangle
        cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 255, 0), -1)

        # Draw the text on top of the background rectangle
        cv2.putText(image, text, (text_x, text_y + text_size[1]), font, font_scale, color, thickness)

    return image


def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image