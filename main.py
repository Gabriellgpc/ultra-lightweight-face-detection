# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-08-03 18:42:33
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-08-03 19:24:27

import time
import click

import cv2

from src.face_detector import FaceDetector
from src import utils

@click.command()
@click.option('-v','--video_source', default='/dev/video0')
@click.option('-c','--confidence', type=float, default=0.5)
def main(video_source, confidence):

    detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=confidence,
                            overlap_thr=0.7)
    video = cv2.VideoCapture(video_source)

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    while True:
        ret, frame = video.read()
        if ret == False:
            print("End of the file or error to read the next frame.")
            break

        start_time = time.perf_counter()
        bboxes, scores = detector.inference(frame)
        end_time = time.perf_counter()

        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
        frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))

        cv2.imshow('video', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    main()