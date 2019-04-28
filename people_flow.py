#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: people_flow.py
@time: 2018/7/9 14:52
"""

import cv2
import colorsys
from timeit import default_timer as timer

import numpy as np
import matplotlib.path as mplPath
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num = 1


class YOLO(object):
    def __init__(self, score_threshold=0.3, iou_threshold=0.45):
        # model path or trained weights path
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = score_threshold
        self.iou = iou_threshold
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, forbid_box=None):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        lbox = []
        lscore = []
        lclass = []
        for i in range(len(out_classes)):
            if out_classes[i] == 0:
                lbox.append(out_boxes[i])
                lscore.append(out_scores[i])
                lclass.append(out_classes[i])
        out_boxes = np.array(lbox)
        out_scores = np.array(lscore)
        out_classes = np.array(lclass)
        print('画面中有{}个人'.format(len(out_boxes)))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500
        thickness = max(thickness, 2)

        font_cn = ImageFont.truetype(font='font/asl.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        draw = ImageDraw.Draw(image)
        forbid_total, video_total = 0, 0
        for i, c in reversed(list(enumerate(out_classes))):
            score = out_scores[i]
            video_total += 1
            # predicted_class = self.class_names[c]
            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{:.2f}'.format(score)
            label_size = draw.textsize(label, font)

            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            # 判断人是否在禁区
            b_center = (int((left+right)/2), bottom)
            color = self.colors[c]
            if forbid_box is not None and forbid_box.contains_point(b_center):
                color = (0, 0, 255)
                forbid_total += 1

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        if forbid_box:
            show_str = '  视频人数：%d， 禁区人数：%d  ' % (video_total, forbid_total)
        else:
            show_str = '  视频人数：%d  ' % video_total

        label_size1 = draw.textsize(show_str, font_cn)
        # print(label_size1)
        draw.rectangle(
            [10, 10, 10 + label_size1[0], 10 + label_size1[1]],
            fill=(255, 255, 0))
        draw.text((10, 10), show_str, fill=(0, 0, 0), font=font_cn)
        del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=0, start=0, end=0,
                 forbid_box=None):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print(output_path)
        out_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, out_fourcc, video_fps, video_size)

    forbid_box_path = None
    if forbid_box is not None:
        forbid_box = np.array(forbid_box)
        forbid_box_path = mplPath.Path(forbid_box)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    start *= 1000
    end = end if end == 0 else end*1000
    width, height = 0, 0
    while True:
        return_value, frame = vid.read()
        if return_value is False:
            break

        msec = int(vid.get(cv2.CAP_PROP_POS_MSEC))
        if msec < start:
            continue
        if end > 0 and msec > end:
            break

        print('当前时间进度：%.2f秒' % (msec/1000))
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, forbid_box=forbid_box_path)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        # 设置禁区
        if forbid_box is not None:
            cv2.polylines(result, forbid_box, isClosed=True,
                          color=(0, 0, 255), thickness=2)

        cv2.putText(result, text=fps, org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        if width == 0:
            height, width = result.shape[:2]
        cv2.putText(result, 'DeeAo AI Team', (width-250, height-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("width: %d, height: %d" % (width, height))
    out.release()
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except Exception as e:
            print('Open Error: %s! Try again!' % str(e))
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()

    yolo.close_session()


if __name__ == '__main__':
    detect_img(YOLO())
