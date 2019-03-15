# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector


import numpy as np
import tensorflow as tf
import cv2
import time
from . import views
from . import models
import os


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for a in range(boxes.shape[1])]
        for j in range(boxes.shape[1]):
            boxes_list[j] = (int(boxes[0, j, 0] * im_height),
                             int(boxes[0, j, 1]*im_width),
                             int(boxes[0, j, 2] * im_height),
                             int(boxes[0, j, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def select_frame_in_frame(captured_frame, x, y, w, h):

    color = (0, 100, 0)             # BGR 0-255
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(captured_frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cropped_frame = captured_frame[y:h, x:w]

    return cropped_frame


def save_in_database(name):
    address = os.path.dirname(os.path.abspath("manage.py"))
    p = models.ImageClass(im_title=name, im_path=address)
    p.save()


# def hit_screenshot_api(screenshot, number):
#     url = 'http://192.168.10.36:8080/Aipl/save/restricted'
#     img_name = "trespasser_alert" + str(number)
#     multipart_form_data = {
#         'image': (img_name, screenshot)
#     }
#     response = requests.post(url, files=multipart_form_data)
#     print(response)


def check_for_trespassers():
    start_time = 0
    count = 0
    model_path = 'human_detection_api/frozen_inference_graph.pb'

    od_api = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(0)
    # rtsp://192.168.10.10:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream
    while True:
        if views.on_off is True:
            r, img = cap.read()
            # img = cv2.resize(img, (1280, 720))
            # selected_frame = select_frame_in_frame(img, 0, 0, 400, 720)
            boxes, scores, classes, num = od_api.process_frame(img)

            # Visualization of the results of a detection.

            for i in range(len(boxes)):

                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    end_time = time.time()
                    if end_time - start_time > 10:
                        start_time = time.time()
                        image_name = "frame" + count + ".jpg"
                        cv2.imwrite(image_name, img)

                        # hit_screenshot_api(img, count)
                        save_in_database(image_name)
                        count += 1
                        time.sleep(10)
                        break
                        # box = boxes[i]
                        # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 4)
                        # break

        else:
            cap.release()
            cv2.destroyAllWindows()
            break




