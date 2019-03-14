# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os
import sys
import numpy as np
import tensorflow as tf
from scipy import misc
from PIL import Image
import align.detect_face
from tensorflow.python.platform import gfile
from sklearn.svm import SVC

facenet_model_checkpoint = "./model/20180402-114759.pb"
classifier_model = "./model/lfw.pkl"

# 人脸结构
class Face:
    def __init__(self):
        self.id = None               # 类别ID
        self.name = None             # 类别名
        self.bounding_box = None     # 矩形框
        self.image = None
        self.container_image = None
        self.embedding = None

# 初始化图
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# 找到人脸
def find_faces(image_np, minsize = 20, threshold = [0.6, 0.7, 0.7], factor = 0.709):
    faces = []

    face_crop_size = 160
    face_crop_margin = 32
    bounding_boxes, _ = align.detect_face.detect_face(image_np, minsize,
                                                          pnet, rnet, onet,
                                                          threshold, factor)
    for bb in bounding_boxes:
        face = Face()
        face.container_image = image_np
        face.bounding_box = np.zeros(5, dtype=np.float32)

        img_size = np.asarray(image_np.shape)[0:2]
        face.bounding_box[0] = np.maximum(bb[0] - face_crop_margin / 2, 0)
        face.bounding_box[1] = np.maximum(bb[1] - face_crop_margin / 2, 0)
        face.bounding_box[2] = np.minimum(bb[2] + face_crop_margin / 2, img_size[1])
        face.bounding_box[3] = np.minimum(bb[3] + face_crop_margin / 2, img_size[0])
        boxes_int = np.zeros(5, dtype=np.int32)
        boxes_int[:] = face.bounding_box[:]
        cropped = image_np[boxes_int[1]:boxes_int[3], boxes_int[0]:boxes_int[2], :]
        face.image = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')

        faces.append(face)

    return faces

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

# 加载pb文件
self_sess = tf.Session()
with self_sess.as_default():
    if (os.path.isfile(facenet_model_checkpoint)):
        print('Model filename: %s' % facenet_model_checkpoint)
        with gfile.FastGFile(facenet_model_checkpoint,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=None, name='')

# 加载pkl文件
def load_pkl():
    infile = open(classifier_model, 'rb')
    self_model, self_class_names = pickle.load(infile)
    return self_model, self_class_names
	
# 人脸分类
def identify(faces, self_model = None, self_class_names = None):
    for i, face in enumerate(faces):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        prewhiten_face = prewhiten(face.image)
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        face.embedding = self_sess.run(embeddings, feed_dict=feed_dict)[0]
        if face.embedding is not None:
            predictions = self_model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            face.id = best_class_indices[0]
            face.name = self_class_names[face.id]
            face.bounding_box[4] = predictions[0, face.id]

    return faces

self_model, self_class_names = load_pkl()

# 检测图像数据
def test_src(image_src):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    image_np = np.array(image_src).astype(np.uint8)
    print('image_np.shape =', image_np.shape)
    faces = find_faces(image_np, minsize, threshold, factor)
    #faces = identify(faces, self_model, self_class_names)
    num = len(faces)
    boxes = np.empty([num, 5], dtype = 'float32')
    cls_names = []
    i = 0
    while i < num:
        boxes[i, :] = faces[i].bounding_box[:]
        cls_names.append(faces[i].name)
        i = i + 1

    return (boxes, cls_names)

# 检测图像文件
def test_image(image_file):
    try:
        print('>> Run test on:', image_file)
        image = Image.open(image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    bounding_boxes, cls_names = test_src(image)
    print('bounding_boxes.shape =', bounding_boxes.shape)
    print('bounding_boxes.dtype =', bounding_boxes.dtype)
    print('bounding_boxes =', bounding_boxes)
    print('cls_names =', cls_names)

# MAIN
if __name__ == '__main__':
    test_image('image.jpg' if (1 == len(sys.argv)) else sys.argv[1])
