#----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rishi Rai
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
#----------------------------------------------------

import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import align.detect_face

# 初始化图
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# 检测图像数据
def test_src(image_src):
    image_np = np.array(image_src).astype(np.uint8)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    crop_rate = 0.125 # frce crop rate

    bounding_boxes, _ = align.detect_face.detect_face(image_np, minsize, pnet, rnet, onet, threshold, factor)
    result = np.empty(bounding_boxes.shape, np.float32)
    img_size = np.asarray(image_np.shape)[0:2]
    index = 0
    for bb in bounding_boxes:
        crop1 = crop_rate * (bb[2] - bb[0])
        crop2 = crop_rate * (bb[3] - bb[1])
        result[index, 0] = np.maximum(bb[0] - crop1, 0)
        result[index, 1] = np.maximum(bb[1] - crop2, 0)
        result[index, 2] = np.minimum(bb[2] + crop1, img_size[1])
        result[index, 3] = np.minimum(bb[3] + crop2, img_size[0])
        result[index, 4] = bb[4]
        index = index + 1

    return result

# 检测图像文件
def test_image(image_file):
    try:
        image = Image.open(image_file)
        print('>>> Run test on image:', image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    bounding_boxes = test_src(image)
    print('bounding_boxes.shape =', bounding_boxes.shape)
    print('bounding_boxes.dtype =', bounding_boxes.dtype)
    print('bounding_boxes =', bounding_boxes)

# 激活GPU
test_image('image.jpg')

# MAIN
if __name__ == '__main__':
    test_image('image.jpg' if (1 == len(sys.argv)) else sys.argv[1])
