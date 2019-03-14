"""
将人脸映射到512维的特征向量.
"""
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

# 人脸图像的宽度
SIZE = 160

# 模型文件及分类文件位置
frozen_graph = "./models/20180402-114759.pb"

##################################################################

# 加载pb文件
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
self_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with self_sess.as_default():
    with tf.gfile.FastGFile(frozen_graph,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=None, name='')

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# 白化预处理
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

# 对人脸图像进行识别
def Identify(image_src):
    image_np = np.array(image_src).astype(np.uint8)
    if SIZE != image_np.shape[0] or SIZE != image_np.shape[1]:
        image_np = misc.imresize(image_np, (SIZE, SIZE), interp='bilinear')
    prewhiten_face = prewhiten(image_np)
    # Run forward pass to calculate embeddings
    feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
    face_embedding = self_sess.run(embeddings, feed_dict=feed_dict)[0]
    return face_embedding.reshape(1, 512)

# 检测图像文件中的人脸
def test_image(image_file):
    try:
        image = Image.open(image_file)
        print('>>> Run test on image:', image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    start_time = time.time()
    face_embedding = Identify(image)
    use_time = time.time() - start_time
    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'elapsed time:', use_time)
    print('face_embedding.shape =', face_embedding.shape)
    print('face_embedding.dtype =', face_embedding.dtype)
    print('face_embedding =', face_embedding)

# 激活GPU
test_image('image.jpg')

# MAIN
if __name__ == '__main__':
    test_image('image.jpg' if (1 == len(sys.argv)) else sys.argv[1])
