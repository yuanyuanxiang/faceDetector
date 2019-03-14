import pickle
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

import socketserver

# 人脸图像的宽度
SIZE = 160
max_len = SIZE*SIZE*3

# 模型文件及分类文件位置
frozen_graph = "./models/20180402-114759.pb"
classifier_model = "./models/lfw.pkl"

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

# 加载pkl文件
with open(classifier_model, 'rb') as infile:
    self_model, self_class_names = pickle.load(infile)

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
    if face_embedding is not None:
        predictions = self_model.predict_proba([face_embedding])
        best_class_indices = np.argmax(predictions, axis=1)
        id = best_class_indices[0]
        name = self_class_names[id]
        prediction = predictions[0, id]
        return (prediction, name)

    return (0, '')

# 检测图像文件中的人脸
def test_image(image_file):
    try:
        image = Image.open(image_file)
        print('>>> Run test on image:', image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    start_time = time.time()
    prediction, name = Identify(image)
    use_time = time.time() - start_time
    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'elapsed time:', use_time)
    print('prediction =', prediction, 'name =', name)

# 激活GPU
test_image('image.jpg')

# SOCKET SERVER
class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print (self.client_address[0], 'start detecting.')
        recv_len = 0
        recv_buf = b''
        while True:
            try:
                data = self.request.recv(max_len - recv_len)
                if (b'' == data):
                    self.request.close()
                    break
                recv_len = recv_len + len(data)
                recv_buf = recv_buf + data
                if max_len == recv_len:
                    start_time = time.time()
                    image = Image.frombuffer('RGB', (SIZE, SIZE), recv_buf, 'raw', 'RGB', 0, 1)
                    prediction, name = Identify(image)
                    recv_len = 0
                    recv_buf = b''
                    use_time = time.time() - start_time
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'elapsed time:', use_time, 
                        'prediction =', prediction, 'name =', name)
                    self.request.sendall(bytes(name + ':' + str(prediction) + ';', 'utf8'))
                elif max_len > recv_len:
                    recv_len = 0
                    recv_buf = b''
            except KeyboardInterrupt:
                print('Ctrl+C is pressed.')
                break
        print (self.client_address[0], 'stop detecting.')

# MAIN
if __name__ == '__main__':
    if (1 == len(sys.argv)):
        HOST, PORT = '127.0.0.1', 9999
    elif (2 == len(sys.argv)):
        HOST, PORT = sys.argv[1], 9999
    elif (3 <= len(sys.argv)):
        HOST, PORT = sys.argv[1], int(sys.argv[2])
    server = socketserver.ThreadingTCPServer((HOST, PORT), RequestHandler)
    print('Start server: host =', HOST, 'port =', PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Ctrl+C is pressed.')
        sys.exit(0)
