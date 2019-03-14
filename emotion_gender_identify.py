"""
对图像中的人脸进行检测，然后识别其性别、表情。
用法：python image_emotion_gender_demo.py IMAGE_FILE
源代码摘自：face_classification\src
2018年6月10日 by yuanyuanxiang
"""

import sys
import cv2
import time
import numpy as np
from PIL import Image
from keras.models import load_model

import socketserver

# 人脸图像的宽度
SIZE = 160
max_len = SIZE*SIZE*3
margin = 32

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# parameters for loading data and images
emotion_model_path = './models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = './models/simple_CNN.81-0.96.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
gender_labels = {0:'woman', 1:'man'}

# hyper-parameters for bounding boxes shape
gender_offsets = (0, 0)
emotion_offsets = (0, 0)
face_coordinates = [int(margin/2), int(margin/2), SIZE-margin, SIZE-margin]
x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
s1, s2, t1, t2 = apply_offsets(face_coordinates, emotion_offsets)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# 检测图像数据
def test_src(image_src):
    rgb_image = np.array(image_src).astype(np.float32)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    rgb_face = rgb_image[y1:y2, x1:x2]
    gray_face = gray_image[t1:t2, s1:s2]

    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        return (0, '')

    rgb_face = preprocess_input(rgb_face, False)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]
    gender_pre = gender_prediction[0, gender_label_arg]

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_pre = emotion_prediction[0, emotion_label_arg]

    prediction = gender_pre * emotion_pre
    result = emotion_text + ' ' + gender_text
    return (emotion_pre, emotion_text)

# 检测图像文件
def test_image(image_file):
    try:
        image = Image.open(image_file)
        print('>>> Run test on image:', image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    start_time = time.time()
    prediction, name = test_src(image)
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
                    prediction, name = test_src(image)
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
