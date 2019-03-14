"""
Face detection
"""
import sys
import time
import numpy as np
from PIL import Image
from scipy import misc
from wide_resnet import WideResNet

import socketserver

# 人脸图像的宽度
SIZE = 160
max_len = SIZE*SIZE*3

WRN_WEIGHTS_PATH = "./models/weights.18-4.06.hdf5"

MODEL = WideResNet(64, depth=16, k=8)()
MODEL.load_weights(WRN_WEIGHTS_PATH)

# 检测图像数据
def test_src(image_src):
    image_np = np.array(image_src).astype(np.uint8)
    image_np = misc.imresize(image_np, (64, 64), interp='bilinear')
    image_np_expanded = np.expand_dims(image_np, axis=0)
    results = MODEL.predict(image_np_expanded)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    predition = max(predicted_genders[0][0], predicted_genders[0][1])
    classify = 'F' if (predicted_genders[0][0] > 0.5) else 'M'
    return (predition, classify + '_' + ('%.1f' % predicted_ages[0]))

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
