这是一个MFC(C++)调用Python（TensorFlow）的Visual Studio项目。

#identify
一个C++通过Socket发送人脸图像到服务器进行人脸识别的例子。

#faceViewer
一个显示人脸照片墙的MFC程序。

#objDetector
一个MFC(C++)调用Python（TensorFlow）facenet进行人脸检测、识别的程序。

必要条件：
		align
		detect.py
		label_map.ini
		simfang.ttf

可选检测项：
	人脸识别：identify.py + modles(pb文件和pkl文件)
	年龄、性别检测：age_gender_identify.py
	表情识别：emotion_gender_identify.py

#编译前置条件
0、已训练好的模型（frozen_inference_graph.pb）
1、安装python3.5 64位，安装OpenCV 64位
2、pip install numpy
3、pip install pillow
4、pip install matplotlib
5、pip install tensorflow
6、其他包视情况而定

#编译注意事项

1、为项目添加Python的附加包含目录及库目录，复制pythonXX.lib的备份，并重命名为pythonXX_d.lib;

1、根据个人计算机，编译好openCV，并在stdafx.h或其他链接的位置配置OpenCV版本及路径；

2、error LNK2019: 无法解析的外部符号 __imp___Py_NegativeRefcount、__imp___Py_RefTotal解决方案：

    注释掉object.h文件第56行 #define Py_TRACE_REFS
	
3、根据个人设置，修改Py_SetPythonHome中python的目录（宏：PYTHON_HOME）；

4、需要将被调用的 *.py 文件拷贝到生成目录；

5、需要给定目标检测的模型文件，并在相应的 *.py 文件中填写。

有问题，请联系：yuan_yuanxiang@163.com

															2018-4-16

															袁沅祥 注
