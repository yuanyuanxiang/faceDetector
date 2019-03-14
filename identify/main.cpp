#include "..\socket\BcecrSocket.h"
#include "CodeTransform.h"
#include "..\config.h"
#include <process.h>

#pragma comment(lib, "ws2_32.lib")

bool g_bExit = false; // 线程状态

CRITICAL_SECTION g_cs;// 临界区

#define __OUT 0

// 收发数据的缓存区
struct RecvBuffer
{
	char buf[4096];
	char *pos;
	RecvBuffer() { memset(buf, 0, sizeof(buf)); pos = buf; }
	void PushData(const char *data, int n)
	{
		EnterCriticalSection(&g_cs);
		memcpy(pos, data, n);
		pos += n;
		LeaveCriticalSection(&g_cs);
	}
	bool PopData(char *data, int n)
	{
		EnterCriticalSection(&g_cs);
		int count = 0;
		const char *p = buf;
		while (*p && count != n)
			if (';' == *p++) ++count;
		if (n == count)
		{
			strcpy(data, buf);
			memset(buf, 0, p - buf);
			if (*p) strcpy(buf, p + 1);
			pos = buf;
			while (*pos) ++pos;
		}
		LeaveCriticalSection(&g_cs);
		return n == count;
	}
};

RecvBuffer g_buf; // 收缓存

void RecvThread(void *param)
{
	CBcecrSocket *pSock = (CBcecrSocket*)param;
	OutputDebugStringA("===> RecvThread Start\n");
	while (!g_bExit)
	{
		char buf[128] = { 0 };
		int n = pSock->recvData(buf, 128, 200);
		if (0 < n)
		{
			WCHAR* unicode = UTF8Convert2Unicode(buf);
			char* ansi = UnicodeConvert2ANSI(unicode);
			if (__OUT) printf("Facedetect result: %s\n", ansi);
			g_buf.PushData(ansi, strlen(ansi));
			delete [] ansi;
			delete [] unicode;
			continue;
		}
		Sleep(5);
	}
	OutputDebugStringA("===> RecvThread Stop\n");
}

// 对图像进行人脸识别
void identify(CBcecrSocket *pSock, const cv::Mat &m, int times)
{
	if (times < 0 || times > 100)
		return;
	//////////////////////////////////////////////////////////////////////////
	// 方式1：发送完立即收取，接着进行第二次
	clock_t t = clock();
	for (int i = 0; i < times; ++i)
	{
		pSock->sendData((const char*)m.data, 160 * 160 * 3);
		char buf[128] = { 0 };
		int max_try = 10;
		do {
			if (pSock->recvData(buf, 128, 200) < 0)
			{
				pSock->DisConnect();
				break;
			}
		}while (0 == buf[0] && --max_try);
		if (buf[0])
		{
			WCHAR* unicode = UTF8Convert2Unicode(buf);
			char* ansi = UnicodeConvert2ANSI(unicode);
			if (__OUT) printf("Facedetect result: %s\n", ansi);
			delete [] ansi;
			delete [] unicode;
		}
	}
	printf("Using time = %f ms.\n", (clock() - t) / float(times ? times : 1));
	Sleep(1000);
	//////////////////////////////////////////////////////////////////////////
	g_bExit = false;
	_beginthread(&RecvThread, 0, pSock);
	// 方式2：发送全部完成时，再统一进行收取
	t = clock();
	for (int i = 0; i < times; ++i)
	{
		pSock->sendData((const char*)m.data, 160 * 160 * 3);
	}
	printf("Send using = %d ms.\n", (clock() - t));// 服务器处理速度跟不上发送速度
	char buf[4096];
	while (false == g_buf.PopData(buf, times))
		Sleep(5);
	printf("Using time = %f ms.\n", (clock() - t) / float(times ? times : 1));
	g_bExit = true;
	Sleep(2000);
	char tmp[128] = { 0 };
	pSock->recvData(tmp, 128, 200);
	if (tmp[0]) printf("Recv data: %s\n", tmp);
}

// 测试2种连接服务器进行人脸识别的算法的时间
int main(int argc, char *argv[])
{
	InitializeCriticalSection(&g_cs);
	WSADATA wsaData; // Socket
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	CBcecrSocket *pSock = new CBcecrSocket();
	pSock->SetServerInfo(argc > 1 ? argv[1] : "127.0.0.1", argc > 2 ? atoi(argv[2]) : 9999);
	cv::Mat m = imread("image.jpg");
	if (false == m.empty() && m.step[1] == 3)
	{
		if (m.rows != 160 || m.cols != 160)
			cv::resize(m, m, cv::Size(160, 160));
		while (true)
		{
			char in[128] = { 0 };
			printf("人脸检测次数: ");
			std::cin >> in;
			if ('q' == in[0] || 'Q' == in[0])
				break;
			if (0 == strcmp("cls", in) || 0 == strcmp("CLS", in))
			{
				system("cls");
				continue;
			}
			int times = atoi(in);
			identify(pSock, m, times);
		}
	}

	g_bExit = true;
	Sleep(1000);
	delete pSock;
	WSACleanup();
	DeleteCriticalSection(&g_cs);

	return 0;
}
