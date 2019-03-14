#include "BcecrSocket.h"   
#include <stdio.h>
#include <iostream>
#include <process.h>
#include "..\identify\CodeTransform.h"

#ifndef WAIT
// 在条件C成立时等待T秒(步长10ms)
#define WAIT(C, T) { int s=100*(T); do{ Sleep(10); } while( (C) && (--s) ); }
#endif

void CBcecrSocket::ConnectServer(void *param)
{
	OutputDebugStringA("======> Thread ConnectServer Start.\n");
	CBcecrSocket *pSocket = (CBcecrSocket*)param;
	pSocket->m_nThreadState = THREADSTATE_START;
	while (THREADSTATE_START == pSocket->m_nThreadState)
	{
		if (false == pSocket->IsConnected() && false == pSocket->Connect())
			WAIT(THREADSTATE_START == pSocket->m_nThreadState, 5);
		Sleep(50);
	}
	pSocket->m_nThreadState = THREADSTATE_STOP;
	OutputDebugStringA("======> Thread ConnectServer Stop.\n");
}


CBcecrSocket::CBcecrSocket()
{
	m_nToport = 0;
	m_Socket = INVALID_SOCKET;
	memset(m_chToIp, 0, sizeof(m_chToIp));
	m_nThreadState = THREADSTATE_UNKNOWN;
	m_bConnected = false;
	_beginthread(&ConnectServer, 0, this);
}


CBcecrSocket::~CBcecrSocket()
{
	if (THREADSTATE_STOP != m_nThreadState)
		m_nThreadState = APP_TERMINATE;
	while (THREADSTATE_STOP != m_nThreadState)
		Sleep(10);
	DisConnect();
}


void CBcecrSocket::SetServerInfo(const char *pIp, int nPort)
{
	strcpy_s(m_chToIp, pIp);
	m_nToport = nPort;
}


bool CBcecrSocket::Connect()
{
	do 
	{
		if (0 == m_chToIp[0] || 0 == m_nToport)
			break;

		m_Socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
		if (INVALID_SOCKET == m_Socket)
			break;
		
		/// 发送缓冲区
		const int nSendBuf = 1024 * 750;
		::setsockopt(m_Socket, SOL_SOCKET, SO_SNDBUF, (const char*)&nSendBuf, sizeof(int));
		/// 接收缓冲区
		const int nRecvBuf = 1024 * 750;
		::setsockopt(m_Socket, SOL_SOCKET, SO_RCVBUF, (const char*)&nRecvBuf, sizeof(int));

		sockaddr_in addrAdpter;
		memset(&addrAdpter, 0, sizeof(addrAdpter));
		addrAdpter.sin_family = AF_INET;
		addrAdpter.sin_port = htons(m_nToport);
		addrAdpter.sin_addr.s_addr = inet_addr(m_chToIp);

		/// 和服务端建立连接
		if (SOCKET_ERROR == ::connect(m_Socket, (const sockaddr *)&addrAdpter, sizeof(addrAdpter)))
		{
			::closesocket(m_Socket);
			m_Socket = INVALID_SOCKET;

			break;
		}

		ULONG ul = 0;   // 阻塞模式
		if (SOCKET_ERROR == ioctlsocket(m_Socket, FIONBIO, &ul))
			break;
		m_bConnected = true;
	} while (false);

	return m_bConnected;
}


void CBcecrSocket::DisConnect()
{
	m_bConnected = false;
	if (INVALID_SOCKET != m_Socket)
	{
		closesocket(m_Socket);
		m_Socket = INVALID_SOCKET;
	}
}

// 返回收到的数据长度
int CBcecrSocket::recvData(char *pBuf, int nReadLen, int nTimeOut)
{
	pBuf[0] = 0;
	if (INVALID_SOCKET == m_Socket)
		return -1;

	const struct timeval time = { nTimeOut/1000, (nTimeOut%1000) * 1000 };

	fd_set fd;
	FD_ZERO(&fd);
	FD_SET(m_Socket, &fd);

	int ret = ::select(m_Socket+1, &fd, NULL, NULL, &time);
	if ( ret )
	{
		if ( FD_ISSET(m_Socket, &fd) )
		{
			ret = ::recv(m_Socket, pBuf, nReadLen, 0);
			ret = (ret <= 0) ? -1 : ret;
		}
	}
	else if(ret < 0)
	{
		ret = -1;
	}

	return ret;
}

// 成功返回0
int CBcecrSocket::sendData(const char *pData, int nSendLen)
{
	if (false == m_bConnected || INVALID_SOCKET == m_Socket)
		return -1;

	const struct timeval time = { 0, 200 * 1000 };

	fd_set fdSend;
	int nLen = nSendLen;
	const char *pTmp = pData;
	int nRet = 0;
	while (nLen)
	{
		FD_ZERO(&fdSend);
		FD_SET(m_Socket, &fdSend);

		int ret = ::select(m_Socket+1, NULL, &fdSend, NULL, &time);
		if ( 1 == ret )
		{
			if ( FD_ISSET(m_Socket, &fdSend) )
			{
				ret = ::send(m_Socket, pTmp, nLen, 0);
				if (ret <= 0)
				{
					nRet = -1;
					break;
				}

				nLen -= ret;
				pTmp += ret;
			}
		}
		else if ( ret <= 0)
		{
			nRet = ret;
			break;
		}
	}

	return nRet;
}

// 返回和检测结果name的相似度
double CBcecrSocket::Identify(char **name)
{
	char buf[128] = { 0 };
	int max_try = 10;
	do {
		if (recvData(buf, 128, 200) < 0)
		{
			DisConnect();
			break;
		}
	}while (0 == buf[0] && --max_try);
	double f_score = 0;
	if (buf[0])
	{
		const WCHAR* unicode = UTF8Convert2Unicode(buf);
		char *s = *name = UnicodeConvert2ANSI(unicode);
		while (*s && ':' != *s) ++s;
		if (*s) f_score = atof(s + 1);
		*s = '\0';
		if(unicode) delete [] unicode;
	}
	return f_score;
}
