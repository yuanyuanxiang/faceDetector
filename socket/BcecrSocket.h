#pragma once
#include <WinSock2.h>
#include <string>

enum 
{
	THREADSTATE_UNKNOWN, // 未知
	THREADSTATE_START,	// 线程启动
	THREADSTATE_STOP,	// 线程退出
	APP_TERMINATE,		// 程序退出
};

/** 
* @class	CBcecrSocket 
* @brief    建立socket通信客户端程序
* @details	实现基本的收/发数据的功能
*/
class CBcecrSocket
{
public:
	/// 构造函数
	CBcecrSocket();
	/// 析构
	~CBcecrSocket();
	/// 设置服务端IP及端口
	void SetServerInfo(const char *pIp, int nPort);
	/// 断开与Server的连接
	void DisConnect();
	/// 接收数据
	int recvData(char *pBuf, int nReadLen, int nTimeOut = 1000); //nTimeOut单位毫秒
	/// 发送数据
	int sendData(const char *pData, int nSendLen);
	/// 是否已连接成功
	bool IsConnected() const { return m_bConnected; }
	// 识别图像
	double Identify(char **name);

private:
	bool Connect(); // 重连

	static void  ConnectServer(void *param); // 重连线程

private:

	SOCKET m_Socket;		/**< 作为客户端连接的socket */

	char m_chToIp[32];				/**< 对方的IP */
	int  m_nToport;					/**< 对方的端口 */
	int m_nThreadState;				/**< 线程状态 */
	bool m_bConnected;				/**< 是否连接成功 */
};
