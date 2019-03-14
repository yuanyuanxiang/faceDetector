
// faceViewerDlg.h : 头文件
//

#pragma once
#include "PicControl.h"
#include <map>
#include <queue>
#include "pyCaller.h"

// 每行/列最大支持播放的窗口
#define MAX_WNDSIZE 12

enum 
{
	Thread_Unknown = 0, 
	Thread_Start, 
	Thread_Stop, 
	App_Exit, 
};

//////////////////////////////////////////////////////////////////////////
// 检测指定目录及其子目录下的图片(按名称排序)，定期对其进行刷新显示

// CfaceViewerDlg 对话框
class CfaceViewerDlg : public CDialogEx
{
// 构造
public:
	CfaceViewerDlg(CWnd* pParent = NULL);	// 标准构造函数
	~CfaceViewerDlg();

// 对话框数据
	enum { IDD = IDD_FACEVIEWER_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

	// 监视的目录
	char m_path[_MAX_PATH];
	// 当前显示的窗口行/列向个数
	int m_nSize;
	// 刷新间隔（s）
	int m_nTime;
	// 目录及子目录图像数
	int m_nCount;
	// 最多可显示的图像数
	CPicControl *m_Pictures[MAX_WNDSIZE * MAX_WNDSIZE];
	// 目录及图像个数映射关系
	std::map<CString, int> m_FolderMap;

	// 调整窗口位置
	void reSizeWindow();
	// 刷新图像
	int refreshFaces(const CString &folder, pyCaller *py);
	// 获取目录下"jpg"图片个数
	int getFolderPicNum(const CString &folder, std::queue<CString> &result) const;
	// 获取目录上一次"jpg"图片个数
	int getLastPicNum(const CString &folder) const;
	bool m_bFullScreen;
	WINDOWPLACEMENT m_struOldWndpl;
	// 全屏处理函数
	void FullScreenProc();

	// 刷新图像的线程
	int m_nThreadState;
	static void RefreshThread(void *param);

// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	virtual BOOL PreTranslateMessage(MSG* pMsg);
};
