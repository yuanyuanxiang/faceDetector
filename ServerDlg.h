#pragma once
#include "afxwin.h"

enum 
{
	SERVER_FACE_IDENTIFY = 0,		// 人脸识别
	SERVER_EMOTION_IDENTIFY = 1,	// 表情识别
	SERVER_AGE_IDENTIFY = 2,		// 年龄识别
	SERVER_MAX, 
};

// CServerDlg 对话框

class CServerDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CServerDlg)

public:
	CServerDlg(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CServerDlg();

// 对话框数据
	enum { IDD = IDD_SERVER_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	CEdit m_EditServerIP;
	CEdit m_EditServerPort;
	CString m_strServerIP;
	int m_nServerPort;
	CComboBox m_ComboType;
	int m_nType;
	virtual BOOL OnInitDialog();
	afx_msg void OnCbnSelchangeComboType();
};
