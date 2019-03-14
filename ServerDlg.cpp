// ServerDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "objDetector.h"
#include "ServerDlg.h"
#include "afxdialogex.h"


// CServerDlg 对话框

IMPLEMENT_DYNAMIC(CServerDlg, CDialogEx)

CServerDlg::CServerDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CServerDlg::IDD, pParent)
	, m_strServerIP(_T(""))
	, m_nServerPort(0)
	, m_nType(SERVER_FACE_IDENTIFY)
{

}

CServerDlg::~CServerDlg()
{
}

void CServerDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SERVER_IP, m_EditServerIP);
	DDX_Control(pDX, IDC_SERVER_PORT, m_EditServerPort);
	DDX_Text(pDX, IDC_SERVER_IP, m_strServerIP);
	DDX_Text(pDX, IDC_SERVER_PORT, m_nServerPort);
	DDV_MinMaxInt(pDX, m_nServerPort, 0, 65535);
	DDX_Control(pDX, IDC_COMBO_TYPE, m_ComboType);
}


BEGIN_MESSAGE_MAP(CServerDlg, CDialogEx)
	ON_CBN_SELCHANGE(IDC_COMBO_TYPE, &CServerDlg::OnCbnSelchangeComboType)
END_MESSAGE_MAP()


// CServerDlg 消息处理程序


BOOL CServerDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	m_ComboType.InsertString(SERVER_FACE_IDENTIFY, _T("人脸识别"));
	m_ComboType.InsertString(SERVER_EMOTION_IDENTIFY, _T("表情识别"));
	m_ComboType.InsertString(SERVER_AGE_IDENTIFY, _T("年龄识别"));
	m_ComboType.SetCurSel(m_nType);

	return TRUE;
}


void CServerDlg::OnCbnSelchangeComboType()
{
	m_nType = m_ComboType.GetCurSel();
}
