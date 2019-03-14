
// faceViewerDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "faceViewer.h"
#include "faceViewerDlg.h"
#include "afxdialogex.h"
#include <io.h>
#include "SqliteManager.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CfaceViewerDlg 对话框



CfaceViewerDlg::CfaceViewerDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CfaceViewerDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_nSize = 8;
	m_nTime = 10;
	m_nCount = 0;
	m_bFullScreen = false;
	m_nThreadState = Thread_Unknown;
	memset(m_Pictures, 0, MAX_WNDSIZE * MAX_WNDSIZE * sizeof(CPicControl*));
}


CfaceViewerDlg::~CfaceViewerDlg()
{
	if (Thread_Stop != m_nThreadState)
		m_nThreadState = App_Exit;
	while (Thread_Stop != m_nThreadState)
		Sleep(10);
}


void CfaceViewerDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CfaceViewerDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DESTROY()
	ON_WM_SIZE()
END_MESSAGE_MAP()


// CfaceViewerDlg 消息处理程序

void CfaceViewerDlg::reSizeWindow()
{
	if (m_Pictures)
	{
		CRect rect;
		GetClientRect(&rect);
		for (int i = 0; i < m_nSize; ++i)
		{
			int h = rect.Height() / m_nSize;
			int row = float(i) * rect.Height() / m_nSize;
			for (int j = 0; j < m_nSize; ++j)
			{
				int w = rect.Width() / m_nSize;
				int col = float(j) * rect.Width() / m_nSize;
				if (m_Pictures[j + i * m_nSize]->GetSafeHwnd())
					m_Pictures[j + i * m_nSize]->MoveWindow(col, row, w, h);
			}
		}
	}
}


void DoDetect(pyCaller *py, const char *folder, const char *name)
{
	char file[_MAX_PATH] = { 0 };
	strcpy_s(file, folder);
	strcat(file, name);
	cv::Mat m = imread(file);
	int id = 0;
	State s = m.empty() ? Embbed_True : g_dbManager.embeddingCheck(name, id);
	if (Embbed_True != s)
	{
		npy_intp dims[] = { m.rows, m.cols, 3 }; // 给定维度信息
		// 生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数
		// 第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
		PyObject *PyArray  = PyArray_SimpleNewFromData(3, dims, NPY_UBYTE, m.data);
		// 同样定义大小与Python函数参数个数一致的PyTuple对象
		PyObject *ArgArray = PyTuple_New(1);
		PyTuple_SetItem(ArgArray, 0, PyArray); 
		tfOutput tf = py->CallFunction("Identify", ArgArray);
		switch (s)
		{
		case Not_Exist: case State_Error:
			if (!g_dbManager.InsertEmbedding(name, tf.boxes))
				TRACE("======> InsertEmbedding path=%s failed.\n", name);
			break;
		case Embbed_False:
			if (!g_dbManager.UpdateEmbbedding(id, tf.boxes))
				TRACE("======> InsertEmbedding id=%d failed.\n", id);
			break;
		}
	}
}

int CfaceViewerDlg::refreshFaces(const CString &folder, pyCaller *py)
{
	clock_t t = clock();
	CFileFind ff;
	BOOL bFind = ff.FindFile(folder + _T("\\*.*"));
	std::queue<CString> result;
	int count = 0;
	USES_CONVERSION;
	char folder_s[_MAX_PATH];
	sprintf_s(folder_s, "%s\\", W2A(folder));
	g_dbManager.begin();
	while (bFind)
	{
		bFind = ff.FindNextFile();
		if (ff.IsDots()) continue;
		else if (ff.IsDirectory())
		{
			CString sub_dir = ff.GetFilePath();
			int n = getFolderPicNum(sub_dir, result);
			count += n;
			m_FolderMap[sub_dir] = n;
			continue;
		}
		if (result.size() == m_nSize * m_nSize)
			result.pop();
		result.push(ff.GetFilePath());
		if (m_nCount < ++count && py->IsModuleLoaded())// do embedding
		{
			const char *path = W2A(ff.GetFileName());
			DoDetect(py, folder_s, path);
			if (0 == count % 1000) // 处理完一批图像保存一下
				break;
		}
	}
	ff.Close();
	if (count != m_nCount)
	{
		
		for(int i = 0; result.size() && i != m_nSize * m_nSize; ++i)
		{
			const CString & cur = result.front();
			m_Pictures[i]->SetImagePath(cur);
			result.pop();
		}
		Invalidate(TRUE);
		t = clock() - t;
		TRACE("======> embedding %d faces using time: %d ms.\n", count-m_nCount, t);
		m_nCount = count;
	}else {
		t = clock() - t;
		TRACE("======> Just visit faceInfo database using time: %d ms.\n", t);
	}
	g_dbManager.commit();
	return bFind;
}


int CfaceViewerDlg::getFolderPicNum(const CString &folder, std::queue<CString> &result) const
{
	CFileFind ff;
	BOOL bFind = ff.FindFile(folder + _T("\\*.jpg"));
	int count = 0, nLastPicNum = getLastPicNum(folder);
	while (bFind)
	{
		bFind = ff.FindNextFile();
		if (++ count > nLastPicNum)
		{
			if (result.size() == m_nSize * m_nSize)
				result.pop();
			result.push(ff.GetFilePath());
		}
	}
	ff.Close();
	return count;
}


int CfaceViewerDlg::getLastPicNum(const CString &folder) const
{
	for (std::map<CString, int>::const_iterator pos = m_FolderMap.begin(); pos != m_FolderMap.end(); ++pos)
		if (folder == pos->first)
			return pos->second;
	return 0;
}


void CfaceViewerDlg::FullScreenProc()
{
	if (false == m_bFullScreen)
	{
		//get current system resolution
		int g_iCurScreenWidth = GetSystemMetrics(SM_CXSCREEN);
		int g_iCurScreenHeight = GetSystemMetrics(SM_CYSCREEN);

		//for full screen while backplay
		GetWindowPlacement(&m_struOldWndpl);

		CRect rectWholeDlg;//entire client(including title bar)
		CRect rectClient;//client area(not including title bar)
		CRect rectFullScreen;
		GetWindowRect(&rectWholeDlg);
		RepositionBars(0, 0xffff, AFX_IDW_PANE_FIRST, reposQuery, &rectClient);
		ClientToScreen(&rectClient);

		rectFullScreen.left = rectWholeDlg.left-rectClient.left;
		rectFullScreen.top = rectWholeDlg.top-rectClient.top;
		rectFullScreen.right = rectWholeDlg.right+g_iCurScreenWidth - rectClient.right;
		rectFullScreen.bottom = rectWholeDlg.bottom+g_iCurScreenHeight - rectClient.bottom;
		//enter into full screen;
		WINDOWPLACEMENT struWndpl;
		struWndpl.length = sizeof(WINDOWPLACEMENT);
		struWndpl.flags = 0;
		struWndpl.showCmd = SW_SHOWNORMAL;
		struWndpl.rcNormalPosition = rectFullScreen;
		SetWindowPlacement(&struWndpl);

		m_bFullScreen = true;
	}else
	{
		SetWindowPlacement(&m_struOldWndpl);
		m_bFullScreen = false;
	}
}


void CfaceViewerDlg::RefreshThread(void *param)
{
	OutputDebugStringA("======> RefreshThread Start.\n");
	CfaceViewerDlg *pThis = (CfaceViewerDlg *)param;
	pThis->m_nThreadState = Thread_Start;
	pyCaller py("embedding");
	py.ActivateFunc("Identify");
	char path[_MAX_PATH], *p = path;
	if (':' != pThis->m_path[1])// 如果没有填写完整的路径
	{
		GetModuleFileNameA(NULL, path, sizeof(path));
		while (*p) ++p;
		while ('\\' != *p) --p;
		strcpy(p + 1, pThis->m_path);
	}else
		strcpy(path, pThis->m_path);

	do 
	{
		if (-1 == _access(path, 0))
		{
			AfxMessageBox(_T("以下目录不存在:\r\n")+CString(path), MB_ICONINFORMATION | MB_OK);
			break;
		}
		BOOL bFind = FALSE;
		if (Thread_Start == pThis->m_nThreadState)
			bFind = pThis->refreshFaces(CString(path), &py);
		if (FALSE == bFind)
		{
			for (int k = 50 * pThis->m_nTime; k && (Thread_Start == pThis->m_nThreadState); --k)
				Sleep(20);
		}
	} while (Thread_Start == pThis->m_nThreadState);

	pThis->m_nThreadState = Thread_Stop;
	OutputDebugStringA("======> RefreshThread Stop.\n");
	if (-1 == _access(path, 0))
		pThis->SendMessage(WM_CLOSE, 0, 0);
}


BOOL CfaceViewerDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	char path[_MAX_PATH], *p = path;
	GetModuleFileNameA(NULL, path, sizeof(path));
	while (*p) ++p;
	while ('\\' != *p) --p;
	strcpy(p + 1, "faceViewer.ini");
	m_nSize = GetPrivateProfileIntA("settings", "wnd_per_row", 8, path);
	m_nSize = max(m_nSize, 1);
	m_nSize = min(m_nSize, MAX_WNDSIZE);
	m_nTime = GetPrivateProfileIntA("settings", "fresh_elapse", 8, path);
	m_nTime = max(m_nTime, 3);
	GetPrivateProfileStringA("settings", "watch_folder", "object_detection\\face", m_path, sizeof(m_path), path);
	if (0 == m_path[0])
		strcpy_s(m_path, "object_detection\\face");
	for (int i = 0; i < m_nSize; ++i)
	{
		for (int j = 0; j < m_nSize; ++j)
		{
			m_Pictures[j + i * m_nSize] = new CPicControl();
			CString text;
			text.Format(_T("%d-%d"), i, j);
			m_Pictures[j + i * m_nSize]->Create(text, WS_CHILD | WS_VISIBLE, CRect(), this);
		}
	}
	ShowWindow(SW_SHOWMAXIMIZED);
	reSizeWindow();
	g_dbManager.initSqliteDB();
	_beginthread(&RefreshThread, 0, this);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CfaceViewerDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CfaceViewerDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CfaceViewerDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CfaceViewerDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	for (int i = 0; i < MAX_WNDSIZE * MAX_WNDSIZE; ++i)
	{
		if (m_Pictures[i])
			delete m_Pictures[i];
		m_Pictures[i] = NULL;
	}
}


void CfaceViewerDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);

	reSizeWindow();
}


BOOL CfaceViewerDlg::PreTranslateMessage(MSG* pMsg)
{
	if( pMsg->message == WM_KEYDOWN)
	{
		switch (pMsg->wParam)
		{
		case VK_F11:// 全屏处理
			BeginWaitCursor();
			FullScreenProc();
			EndWaitCursor();
			return TRUE;
		case VK_ESCAPE: case VK_RETURN:// 屏蔽 ESC/Enter 关闭窗体
			return TRUE;
		default:
			break;
		}
	}

	return CDialogEx::PreTranslateMessage(pMsg);
}
