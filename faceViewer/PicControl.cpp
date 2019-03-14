// PicControl.cpp : 实现文件
//

#include "stdafx.h"
#include "faceViewer.h"
#include "PicControl.h"


// CPicControl

IMPLEMENT_DYNAMIC(CPicControl, CStatic)

CPicControl::CPicControl()
{

}

CPicControl::~CPicControl()
{
}


void CPicControl::SetImagePath(const CString &path)
{
	if (!m_image.IsNull())
		m_image.Destroy();
	m_image.Load(path);
}


BEGIN_MESSAGE_MAP(CPicControl, CStatic)
	ON_WM_PAINT()
END_MESSAGE_MAP()


// CPicControl 消息处理程序


void CPicControl::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	if (!m_image.IsNull())
	{
		CRect rect;
		GetClientRect(&rect);
		m_image.Draw(dc.GetSafeHdc(), rect);
	}
}
