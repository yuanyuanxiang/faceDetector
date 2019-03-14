#pragma once


// CPicControl

class CPicControl : public CStatic
{
	DECLARE_DYNAMIC(CPicControl)

private:
	CImage m_image;

public:
	CPicControl();
	virtual ~CPicControl();

	void SetImagePath(const CString &path);

protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnPaint();
};


