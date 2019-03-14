#include <wchar.h>  
#include <assert.h>  
#include <locale.h>  
#include <ctype.h>  
#include "CvxText.h"  

#pragma comment(lib, "freetype-2.9.1/win64/freetype.lib")

#define FT_SUCCESS 0

bool CvxText::LoadFont(const char *path)
{
	if (!m_bOk)
	{
		char full_path[_MAX_PATH];
		strcpy_s(full_path, (path && path[0]) ? path : "simfang.ttf");
		if (full_path[1] && ':' != full_path[1])
		{
			char buf[_MAX_PATH];
			strcpy_s(buf, full_path);
			sprintf(full_path, "C:\\Windows\\Fonts\\%s", buf);
		}

		// 打开字库文件, 创建一个字体
		if (FT_SUCCESS == FT_Init_FreeType(&m_library))
		{
			if (FT_SUCCESS == FT_New_Face(m_library, full_path, 0, &m_face))
			{
				// 设置字体输出参数
				restoreFont();
				// 设置C语言的字符集环境
				setlocale(LC_ALL, "");
				m_bOk = true;
			}
		}
		if (!m_bOk)
			OUTPUT("======> 加载Freetype字体\"%s\"失败!\n", full_path);
	}
	return m_bOk;
}


CvxText::CvxText(const char *freeType)
{
	m_bOk = false;

	LoadFont((NULL == freeType || 0 == *freeType) ? "simfang.ttf" : freeType);
}

// 释放FreeType资源
CvxText::~CvxText()
{
	FT_Done_Face(m_face);
	FT_Done_FreeType(m_library);
}

// 设置字体参数:  
//  
// font         - 字体类型, 目前不支持  
// size         - 字体大小/空白比例/间隔比例/旋转角度  
// underline   - 下画线  
// diaphaneity   - 透明度  
void CvxText::getFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	if (type) *type = m_fontType;
	if (size) *size = m_fontSize;
	if (underline) *underline = m_fontUnderline;
	if (diaphaneity) *diaphaneity = m_fontDiaphaneity;
}

void CvxText::setFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	// 参数合法性检查
	if (type)
	{
		if (type >= 0) m_fontType = *type;
	}
	if (size)
	{
		m_fontSize.val[0] = fabs(size->val[0]);
		m_fontSize.val[1] = fabs(size->val[1]);
		m_fontSize.val[2] = fabs(size->val[2]);
		m_fontSize.val[3] = fabs(size->val[3]);
	}
	if (underline)
	{
		m_fontUnderline = *underline;
	}
	if (diaphaneity)
	{
		m_fontDiaphaneity = *diaphaneity;
	}
}

// 恢复原始的字体设置
void CvxText::restoreFont()
{
	m_fontType = 0;            // 字体类型(不支持)  

	m_fontSize.val[0] = 20;      // 字体大小  
	m_fontSize.val[1] = 0.5;   // 空白字符大小比例  
	m_fontSize.val[2] = 0.1;   // 间隔大小比例  
	m_fontSize.val[3] = 0;      // 旋转角度(不支持)  

	m_fontUnderline = false;   // 下画线(不支持)  

	m_fontDiaphaneity = 1.0;   // 色彩比例(可产生透明效果)  

	// 设置字符大小  
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}

void CvxText::putText(cv::Mat &frame, const char *text, const CvPoint &pos, const CvScalar &color)
{
	if (frame.empty() || NULL == text) return;

	if (m_bOk)
	{
		CvPoint scan(pos);
		for (int i = 0; text[i] != '\0'; ++i)
		{
			wchar_t wc = text[i];
			if (!isascii(wc)) mbtowc(&wc, &text[i++], 2); // 解析双字节符号
			// 输出当前的字符
			putWChar(frame, wc, scan, color);
		}
	}else
		cv::putText(frame, text, pos, CV_FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
}

void CvxText::putText(cv::Mat &frame, const wchar_t *text, const CvPoint &pos, const CvScalar &color)
{
	if (frame.empty() || NULL == text) return;


	if (m_bOk)
	{
		CvPoint scan(pos);
		for (int i = 0; text[i] != '\0'; ++i)
		{
			// 输出当前的字符  
			putWChar(frame, text[i], scan, color);
		}
	}else
	{
		char buf[256];
		size_t count = 0;
		wcstombs_s(&count, buf, text, 64);
		cv::putText(frame, buf, pos, CV_FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
	}
}

// 输出当前字符, 更新pos位置
void CvxText::putWChar(cv::Mat &frame, wchar_t wc, CvPoint &pos, CvScalar color)
{
	// 根据unicode生成字体的二值位图
	IplImage* img = &(IplImage)frame;

	FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
	FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
	FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_MONO);

	FT_GlyphSlot slot = m_face->glyph;

	// 行列数
	int rows = slot->bitmap.rows;
	int cols = slot->bitmap.width;

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			int off = ((img->origin == 0) ? i : (rows - 1 - i))* slot->bitmap.pitch + j / 8;
			if (slot->bitmap.buffer[off] & (0xC0 >> (j % 8)))
			{
				int r = (img->origin == 0) ? pos.y - (rows - 1 - i) : pos.y + i;
				int c = pos.x + j;
				if (r >= 0 && r < img->height && c >= 0 && c < img->width)
				{
					CvScalar scalar = cvGet2D(img, r, c);
					// 进行色彩融合
					double p = m_fontDiaphaneity;
					scalar.val[0] = scalar.val[0] * (1 - p) + color.val[0] * p;
					scalar.val[1] = scalar.val[1] * (1 - p) + color.val[1] * p;
					scalar.val[2] = scalar.val[2] * (1 - p) + color.val[2] * p;
					scalar.val[3] = scalar.val[3] * (1 - p) + color.val[3] * p;
					cvSet2D(img, r, c, scalar);
				}
			}
		} // end for  
	} // end for  

	// 修改下一个字的输出位置  
	double space = m_fontSize.val[0] * m_fontSize.val[1];
	double sep = m_fontSize.val[0] * m_fontSize.val[2];
	pos.x += (int)((cols ? cols : space) + sep);
}
