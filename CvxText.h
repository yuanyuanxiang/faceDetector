//====================================================================  
//====================================================================  
// 文件: CvxText.h   
// 说明: OpenCV 3.x 汉字输出    
// 时间:  2016-11-11 
// 改写作者: zmdsjtu@163.com  
//博客地址:http://blog.csdn.net/zmdsjtu/article/category/6371625

//原作者： chaishushan@gmail.com   2007-8-21
//====================================================================  
//====================================================================  

#ifndef OPENCV_Cv310Text_2016_11_11_ZMD 
#define OPENCV_Cv310Text_2016_11_11_ZMD 


#include "ft2build.h"
#include "freetype/freetype.h"
#include "config.h"


// Freetype 图片叠加字体类
class CvxText
{
public:

	/**
	* 装载字库文件，默认为仿宋"simfang.ttf"
	*/
	CvxText(const char *freeType = NULL);
	virtual ~CvxText();

	/**
	* 获取字体。目前有些参数尚不支持。
	*
	* \param font        字体类型, 目前不支持
	* \param size        字体大小/空白比例/间隔比例/旋转角度
	* \param underline   下画线
	* \param diaphaneity 透明度
	*
	* \sa setFont, restoreFont
	*/
	void getFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);

	/**
	* 设置字体。目前有些参数尚不支持。
	*
	* \param font        字体类型, 目前不支持
	* \param size        字体大小/空白比例/间隔比例/旋转角度
	* \param underline   下画线
	* \param diaphaneity 透明度
	*
	* \sa getFont, restoreFont
	*/
	void setFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);

	/**
	* 恢复原始的字体设置。
	*
	* \sa getFont, setFont
	*/
	void restoreFont();

	/**
	* 输出汉字。遇到不能输出的字符将停止。
	*
	* \param img   输出的影象
	* \param text  文本内容
	* \param pos   文本位置
	* \param color 文本颜色
	* 
	*/
	void putText(cv::Mat &frame, const char *text, const CvPoint &pos, const CvScalar &color);

	/**
	* 输出汉字。遇到不能输出的字符将停止。
	*
	* \param img   输出的影象
	* \param text  文本内容
	* \param pos   文本位置
	* \param color 文本颜色
	* 
	*/
	void putText(cv::Mat &frame, const wchar_t *text, const CvPoint &pos, const CvScalar &color);

private:

	// 输出当前字符, 更新pos位置  
	void putWChar(cv::Mat &frame, wchar_t wc, CvPoint &pos, CvScalar color);

	// 加载字体
	bool LoadFont(const char *path);

private:

	FT_Library		m_library;   // 字库
	FT_Face			m_face;      // 字体

	bool			m_bOk;		// freetype字体是否就绪
	int				m_fontType; // 字体类型
	CvScalar		m_fontSize; // 字体大小
	bool			m_fontUnderline; // 下划线
	float			m_fontDiaphaneity;// 融合
};

#endif // OPENCV_Cv310Text_2016_11_11_ZMD
