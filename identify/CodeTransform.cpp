#include "CodeTransform.h"

/// 将Unicode编码转换成ANSI编码
char* UnicodeConvert2ANSI(LPCWCH strUnicode)
{
	int ncLength = WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, NULL, 0, NULL, NULL);
	char *strANSI = new char[ncLength];
	ncLength = WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, strANSI, ncLength, NULL, NULL);
	strANSI[ncLength - 1] = 0;
	return strANSI;
}

/// 将utf-8编码转换成Unicode编码
WCHAR* UTF8Convert2Unicode(const char* strUtf8)
{
	int ncLength = MultiByteToWideChar(CP_UTF8, 0, strUtf8, -1, NULL, 0);
	WCHAR *strUnicode = new WCHAR[ncLength];
	ncLength = MultiByteToWideChar(CP_UTF8, 0, strUtf8, -1, strUnicode, ncLength);
	strUnicode[ncLength - 1] = 0;
	return strUnicode;
}
