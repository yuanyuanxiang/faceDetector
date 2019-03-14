
#include <wtypes.h>

#ifdef _WIN32

#pragma once

// 将Unicode编码转换成ANSI编码
char* UnicodeConvert2ANSI(LPCWCH strEncodeData);

// 将utf-8编码转换成Unicode编码
WCHAR* UTF8Convert2Unicode(const char* strUtf8);

#endif