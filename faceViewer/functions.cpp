#include "stdafx.h"
#include "functions.h"
#include <io.h>


std::string getExePath()
{
	char cPath[MAX_PATH] = {0}, *p = cPath;

	::GetModuleFileNameA(NULL, cPath, MAX_PATH);
	while ('\0' != *p) ++p;
	while (cPath != p && '\\' != *p) --p;
	*(p + 1) = 0;

	return cPath;
}


bool isFileExist(const char* path)
{
	return (_access(path, 0) == -1) ? false : true;
}
