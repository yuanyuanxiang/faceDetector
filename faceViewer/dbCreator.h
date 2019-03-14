#pragma once

#include "cppsqlite3/CppSQLite3.h"

bool CreateDataBase(const char* strPath, CppSQLite3DB *m_pDatabase = NULL);
