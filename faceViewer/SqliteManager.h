#pragma once

#include "mString.h"

#if USING_ACL_SQLITE
#include "acl_cpp/db/db_sqlite.hpp"
#endif

enum State
{
	State_Error = -2, 
	Not_Exist = -1, 
	Embbed_False = 0, 
	Embbed_True = 1, 
};

struct buffer
{
	char sq[1024+32*512];
	char *p;
	buffer() { sq[0] = 0; p = sq; }
	buffer(const buffer &o)
	{
		memcpy(sq, o.sq, sizeof(sq));
		p = sq + (o.p - o.sq);
	}
	int length() const { return p - sq; }
	void append(const char* fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		vsprintf(p, fmt, ap);
		p += strlen(p);
		va_end(ap);
	}
};

/**
* @class CSqliteManager
* @brief 数据库管理器
* @date 2018-7-7 袁沅祥
*/
class CSqliteManager
{
private:
	NAMES::db_handle*	m_pDatabase; // 数据库连接句柄
	sqlite3_stmt		*m_pStmt[2];	// 执行准备

public:
	CSqliteManager(void);
	~CSqliteManager(void);

	// 初始化数据库管理器
	bool initSqliteDB();
	// 执行sql语句，适用于insert、update、delete不注重数据内容的命令
	// 未加锁，防止嵌套调用出现死锁
	bool executeSql(const char* sql)
	{
		return m_pDatabase ? m_pDatabase->sql_update(sql) : false;
	}

	// 检查是否已提取人脸特征向量
	State embeddingCheck(const char *path, int &id);
	// 更新人脸数据库
	bool InsertEmbedding(const char *path, const float *embed);
	bool UpdateEmbbedding(int id, const float *embed);

	void begin();

	void commit();
};

extern	CSqliteManager	g_dbManager;
