#include "stdafx.h"
#include "SqliteManager.h"
#include "functions.h"
#include "cppsqlite3\CppSQLite3.h"

CSqliteManager g_dbManager;

// sqlite3动态库
#define DB_ENGINE "sqlite3.dll"

// 图编号 名称 类别 是否映射 特征向量
// 数据库：id path class embedding p1 p2 ... p512
#define  DB_FILE_NAME  "face.db"


CSqliteManager::CSqliteManager(void)
{
	m_pDatabase = NULL;
	m_pStmt[0] = m_pStmt[1] = NULL;
}


CSqliteManager::~CSqliteManager(void)
{
	SAFEDELETE(m_pDatabase);
}


bool CSqliteManager::initSqliteDB()
{
	// check db file exist
	std::string exePath = getExePath();
	std::string sEnginePath = exePath + DB_ENGINE;
	std::string sDBFilePath = exePath + DB_FILE_NAME;

	if (!isFileExist(sDBFilePath.c_str()))
	{
		if (!CreateDataBase(sDBFilePath.c_str()))
		{
			printf("CSqliteManager: Create Database file failed.\n");
			MY_LOG_INFO("CSqliteManager: Create Database file failed.\n");
			return false;
		}
	}

	NAMES::db_handle::set_loadpath(sEnginePath.c_str());
	if (NULL == m_pDatabase)
		m_pDatabase = new NAMES::db_sqlite(sDBFilePath.c_str());

	if (!m_pDatabase->open())
	{
		printf("CSqliteManager: Open database error.\n");
		MY_LOG_INFO("CSqliteManager: Open database error.\n");

		SAFEDELETE(m_pDatabase);
		return false;
	}
	m_pDatabase->sync_off();

	return true;
}


State CSqliteManager::embeddingCheck(const char *path, int &id)
{
	id = 0;
	char sq[256];
	sprintf_s(sq, "select id, embedding from faceInfo where path = \"%s\";", path);
	if (m_pDatabase->sql_select(sq))
	{
		const NAMES::db_rows* result = m_pDatabase->get_result();
		if (result)
		{
			const std::vector<NAMES::db_row*>& rows = result->get_rows();
			const NAMES::db_row* row = rows[0];
			id = atoi((*row)["id"]);
			return atoi((*row)["embedding"]) ? Embbed_True : Embbed_False;
		}
		return Not_Exist;
	}
	return State_Error;
}


bool CSqliteManager::InsertEmbedding(const char *path, const float *embed)
{
	static buffer insert;
	if (NULL == m_pStmt[0])
	{
		if (0 == insert.sq[0])
		{
			insert.append("insert into faceInfo (path, embedding, ");
			for (int i = 0; i < 511; ++i)
				insert.append("p%d, ", i + 1);
			insert.append("p512) values (?, 1, ");
			for (int i = 0; i < 511; ++i)
				insert.append("?, ");
			insert.append("?);");
		}
		sqlite3* db = m_pDatabase->getHandle();
		int s = sqlite3_prepare_v2(db, insert.sq, insert.length(), &m_pStmt[0], 0);
	}
	sqlite3_reset(m_pStmt[0]);
	sqlite3_bind_text(m_pStmt[0], 1, path, -1, SQLITE_STATIC);
	for (int i = 0; i < 512; ++i)
		sqlite3_bind_double(m_pStmt[0], i+2, embed[i]);
	sqlite3_step(m_pStmt[0]);

	return true;
}


bool CSqliteManager::UpdateEmbbedding(int id, const float *embed)
{
	static buffer update;
	if (NULL == m_pStmt[1])
	{
		if (0 == update.sq[0])
		{
			update.append("update faceInfo set embedding=1, ");
			for (int i = 0; i < 511; ++i)
				update.append("p%d=?, ", i + 1);
			update.append("p512=? where id=?;");
		}
		sqlite3* db = m_pDatabase->getHandle();
		int s = sqlite3_prepare_v2(db, update.sq, update.length(), &m_pStmt[1], 0);
	}
	sqlite3_reset(m_pStmt[1]);
	for (int i = 0; i < 512; ++i)
		sqlite3_bind_double(m_pStmt[1], i + 1, embed[i]);
	sqlite3_bind_int(m_pStmt[1], 513, id);
	sqlite3_step(m_pStmt[1]);

	return true;
}


void CSqliteManager::begin()
{
	if (m_pDatabase)
		m_pDatabase->begin();
}


void CSqliteManager::commit()
{
	if (m_pStmt[0])
	{
		sqlite3_finalize(m_pStmt[0]);
		m_pStmt[0]= NULL;
	}
	if (m_pStmt[1])
	{
		sqlite3_finalize(m_pStmt[1]);
		m_pStmt[0]= NULL;
	}
	if (m_pDatabase)
		m_pDatabase->commit();
}
