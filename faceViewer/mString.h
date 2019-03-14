/**	
* my::string 该类用来替代 acl::string, my::locker 用来替代
* acl::locker。
* 若将来不使用ACL，方便进行替代。在 mString.h 文件中定义了
* #define USING_ACL 1，表示是否使用ACL字符串。同步更新了ESU。
* 若 USING_ACL 设置为1，将使用ACL的字符串及互斥锁，否则使用
* 本文件的字符串及互斥锁。
* 将来若不使用ACL，仅需对CSqliteManager类进行有关数据库的修改。
*/

#pragma once

// 是否使用ACL
#define USING_ACL 0

// 是否使用ACL的Sqlite数据库
#define USING_ACL_SQLITE 0

// my string开始时的默认值（bytes）
#define DEFAULT_SIZE 1024

#include "dbCreator.h"

#if USING_ACL

//acl库文件
#include "acl_cpp/lib_acl.hpp"
#include "acl_cpp/stdlib/locker.hpp"
#include "acl_cpp/stdlib/string.hpp"
#include "acl_cpp/db/db_sqlite.hpp"

#define my acl
#define NAMES acl

#else

#define MY_API

#include <stdlib.h>
#include <string>
#include <stdarg.h>
#include <windows.h>
#include <process.h>
#include "cppsqlite3/CppSQLite3.h"
#include "StrTransfer.h"
#include <vector>

namespace my
{
	/**
	* @class locker
	* @brief 互斥锁，替代acl::locker
	*/
	class locker
	{
	public:
		locker() { ::InitializeCriticalSection(&m_csLock); }
		~locker(){ ::DeleteCriticalSection(&m_csLock); }

		void lock() { EnterCriticalSection(&m_csLock); }
		void unlock() { LeaveCriticalSection(&m_csLock); }

	private:
		CRITICAL_SECTION	m_csLock;
	};

	/**
	* @class string
	* @brief 自定义字符串，替代acl::string
	* @details 该类用来替代 acl::string。
	* 若将来不使用ACL，方便进行替代。在 mString.h 文件中定义了
	* #define USING_ACL 1，表示是否使用ACL字符串。同步更新了ESU。
	* 若 USING_ACL 设置为1，将使用ACL的字符串及互斥锁，否则使用
	* 本文件的字符串及互斥锁。
	* 将来若不使用ACL，仅需对CSqliteManager类进行有关数据库的修改。
	*/
	class string
	{
	private:
		int *m_nRef;		// 引用计数
		void AddRef() const { ++ *m_nRef; }
		int RemoveRef() const { return -- *m_nRef; }

	protected:
		int m_nSize;			// 缓存区长度
		char *m_pBuf;			// 字符串指针
		void resize(int nSize, int nLen) // 大小增加
		{
			if (nSize > m_nSize)
			{
				m_nSize = nSize;
				const char *t = m_pBuf;
				m_pBuf = new char[m_nSize];
				memcpy(m_pBuf, t, nLen + 1);
				delete [] t;
			}
		}
		char* addr(int i) const { return m_pBuf + i; }
		char& at(int i) const { return m_pBuf[i]; }

	public:
		void print(const char *method = NULL) const
		{
			printf("%s %s (Ref=%d).\n", method ? method : "", 
				*m_pBuf ? m_pBuf : "null", *m_nRef);
		}
		int GetRef() const { return *m_nRef; }
		char* c_str() const { return m_pBuf; }
		bool empty() const { return 0 == m_pBuf[0]; }

		explicit string(int nLen = DEFAULT_SIZE)
		{
			m_nRef = new int(1);
			m_nSize = max(nLen, 1);
			m_pBuf = new char[m_nSize];
			m_pBuf[0] = 0;
#if PRINT_REF
			print("默认构造");
#endif
		}

		string(const char *pSrc)
		{
			m_nRef = new int(1);
			m_nSize = pSrc ? strlen(pSrc) + 1 : DEFAULT_SIZE;
			m_pBuf = new char[m_nSize];
			strcpy(m_pBuf, pSrc ? pSrc : "");
#if PRINT_REF
			print("构造");
#endif
		}

		string(const string &other)
		{
			m_nSize = other.m_nSize;
			m_pBuf = other.m_pBuf;
			m_nRef = other.m_nRef;
			AddRef();
#if PRINT_REF
			print("拷贝构造");
#endif
		}

		string& operator = (const string &other)
		{
			int n = other.length();
			resize(n + 1, n);
			memcpy(m_pBuf, other.c_str(), n);
			m_pBuf[n] = 0;
			return *this;
		}

		~string()
		{
#if PRINT_REF
			print("析构");
#endif
			if (0 == RemoveRef())
			{
				delete [] m_pBuf;
				delete m_nRef;
			}
		}

		operator std::string()
		{
			return std::string(m_pBuf);
		}

		string& operator += (const string &other)
		{
			int n = length(), n0 = other.length();
			resize(n + n0 + 1, n);
			memcpy(m_pBuf + n, other.c_str(), n0);
			m_pBuf[n + n0] = 0;
			return *this;
		}

		/**
		* 清空当前对象的数据缓冲区
		* @return {string&} 当前对象的引用
		*/
		string& clear()
		{
			memset(m_pBuf, 0, m_nSize);
			return *this;
		}

		/**
		* 将指定字符串添加在当前对象数据缓冲区数据的尾部
		* @param s {const string&} 源数据对象指针
		* @return {string&} 当前对象的引用
		*/
		string& append(const char* s)
		{
			const char *p = s ? s : "";
			int n = length(), n0 = strlen(p);
			resize(n + n0 + 1, n);
			memcpy(m_pBuf + n, p, n0);
			m_pBuf[n + n0] = 0;
			return *this;
		}

		/**
		* 将指定字符串数据添加在当前对象数据缓冲区数据的首部
		* @param s {const char*} 源数据地址
		* @return {string&} 当前对象的引用
		*/
		string& prepend(const char* s)
		{
			const char *p = s ? s : "";
			int n = length(), n0 = strlen(p);
			resize(n + n0 + 1, n);
			memcpy(m_pBuf+n0, m_pBuf, n+1);// 是否安全
			memcpy(m_pBuf, p, n0);
			return *this;
		}

		friend 	bool operator == (const string &s1, const string &s2)
		{
			return 0 == strcmp(s1.c_str(), s2.c_str());
		}

		friend 	bool operator != (const string &s1, const string &s2)
		{
			return 0 != strcmp(s1.c_str(), s2.c_str());
		}

		/**
		* 查找指定字符在当前对象缓冲区的位置（下标从 0 开始）
		* @param n {char} 要查找的有符号字符
		* @return {int} 字符在缓冲区中的位置，若返回值 < 0 则表示不存在
		*/
		int find(char n) const
		{
			int i = 0;
			for (const char *p = m_pBuf; (n != *p && i < m_nSize); ++i)
				++p;
			return i == m_nSize ? -1 : i;
		}

		/**
		* 从尾部向前查找指定字符吕在当前对象缓冲区的起始位置（下标从 0 开始）
		* @param needle {const char*} 要查找的有符号字符串
		* @param case_sensitive {bool} 为 true 表示区分大小写
		* @return {char*} 字符串在缓冲区中的起始位置，若返回值为空指针则表示不存在
		*/
		char* rfind(const char* needle, bool case_sensitive=true) const
		{
			int n = length(), n0 = needle ? strlen(needle) : 0;
			char *p = m_pBuf + n - n0, *p0 = needle ? needle : "";
			do{
				if (0 == strncmp(p, p0, n0))
					return p;
			}while (--p >= m_pBuf);
			return NULL;
		}

		/**
		* 返回从当前字符串对象中缓冲区指定位置以左的内容
		* @param npos {size_t} 下标位置，当该值大于等于当前字符串的数据长度时，
		*  则返回整个字符串对象；返回值不包含该值指定位置的字符内容
		* @return {string} 返回值为一完整的对象，不需要单独释放，该函数的效率
		*  可能并不太高
		*/
		string left(int npos)
		{
			string ret(npos + 1);
			memcpy(ret.c_str(), m_pBuf, npos);
			ret.at(npos) = 0;
			return ret;
		}

		/**
		* 返回从当前字符串对象中缓冲区指定位置以右的内容
		* @param npos {size_t} 下标位置，当该值大于等于当前字符串的数据长度时，
		*  则返回的字符串对象内容为空；返回值不包含该值指定位置的字符内容
		* @return {const string} 返回值为一完整的对象，不需要单独释放，该
		*  函数的效率可能并不太高
		*/
		string right(int npos)
		{
			int len = length() - npos;
			len = len < 0 ? 0 : len;
			string ret(len + 1);
			memcpy(ret.c_str(), addr(npos), len);
			ret.at(len) = 0;
			return ret;
		}

		/**
		* 返回当前对象字符串的长度（不含\0）
		* @return {size_t} 返回值 >= 0
		*/
		int length() const
		{
			const char *p = m_pBuf;
			while (*p)
				++p;
			return p - m_pBuf;
		}

		/**
		* 带格式方式的添加数据（类似于 sprintf 接口方式）
		* @param fmt {const char*} 格式字符串
		* @param ... 变参数据
		* @return {string&} 当前对象的引用
		*/
		string& format(const char* fmt, ...)
		{
			va_list ap;
			va_start(ap, fmt);
			vsprintf(m_pBuf, fmt, ap);
			va_end(ap);
			return *this;
		}
	};

	/**
	* 纯虚函数：线程任务类，该类实例的 run 方法是在子线程中被执行的
	* 该类参考自ACL
	*/
	class MY_API thread_job
	{
	public:
		thread_job() {}
		virtual ~thread_job() {}

		/**
		* 纯虚函数，子类必须实现此函数，该函数在子线程中执行
		* @return {void*} 线程退出前返回的参数
		*/
		virtual void* run() = 0;
	};

	/**
	* 线程纯虚类，该类的接口定义类似于 Java 的接口定义，子类需要实现
	* 基类的纯虚函数，使用者通过调用 thread::start() 启动线程过程
	* 该类参考自ACL
	*/
	class MY_API thread : public thread_job
	{
	public:
		thread() : is_running(false), detachable(false) { }
		virtual ~thread() { }

		/**
		* 开始启动线程过程，一旦该函数被调用，则会立即启动一个新的
		* 子线程，在子线程中执行基类 thread_job::run 过程
		* @return {bool} 是否成功创建线程
		*/
		bool start() { is_running = true; _beginthread(thread_run, 0, this);  return is_running; }

		/**
		* 当创建线程时为非 detachable 状态，则必须调用此函数等待线程结束；
		* 若创建线程时为 detachable 状态时，禁止调用本函数
		* @param out {void**} 当该参数非空指针时，该参数用来存放
		*  线程退出前返回的参数
		* @return {bool} 是否成功
		*/
		void wait(void** out = NULL) { if (false == detachable) while(is_running) Sleep(1); }

		/**
		* 在调用 start 前调用此函数可以设置所创建线程是否为
		* 分离 (detachable) 状态；如果未调用此函数，则所创建
		* 的线程默认为分离状态
		* @param yes {bool} 是否为分离状态
		* @return {thread&}
		*/
		thread& set_detachable(bool yes) { detachable = yes; return *this; }

	private:
		bool is_running;
		bool detachable;
		void end() { is_running = false; }
		static void thread_run(void* arg) { thread *pThis = (thread*)arg; if (pThis){ pThis->run(); pThis->end(); } }
	};

#if USING_ACL_SQLITE

#define NAMES acl

#else

#define NAMES my

	/**
	* 32字节字符串
	*/
	typedef struct str32 
	{
		char data[32];
		str32(const char *pStr) { strcpy_s(data, pStr ? pStr : ""); };
		str32(const str32 &other) { memcpy(data, other.data, 32 * sizeof(char)); }
		const char* c_str() const { return data; }
	}str32;

	/**
	* 数据库查询结果集的行记录类型定义
	*/
	class MY_API db_row
	{
	public:
		/**
		* 构造函数
		* @param names {const std::vector<const char*>&} 数据库表中字段名列表
		*/
		db_row(const std::vector<str32>& n, const std::vector<str32>& v)
		{
			names.assign(n.begin(), n.end());
			values.assign(v.begin(), v.end());
		}
		~db_row() { }

		/**
		* 从查询结果的记录行中根据字段名取得相应的字段值，
		* 功能与 field_value 相同
		* @param name {const char*} 数据表的字段名
		* @return {const char*} 对应的字段值，为空则表示字段值不存在或
		*  字段名非法
		*/
		const char* operator[](const char* n) const
		{
			std::vector<str32>::const_iterator pos = names.begin();
			for (int i = 0; pos != names.end(); ++i, ++pos)
				if (0 == strcmp(pos->c_str(), n))
					return values.at(i).c_str();
			return "";
		}

	private:
		// 数据表的字段名集合
		std::vector<str32> names;

		// 数据结果行的字段集合
		std::vector<str32> values;
	};

	/**
	* 数据库查询结果的行记录集合类型定义
	*/
	class MY_API db_rows
	{
	public:
		db_rows() { }
		virtual ~db_rows()
		{ 
			for (std::vector<db_row*>::const_iterator p = rows.begin(); p != rows.end(); ++p)
				if (*p) delete *p;
		}

		/**
		* 取得所有的查询结果集
		* @return {const std::vector<db_row*>&} 返回行记录集合类型对象，
		*  可以通过调用 db_rows.empty() 来判断结果是否为空
		*/
		const std::vector<db_row*>& get_rows() const { return rows; }

		/**
		* 从查询的行记录集合中根据索引下标取得对应的某行记录
		* @param idx {size_t} 索引下标，该值应该 < 结果集大小
		* @return {const db_row*} 返回空表示输入下标值非法或字段值本身
		*  为空
		*/
		const db_row* operator[](size_t idx) const { return rows.at(idx); }

		/**
		* 判断结果集是否为空
		* @return {bool} 是否为空
		*/
		bool empty() const { return rows.empty(); }

		/**
		* 结果集的行记录个数
		* @return {size_t} 行记录个数
		*/
		size_t length() const { return rows.size(); }

		/**
		* 向结果集中push_back一个结果
		*/
		void push(db_row* row) { rows.push_back(row); }

	protected:
		// 查询结果行集合，其中的元素 db_row 必须是动态添加进去的，
		// 因为在本类对象析构时会自动 delete rows 中的所有元素对象
		std::vector<db_row*> rows;
	};

	/**
	* @class db_handle
	* @brief 该类替换acl::db_handle
	*/
	class db_handle
	{
	protected:
		CppSQLite3DB *m_pDatabase;

	public:
		db_handle() : m_pDatabase(NULL) { }
		virtual ~db_handle() { if (m_pDatabase) { m_pDatabase->close(); delete m_pDatabase; } }

		/**
		* 当采用动态加载方式加载动态库时，可以使用此函数设置动态库的加载全路径
		*/
		static void set_loadpath(const char* path) { printf("Load \"%s\". \n", path); }

		/**
		* 基类 connect_client 虚函数的实现
		* @return {bool} 打开数据库连接是否成功
		*/
		virtual bool open() = 0;

		/**
		* 纯虚接口，子类必须实现此接口用于执行 SELECT SQL 语句
		* @param sql {const char*} 标准的 SQL 语句，非空，并且一定得要注意该
		*  SQL 语句必须经过转义处理，以防止 SQL 注入攻击
		* @return {bool} 执行是否成功
		*/
		virtual bool sql_select(const char* sql) = 0;

		/**
		* @brief 判断表格中是否存在字段
		*/
		virtual bool IsColumnExist(const char *tbl_name, const char *col_name) { return false; }

		/**
		* 获得执行 SQL 语句后的结果
		* @return {const db_rows*}，返回结果若非空，则用完后需要调用
		*  free_result() 以释放结果对象
		*/
		virtual const db_rows* get_result() const = 0;

		/**
		* 释放上次查询的结果，当查询完成后，调用该函数来释放上次查询的结果，该函数被
		* 多次调用并无害处，因为当第一次调用时会自动将内部变量 result_ 置空,
		* 另外，要求子类必须在每次执行 SQL 查询前先调用此方法，以免用户忘记
		* 调用而造成内存泄露；此外，本类对象在析构时会自动再调用本方法释放可能
		* 未释放的内存
		*/
		virtual void free_result() = 0;

		/**
		* 纯虚接口，子类必须实现此接口用于执行 INSERT/UPDATE/DELETE SQL 语句
		* @param sql {const char*} 标准的 SQL 语句，非空，并且一定得要注意该
		*  SQL 语句必须经过转义处理，以防止 SQL 注入攻击
		* @return {bool} 执行是否成功
		*/
		virtual bool sql_update(const char* sql) = 0;

		// 开启事务
		inline void begin() { if (m_pDatabase) m_pDatabase->beginTransaction(); }

		// 提交事务
		inline void commit() { if (m_pDatabase) m_pDatabase->commitTransaction(); }

		// 关闭同步写
		inline void sync_off() { if (m_pDatabase) m_pDatabase->sync_off(); }

		// 获取sqlite3句柄
		sqlite3* getHandle() { return m_pDatabase ? m_pDatabase->getHandle() : NULL; }
	};

	class db_sqlite : public db_handle
	{
	private:
		string m_sDbFile;
		string m_sCharset;
		db_rows *m_rows;

	public:
		db_sqlite(const char* dbfile, const char* charset = "utf-8") 
			: m_sDbFile(dbfile), m_sCharset(charset), m_rows(NULL) { }
		~db_sqlite(void) { if(m_rows) delete m_rows; }

		/**
		* 基类 connect_client 虚函数的实现
		* @return {bool} 打开数据库连接是否成功
		*/
		virtual bool open()
		{
			m_pDatabase	= m_pDatabase ? m_pDatabase : new CppSQLite3DB;

			return CreateDataBase(m_sDbFile.c_str(), m_pDatabase);
		}

		/**
		* 纯虚接口，子类必须实现此接口用于执行 SELECT SQL 语句
		* @param sql {const char*} 标准的 SQL 语句，非空，并且一定得要注意该
		*  SQL 语句必须经过转义处理，以防止 SQL 注入攻击
		* @return {bool} 执行是否成功
		*/
		virtual bool sql_select(const char* sql)
		{
			if (NULL == m_pDatabase)
				return false;
			// 查询结果
			CppSQLite3Query	querySQLite3 = m_pDatabase->execQuery(sql);
			if(m_rows) delete m_rows;
			m_rows = new db_rows;
			while ( !querySQLite3.eof() )
			{
				int cols = querySQLite3.numFields();
				std::vector<str32> names, values;
				for (int nIdx = 0; nIdx < cols; ++nIdx)
				{
					const char *n = querySQLite3.fieldName(nIdx);
					const char *v = querySQLite3.getStringField(nIdx);
					if (!n && !v) break;
					names.push_back(str32(n));
					values.push_back(str32(v));
				}
				m_rows->push(new db_row(names, values));
				querySQLite3.nextRow();
			}
			querySQLite3.finalize();
			return true;
		}

		/**
		* @brief 判断表格中是否存在字段
		*/
		virtual bool IsColumnExist(const char *tbl_name, const char *col_name)
		{
			bool ret = false;
			if (m_pDatabase)
			{
				string sql;
				sql.format("SELECT sql from sqlite_master WHERE tbl_name=\"%s\" and type=\"table\";", tbl_name);
				CppSQLite3Query	querySQLite3 = m_pDatabase->execQuery(sql.c_str());
				if (!querySQLite3.eof())
				{
					const char* sess = querySQLite3.getStringField(0);
					ret = strstr(sess, col_name);
				}
				querySQLite3.finalize();
			}
			return ret;
		}

		/**
		* 获得执行 SQL 语句后的结果
		* @return {const db_rows*}，返回结果若非空，则用完后需要调用
		*  free_result() 以释放结果对象
		*/
		virtual const db_rows* get_result() const
		{
			return m_rows ? (m_rows->empty() ? NULL : m_rows) : NULL;
		}

		/**
		* 释放上次查询的结果，当查询完成后，调用该函数来释放上次查询的结果，该函数被
		* 多次调用并无害处，因为当第一次调用时会自动将内部变量 result_ 置空,
		* 另外，要求子类必须在每次执行 SQL 查询前先调用此方法，以免用户忘记
		* 调用而造成内存泄露；此外，本类对象在析构时会自动再调用本方法释放可能
		* 未释放的内存
		*/
		virtual void free_result()
		{
			if(m_rows)
			{
				delete m_rows;
				m_rows = NULL;
			}
		}

		/**
		* 纯虚接口，子类必须实现此接口用于执行 INSERT/UPDATE/DELETE SQL 语句
		* @param sql {const char*} 标准的 SQL 语句，非空，并且一定得要注意该
		*  SQL 语句必须经过转义处理，以防止 SQL 注入攻击
		* @return {bool} 执行是否成功
		*/
		virtual bool sql_update(const char* sql)
		{
			if (NULL == m_pDatabase)
				return false;
			const char *pUtf8 = NULL;
			try
			{
				pUtf8 = MByteToUtf8(sql);
				int ret = m_pDatabase->execDML(pUtf8);
				if (pUtf8) delete [] pUtf8;
				return ret > 0;
			}
			catch(CppSQLite3Exception ex)
			{
				TRACE("CppSQLite3Exception: %s \n", ex.errorMessage());
				if (pUtf8) delete [] pUtf8;
				return false;
			}
		}
	};
#endif
}

#endif
