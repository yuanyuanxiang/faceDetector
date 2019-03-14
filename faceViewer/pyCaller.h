#include "python.h"
#include <map>
#include <string>
#include "StrTransfer.h"
#include "../config.h"

#pragma once

// 最多可检测目标个数
#define DIM 512

// Py_DECREF 导致程序二次运行时崩溃
#define My_DECREF(p) 

// 包含 numpy 中的头文件arrayobject.h
#include "..\Lib\site-packages\numpy\core\include\numpy\arrayobject.h"

class tfOutput
{
public:
	float boxes[DIM];			// 人脸特征向量

	inline void zeros() { memset(boxes, 0, sizeof(boxes)); }

	/**
	* @brief 构造一个n个人脸的模型参数
	* @note 默认构造的为空参数
	*/
	tfOutput() { zeros(); }
	~tfOutput() { }
};

/************************************************************************
* @class pyCaller
* @brief python调用类: 适用于对图片、视频进行识别
* @author 袁沅祥, 2018-4-11
************************************************************************/
class pyCaller
{
private:
	static wchar_t pyHome[_MAX_PATH];			// python路径
	PyObject* pModule;							// python模块
	std::map<std::string, PyObject*> pFunMap;	// 函数列表
	bool bMultiThread;							// 多线程

	// 初始化 numpy 执行环境，主要是导入包
	// python2.7用void返回类型，python3.0以上用int返回类型
	inline int init_numpy()
	{
		import_array();
		return 1;
	}

public:
	// 设置python安装目录
	static bool SetPythonHome(const char *py)
	{
		char pyExe[_MAX_PATH] = { 0 };
		strcat_s(pyExe, py);
		strcat_s(pyExe, "\\python.exe");
		if (-1 != _access(pyExe, 0))
		{
			size_t s;
			mbstowcs_s(&s, pyHome, py, strlen(py));
			OUTPUT("======> SetPythonHome: %s\n", py);
			return true;
		}
		else
		{
			OUTPUT("======> SetPythonHome: \"%s\" don't include python.exe.\n", py);
			return false;
		}
	}

	/**
	* @brief 构造一个pyCaller对象
	*/
	pyCaller(const char *module, bool multi_thread = true)
	{
		pModule = NULL;
		bMultiThread = multi_thread;
		Init(module);
	}

	/**
	* @brief 初始化pyCaller对象，接收py脚本名称作为传入参数
	*/
	bool Init(const char * module_name)
	{
		// 获取路径信息
		char path[_MAX_PATH], *p = path, home[_MAX_PATH];
		GetModuleFileNameA(NULL, path, _MAX_PATH);
		while('\0' != *p) ++p;
		while('\\' != *p) --p;
		strcpy(p + 1, "settings.ini");
		if (-1 == _access(path, 0))
			OUTPUT("程序配置文件不存在!\n");
		GetPrivateProfileStringA("settings", "python_home", "", home, _MAX_PATH, path);
		if(false == pyCaller::SetPythonHome(home))
			OUTPUT("python_home配置错误!\n");

		if (pyHome[0] && NULL == pModule)
		{
			clock_t t = clock();
			Py_SetPythonHome(pyHome);
			Py_Initialize();
			if (0 == Py_IsInitialized())
			{
				OUTPUT("Py_IsInitialized = 0.\n");
				return false;
			}
			if (bMultiThread)
				PyEval_InitThreads(); // 多线程支持
			if (NUMPY_IMPORT_ARRAY_RETVAL == init_numpy())
			{
				OUTPUT("init_numpy failed.\n");
				return false;
			}
			PyObject *py = PyImport_ImportModule(module_name);
			if (bMultiThread && PyEval_ThreadsInitialized())
				PyEval_SaveThread();
			if (NULL == py)
				OUTPUT("PyImport_ImportModule failed.\n");
			t = clock() - t;
			char szOut[128];
			sprintf_s(szOut, "PyImport_ImportModule using %d ms.\n", t);
			OutputDebugStringA(szOut);
#ifndef _AFX
			printf(szOut);
#endif
			pModule = py;
		}
		else
		{
			OUTPUT("Py_SetPythonHome is not called.\n");
		}
		return pModule;
	}

	/**
	* @brief 反初始化python环境，释放python对象的内存
	*/
	~pyCaller()
	{
		if (pModule)
			Py_DECREF(pModule);
		for (std::map<std::string, PyObject*>::iterator p = pFunMap.begin(); 
			p != pFunMap.end(); ++p)
			if (p->second) Py_DECREF(p->second);
		if (pyHome[0])
		{
			if (bMultiThread && !PyGILState_Check())
				PyGILState_Ensure();
			Py_Finalize();
		}
	}

	// 是否加载了指定模块
	bool IsModuleLoaded() const { return pModule; }

	/**
	* @brief 使用前激活指定名称的函数
	*/
	bool ActivateFunc(const char * func_name)
	{
		bool bFind = false;
		std::string fun(func_name);
		for (std::map<std::string, PyObject*>::iterator p = pFunMap.begin(); 
			p != pFunMap.end(); ++p)
		{
			if (p->first == fun)
			{
				bFind = true;
				break;
			}
		}
		if (bFind)
			return true;
		PyObject *pFunc = NULL;

		PyGILState_STATE gstate;
		int nHold = bMultiThread ? PyGILState_Check() : TRUE;
		if (!nHold) gstate = PyGILState_Ensure();
		Py_BEGIN_ALLOW_THREADS;
		Py_BLOCK_THREADS;

		pFunc =  pModule ? PyObject_GetAttrString(pModule, func_name) : 0;

		Py_UNBLOCK_THREADS;
		Py_END_ALLOW_THREADS;
		if (!nHold) PyGILState_Release(gstate);

		pFunMap.insert(std::make_pair(func_name, pFunc));

		return pFunc;
	}

	/**
	* @brief 调用python脚本中的指定函数
	*/
	tfOutput CallFunction(const char * func_name, const char *arg)
	{
		tfOutput out;
		PyObject *pFunc = pFunMap[func_name];
		if (pFunc)
		{
			const char *utf8 = MByteToUtf8(arg);
			PyObject* pArg = Py_BuildValue("(s)", utf8);
			delete [] utf8;
			if (NULL == pArg)
				return out;
			PyObject* pRetVal = PyEval_CallObject(pFunc, pArg);
			if (NULL == pRetVal)
				return out;
			PyArrayObject *pMatrix = (PyArrayObject *) pRetVal;

			int x1 = pMatrix->dimensions[0], x2 = pMatrix->dimensions[1];
			if (x1 * x2 <= DIM)
				memcpy(out.boxes, pMatrix->data, x1 * x2 * sizeof(float));
			My_DECREF(pMatrix);
		}
		return out;
	}
	/**
	* @brief 调用python脚本中的指定函数
	*/
	tfOutput CallFunction(const char * func_name, PyObject *arg)
	{
		tfOutput out;
#if STATIC_DETECTING
		static PyObject *pFunc = pFunMap[func_name];
#else 
		PyObject *pFunc = pFunMap[func_name];
#endif
		if (pFunc)
		{
			PyGILState_STATE gstate;
			int nHold = bMultiThread ? PyGILState_Check() : TRUE;
			if (!nHold)gstate = PyGILState_Ensure();
			Py_BEGIN_ALLOW_THREADS;
			Py_BLOCK_THREADS;

 			PyObject* pRetVal = PyEval_CallObject(pFunc, arg);
			if (pRetVal)
			{
				PyArrayObject *pMatrix = (PyArrayObject *) pRetVal;

				int x1 = pMatrix->dimensions[0], x2 = pMatrix->dimensions[1];
				if (x1 * x2 <= DIM)
					memcpy(out.boxes, pMatrix->data, x1 * x2 * sizeof(float));
				My_DECREF(pMatrix);
			}

			Py_UNBLOCK_THREADS;
			Py_END_ALLOW_THREADS;
			if (!nHold)PyGILState_Release(gstate);
		}
		return out;
	}
};
