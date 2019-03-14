#include "pyCaller.h"

wchar_t pyCaller::pyHome[] = { 0 };

tfOutput pyCaller::ParseResult(PyObject *pRetVal, tfOutput *tf)
{
	tfOutput out;
	out = tf ? *tf : out;
	PyArrayObject *pMatrix = (PyArrayObject *) pRetVal;

	int x1 = pMatrix->dimensions[0], x2 = pMatrix->dimensions[1];
	if (x1 * x2 <= out.n * 5)
		memcpy(out.boxes, pMatrix->data, x1 * x2 * sizeof(float));
	My_DECREF(pMatrix);

	return out;
}
