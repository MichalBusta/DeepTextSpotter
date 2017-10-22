
#include "trie_api.h"

static PyObject *TrieError;

static PyObject* decode_sofmax_cfun (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyObject *out=NULL;
	PyArrayObject *arr1=NULL;
	PyArrayObject* img = NULL;
	npy_intp* img_dims = NULL;

	const char * imageName;
	const char * outputDir = NULL;
	int instance = 0;
	int minHeight = 0;
	if (!PyArg_ParseTuple(args, "O",  &arg1))
		return NULL;

	img = (PyArrayObject *) arg1;
	img_dims = PyArray_DIMS(img);
	int numOfDim = PyArray_NDIM(img);

	out =  decode_sofmax(img, numOfDim, img_dims);

	return out;
}

static PyObject* load_dict_cfun (PyObject *dummy, PyObject *args)
{
	const char * fileName;
	if (!PyArg_ParseTuple(args, "s", &fileName))
		return NULL;

	load_dict(fileName);

	return Py_BuildValue("");
}

static PyObject* is_dict_cfun (PyObject *dummy, PyObject *args)
{
	const char * word;
	if (!PyArg_ParseTuple(args, "s", &word))
		return NULL;

	int isd = is_dict_word(word);

	return Py_BuildValue("i", isd);
}

static PyObject* assign_lines_cfun (PyObject *dummy, PyObject *args)
{

	PyArrayObject *arr1=NULL;
	PyArrayObject *arr2=NULL;
	PyArrayObject* img = NULL;
	npy_intp* img_dims = NULL;

	if (!PyArg_ParseTuple(args, "OO", &arr1, &arr2))
		return NULL;

	img = (PyArrayObject *) arr1;
	img_dims = PyArray_DIMS(img);
	int numOfDims = PyArray_NDIM(img);

	PyArrayObject * det = (PyArrayObject *) arr2;
	npy_intp* det_dims = PyArray_DIMS(det);;


	return (PyObject *) assign_lines(img, numOfDims, img_dims, det, det_dims);
}

static PyMethodDef TrieMethods[] = {

		{"decode_sofmax",  decode_sofmax_cfun, METH_VARARGS, "Decode softmax"},
		{"load_dict",  load_dict_cfun, METH_VARARGS, "Loads dictionary"},
		{"is_dict",  is_dict_cfun, METH_VARARGS, "returns true if input is dictionary entry"},
		{"assign_lines",  assign_lines_cfun, METH_VARARGS, "returns the text displays"},
		{NULL, NULL, 0, NULL}        /* Sentinel */
};


#ifdef PYTHON3

static struct PyModuleDef spammodule = {
   PyModuleDef_HEAD_INIT,
   "cmp_trie",   /* name of module */
   NULL,         /* module documentation, may be NULL */
   -1,           /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	 TrieMethods
};

PyMODINIT_FUNC
PyInit_cmp_trie(void)
{
    return PyModule_Create(&spammodule);
}

#else

PyMODINIT_FUNC
initcmp_trie(void)
{
	PyObject *m;

	m = Py_InitModule("cmp_trie", TrieMethods);
	import_array();
	if (m == NULL)
		return;

	TrieError = PyErr_NewException((char*) "trie.error", NULL, NULL);
	Py_INCREF(TrieError);
	PyModule_AddObject(m, "error", TrieError);
}

#endif
