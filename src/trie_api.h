#ifndef TRIE_API_H_
#define TRIE_API_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

void load_dict(const char * fileName);

PyObject* decode_sofmax(PyArrayObject* data, int numOfDims,  npy_intp* img_dims);

int is_dict_word(const char* word);

PyArrayObject* assign_lines(PyArrayObject* img, int numOfDims,  npy_intp* img_dims, PyArrayObject* det, npy_intp* det_dims);

#ifdef __cplusplus
}
#endif

#endif /* TRIE_API_H_ */
