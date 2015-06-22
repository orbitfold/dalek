#include "dalek_likelihood.hpp"
#include <python2.7/Python.h>


double cpp_loglikelihood (double theta[], int& nDims, double phi[], int& nDerived)
{
  double result_double;
  PyObject* tardisModuleString = PyString_FromString((char*)"dalek.dalek_helper");
  PyObject* tardisModule = PyImport_Import(tardisModuleString);
  Py_DECREF(tardisModuleString);
  PyObject* result;
  if (tardisModule != NULL) 
    {
      PyObject* runTardis = PyObject_GetAttrString(tardisModule, (char*)"get_fitness");
      if (runTardis && PyCallable_Check(runTardis))
	{
	  PyObject* args = PyTuple_New(13);
	  for (int i = 0; i < 13; i++)
	    {
	      PyObject* arg = PyFloat_FromDouble(theta[i]);
	      PyTuple_SetItem(args, i, arg);
	    }
	  result = PyObject_CallObject(runTardis, args);
	  Py_DECREF(args);
	}
      else
	{
	  if (PyErr_Occurred())
	    {
	      PyErr_Print();
	    }
	  fprintf(stderr, "Cannot find get_fitness function!\n");
	}
      Py_XDECREF(runTardis);
      Py_DECREF(tardisModule);
    }
  else
    {
      fprintf(stderr, "DALEK helper module not found!\n");
    }
  result_double = PyFloat_AsDouble(result);
  //Py_Finalize();
  return result_double;
}


void cpp_loglikelihood_setup ()
{
  // Apparently numpy does not work when doing Py_Initialize/Py_Finalize more than once.
  // So we will do this here and manually make sure there are no leaks.
  Py_Initialize();
  return;
}
