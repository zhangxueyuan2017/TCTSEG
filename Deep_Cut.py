# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_Deep_Cut')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_Deep_Cut')
    _Deep_Cut = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_Deep_Cut', [dirname(__file__)])
        except ImportError:
            import _Deep_Cut
            return _Deep_Cut
        try:
            _mod = imp.load_module('_Deep_Cut', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _Deep_Cut = swig_import_helper()
    del swig_import_helper
else:
    import _Deep_Cut
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

TT = _Deep_Cut.TT
MAX_CORNERS = _Deep_Cut.MAX_CORNERS
class ContourInformation(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ContourInformation, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ContourInformation, name)
    __repr__ = _swig_repr
    __swig_setmethods__["LeftTop"] = _Deep_Cut.ContourInformation_LeftTop_set
    __swig_getmethods__["LeftTop"] = _Deep_Cut.ContourInformation_LeftTop_get
    if _newclass:
        LeftTop = _swig_property(_Deep_Cut.ContourInformation_LeftTop_get, _Deep_Cut.ContourInformation_LeftTop_set)
    __swig_setmethods__["RightDown"] = _Deep_Cut.ContourInformation_RightDown_set
    __swig_getmethods__["RightDown"] = _Deep_Cut.ContourInformation_RightDown_get
    if _newclass:
        RightDown = _swig_property(_Deep_Cut.ContourInformation_RightDown_get, _Deep_Cut.ContourInformation_RightDown_set)
    __swig_setmethods__["PixelNumber"] = _Deep_Cut.ContourInformation_PixelNumber_set
    __swig_getmethods__["PixelNumber"] = _Deep_Cut.ContourInformation_PixelNumber_get
    if _newclass:
        PixelNumber = _swig_property(_Deep_Cut.ContourInformation_PixelNumber_get, _Deep_Cut.ContourInformation_PixelNumber_set)

    def __init__(self):
        this = _Deep_Cut.new_ContourInformation()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _Deep_Cut.delete_ContourInformation
    __del__ = lambda self: None
ContourInformation_swigregister = _Deep_Cut.ContourInformation_swigregister
ContourInformation_swigregister(ContourInformation)


def testWaterSeg(path, save_dir):
    return _Deep_Cut.testWaterSeg(path, save_dir)
testWaterSeg = _Deep_Cut.testWaterSeg

def WatershedSegment(path, image, save_dir):
    return _Deep_Cut.WatershedSegment(path, image, save_dir)
WatershedSegment = _Deep_Cut.WatershedSegment

def Morphology(image):
    return _Deep_Cut.Morphology(image)
Morphology = _Deep_Cut.Morphology

def cropPoly(image, points, size, pathName):
    return _Deep_Cut.cropPoly(image, points, size, pathName)
cropPoly = _Deep_Cut.cropPoly

def splitImage(image, dstimage):
    return _Deep_Cut.splitImage(image, dstimage)
splitImage = _Deep_Cut.splitImage

def xmult(a, b, c):
    return _Deep_Cut.xmult(a, b, c)
xmult = _Deep_Cut.xmult

def Area1(p, nv):
    return _Deep_Cut.Area1(p, nv)
Area1 = _Deep_Cut.Area1
# This file is compatible with both classic and new-style classes.

