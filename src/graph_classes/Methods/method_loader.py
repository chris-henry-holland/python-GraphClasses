#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict, 
    List,
    Tuple,
    Callable,
    Union,
)

if TYPE_CHECKING:
    from graph_classes.generic_graph_types import (
        GenericGraphTemplate,
    )

import importlib
import inspect



#from Graph_classes import\
#        GenericGraphTemplate

def loadMethodsMultipleModules(
    methodname_dicts: Dict[str, Dict[GenericGraphTemplate, List[Union[Tuple[str], str]]]]
) -> Dict[GenericGraphTemplate, Dict[str, Callable]]:
    method_dicts = {}
    subpackage_name =\
            inspect.currentframe().f_back.f_globals["__name__"]
    for module_name, methodname_dict in methodname_dicts.items():
        method_import_dict = loadMethodsSingleModule(module_name,\
                methodname_dict, subpackage_name=subpackage_name)
        for cls, method_dict in method_import_dict.items():
            method_dicts.setdefault(cls, {})
            method_dicts[cls].update(method_dict)
    return method_dicts

def loadMethodsSingleModule(
    module_name: str,
    methodname_dict: Dict[GenericGraphTemplate, List[Union[Tuple[str], str]]],
    subpackage_name: str
) -> Dict[GenericGraphTemplate, Dict[str, Callable]]:
    module = importlib.import_module(
        f".{module_name}", package=subpackage_name
    )
    
    method_dict = {}
    for cls, methodname_lst in methodname_dict.items():
        method_dict.setdefault(cls, {})
        for methodname in methodname_lst:
            if not isinstance(methodname, str):
                method_dict[cls][methodname[0]] =\
                        getattr(module, methodname[1])
            else:
                method_dict[cls][methodname] =\
                        getattr(module, methodname)
    return method_dict
    
    """
    if global_dict_names is None:
        global_dict_names = ["_submethod_times_indicts"]
    if methodname_dict is None:
        methodname_dict = {}
    if convmethodname_dict is None:
        convmethodname_dict = {}

    import_dict = {"global_dicts": {x: {} for x in global_dict_names}}

    subpackage_name = inspect.currentframe().f_back.f_globals["__name__"]

    for (in_dict, import_dict_key) in (
        (methodname_dict, "method"),
        (convmethodname_dict, "conv_method"),
    ):
        import_dict[import_dict_key] = {}
        for module_name, meth_list in in_dict.items():
            module = importlib.import_module(
                ".{}".format(module_name), package=subpackage_name
            )
            for meth_name in meth_list:
                import_dict[import_dict_key][meth_name] = getattr(
                    module, meth_name
                )
            for dict_name in global_dict_names:
                global_dict = getattr(module, dict_name, None)
                if isinstance(global_dict, dict):
                    import_dict["global_dicts"][dict_name].update(global_dict)
    return import_dict
    """
