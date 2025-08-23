#!/usr/bin/env python

from typing import (
    Dict,
    Tuple,
    Optional,
    Union,
    Any,
    Callable,
)

import unittest

from abc import ABC, abstractmethod

from unittest_templates.utils import methodStrings

class TestMethodTemplate(unittest.TestCase, ABC):
    # Need to define class attributes method_name, res_type_alias_lst
    
    @abstractmethod
    def resultString(self, res: Any, method_args: Tuple[Any],\
            method_kwargs: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def objectDescriptionFunction(self, obj: Any) -> str:
        pass
        
    
    def _methodResultTest(self, obj: Any, test_func: Callable[[Any],\
            Tuple[Union[bool, str]]],\
            method_args: Optional[Tuple[Any]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[Any]=None,\
            full_chk_if_known: bool=True)\
            -> Any:
        
        method_str_short, method_str_long =\
                methodStrings(self.method_name, method_args,\
                method_kwargs)
        if method_args is None: method_args = ()
        if method_kwargs is None: method_kwargs = {}
        equality_func = getattr(self, "resultEqualityFunction", None)
        res = getattr(obj, self.method_name)(*method_args,\
                **method_kwargs)
        #print(obj.fullAdj())
        #print()
        #print(self.resultString(res))
        #print(res)
        
        msg_stem_func = lambda: f"Call of the method "\
                f"{method_str_long} for the "\
                f"{self.objectDescriptionFunction(obj)}gave "\
                f"{self.resultString(res, method_args, method_kwargs)}"
        
        if known_result is not None:
            if equality_func is None:
                eq_test = self.assertEqual
                eq_test_args = (res, known_result)
            else:
                eq_test = self.assertTrue
                eq_test_args = (equality_func(res, known_result),)
            with self.subTest(
                msg=f"Call of method {method_str_short}() did not "
                        "give the known correct result"
            ):
                eq_test(
                    *eq_test_args,
                    msg=f"{msg_stem_func()}which did not match the "
                            "expected (correct) result of "
                            f"{self.resultString(known_result, method_args, method_kwargs)}"
                )
            if not full_chk_if_known:
                return res
        
        test_pass, test_str = test_func(res)
        with self.subTest(
            msg=f"Call of method {method_str_short}() gave an "
                    "incorrect result"
        ):
            self.assertTrue(
                test_pass,
                msg=f"{msg_stem_func()}{test_str}",
            )
        return res
    
    @abstractmethod
    def methodResultTest(self, obj: Any, test_args: Tuple[Any],\
            test_kwargs: Dict[str, Any],\
            known_result: Optional[Any]=None)\
            -> bool:
        pass
    
    def knownGoodResultTestTemplate(self) -> None:
        method_name = "knownGoodResults"
        for unittest_cls in type(self).mro():
            #getattr(self, "known_good_results", {}).items():
            if not method_name in unittest_cls.__dict__.keys():
                continue
            for cls, obj_dict_lst in\
                    getattr(unittest_cls, method_name)().items():
                for obj_dict in obj_dict_lst:
                    obj = obj_dict["obj_func"](cls)
                    for opt_dict in obj_dict.get("opts", {}):
                        self.methodResultTest(obj,\
                                method_args=opt_dict.get("args", None),\
                                method_kwargs=opt_dict.get("kwargs", None),\
                                known_result=opt_dict.get("result", None))
        return
    
    def knownErrorTestTemplate(self) -> None:
        method_name = "knownError"
        for unittest_cls in type(self).mro():
            #getattr(self, "known_good_results", {}).items():
            #for cls, obj_dict_lst in getattr(self, "known_err", {}).items():
            if not method_name in unittest_cls.__dict__.keys():
                continue
            for cls, obj_dict_lst in\
                    getattr(unittest_cls, method_name)().items():
                for obj_dict in obj_dict_lst:
                    obj = obj_dict["obj_func"](cls)
                    for opt_dict in obj_dict.get("opts", {}):
                        method_str_short, method_str_long =\
                                methodStrings(self.method_name,\
                                opt_dict.get("args", None), opt_dict.get("kwargs", None))
                        
                        with self.subTest(
                            msg=f"Call of method {method_str_short}() "
                                    "did not result in the expected error"
                        ):
                            with self.assertRaises(
                                opt_dict["err"],
                                msg="Call of the method {method_str_long} for "\
                                        f"the {self.objectDescriptionFunction(obj)}"\
                                        "should have given a "\
                                        f"{opt_dict['err'].__name__} "
                                        "exception. This did not occur.",
                            ):
                                getattr(obj, self.method_name)(*opt_dict.get("args", ()), opt_dict.get("kwargs", {}))
        return
