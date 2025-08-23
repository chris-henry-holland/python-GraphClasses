#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
)

def toString(x, not_str: bool=False) -> str:
    qstr = '"' if not not_str and isinstance(x, str) else ''
    return f"{qstr}{x}{qstr}"

def argsKwargsStrings(args: Optional[Tuple[Any]], kwargs: Optional[Dict[str, Any]]) -> List[str]:
    res = []
    if args:
        res.append(", ".join([toString(x) for x in args]))
    if kwargs:
        res.append(", ".join([f"{toString(x, not_str=True)}={toString(y)}" for x, y in kwargs.items()]))
    return res

def methodStrings(method_name: str,\
        method_args: Optional[Tuple[Any]]=None,\
        method_kwargs: Optional[Dict[str, Any]]=None) -> str:
    args_kwargs_strs = argsKwargsStrings(method_args, method_kwargs)
    method_str_short = method_name
    return method_str_short,\
            f"{method_str_short}({', '.join(args_kwargs_strs)})"