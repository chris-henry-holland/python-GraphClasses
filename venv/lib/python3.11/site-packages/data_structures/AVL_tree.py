#!/usr/bin/env python

from typing import (
    Any,
    Callable,
    Optional,
    Tuple, 
    Union,
)

class AVLTreeNode:
    # Based on https://www.geeksforgeeks.org/introduction-to-avl-tree/?ref=lbp
    def __init__(
        self,
        val: Any,
        comp_func: Callable[[Any, Any], int],
    ):
        self.val = val
        self.comp_func = comp_func
        self.left = None
        self.right = None
        self.freq = 1
        self.depth = 1
    
    def __str__(self):
        return str({"val": self.val, "left": str(self.left),\
                "right": str(self.right)})
    
    def getDepth(self) -> int:
        res = 0 if self.left is None else self.left.depth
        res2 = 0 if self.right is None else self.right.depth
        return max(res, res2) + 1
    
    def getBalance(self) -> int:
        d1 = 0 if self.left is None else self.left.depth
        d2 = 0 if self.right is None else self.right.depth
        return d1 - d2
    
    def leftRotate(self) -> "AVLTreeNode":
        if self.right is None:
            return self
        node2 = self.right
        self.right = node2.left
        node2.left = self
        self.depth = self.getDepth()
        node2.depth = node2.getDepth()
        return node2
    
    def rightRotate(self) -> "AVLTreeNode":
        if self.left is None:
            return self
        node2 = self.left
        self.left = node2.right
        node2.right = self
        self.depth = self.getDepth()
        node2.depth = node2.getDepth()
        return node2
    
    def leftRightRotate(self) -> "AVLTreeNode":
        if self.left is None:
            return self
        self.left = self.left.leftRotate()
        return self.rightRotate()
    
    def rightLeftRotate(self) -> "AVLTreeNode":
        if self.right is None:
            return self
        self.right = self.right.rightRotate()
        return self.leftRotate()
    
    def insert(
        self,
        node: "AVLTreeNode",
    ) -> Tuple[Union["AVLTreeNode", bool]]:
        # Based on https://www.geeksforgeeks.org/insertion-in-an-avl-tree/
        v = self.comp_func(self.val, node.val)
        if not v:
            self.freq += node.freq
            return self, False
        if v < 0:
            if self.left is None:
                self.left = node
            else:
                self.left, b = self.left.insert(node)
                if not b: return self, False
            if self.getBalance() <= 1:
                self.depth = self.getDepth()
                return self, True
            if self.comp_func(self.left.val, node.val) < 0:
                return self.rightRotate(), True
            return self.leftRightRotate(), True
        if self.right is None:
            self.right = node
        else:
            self.right, b = self.right.insert(node)
            if not b: return self
        if self.getBalance() >= -1:
            self.depth = self.getDepth()
            return self, True
        if self.comp_func(self.right.val, node.val) > 0:
            return self.leftRotate(), True
        return self.rightLeftRotate(), True
    
    def delete(
        self,
        val: Any,
    ) -> Tuple[Union["AVLTreeNode", bool]]:
        # Based on https://www.geeksforgeeks.org/deletion-in-an-avl-tree/
        v = self.comp_func(self.val, val)
        if not v:
            self.freq -= 1
            if self.freq:
                return self, False
            if self.left is None:
                return self.right, True
            elif self.right is None:
                return self.left, True
            
            d1 = self.left.depth
            d2 = self.right.depth
            if d1 >= d2:
                node = self.left
                while node.right is not None:
                    node = node.right
                self.val = node.val
                self.freq = node.freq
                node.freq = 1
                node.left = self.left.delete(node.val)[0]
            else:
                node = self.right
                while node.left is not None:
                    node = node.left
                self.val = node.val
                self.freq = node.freq
                node.freq = 1
                node.right = self.right.delete(node.val)[0]
        elif v < 0:
            if self.left is None:
                return self, False
            self.left, b = self.left.delete(val)
            if not b: return self
            if self.getBalance() >= -1:
                self.depth = self.getDepth()
                return self, True
            if self.right.getBalance() <= 0:
                return self.leftRotate(), True
            return self.rightLeftRotate(), True
        if self.right is None:
            return self, False
        self.right, b = self.right.delete(val)
        if not b: return self
        if self.getBalance() <= 1:
            self.depth = self.getDepth()
            return self, True
        if self.left.getBalance() >= 0:
            return self.rightRotate(), True
        return self.leftRightRotate(), True
    
    def search(self, val: Any) -> bool:
        v = self.comp_func(self.val, val)
        if not v: return True
        if v < 0:
            return self.left is not None and self.left.search(val)
        return self.right is not None and self.right.search(val)

def defaultComparisonFunction(val1: Any, val2: Any) -> int:
    if val1 > val2: return -1
    elif val1 < val2: return 1
    return 0

class AVLTree:
    def __init__(
        self,
        comp_func: Optional[Callable[[Any, Any], int]]=None,
    ):
        self.comp_func = defaultComparisonFunction if comp_func is None\
                else comp_func
        self.root = None
    
    def __str__(self):
        return f"AVL tree {self.root}"
    
    def insert(self, val) -> None:
        node = AVLTreeNode(val, comp_func=self.comp_func)
        if self.root is None:
            self.root = node
            return
        self.root = self.root.insert(node)[0]
        return
    
    def delete(self, val) -> None:
        if self.root is None: return
        self.root = self.root.delete(val)[0]
    
    def search(self, val) -> bool:
        return self.root is not None and self.root.search(val)
