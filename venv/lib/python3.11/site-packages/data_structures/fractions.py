#! /usr/bin/env python

# src/data_structures/fractions.py

from typing import (
    Union,
)

from algorithms.number_theory_algorithms import (
    gcd,
    lcm,
)

Real = Union[int, float]

class CustomFraction(object):

    def __init__(self, numerator: int, denominator: int):
        if not numerator and not denominator:
            raise ValueError("0 / 0 is indeterminate")
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
        g = gcd(abs(numerator), abs(denominator))
        self.numerator = numerator // g
        self.denominator = denominator // g
    
    def __hash__(self):
        return hash((self.numerator, self.denominator))

    def __str__(self):
        return f"{self.numerator} / {self.denominator}"
    
    def __repr__(self):
        return f"{self.numerator} / {self.denominator}"

    def __name__(self):
        return f"{self.numerator} / {self.denominator}"
    
    def __eq__(self, other: Union["CustomFraction", int]) -> bool:
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        return self.numerator == other.numerator and self.denominator == other.denominator
    
    def __neq__(self, other: Union["CustomFraction", int]) -> bool:
        return not self.__eq__(other)

    def __neg__(self) -> "CustomFraction":
        return CustomFraction(-self.numerator, self.denominator)

    def __add__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator / self.denominator + other
        if self.denominator == 0:
            if self != other and other.denominator: raise ValueError("Indeterminate value for addition of +inf and -inf")
            return self
        elif other.denominator == 0:
            return other
        denom = lcm(abs(self.denominator), abs(other.denominator))
        numer = (self.numerator * denom // self.denominator) + (other.numerator * denom // other.denominator) 
        return CustomFraction(numer, denom)
    
    def __radd__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        return self.__add__(other)

    def __sub__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator / self.denominator - other
        if self.denominator == 0:
            if self == other and other.denominator:
                sgn = "+" if self.numerator > 0 else "-"
                raise ValueError(f"Indeterminate value for subtraction of {sgn}inf from {sgn}inf")
            return self
        elif other.denominator == 0:
            return CustomFraction(-other.numerator, other.denominator)
        denom = lcm(abs(self.denominator), abs(other.denominator))
        numer = (self.numerator * denom // self.denominator) - (other.numerator * denom // other.denominator) 
        return CustomFraction(numer, denom)
    
    def __rsub__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        cp = CustomFraction(-self.numerator, self.denominator)
        return cp.__add__(other)
    
    def __mul__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator / self.denominator * other
        return CustomFraction(self.numerator * other.numerator, self.denominator * other.denominator)
    
    def __rmul__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        return self.__mul__(other)

    def __truediv__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator / (self.denominator * other)
        return CustomFraction(self.numerator * other.denominator, self.denominator * other.numerator)
    
    def __rtruediv__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        res = self.__truediv__(other)
        cp = CustomFraction(self.denominator, self.numerator)
        return cp.__mul__(other)
    
    def __lt__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator < self.denominator * other
        elif not self.denominator and not other.denominator: return self.numerator < other.numerator
        return self.numerator * other.denominator < self.denominator * other.numerator
    
    def __le__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator <= self.denominator * other
        return self.numerator * other.denominator <= self.denominator * other.numerator
    
    def __gt__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator > self.denominator * other
        elif not self.denominator and not other.denominator: return self.numerator > other.numerator
        return self.numerator * other.denominator > self.denominator * other.numerator
    
    def __ge__(self, other: Union["CustomFraction", Real]) -> "CustomFraction":
        if isinstance(other, int):
            other = CustomFraction(other, 1)
        elif isinstance(other, float):
            return self.numerator >= self.denominator * other
        return self.numerator * other.denominator >= self.denominator * other.numerator
