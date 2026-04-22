# Python In-Depth - Complete Guide

## 1. Python Memory Model & Object System

### Everything is an Object
```python
# All values in Python are objects with identity, type, and value
x = 42
print(id(x))      # Identity (memory address)
print(type(x))    # Type
print(x)          # Value

# Objects have attributes and methods
print(dir(x))     # List all attributes
```

### Mutable vs Immutable
```python
# Immutable: int, float, str, tuple, frozenset, bytes
x = 10
print(id(x))  # e.g., 140234567890
x += 1
print(id(x))  # Different id - new object created

# Mutable: list, dict, set, bytearray
lst = [1, 2, 3]
print(id(lst))  # e.g., 140234567900
lst.append(4)
print(id(lst))  # Same id - modified in place
```

### Reference Semantics
```python
# Variables are references to objects
a = [1, 2, 3]
b = a           # b references same object
b.append(4)
print(a)        # [1, 2, 3, 4] - both affected

# Shallow vs Deep Copy
import copy
original = [[1, 2], [3, 4]]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

original[0][0] = 'X'
print(shallow)  # [['X', 2], [3, 4]] - nested affected
print(deep)     # [[1, 2], [3, 4]] - independent copy
```

### Interning & Caching
```python
# Small integers (-5 to 256) are cached
a = 256
b = 256
print(a is b)  # True

a = 257
b = 257
print(a is b)  # False (usually)

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True - interned
```

---

## 2. Data Structures Deep Dive

### Lists - Dynamic Arrays
```python
# Implementation: Dynamic array with overallocation
# Amortized O(1) append, O(n) insert/delete

import sys
lst = []
for i in range(10):
    lst.append(i)
    print(f"Length: {len(lst)}, Size: {sys.getsizeof(lst)}")
# Size grows in chunks for amortization

# List comprehensions vs loops (prefer comprehensions)
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(3)] for i in range(3)]
```

### Dictionaries - Hash Tables
```python
# Implementation: Hash table with open addressing
# Average O(1) lookup, insert, delete
# Maintains insertion order (Python 3.7+)

# Dict comprehensions
squares = {x: x**2 for x in range(5)}

# Useful methods
d = {'a': 1, 'b': 2}
d.get('c', 0)           # Default value
d.setdefault('c', 3)    # Set if not exists
d.update({'d': 4})      # Merge
d.pop('a')              # Remove and return

# Dictionary views
d.keys()    # dict_keys - view object
d.values()  # dict_values
d.items()   # dict_items
```

### Sets - Hash Sets
```python
# Implementation: Hash table (only keys)
# O(1) average for add, remove, lookup

s1 = {1, 2, 3}
s2 = {2, 3, 4}

# Set operations
s1 | s2     # Union: {1, 2, 3, 4}
s1 & s2     # Intersection: {2, 3}
s1 - s2     # Difference: {1}
s1 ^ s2     # Symmetric difference: {1, 4}

# Frozen sets (immutable, hashable)
fs = frozenset([1, 2, 3])
d = {fs: "value"}  # Can be dict key
```

### Collections Module
```python
from collections import (
    defaultdict, Counter, OrderedDict, 
    deque, namedtuple, ChainMap
)

# defaultdict - automatic default values
dd = defaultdict(list)
dd['key'].append(1)  # No KeyError

# Counter - counting hashable objects
c = Counter(['a', 'b', 'a', 'c', 'a'])
c.most_common(2)     # [('a', 3), ('b', 1)]
c.update(['a', 'a']) # Add more counts

# deque - double-ended queue
dq = deque([1, 2, 3], maxlen=5)
dq.appendleft(0)     # O(1) left append
dq.popleft()         # O(1) left pop
dq.rotate(1)         # Rotate right

# namedtuple - lightweight immutable class
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)      # Attribute access
print(p[0], p[1])    # Index access

# ChainMap - combine multiple dicts
defaults = {'color': 'red', 'user': 'guest'}
env = {'user': 'admin'}
config = ChainMap(env, defaults)
print(config['user'])  # 'admin' (first found)
```

---

## 3. Functions & Closures

### First-Class Functions
```python
# Functions are objects
def greet(name):
    return f"Hello, {name}"

# Assign to variable
say_hello = greet
print(say_hello("World"))

# Pass as argument
def apply(func, value):
    return func(value)

# Return from function
def multiplier(n):
    def multiply(x):
        return x * n
    return multiply

double = multiplier(2)
print(double(5))  # 10
```

### Closures
```python
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

c = counter()
print(c())  # 1
print(c())  # 2

# Examining closure
print(c.__closure__)
print(c.__closure__[0].cell_contents)
```

### *args and **kwargs
```python
def func(*args, **kwargs):
    print(f"args: {args}")      # Tuple
    print(f"kwargs: {kwargs}")  # Dict

func(1, 2, 3, a=4, b=5)

# Unpacking
def add(a, b, c):
    return a + b + c

args = [1, 2, 3]
kwargs = {'a': 1, 'b': 2, 'c': 3}
add(*args)
add(**kwargs)
```

### Lambda Functions
```python
# Anonymous functions
square = lambda x: x ** 2
add = lambda x, y: x + y

# Common with higher-order functions
sorted(data, key=lambda x: x['name'])
list(filter(lambda x: x > 0, nums))
list(map(lambda x: x * 2, nums))
```

---

## 4. Decorators

### Basic Decorator Pattern
```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserve metadata
    def wrapper(*args, **kwargs):
        print("Before call")
        result = func(*args, **kwargs)
        print("After call")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    """Greet someone"""
    return f"Hello, {name}"

# Equivalent to: say_hello = my_decorator(say_hello)
```

### Decorator with Arguments
```python
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}")
```

### Class-based Decorators
```python
class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")
```

### Practical Decorators
```python
# Timing decorator
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

# Memoization decorator
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Or use built-in
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
```

---

## 5. Generators & Iterators

### Iterators
```python
# Iterator protocol: __iter__ and __next__
class CountUp:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        self.current += 1
        return self.current

for num in CountUp(5):
    print(num)
```

### Generators
```python
# Generator function (uses yield)
def count_up(limit):
    current = 0
    while current < limit:
        current += 1
        yield current

# Generator expression
squares = (x**2 for x in range(10))

# Lazy evaluation - memory efficient
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# Generator methods
def generator():
    value = yield 1
    yield value * 2

g = generator()
print(next(g))       # 1
print(g.send(10))    # 20
```

### yield from
```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

list(flatten([1, [2, [3, 4], 5]]))  # [1, 2, 3, 4, 5]
```

### itertools Module
```python
import itertools

# Infinite iterators
itertools.count(10, 2)      # 10, 12, 14, ...
itertools.cycle('ABC')      # A, B, C, A, B, C, ...
itertools.repeat(10, 3)     # 10, 10, 10

# Combinatoric
itertools.permutations('ABC', 2)    # AB, AC, BA, BC, CA, CB
itertools.combinations('ABC', 2)     # AB, AC, BC
itertools.product('AB', '12')        # A1, A2, B1, B2

# Terminating
itertools.chain([1,2], [3,4])        # 1, 2, 3, 4
itertools.islice(range(100), 5)      # 0, 1, 2, 3, 4
itertools.takewhile(lambda x: x<5, range(10))  # 0,1,2,3,4
itertools.groupby(data, key=func)    # Group consecutive
itertools.accumulate([1,2,3,4])      # 1, 3, 6, 10
```

---

## 6. Context Managers

### Using with Statement
```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed

# Multiple context managers
with open('in.txt') as fin, open('out.txt', 'w') as fout:
    fout.write(fin.read())
```

### Creating Context Managers
```python
# Class-based
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False  # Don't suppress exceptions

with Timer() as t:
    time.sleep(1)
print(f"Elapsed: {t.elapsed}")

# Generator-based with contextlib
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Elapsed: {end - start}")

with timer():
    time.sleep(1)
```

---

## 7. Object-Oriented Programming

### Class Basics
```python
class Dog:
    species = "Canis familiaris"  # Class attribute
    
    def __init__(self, name, age):
        self.name = name          # Instance attribute
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    @classmethod
    def from_birth_year(cls, name, birth_year):
        return cls(name, 2024 - birth_year)
    
    @staticmethod
    def is_adult(age):
        return age >= 2
    
    def __str__(self):
        return f"{self.name} ({self.age})"
    
    def __repr__(self):
        return f"Dog('{self.name}', {self.age})"
```

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Multiple inheritance
class FlyingMixin:
    def fly(self):
        return f"{self.name} is flying"

class Bird(Animal, FlyingMixin):
    def speak(self):
        return f"{self.name} says Tweet!"

# Method Resolution Order (MRO)
print(Bird.__mro__)
```

### Properties
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

### Dunder/Magic Methods
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
    
    def __getitem__(self, index):
        return (self.x, self.y)[index]
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x!r}, {self.y!r})"
```

### Dataclasses (Python 3.7+)
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.name = self.name.title()

@dataclass(frozen=True)  # Immutable
class Point:
    x: float
    y: float

@dataclass(order=True)  # Enables comparison
class Student:
    sort_index: float = field(init=False, repr=False)
    name: str
    grade: float
    
    def __post_init__(self):
        self.sort_index = self.grade
```

---

## 8. Metaclasses & Descriptors

### Descriptors
```python
class Validator:
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        self.validate(value)
        obj.__dict__[self.name] = value
    
    def validate(self, value):
        pass

class PositiveNumber(Validator):
    def validate(self, value):
        if value < 0:
            raise ValueError(f"{self.name} must be positive")

class Person:
    age = PositiveNumber()
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

### Metaclasses
```python
# Classes are instances of metaclasses
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Called when class is created
        namespace['created_by'] = 'MyMeta'
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace):
        # Called after class is created
        super().__init__(name, bases, namespace)

class MyClass(metaclass=MyMeta):
    pass

print(MyClass.created_by)  # 'MyMeta'

# Practical: Singleton metaclass
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    pass
```

---

## 9. Concurrency & Parallelism

### Threading
```python
import threading

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)
    print(f"Worker {name} done")

# Create and start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for completion
for t in threads:
    t.join()

# Thread synchronization
lock = threading.Lock()
counter = 0

def increment():
    global counter
    with lock:
        counter += 1

# Thread-safe queue
from queue import Queue
q = Queue()
q.put(item)
item = q.get()
```

### Multiprocessing
```python
from multiprocessing import Process, Pool, Queue, Manager

def worker(x):
    return x * x

# Process pool
with Pool(4) as p:
    results = p.map(worker, range(10))

# Shared state
manager = Manager()
shared_list = manager.list()
shared_dict = manager.dict()
```

### Async/Await
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulated I/O
    return f"Data from {url}"

async def main():
    # Sequential
    result1 = await fetch_data("url1")
    result2 = await fetch_data("url2")
    
    # Concurrent
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    return results

# Run async code
asyncio.run(main())

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, *args):
        await asyncio.sleep(0.1)

# Async generator
async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i
```

### GIL (Global Interpreter Lock)
```python
# GIL prevents true parallelism for CPU-bound Python code
# - Only one thread executes Python bytecode at a time
# - Threading still useful for I/O-bound tasks
# - Use multiprocessing for CPU-bound parallelism
# - NumPy and other C extensions can release GIL
```

---

## 10. Type Hints

```python
from typing import (
    List, Dict, Tuple, Set, Optional, Union,
    Callable, TypeVar, Generic, Any, Literal
)

# Basic annotations
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections
def process(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items)}

# Optional and Union
def find(id: int) -> Optional[str]:  # str | None
    pass

def handle(data: Union[str, bytes]) -> None:  # str | bytes
    pass

# Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# TypeVar for generics
T = TypeVar('T')

def first(items: List[T]) -> T:
    return items[0]

# Generic classes
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Literal types
Mode = Literal['r', 'w', 'a']

def open_file(path: str, mode: Mode) -> None:
    pass

# Python 3.10+ syntax
def process(x: int | str) -> list[int]:
    pass
```

---

## 11. Testing

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def tearDown(self):
        pass
    
    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(1, 0)
    
    @patch('module.external_api')
    def test_with_mock(self, mock_api):
        mock_api.return_value = {'status': 'ok'}
        result = self.calc.call_api()
        mock_api.assert_called_once()

# pytest style
import pytest

def test_add():
    assert add(2, 3) == 5

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def test_with_fixture(sample_data):
    assert sum(sample_data) == 6

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
```

---

## 12. Common Gotchas

```python
# 1. Mutable default arguments
def bad(items=[]):       # Same list reused!
    items.append(1)
    return items

def good(items=None):
    if items is None:
        items = []
    items.append(1)
    return items

# 2. Late binding closures
funcs = [lambda x: x * i for i in range(3)]
[f(2) for f in funcs]  # [4, 4, 4] - all use i=2!

# Fix: capture value
funcs = [lambda x, i=i: x * i for i in range(3)]
[f(2) for f in funcs]  # [0, 2, 4]

# 3. Modifying list while iterating
lst = [1, 2, 3, 4, 5]
for item in lst:      # Don't do this!
    if item % 2 == 0:
        lst.remove(item)

# Fix: iterate over copy or use comprehension
lst = [x for x in lst if x % 2 != 0]

# 4. Integer division
5 / 2   # 2.5 (float division)
5 // 2  # 2 (integer division)

# 5. is vs ==
a = [1, 2, 3]
b = [1, 2, 3]
a == b  # True (same value)
a is b  # False (different objects)
```
