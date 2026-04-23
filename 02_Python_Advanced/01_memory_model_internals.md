# Python Memory Model & Internals - Complete Guide

## Table of Contents
1. [Object Model](#object-model)
2. [Memory Management](#memory-management)
3. [Reference Counting & GC](#reference-counting--gc)
4. [CPython Internals](#cpython-internals)
5. [Data Type Internals](#data-type-internals)
6. [Performance Optimization](#performance-optimization)
7. [Interview Questions](#interview-questions)

---

## 1. Object Model

### Everything is an Object

```python
# In Python, everything is an object with:
# - Identity (memory address)
# - Type (what it is)
# - Value (the data)

x = 42
print(id(x))        # Identity: 140234567890 (memory address)
print(type(x))      # Type: <class 'int'>
print(x)            # Value: 42

# Even functions and classes are objects
def greet():
    pass

print(type(greet))  # <class 'function'>
print(id(greet))    # Functions have memory addresses

# Classes are objects too (instances of 'type')
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
print(isinstance(MyClass, object))  # True


# Object attributes
print(dir(x))  # List all attributes of integer object
# ['__abs__', '__add__', '__and__', '__bool__', ...]

# Get object size
import sys
print(sys.getsizeof(42))       # ~28 bytes for int
print(sys.getsizeof("hello"))  # ~54 bytes for str
print(sys.getsizeof([1,2,3]))  # ~88 bytes for list
```

### PyObject Structure

```python
"""
Every Python object in CPython is represented by PyObject struct:

typedef struct {
    Py_ssize_t ob_refcnt;    // Reference count
    PyTypeObject *ob_type;    // Pointer to type object
} PyObject;

For variable-size objects (like lists), PyVarObject is used:
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size;      // Number of items
} PyVarObject;
"""

# We can inspect these using ctypes
import ctypes

class PyObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
    ]

x = 42
obj = PyObject.from_address(id(x))
print(f"Reference count: {obj.ob_refcnt}")
```

### Mutable vs Immutable Objects

```python
# ============== IMMUTABLE TYPES ==============
# int, float, str, tuple, frozenset, bytes, bool, NoneType

# Immutable: New object created on modification
x = 10
print(f"Before: id={id(x)}")  # e.g., 140234567890
x += 1
print(f"After:  id={id(x)}")  # Different id - NEW object

# String immutability
s = "hello"
# s[0] = 'H'  # TypeError: 'str' object does not support item assignment
s = "H" + s[1:]  # Creates new string

# Tuple immutability (but can contain mutable objects!)
t = ([1, 2], [3, 4])
# t[0] = [5, 6]  # TypeError
t[0].append(3)   # OK! The list inside is mutable
print(t)  # ([1, 2, 3], [3, 4])


# ============== MUTABLE TYPES ==============
# list, dict, set, bytearray, custom objects

# Mutable: Modified in place
lst = [1, 2, 3]
print(f"Before: id={id(lst)}")  # e.g., 140234567900
lst.append(4)
print(f"After:  id={id(lst)}")  # SAME id - same object modified

# Dictionary mutation
d = {'a': 1}
original_id = id(d)
d['b'] = 2
d.update({'c': 3})
assert id(d) == original_id  # Same object


# ============== WHY DOES THIS MATTER? ==============

# 1. Function arguments
def modify_list(lst):
    lst.append(4)  # Modifies original!

def modify_int(x):
    x += 1  # Creates new local variable

my_list = [1, 2, 3]
my_int = 10

modify_list(my_list)
modify_int(my_int)

print(my_list)  # [1, 2, 3, 4] - Modified!
print(my_int)   # 10 - Unchanged

# 2. Dictionary keys (must be hashable = immutable)
d = {}
d[(1, 2)] = "tuple key"  # OK - tuple is hashable
# d[[1, 2]] = "list key"   # TypeError - list is not hashable

# 3. Default arguments trap
def bad_default(items=[]):  # DANGER: Same list reused!
    items.append(1)
    return items

print(bad_default())  # [1]
print(bad_default())  # [1, 1] - Oops!
print(bad_default())  # [1, 1, 1]

def good_default(items=None):
    if items is None:
        items = []
    items.append(1)
    return items
```

### Reference Semantics

```python
# Variables are references (pointers) to objects

# Assignment creates reference, not copy
a = [1, 2, 3]
b = a           # b points to SAME object as a
b.append(4)
print(a)        # [1, 2, 3, 4] - Both affected!
print(a is b)   # True - same object

# Creating actual copies
import copy

original = [[1, 2], [3, 4], {'a': 1}]

# Shallow copy - new container, same nested objects
shallow = copy.copy(original)
shallow = original[:]           # For lists
shallow = list(original)        # For lists
shallow = original.copy()       # For lists/dicts

print(original is shallow)      # False - different containers
print(original[0] is shallow[0])  # True - same nested list!

original[0][0] = 'X'
print(shallow[0][0])  # 'X' - nested object affected!

# Deep copy - recursively copy everything
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)

print(original is deep)         # False
print(original[0] is deep[0])   # False - different nested lists

original[0][0] = 'X'
print(deep[0][0])  # 1 - Independent copy


# Custom copy behavior
class MyClass:
    def __init__(self):
        self.data = [1, 2, 3]
    
    def __copy__(self):
        # Shallow copy
        new = MyClass()
        new.data = self.data  # Same list
        return new
    
    def __deepcopy__(self, memo):
        # Deep copy
        new = MyClass()
        new.data = copy.deepcopy(self.data, memo)
        return new
```

---

## 2. Memory Management

### Small Object Allocator

```python
"""
CPython uses a hierarchical memory allocator:

1. OS Memory (malloc/free)
   └── 2. Python Memory Allocator (PyMem_*)
       └── 3. Object-specific Allocators
           └── Small Object Allocator (for < 512 bytes)

Small Object Allocator uses:
- Arenas (256 KB chunks from OS)
- Pools (4 KB blocks within arenas)  
- Blocks (8-512 bytes within pools)

Block sizes are multiples of 8: 8, 16, 24, 32, ... 512
"""

import sys

# Small integers (-5 to 256) are pre-allocated and cached
a = 256
b = 256
print(a is b)  # True - same cached object

a = 257
b = 257
print(a is b)  # False - new objects created (usually)

# Check memory usage
print(sys.getsizeof(1))        # 28 bytes
print(sys.getsizeof(10**100))  # Much larger for big ints
print(sys.getsizeof([]))       # 56 bytes (empty list)
print(sys.getsizeof([1,2,3]))  # 88 bytes


# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True - interned (automatic for identifiers)

s1 = "hello world"
s2 = "hello world"
print(s1 is s2)  # May be True or False (depends on context)

# Force interning
s1 = sys.intern("hello world")
s2 = sys.intern("hello world")
print(s1 is s2)  # True - explicitly interned
```

### Memory Profiling

```python
import sys
import tracemalloc

# Method 1: sys.getsizeof (shallow size only)
lst = [1, 2, 3, [4, 5]]
print(sys.getsizeof(lst))  # Only counts list object, not nested!

# Deep size calculation
def deep_getsizeof(obj, seen=None):
    """Recursively calculate total memory of object."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) 
                   for k, v in obj.items())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        size += sum(deep_getsizeof(i, seen) for i in obj)
    
    return size

print(deep_getsizeof([1, 2, 3, [4, 5]]))


# Method 2: tracemalloc (track allocations)
tracemalloc.start()

# Your code here
data = [i ** 2 for i in range(10000)]
more_data = {i: str(i) for i in range(10000)}

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.2f} KB")
print(f"Peak: {peak / 1024:.2f} KB")

tracemalloc.stop()


# Method 3: memory_profiler (pip install memory_profiler)
# @profile  # Decorator for line-by-line memory profiling
# def my_function():
#     a = [1] * 1000000
#     b = [2] * 2000000
#     del b
#     return a
```

### Memory Leaks

```python
# Common causes of memory leaks in Python

# 1. Circular references (usually handled by GC, but not always)
class Node:
    def __init__(self):
        self.parent = None
        self.children = []

parent = Node()
child = Node()
parent.children.append(child)
child.parent = parent  # Circular reference!

# Fix: Use weakref
import weakref

class Node:
    def __init__(self):
        self._parent = None
        self.children = []
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, value):
        self._parent = weakref.ref(value) if value else None


# 2. Caches without limits
cache = {}  # Grows forever!

def cached_compute(x):
    if x not in cache:
        cache[x] = expensive_computation(x)
    return cache[x]

# Fix: Use LRU cache with maxsize
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_compute(x):
    return expensive_computation(x)


# 3. Global variables holding references
_global_data = []

def process():
    _global_data.append(large_object())  # Never cleared!

# Fix: Clear when done or use context manager


# 4. Closures capturing large objects
def create_processor(large_data):
    def process(x):
        return x in large_data  # Captures large_data!
    return process

# Fix: Only capture what's needed
def create_processor(large_data):
    data_set = set(large_data)  # Convert to set
    def process(x):
        return x in data_set
    return process
```

---

## 3. Reference Counting & Garbage Collection

### Reference Counting

```python
import sys

# Every object has a reference count
x = [1, 2, 3]
print(sys.getrefcount(x))  # Usually 2 (x + getrefcount's reference)

# Reference count increases when:
y = x                      # Assignment
my_list = [x]              # Added to container
def func(param): pass
func(x)                    # Passed as argument

# Reference count decreases when:
del y                      # Explicit deletion
my_list = []               # Removed from container
# Variable goes out of scope (function returns)

# Object is deallocated when refcount reaches 0


# Viewing reference count
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2

b = a
print(sys.getrefcount(a))  # 3

c = [a, a, a]
print(sys.getrefcount(a))  # 6 (a, b, 3 in list c, + getrefcount)

del b
print(sys.getrefcount(a))  # 5

del c
print(sys.getrefcount(a))  # 2
```

### Garbage Collection

```python
import gc

# Reference counting can't handle circular references
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Create circular reference
a = Node(1)
b = Node(2)
a.next = b
b.next = a  # Circular!

# Even after deleting, objects aren't freed (refcount > 0)
del a
del b
# Objects still exist in memory!

# Garbage collector handles this
gc.collect()  # Force garbage collection


# GC Configuration
print(gc.get_threshold())  # (700, 10, 10) - generation thresholds

# Generation 0: New objects, collected frequently
# Generation 1: Survived one collection
# Generation 2: Long-lived objects, collected rarely

gc.set_threshold(1000, 15, 15)  # Customize thresholds

# Disable GC (careful!)
gc.disable()
# ... do time-critical work without GC pauses ...
gc.enable()

# Get GC statistics
print(gc.get_stats())


# Finding references
x = [1, 2, 3]
y = {'data': x}
z = [x, x]

# Find what refers to x
print(gc.get_referrers(x))  # [y, z, ...]

# Find what x refers to
print(gc.get_referents(x))  # [1, 2, 3]


# Weak references (don't increment refcount)
import weakref

class Data:
    pass

d = Data()
weak_d = weakref.ref(d)

print(weak_d())  # <Data object>
del d
print(weak_d())  # None - object was collected


# WeakValueDictionary - values don't prevent GC
cache = weakref.WeakValueDictionary()
obj = Data()
cache['key'] = obj
print(cache.get('key'))  # <Data object>
del obj
print(cache.get('key'))  # None - automatically removed
```

### `__del__` and Destructor Patterns

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Creating {name}")
    
    def __del__(self):
        """Called when object is about to be destroyed.
        
        WARNING: Don't rely on __del__ for cleanup!
        - May not be called immediately
        - May not be called at all (circular references)
        - Can't access global variables reliably
        """
        print(f"Destroying {self.name}")

r = Resource("test")
del r  # __del__ called (usually immediately)


# Better pattern: Context managers
class Resource:
    def __init__(self, name):
        self.name = name
        self.file = open(name, 'w')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()  # Guaranteed cleanup
        return False

with Resource("test.txt") as r:
    r.file.write("data")
# File is closed here, guaranteed


# Or explicit close method
class Resource:
    def __init__(self, name):
        self.name = name
        self._closed = False
    
    def close(self):
        if not self._closed:
            # cleanup
            self._closed = True
    
    def __del__(self):
        self.close()  # Fallback only
```

---

## 4. CPython Internals

### Bytecode and the VM

```python
import dis

# Python compiles to bytecode, executed by the VM
def add(a, b):
    return a + b

# Disassemble to see bytecode
dis.dis(add)
"""
  2           0 LOAD_FAST                0 (a)
              2 LOAD_FAST                1 (b)
              4 BINARY_ADD
              6 RETURN_VALUE
"""

# Access code object
code = add.__code__
print(code.co_varnames)    # ('a', 'b')
print(code.co_consts)      # (None,)
print(code.co_code)        # Raw bytecode bytes


# Bytecode optimization
def example():
    x = 1 + 2        # Constant folding: becomes 3
    y = "a" * 5      # Becomes "aaaaa"
    z = [1, 2, 3]    # List created at runtime (mutable)
    
dis.dis(example)


# Peephole optimizer
def before():
    if True:
        return 1
    return 2

# Optimizer removes dead code
dis.dis(before)
```

### Frame Objects and Call Stack

```python
import sys
import inspect

def outer():
    x = 10
    inner()

def inner():
    frame = sys._getframe()
    
    # Current frame info
    print(f"Function: {frame.f_code.co_name}")
    print(f"Line: {frame.f_lineno}")
    print(f"Locals: {frame.f_locals}")
    
    # Caller frame
    caller = frame.f_back
    print(f"Caller: {caller.f_code.co_name}")
    print(f"Caller locals: {caller.f_locals}")

outer()


# Using inspect module (safer)
def get_caller_info():
    stack = inspect.stack()
    caller = stack[1]
    return {
        'function': caller.function,
        'filename': caller.filename,
        'line': caller.lineno,
    }


# Stack trace manipulation
def format_stack():
    import traceback
    return ''.join(traceback.format_stack())
```

### Global Interpreter Lock (GIL)

```python
"""
GIL (Global Interpreter Lock):
- Mutex that protects access to Python objects
- Only one thread can execute Python bytecode at a time
- Prevents true parallelism for CPU-bound Python code

Why does GIL exist?
- Simplifies CPython's memory management
- Makes single-threaded programs faster
- Makes C extensions easier to write

When is GIL released?
- I/O operations (file, network)
- time.sleep()
- Some C extensions (NumPy operations)
"""

import threading
import time

# CPU-bound work (limited by GIL)
def cpu_bound(n):
    count = 0
    for i in range(n):
        count += i
    return count

# Threading doesn't help for CPU-bound
start = time.time()
threads = [threading.Thread(target=cpu_bound, args=(10**7,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threaded CPU-bound: {time.time() - start:.2f}s")

# Sequential is similar or faster!
start = time.time()
for _ in range(4):
    cpu_bound(10**7)
print(f"Sequential CPU-bound: {time.time() - start:.2f}s")


# I/O-bound work (GIL released, threading helps)
import urllib.request

def io_bound(url):
    with urllib.request.urlopen(url) as response:
        return len(response.read())

# Threading helps for I/O
urls = ["http://example.com"] * 4

start = time.time()
threads = [threading.Thread(target=io_bound, args=(url,)) for url in urls]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threaded I/O: {time.time() - start:.2f}s")


# Solutions for CPU-bound parallelism:
# 1. multiprocessing (separate processes, no GIL sharing)
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(cpu_bound, [10**7] * 4)

# 2. C extensions that release GIL (NumPy, etc.)
# 3. Use PyPy, Cython, or other implementations
```

---

## 5. Data Type Internals

### Integer Internals

```python
"""
Python integers are arbitrary precision:
- Small integers: Fixed size, fast
- Large integers: Variable size, uses arrays of "digits"

PyLongObject structure:
- ob_refcnt: Reference count
- ob_type: Pointer to int type
- ob_size: Number of digits (negative = negative number)
- ob_digit[]: Array of digits (base 2^30)
"""

import sys

# Small integer cache (-5 to 256)
a = 100
b = 100
print(a is b)  # True - cached

# Large integers
big = 10 ** 100
print(sys.getsizeof(big))  # Much larger

# Integer operations create new objects
x = 5
print(id(x))
x += 1
print(id(x))  # Different! New object created


# Bit operations on integers
n = 42
print(bin(n))          # '0b101010'
print(n.bit_length())  # 6 (bits needed to represent)
print(n.bit_count())   # 3 (number of 1s) - Python 3.10+
```

### String Internals

```python
"""
Strings in Python 3 use flexible representation:
- ASCII only: 1 byte per character
- Latin-1 (≤ U+00FF): 1 byte per character  
- BMP (≤ U+FFFF): 2 bytes per character
- Full Unicode: 4 bytes per character

This is called "flexible string representation" (PEP 393).
"""

import sys

# ASCII string
s1 = "hello"
print(sys.getsizeof(s1))  # ~54 bytes

# Unicode string
s2 = "héllo"  # Latin-1 characters
print(sys.getsizeof(s2))  # Similar

s3 = "hello 世界"  # BMP characters
print(sys.getsizeof(s3))  # Larger (2 bytes/char)

s4 = "hello 🎉"  # Full Unicode (emoji)
print(sys.getsizeof(s4))  # Larger (4 bytes/char)


# String interning
s1 = sys.intern("hello_world")
s2 = sys.intern("hello_world")
print(s1 is s2)  # True - same object

# Automatic interning for identifiers
s1 = "valid_identifier"
s2 = "valid_identifier"
print(s1 is s2)  # Usually True


# String methods create new strings (immutable!)
s = "hello"
s_upper = s.upper()  # New string created
s_concat = s + " world"  # New string created

# Efficient string building
parts = []
for i in range(1000):
    parts.append(str(i))
result = ''.join(parts)  # Single string creation

# NOT: result = "" ; for i: result += str(i)  # O(n²)!
```

### List Internals

```python
"""
Lists are dynamic arrays:
- Contiguous memory for pointers to objects
- Over-allocation for amortized O(1) append
- Growth pattern: 0, 4, 8, 16, 24, 32, 40, 52, 64, 76, ...

PyListObject:
- ob_refcnt, ob_type, ob_size
- ob_item: Pointer to array of PyObject*
- allocated: Number of slots allocated
"""

import sys

# Watch list growth
lst = []
prev_size = 0
for i in range(20):
    lst.append(i)
    size = sys.getsizeof(lst)
    if size != prev_size:
        print(f"Length {len(lst)}: {size} bytes (allocated ~{(size-56)//8} slots)")
        prev_size = size

"""
Output shows over-allocation:
Length 1: 88 bytes (allocated ~4 slots)
Length 5: 120 bytes (allocated ~8 slots)
Length 9: 184 bytes (allocated ~16 slots)
...
"""


# List operations complexity
lst = list(range(1000))

# O(1) operations
lst.append(x)           # Amortized O(1)
lst.pop()               # O(1)
lst[i]                  # O(1)
lst[i] = x              # O(1)
len(lst)                # O(1)

# O(n) operations  
lst.pop(0)              # O(n) - shifts all elements
lst.insert(0, x)        # O(n) - shifts all elements
x in lst                # O(n) - linear search
lst.remove(x)           # O(n)
lst.index(x)            # O(n)

# O(n) with copying
lst.copy()              # O(n)
lst[:]                  # O(n) - slice copy
lst + other             # O(n + m)
lst * k                 # O(n * k)
```

### Dictionary Internals

```python
"""
Dicts are hash tables with open addressing:
- Hash function maps keys to indices
- Collision resolution: Open addressing (probing)
- Maintains insertion order (Python 3.7+)

Key features:
- Average O(1) lookup, insert, delete
- Worst case O(n) with many collisions
- Load factor triggers resize (~2/3 full)
"""

# Hash function
print(hash("hello"))    # Consistent within session
print(hash(42))         # For small ints: hash(n) == n
print(hash((1, 2, 3)))  # Tuples are hashable

# Custom hash
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# Dict operations complexity
d = {i: i for i in range(1000)}

# O(1) average
d[key]                  # Lookup
d[key] = value          # Insert/update  
key in d                # Membership
d.get(key, default)     # Lookup with default
del d[key]              # Delete

# O(n)
len(d)                  # Actually O(1) - cached
list(d.keys())          # O(n) - creates list
list(d.values())        # O(n)
list(d.items())         # O(n)


# Dict memory
import sys

d = {}
for i in range(20):
    d[i] = i
    print(f"Size {len(d)}: {sys.getsizeof(d)} bytes")
# Shows resize at certain thresholds
```

---

## 6. Performance Optimization

### Profiling

```python
import cProfile
import pstats
from io import StringIO

def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

# Basic profiling
cProfile.run('slow_function()')

# Detailed profiling
profiler = cProfile.Profile()
profiler.enable()
slow_function()
profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)


# Line profiler (pip install line_profiler)
# @profile
# def slow_function():
#     ...
# Run: kernprof -l -v script.py


# Memory profiler
# @profile
# def memory_heavy():
#     ...
# Run: python -m memory_profiler script.py


# Timing
import timeit

# Time a statement
time = timeit.timeit('sum(range(1000))', number=10000)
print(f"Average: {time/10000:.6f}s")

# Time a function
def my_func():
    return sum(range(1000))

time = timeit.timeit(my_func, number=10000)
```

### Common Optimizations

```python
# 1. Use built-in functions (implemented in C)
# Bad
total = 0
for x in numbers:
    total += x

# Good
total = sum(numbers)


# 2. Use list comprehensions
# Bad
squares = []
for x in range(1000):
    squares.append(x ** 2)

# Good
squares = [x ** 2 for x in range(1000)]


# 3. Use generators for large sequences
# Bad (creates full list in memory)
total = sum([x ** 2 for x in range(1000000)])

# Good (generator expression)
total = sum(x ** 2 for x in range(1000000))


# 4. Use appropriate data structures
# Bad: O(n) lookup
if item in my_list:
    ...

# Good: O(1) lookup
my_set = set(my_list)
if item in my_set:
    ...


# 5. Avoid global variable lookups in loops
import math

# Bad
def compute():
    for x in range(1000):
        y = math.sqrt(x)  # Global lookup each time

# Good  
def compute():
    sqrt = math.sqrt  # Local reference
    for x in range(1000):
        y = sqrt(x)


# 6. Use local variables
# Bad
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment_slow(self):
        for _ in range(1000000):
            self.count += 1  # Attribute lookup each time

# Good
    def increment_fast(self):
        count = self.count
        for _ in range(1000000):
            count += 1
        self.count = count


# 7. String concatenation
# Bad: O(n²)
s = ""
for word in words:
    s += word

# Good: O(n)
s = ''.join(words)


# 8. Use __slots__ for memory
class PointWithSlots:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointWithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
print(sys.getsizeof(PointWithSlots(1, 2).__dict__))  # Error - no __dict__
print(sys.getsizeof(PointWithoutSlots(1, 2).__dict__))  # ~104 bytes
```

---

## 7. Interview Questions

### Common Questions

```python
# Q1: What's the difference between is and ==?
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True - same values
print(a is b)  # False - different objects
print(a is c)  # True - same object
# 'is' checks identity (id), '==' checks equality (value)


# Q2: Explain mutable default argument problem
def bad(items=[]):
    items.append(1)
    return items

print(bad())  # [1]
print(bad())  # [1, 1] - Same list reused!

# Fix: Use None as default
def good(items=None):
    if items is None:
        items = []
    items.append(1)
    return items


# Q3: How does Python handle memory?
"""
- Reference counting: Every object has refcount
- Garbage collection: Handles circular references
- Memory pools: Small object allocator
- GIL: Protects Python objects in multithreading
"""


# Q4: What is GIL and its implications?
"""
GIL = Global Interpreter Lock
- Only one thread executes Python at a time
- CPU-bound: Use multiprocessing
- I/O-bound: Threading still useful
- Released during I/O and some C extensions
"""


# Q5: Explain Python's object model
"""
Everything is object with:
- Identity: id(obj) - memory address
- Type: type(obj) - class
- Value: the data

Variables are references to objects.
Assignment creates new reference, not copy.
"""


# Q6: Deep vs shallow copy
import copy

original = [[1, 2], [3, 4]]

# Shallow: New container, same nested objects
shallow = copy.copy(original)
shallow[0][0] = 'X'
print(original)  # [['X', 2], [3, 4]] - affected!

# Deep: Recursively copy everything
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)
deep[0][0] = 'X'
print(original)  # [[1, 2], [3, 4]] - unaffected


# Q7: How are dictionaries implemented?
"""
Hash table with open addressing:
- Keys hashed to index
- O(1) average lookup/insert/delete
- Maintains insertion order (3.7+)
- Resizes at ~2/3 full
"""


# Q8: What happens when you do x += 1?
x = 5
print(id(x))  # e.g., 12345
x += 1
print(id(x))  # Different! New object created

# For integers, += creates new object (immutable)
# For lists, += modifies in place (mutable)

lst = [1, 2]
print(id(lst))
lst += [3]
print(id(lst))  # Same! Modified in place


# Q9: Interning and caching
# Small integers (-5 to 256) are cached
a = 100
b = 100
print(a is b)  # True

# String literals may be interned
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # Usually True


# Q10: Why can't lists be dictionary keys?
"""
Dictionary keys must be hashable (immutable).
Lists are mutable, so their hash could change.
Use tuples instead: they're immutable and hashable.
"""
d = {}
d[(1, 2)] = "ok"      # Tuple key - works
# d[[1, 2]] = "error"  # List key - TypeError
```
