# Iterators, Generators & Context Managers - Complete Guide

## Table of Contents
1. [Iterator Protocol](#iterator-protocol)
2. [Generators](#generators)
3. [Generator Expressions](#generator-expressions)
4. [itertools Module](#itertools-module)
5. [Context Managers](#context-managers)
6. [Interview Questions](#interview-questions)

---

## 1. Iterator Protocol

### Understanding Iteration

```python
"""
Iterator Protocol:
- __iter__(): Returns the iterator object itself
- __next__(): Returns the next item, raises StopIteration when exhausted

Iterable: Object that can return an iterator (__iter__)
Iterator: Object that produces values one at a time (__next__)
"""

# How for loop works internally
my_list = [1, 2, 3]

# This:
for item in my_list:
    print(item)

# Is equivalent to:
iterator = iter(my_list)  # Calls my_list.__iter__()
while True:
    try:
        item = next(iterator)  # Calls iterator.__next__()
        print(item)
    except StopIteration:
        break
```

### Creating Custom Iterators

```python
# Method 1: Class-based iterator
class CountUp:
    """Iterator that counts from start to end."""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

for num in CountUp(1, 5):
    print(num)  # 1, 2, 3, 4, 5


# Method 2: Separate iterable and iterator classes
class Range:
    """Iterable that creates new iterators."""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return RangeIterator(self.start, self.end)

class RangeIterator:
    """Iterator for Range."""
    
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can iterate multiple times (creates new iterator each time)
r = Range(1, 3)
print(list(r))  # [1, 2, 3]
print(list(r))  # [1, 2, 3] (fresh iterator)


# Method 3: Using __getitem__ (sequence protocol)
class Squares:
    """Iterable using __getitem__."""
    
    def __init__(self, n):
        self.n = n
    
    def __getitem__(self, index):
        if index >= self.n:
            raise IndexError
        return index ** 2
    
    def __len__(self):
        return self.n

for sq in Squares(5):
    print(sq)  # 0, 1, 4, 9, 16
```

### Infinite Iterators

```python
class InfiniteCounter:
    """Counts forever from start."""
    
    def __init__(self, start=0, step=1):
        self.current = start
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.current
        self.current += self.step
        return value

# Use with caution - need to break manually
counter = InfiniteCounter()
for i, num in enumerate(counter):
    print(num)
    if i >= 5:
        break


class Cycle:
    """Cycles through items forever."""
    
    def __init__(self, iterable):
        self.items = list(iterable)
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.items:
            raise StopIteration
        value = self.items[self.index]
        self.index = (self.index + 1) % len(self.items)
        return value
```

---

## 2. Generators

### Generator Functions

```python
# Generator function uses 'yield' instead of 'return'
def count_up(start, end):
    """Generator that counts from start to end."""
    current = start
    while current <= end:
        yield current  # Pauses and returns value
        current += 1

# Creates generator object
gen = count_up(1, 5)
print(type(gen))  # <class 'generator'>

# Iterate
for num in gen:
    print(num)

# Or manually
gen = count_up(1, 3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# print(next(gen))  # StopIteration


# Generator vs regular function
def get_squares_list(n):
    """Returns list - stores all in memory."""
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

def get_squares_gen(n):
    """Generator - produces one at a time."""
    for i in range(n):
        yield i ** 2

# Memory comparison
import sys
list_result = get_squares_list(1000000)
gen_result = get_squares_gen(1000000)

print(sys.getsizeof(list_result))  # ~8MB
print(sys.getsizeof(gen_result))   # ~112 bytes!
```

### Generator with Multiple Yields

```python
def fibonacci(n):
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(list(fibonacci(10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def flatten(nested_list):
    """Flatten nested lists."""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # Delegate to sub-generator
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
print(list(flatten(nested)))
# [1, 2, 3, 4, 5, 6, 7, 8, 9]


def read_large_file(file_path, chunk_size=1024):
    """Read file in chunks."""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
```

### Generator Methods

```python
# send() - Send value into generator
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

acc = accumulator()
next(acc)          # Initialize (advance to first yield)
print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35


# throw() - Raise exception in generator
def careful_generator():
    try:
        yield 1
        yield 2
        yield 3
    except ValueError:
        yield "Caught ValueError"
    yield "Done"

gen = careful_generator()
print(next(gen))           # 1
print(gen.throw(ValueError))  # Caught ValueError
print(next(gen))           # Done


# close() - Close generator
def cleanup_generator():
    try:
        yield 1
        yield 2
        yield 3
    finally:
        print("Cleaning up!")

gen = cleanup_generator()
print(next(gen))  # 1
gen.close()       # Cleaning up!


# Generator with return value
def gen_with_return():
    yield 1
    yield 2
    return "All done"

gen = gen_with_return()
print(next(gen))  # 1
print(next(gen))  # 2
try:
    next(gen)
except StopIteration as e:
    print(e.value)  # "All done"
```

### Coroutines with Generators

```python
# Generator-based coroutine (old style, before async/await)
def coroutine():
    print("Coroutine started")
    while True:
        value = yield
        print(f"Received: {value}")

co = coroutine()
next(co)           # Initialize
co.send("Hello")   # Received: Hello
co.send("World")   # Received: World


# Pipeline pattern
def producer(target):
    """Send values to target."""
    for i in range(5):
        target.send(i)
    target.close()

def processor(target):
    """Process and forward to target."""
    try:
        while True:
            value = yield
            target.send(value * 2)
    except GeneratorExit:
        target.close()

def consumer():
    """Final consumer."""
    try:
        while True:
            value = yield
            print(f"Consumed: {value}")
    except GeneratorExit:
        print("Consumer closed")

# Build pipeline
cons = consumer()
next(cons)

proc = processor(cons)
next(proc)

producer(proc)
# Consumed: 0, 2, 4, 6, 8
```

---

## 3. Generator Expressions

```python
# Generator expression syntax (like list comprehension but lazy)
squares_gen = (x**2 for x in range(10))
squares_list = [x**2 for x in range(10)]

print(type(squares_gen))   # <class 'generator'>
print(type(squares_list))  # <class 'list'>


# Memory efficient for large datasets
# Bad: Creates entire list in memory
sum([x**2 for x in range(10**7)])

# Good: Processes one at a time
sum(x**2 for x in range(10**7))


# Conditional generator expression
evens = (x for x in range(20) if x % 2 == 0)
print(list(evens))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


# Nested generator expression
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = (num for row in matrix for num in row)
print(list(flattened))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]


# Generator expression with function call
def process(x):
    return x * 2

results = (process(x) for x in range(5))
print(list(results))  # [0, 2, 4, 6, 8]


# Chaining generators
nums = range(10)
step1 = (x**2 for x in nums)
step2 = (x + 1 for x in step1)
step3 = (x for x in step2 if x > 10)
print(list(step3))  # [17, 26, 37, 50, 65, 82]


# Generator expression as function argument
print(sum(x**2 for x in range(10)))  # No extra parentheses needed
print(max(len(word) for word in ["apple", "banana", "cherry"]))
```

---

## 4. itertools Module

### Infinite Iterators

```python
import itertools

# count(start, step) - infinite counter
for i in itertools.count(10, 2):
    print(i)  # 10, 12, 14, 16, ...
    if i > 20:
        break

# cycle(iterable) - infinite cycling
colors = itertools.cycle(['red', 'green', 'blue'])
for i, color in enumerate(colors):
    print(color)
    if i > 5:
        break

# repeat(elem, n) - repeat element n times (or forever)
for x in itertools.repeat('hello', 3):
    print(x)  # hello, hello, hello
```

### Finite Iterators

```python
import itertools

# chain(*iterables) - combine iterables
combined = itertools.chain([1, 2], [3, 4], [5, 6])
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# chain.from_iterable(iterable) - flatten one level
nested = [[1, 2], [3, 4], [5, 6]]
flat = itertools.chain.from_iterable(nested)
print(list(flat))  # [1, 2, 3, 4, 5, 6]


# islice(iterable, start, stop, step) - slice iterator
result = itertools.islice(range(100), 10, 20, 2)
print(list(result))  # [10, 12, 14, 16, 18]


# takewhile(predicate, iterable) - take while condition true
result = itertools.takewhile(lambda x: x < 5, range(10))
print(list(result))  # [0, 1, 2, 3, 4]


# dropwhile(predicate, iterable) - drop while condition true
result = itertools.dropwhile(lambda x: x < 5, range(10))
print(list(result))  # [5, 6, 7, 8, 9]


# filterfalse(predicate, iterable) - opposite of filter
result = itertools.filterfalse(lambda x: x % 2, range(10))
print(list(result))  # [0, 2, 4, 6, 8]


# compress(data, selectors) - filter by selector
result = itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1])
print(list(result))  # ['A', 'C', 'E', 'F']


# zip_longest(*iterables, fillvalue=None)
result = itertools.zip_longest([1, 2, 3], ['a', 'b'], fillvalue='-')
print(list(result))  # [(1, 'a'), (2, 'b'), (3, '-')]


# accumulate(iterable, func) - cumulative results
result = itertools.accumulate([1, 2, 3, 4, 5])
print(list(result))  # [1, 3, 6, 10, 15] (cumulative sum)

result = itertools.accumulate([1, 2, 3, 4, 5], lambda x, y: x * y)
print(list(result))  # [1, 2, 6, 24, 120] (cumulative product)


# groupby(iterable, key) - group consecutive elements
data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('a', 5)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# a [('a', 1), ('a', 2)]
# b [('b', 3), ('b', 4)]
# a [('a', 5)]  # Note: only consecutive!

# Sort first for complete grouping
data_sorted = sorted(data, key=lambda x: x[0])
for key, group in itertools.groupby(data_sorted, key=lambda x: x[0]):
    print(key, list(group))


# tee(iterable, n) - create n independent iterators
it1, it2, it3 = itertools.tee(range(5), 3)
print(list(it1))  # [0, 1, 2, 3, 4]
print(list(it2))  # [0, 1, 2, 3, 4]
print(list(it3))  # [0, 1, 2, 3, 4]
```

### Combinatoric Iterators

```python
import itertools

# permutations(iterable, r) - all orderings
result = itertools.permutations('ABC', 2)
print(list(result))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]


# combinations(iterable, r) - without repetition
result = itertools.combinations('ABC', 2)
print(list(result))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]


# combinations_with_replacement(iterable, r) - with repetition
result = itertools.combinations_with_replacement('ABC', 2)
print(list(result))
# [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]


# product(*iterables, repeat) - Cartesian product
result = itertools.product('AB', '12')
print(list(result))
# [('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')]

result = itertools.product('AB', repeat=2)
print(list(result))
# [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]


# Practical: Generate all possible passwords
import string
chars = string.ascii_lowercase
passwords = itertools.product(chars, repeat=3)
# All 3-letter lowercase combinations
```

### Practical itertools Recipes

```python
import itertools
from collections import deque

# Sliding window
def sliding_window(iterable, n):
    it = iter(iterable)
    window = deque(itertools.islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

print(list(sliding_window([1, 2, 3, 4, 5], 3)))
# [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


# Pairwise (Python 3.10+ has itertools.pairwise)
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

print(list(pairwise([1, 2, 3, 4, 5])))
# [(1, 2), (2, 3), (3, 4), (4, 5)]


# Chunked iteration
def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

print(list(chunked(range(10), 3)))
# [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


# Unique elements preserving order
def unique_everseen(iterable, key=None):
    seen = set()
    for element in iterable:
        k = key(element) if key else element
        if k not in seen:
            seen.add(k)
            yield element

print(list(unique_everseen([1, 2, 1, 3, 2, 4])))
# [1, 2, 3, 4]


# Round-robin
def roundrobin(*iterables):
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next_func in nexts:
                yield next_func()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))

print(list(roundrobin('ABC', '12', 'xyz')))
# ['A', '1', 'x', 'B', '2', 'y', 'C', 'z']
```

---

## 5. Context Managers

### Using Context Managers

```python
# Basic with statement
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed

# Multiple context managers
with open('in.txt') as fin, open('out.txt', 'w') as fout:
    fout.write(fin.read())

# Python 3.10+ allows parentheses
with (
    open('file1.txt') as f1,
    open('file2.txt') as f2,
    open('file3.txt') as f3,
):
    pass

# Common built-in context managers
import threading
import decimal

with threading.Lock():
    # Thread-safe critical section
    pass

with decimal.localcontext() as ctx:
    ctx.prec = 50
    # High precision calculations
```

### Class-Based Context Managers

```python
class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"Elapsed: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

with Timer() as t:
    sum(range(1000000))
# Elapsed: 0.0234 seconds


class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self, host, database):
        self.host = host
        self.database = database
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.database}@{self.host}")
        self.connection = f"Connection({self.host}, {self.database})"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.connection = None
        
        # Exception handling
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
            # Return True to suppress exception
            # Return False to propagate exception
        return False


class SuppressException:
    """Context manager that suppresses specific exceptions."""
    
    def __init__(self, *exceptions):
        self.exceptions = exceptions
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            return True  # Suppress
        return False  # Propagate

with SuppressException(ZeroDivisionError):
    result = 1 / 0  # Suppressed!
print("Continued execution")
```

### Generator-Based Context Managers

```python
from contextlib import contextmanager

@contextmanager
def timer():
    """Timer using contextmanager decorator."""
    import time
    start = time.perf_counter()
    try:
        yield  # Code block runs here
    finally:
        end = time.perf_counter()
        print(f"Elapsed: {end - start:.4f} seconds")

with timer():
    sum(range(1000000))


@contextmanager
def change_directory(path):
    """Temporarily change directory."""
    import os
    original = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)

with change_directory('/tmp'):
    # Now in /tmp
    pass
# Back to original directory


@contextmanager
def temporary_attribute(obj, name, value):
    """Temporarily set an attribute."""
    original = getattr(obj, name, None)
    has_attr = hasattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if has_attr:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


@contextmanager
def managed_resource():
    """Yielding a resource."""
    print("Acquiring resource")
    resource = "Resource"
    try:
        yield resource
    finally:
        print("Releasing resource")

with managed_resource() as r:
    print(f"Using {r}")
# Acquiring resource
# Using Resource
# Releasing resource
```

### contextlib Utilities

```python
from contextlib import (
    contextmanager, closing, suppress, 
    redirect_stdout, redirect_stderr,
    ExitStack, nullcontext
)
import io

# closing - ensure close() is called
class Resource:
    def close(self):
        print("Resource closed")

with closing(Resource()) as r:
    pass
# Resource closed


# suppress - suppress specific exceptions
with suppress(FileNotFoundError):
    open('nonexistent.txt')
print("Continued")  # No exception raised


# redirect_stdout/redirect_stderr
f = io.StringIO()
with redirect_stdout(f):
    print("This goes to StringIO")
print(f.getvalue())  # "This goes to StringIO\n"


# ExitStack - dynamic context manager management
with ExitStack() as stack:
    files = [
        stack.enter_context(open(f'file{i}.txt', 'w'))
        for i in range(3)
    ]
    # All files automatically closed on exit

# ExitStack for conditional contexts
with ExitStack() as stack:
    if condition:
        db = stack.enter_context(database_connection())
    if another_condition:
        file = stack.enter_context(open('file.txt'))


# nullcontext - no-op context manager (Python 3.7+)
cm = open('file.txt') if use_file else nullcontext(default_value)
with cm as value:
    process(value)


# Reentrant context managers
from contextlib import contextmanager

@contextmanager
def reentrant_lock():
    """Context manager that can be nested."""
    import threading
    lock = threading.RLock()  # Reentrant lock
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
```

### Async Context Managers

```python
import asyncio

class AsyncResource:
    """Async context manager."""
    
    async def __aenter__(self):
        print("Async acquiring")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Async releasing")
        await asyncio.sleep(0.1)
        return False

async def main():
    async with AsyncResource() as r:
        print("Using resource")

asyncio.run(main())


# Using asynccontextmanager
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer():
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"Elapsed: {time.perf_counter() - start:.4f}s")

async def main():
    async with async_timer():
        await asyncio.sleep(1)
```

---

## 6. Interview Questions

```python
# Q1: What's the difference between iterator and iterable?
"""
Iterable: Object with __iter__() that returns an iterator
- Examples: list, tuple, dict, str, set, file

Iterator: Object with __iter__() and __next__()
- __iter__() returns self
- __next__() returns next value or raises StopIteration
- Can only be traversed once
"""


# Q2: What's the difference between generator and iterator?
"""
Generator: A special kind of iterator created using:
- Generator function (def with yield)
- Generator expression (x for x in ...)

Advantages over regular iterators:
- More concise syntax
- Automatic state management
- Lazy evaluation
- Memory efficient
"""


# Q3: When to use generators?
"""
Use generators when:
- Processing large datasets
- Infinite sequences
- Memory is a concern
- Lazy evaluation needed
- Pipeline processing

Don't use when:
- Need random access
- Need to iterate multiple times
- Need length before iteration
"""


# Q4: Explain yield vs return
"""
return: Terminates function, returns value
yield: Pauses function, returns value, can be resumed

def with_return():
    return 1
    return 2  # Never reached

def with_yield():
    yield 1
    yield 2  # Will be reached on second next()
"""


# Q5: What is yield from?
"""
yield from delegates to another generator/iterable.
"""
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)  # Delegate
        else:
            yield item


# Q6: How does context manager work?
"""
Context manager protocol:
__enter__(): Called on entering 'with' block, return value assigned to 'as'
__exit__(exc_type, exc_val, exc_tb): Called on exiting, handles cleanup
- Return True to suppress exception
- Return False to propagate exception
"""


# Q7: When to use contextmanager decorator vs class?
"""
Use @contextmanager:
- Simple setup/teardown logic
- Single use cases
- Quick implementation

Use class:
- Complex state management
- Need to be reusable
- Need access to both __enter__ and __exit__ returns
- Need exception handling granularity
"""


# Q8: What's the difference between itertools.chain and itertools.chain.from_iterable?
chain([1,2], [3,4])           # Takes multiple iterables as args
chain.from_iterable([[1,2], [3,4]])  # Takes single iterable of iterables


# Q9: How to create infinite iterator?
"""
1. Class with __next__ that never raises StopIteration
2. Generator with infinite loop
3. itertools: count, cycle, repeat
"""
def infinite():
    n = 0
    while True:
        yield n
        n += 1


# Q10: What happens if generator is not fully consumed?
"""
Generator object is garbage collected.
- If generator has try/finally, finally runs on GC
- Can explicitly close with gen.close()
- Resources in finally block are cleaned up
"""
def gen():
    try:
        yield 1
        yield 2
    finally:
        print("Cleanup")

g = gen()
next(g)  # 1
del g    # "Cleanup" (when garbage collected)
```
