# Functions, Closures & Decorators - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Functions are first-class objects in Python — they can be passed, returned, and stored. Closures capture enclosing scope; decorators wrap functions to add behavior without modifying source code.

### Closures — The Foundation of Decorators

```python
# A closure: inner function captures outer variable even after outer returns
def make_multiplier(n):
    def multiply(x):
        return x * n    # 'n' is captured from enclosing scope
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))  # 10
print(triple(5))  # 15

# Classic closure pitfall: loop variable captured by reference, not value
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])  # [2, 2, 2]  <- all capture same 'i'

# Fix: capture by value using default argument
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])  # [0, 1, 2]  ✓
```

### Decorator Pattern — Step by Step

```python
import functools
import time

# A decorator is a function that takes a function and returns a function
def timer(func):
    @functools.wraps(func)   # preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)    # call original function
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer                    # equivalent to: my_func = timer(my_func)
def my_func(n):
    return sum(range(n))

my_func(1_000_000)        # automatically times and prints

# Decorator with arguments (factory pattern)
def retry(max_attempts=3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}, retrying...")
        return wrapper
    return decorator

@retry(max_attempts=3, exceptions=(ConnectionError,))
def fetch_data(url):
    pass   # actual implementation
```

### Key Functional Tools

```python
from functools import partial, lru_cache, reduce

# lru_cache: memoization decorator (caches results)
@lru_cache(maxsize=128)  # None = unlimited cache
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)   # O(n) with cache, O(2^n) without

# partial: fix some arguments of a function
def power(base, exp): return base ** exp
square = partial(power, exp=2)  # fix exp=2
print(square(5))  # 25

# Use *args and **kwargs for flexible wrappers
def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):   # accepts any signature
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper
```

### 🚨 Top Interview Pitfalls
- Forgetting `@functools.wraps(func)` in decorators — without it, `func.__name__` becomes `wrapper`, breaking introspection
- Loop variable capture in closures: `lambda: i` captures reference to `i`, not its value at creation time
- `nonlocal` vs `global`: use `nonlocal` to modify enclosing scope variable; `global` for module-level
- Decorator order matters: `@decorator1 @decorator2 def f()` applies `decorator2` first, then `decorator1`

---

## Table of Contents
1. [Function Fundamentals](#function-fundamentals)
2. [Closures & Scope](#closures--scope)
3. [Decorator Patterns](#decorator-patterns)
4. [Advanced Decorators](#advanced-decorators)
5. [Functools Module](#functools-module)
6. [Interview Questions](#interview-questions)

---

## 1. Function Fundamentals

### Functions as First-Class Objects

```python
# Functions are objects with attributes
def greet(name):
    """Greet someone by name."""
    return f"Hello, {name}!"

# Functions have attributes
print(greet.__name__)       # 'greet'
print(greet.__doc__)        # 'Greet someone by name.'
print(greet.__module__)     # '__main__'
print(type(greet))          # <class 'function'>

# Assign to variable
say_hello = greet
print(say_hello("World"))   # "Hello, World!"

# Store in data structures
functions = [greet, str.upper, len]
for f in functions:
    print(f.__name__)

# Pass as argument
def apply_func(func, value):
    return func(value)

result = apply_func(greet, "Alice")  # "Hello, Alice!"

# Return from function
def get_greeter(greeting):
    def greeter(name):
        return f"{greeting}, {name}!"
    return greeter

casual_greet = get_greeter("Hey")
formal_greet = get_greeter("Good evening")
print(casual_greet("Bob"))   # "Hey, Bob!"
print(formal_greet("Sir"))   # "Good evening, Sir!"
```

### Arguments Deep Dive

```python
# Positional and keyword arguments
def func(pos1, pos2, /, pos_or_kw, *, kw_only, **kwargs):
    """
    pos1, pos2: Positional-only (Python 3.8+)
    pos_or_kw: Can be either
    kw_only: Keyword-only (after *)
    **kwargs: Additional keyword arguments
    """
    pass

func(1, 2, 3, kw_only=4)           # Valid
func(1, 2, pos_or_kw=3, kw_only=4) # Valid
# func(pos1=1, pos2=2, ...)        # Error: pos1, pos2 are positional-only


# *args and **kwargs
def flexible(*args, **kwargs):
    print(f"args: {args}")      # Tuple of positional args
    print(f"kwargs: {kwargs}")  # Dict of keyword args

flexible(1, 2, 3, a=4, b=5)
# args: (1, 2, 3)
# kwargs: {'a': 4, 'b': 5}


# Unpacking in function calls
def add(a, b, c):
    return a + b + c

args = [1, 2, 3]
kwargs = {'a': 1, 'b': 2, 'c': 3}

add(*args)      # Unpack list as positional args
add(**kwargs)   # Unpack dict as keyword args


# Default arguments (evaluated once at definition!)
def bad_default(items=[]):
    items.append(1)
    return items

print(bad_default())  # [1]
print(bad_default())  # [1, 1] - Same list!

# Fix: Use None
def good_default(items=None):
    if items is None:
        items = []
    items.append(1)
    return items


# Default values evaluated at definition
import time

def timestamp(t=time.time()):  # Captured once!
    return t

print(timestamp())  # Same value
time.sleep(1)
print(timestamp())  # Same value!

# Fix: Use None and compute in function
def timestamp(t=None):
    if t is None:
        t = time.time()
    return t
```

### Lambda Functions

```python
# Anonymous functions
square = lambda x: x ** 2
add = lambda x, y: x + y
identity = lambda x: x

# Common with higher-order functions
numbers = [3, 1, 4, 1, 5, 9]
sorted(numbers, key=lambda x: -x)         # Descending
list(filter(lambda x: x > 3, numbers))    # [4, 5, 9]
list(map(lambda x: x * 2, numbers))       # [6, 2, 8, 2, 10, 18]

# With reduce
from functools import reduce
reduce(lambda acc, x: acc + x, numbers)   # Sum: 23
reduce(lambda acc, x: acc * x, numbers)   # Product

# Sorting complex data
people = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]
sorted(people, key=lambda p: p['age'])
sorted(people, key=lambda p: (p['age'], p['name']))  # Multiple keys


# Lambda limitations
# - Single expression only (no statements)
# - No assignments
# - No type annotations
# - Harder to debug (no name)

# When to use lambda:
# - Simple one-liners
# - Inline with higher-order functions
# - Callbacks

# When NOT to use:
# - Complex logic
# - Reused in multiple places
# - Need debugging
```

### Function Annotations

```python
# Type hints (annotations)
def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}! " * times).strip()

# Annotations are stored but not enforced
print(greet.__annotations__)
# {'name': <class 'str'>, 'times': <class 'int'>, 'return': <class 'str'>}

greet(123, "hello")  # Python doesn't check - runs (incorrectly)


# Complex type hints
from typing import List, Dict, Optional, Union, Callable, TypeVar

def process_items(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items), "count": len(items)}

def find_user(id: int) -> Optional[str]:  # str | None
    return None

def handle_data(data: Union[str, bytes]) -> None:
    pass

# Python 3.10+ syntax
def handle_data(data: str | bytes) -> None:
    pass

def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)


# TypeVar for generics
T = TypeVar('T')

def first(items: List[T]) -> T:
    return items[0]

# Use with mypy for static type checking
# mypy script.py
```

---

## 2. Closures & Scope

### LEGB Rule

```python
"""
Python variable lookup order (LEGB):
L - Local: Inside current function
E - Enclosing: Inside enclosing function(s)
G - Global: Module level
B - Built-in: Python built-ins
"""

# Built-in
print(len([1, 2, 3]))  # len is built-in

# Global
global_var = "I'm global"

def outer():
    # Enclosing
    enclosing_var = "I'm enclosing"
    
    def inner():
        # Local
        local_var = "I'm local"
        
        print(local_var)      # L - Local
        print(enclosing_var)  # E - Enclosing
        print(global_var)     # G - Global
        print(len)            # B - Built-in
    
    inner()

outer()


# Shadowing
len = 10  # Shadows built-in len
# len([1, 2, 3])  # TypeError: 'int' object is not callable
del len   # Restore access to built-in


# global keyword
counter = 0

def increment():
    global counter  # Modify global variable
    counter += 1

increment()
print(counter)  # 1


# nonlocal keyword (for closures)
def outer():
    count = 0
    
    def inner():
        nonlocal count  # Modify enclosing variable
        count += 1
        return count
    
    return inner

counter = outer()
print(counter())  # 1
print(counter())  # 2
```

### Closures

```python
# Closure: Function that captures variables from enclosing scope
def multiplier(factor):
    def multiply(x):
        return x * factor  # Captures 'factor'
    return multiply

double = multiplier(2)
triple = multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# Examining closures
print(double.__closure__)
print(double.__closure__[0].cell_contents)  # 2


# Practical closure: Counter
def make_counter(start=0):
    count = [start]  # Use list to allow modification
    
    def counter():
        count[0] += 1
        return count[0]
    
    return counter

# Or with nonlocal
def make_counter(start=0):
    count = start
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    return counter

c1 = make_counter()
c2 = make_counter(10)
print(c1(), c1(), c1())  # 1, 2, 3
print(c2(), c2())        # 11, 12


# Closure for configuration
def create_logger(prefix):
    def log(message):
        print(f"[{prefix}] {message}")
    return log

error_log = create_logger("ERROR")
info_log = create_logger("INFO")

error_log("Something went wrong")  # [ERROR] Something went wrong
info_log("Processing complete")    # [INFO] Processing complete


# Late binding gotcha
functions = []
for i in range(3):
    functions.append(lambda: i)

print([f() for f in functions])  # [2, 2, 2] - All captured final i!

# Fix: Capture value immediately
functions = []
for i in range(3):
    functions.append(lambda i=i: i)  # Default argument captures value

print([f() for f in functions])  # [0, 1, 2]


# Closure with mutable state
def make_averager():
    series = []
    
    def averager(new_value):
        series.append(new_value)
        return sum(series) / len(series)
    
    return averager

avg = make_averager()
print(avg(10))   # 10.0
print(avg(20))   # 15.0
print(avg(30))   # 20.0
```

---

## 3. Decorator Patterns

### Basic Decorator

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    """Greet someone."""
    return f"Hello, {name}!"

# Equivalent to: say_hello = my_decorator(say_hello)

print(say_hello("Alice"))
# Before function call
# After function call
# Hello, Alice!

# Metadata preserved by @functools.wraps
print(say_hello.__name__)  # 'say_hello' (not 'wrapper')
print(say_hello.__doc__)   # 'Greet someone.'


# Without @functools.wraps
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def greet(name):
    """Greet someone."""
    pass

print(greet.__name__)  # 'wrapper' - Wrong!
print(greet.__doc__)   # None - Lost!
```

### Decorator with Arguments

```python
# Decorator that accepts arguments
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!


# Decorator with optional arguments
def repeat(func=None, times=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    
    if func is not None:
        return decorator(func)  # Called without arguments
    return decorator            # Called with arguments

@repeat           # No parentheses - uses default times=2
def greet1(name):
    print(f"Hello, {name}!")

@repeat(times=3)  # With arguments
def greet2(name):
    print(f"Hello, {name}!")
```

### Class-Based Decorators

```python
# Using __call__ method
class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call #{self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # Call #1, Hello!
say_hello()  # Call #2, Hello!
print(say_hello.count)  # 2


# Class decorator with arguments
class Retry:
    def __init__(self, max_attempts=3, exceptions=(Exception,)):
        self.max_attempts = max_attempts
        self.exceptions = exceptions
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    if attempt == self.max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}")
        return wrapper

@Retry(max_attempts=3, exceptions=(ValueError, ConnectionError))
def risky_operation():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success"
```

### Method Decorators

```python
# Decorating methods
class Calculator:
    def __init__(self):
        self.history = []
    
    @staticmethod
    def validate_numbers(*args):
        return all(isinstance(a, (int, float)) for a in args)
    
    @classmethod
    def from_string(cls, expression):
        """Create calculator and evaluate expression."""
        calc = cls()
        # Parse and evaluate
        return calc
    
    @property
    def last_result(self):
        return self.history[-1] if self.history else None
    
    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result


# Custom method decorator
def log_method(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        print(f"{class_name}.{func.__name__} called with {args}, {kwargs}")
        return func(self, *args, **kwargs)
    return wrapper

class MyClass:
    @log_method
    def process(self, data):
        return data.upper()

obj = MyClass()
obj.process("hello")  # MyClass.process called with ('hello',), {}
```

---

## 4. Advanced Decorators

### Practical Decorators

```python
import functools
import time
from typing import Callable

# Timer decorator
def timer(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper


# Debug decorator
def debug(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper


# Cache/memoize decorator
def memoize(func: Callable) -> Callable:
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create hashable key
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    wrapper.clear_cache = cache.clear
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# Singleton decorator
def singleton(cls):
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Initializing database connection")

db1 = Database()  # Initializing database connection
db2 = Database()  # No output - same instance
print(db1 is db2)  # True


# Validate arguments decorator
def validate_types(**type_hints):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function parameters
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            
            for param, value in bound.arguments.items():
                if param in type_hints:
                    expected_type = type_hints[param]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{param} must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

create_user("Alice", 30)      # OK
# create_user("Alice", "30")  # TypeError


# Deprecated decorator
import warnings

def deprecated(message=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated("Use new_function() instead")
def old_function():
    pass
```

### Stacking Decorators

```python
# Decorators are applied bottom-up
@decorator1
@decorator2
@decorator3
def func():
    pass

# Equivalent to:
func = decorator1(decorator2(decorator3(func)))


# Example: Combining timer and debug
@timer
@debug
def slow_function(n):
    time.sleep(0.1)
    return n * 2

slow_function(5)
# Calling slow_function(5)
# slow_function returned 10
# slow_function took 0.1001 seconds


# Order matters!
@debug
@timer
def slow_function(n):
    time.sleep(0.1)
    return n * 2

slow_function(5)
# slow_function took 0.1001 seconds
# Calling wrapper(5)  # Wrong name!
# wrapper returned 10
```

### Decorator for Both Functions and Classes

```python
def add_logging(obj):
    """Decorator that works on both functions and classes."""
    if isinstance(obj, type):
        # It's a class
        original_init = obj.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            print(f"Creating instance of {obj.__name__}")
            original_init(self, *args, **kwargs)
        
        obj.__init__ = new_init
        return obj
    else:
        # It's a function
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            print(f"Calling {obj.__name__}")
            return obj(*args, **kwargs)
        return wrapper

@add_logging
class MyClass:
    def __init__(self, value):
        self.value = value

@add_logging
def my_function(x):
    return x * 2

obj = MyClass(10)  # Creating instance of MyClass
my_function(5)     # Calling my_function
```

---

## 5. Functools Module

```python
import functools

# lru_cache - Least Recently Used cache
@functools.lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(100)  # Fast due to caching
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)
fibonacci.cache_clear()


# cache - Unbounded cache (Python 3.9+)
@functools.cache
def expensive_computation(x):
    return x ** 2


# cached_property (Python 3.8+)
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @functools.cached_property
    def area(self):
        print("Computing area...")
        return 3.14159 * self.radius ** 2

c = Circle(5)
print(c.area)  # Computing area... 78.53975
print(c.area)  # 78.53975 (cached, no recomputation)


# partial - Pre-fill function arguments
def power(base, exponent):
    return base ** exponent

square = functools.partial(power, exponent=2)
cube = functools.partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Useful with map/filter
from functools import partial

int_from_binary = partial(int, base=2)
print(int_from_binary("1010"))  # 10


# partialmethod - partial for methods
class Cell:
    def __init__(self):
        self._alive = False
    
    def set_state(self, state):
        self._alive = state
    
    set_alive = functools.partialmethod(set_state, True)
    set_dead = functools.partialmethod(set_state, False)

cell = Cell()
cell.set_alive()
print(cell._alive)  # True


# reduce - Cumulative operation
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda acc, x: acc * x, numbers)  # 120
sum_val = reduce(lambda acc, x: acc + x, numbers)  # 15

# With initial value
result = reduce(lambda acc, x: acc + x, numbers, 100)  # 115


# wraps - Preserve function metadata (shown earlier)
def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# total_ordering - Generate comparison methods
@functools.total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __lt__(self, other):
        return self.age < other.age

# Now Person has __le__, __gt__, __ge__ automatically
p1 = Person("Alice", 30)
p2 = Person("Bob", 25)
print(p1 > p2)   # True
print(p1 >= p2)  # True


# singledispatch - Function overloading by type
@functools.singledispatch
def process(arg):
    print(f"Default: {arg}")

@process.register(int)
def _(arg):
    print(f"Integer: {arg}")

@process.register(list)
def _(arg):
    print(f"List with {len(arg)} items")

@process.register(str)
def _(arg):
    print(f"String: {arg.upper()}")

process(42)        # Integer: 42
process([1, 2, 3]) # List with 3 items
process("hello")   # String: HELLO
process(3.14)      # Default: 3.14
```

---

## 6. Interview Questions

```python
# Q1: What is a closure?
"""
A closure is a function that captures variables from its enclosing scope.
The captured variables persist even after the enclosing function returns.
"""
def make_multiplier(n):
    def multiplier(x):
        return x * n  # 'n' is captured
    return multiplier

double = make_multiplier(2)
print(double(5))  # 10


# Q2: Explain the difference between @staticmethod and @classmethod
class MyClass:
    class_var = "class"
    
    @staticmethod
    def static_method():
        # No access to class or instance
        return "static"
    
    @classmethod
    def class_method(cls):
        # Has access to class, not instance
        return cls.class_var
    
    def instance_method(self):
        # Has access to instance and class
        return self


# Q3: What is late binding in closures?
functions = [lambda: i for i in range(3)]
print([f() for f in functions])  # [2, 2, 2] - All use final i!

# Fix with default argument
functions = [lambda i=i: i for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]


# Q4: What does @functools.wraps do?
"""
@functools.wraps preserves the original function's metadata:
- __name__: Function name
- __doc__: Docstring
- __annotations__: Type hints
- __module__: Module name
- __qualname__: Qualified name
"""


# Q5: How to make a decorator with optional arguments?
def decorator(func=None, *, option=True):
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if option:
                print("Option enabled")
            return func(*args, **kwargs)
        return wrapper
    
    if func is not None:
        return actual_decorator(func)
    return actual_decorator

@decorator           # No args
def func1(): pass

@decorator(option=False)  # With args
def func2(): pass


# Q6: What is the LEGB rule?
"""
Variable lookup order:
L - Local (inside function)
E - Enclosing (enclosing function)
G - Global (module level)
B - Built-in (Python built-ins)
"""


# Q7: Explain global and nonlocal keywords
count = 0

def outer():
    outer_var = 0
    
    def inner():
        global count      # Access module-level variable
        nonlocal outer_var  # Access enclosing function variable
        count += 1
        outer_var += 1


# Q8: What is a decorator factory?
"""
A decorator factory is a function that returns a decorator.
Used when decorator needs to accept arguments.
"""
def repeat(n):              # Factory
    def decorator(func):    # Decorator
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


# Q9: How does @property work?
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

# @property creates a descriptor that intercepts attribute access


# Q10: What is functools.lru_cache?
"""
@lru_cache provides memoization with LRU (Least Recently Used) eviction.
- maxsize: Maximum cache size (default 128)
- typed: If True, cache f(3) and f(3.0) separately
- Provides cache_info() and cache_clear() methods
"""
@functools.lru_cache(maxsize=100, typed=True)
def expensive_function(n):
    return n ** 2
```
