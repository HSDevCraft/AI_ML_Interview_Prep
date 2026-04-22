# DSA & Python Interview Questions Bank

## DSA Questions

### Arrays & Strings

**Q1: Two Sum**
```python
# Given array of integers, return indices of two numbers that add up to target
# Time: O(n), Space: O(n)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**Q2: Best Time to Buy and Sell Stock**
```python
# Find max profit from single buy and sell
# Time: O(n), Space: O(1)
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

**Q3: Longest Substring Without Repeating Characters**
```python
# Time: O(n), Space: O(min(m,n)) where m is charset size
def length_of_longest_substring(s):
    char_index = {}
    max_len = start = 0
    
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_len = max(max_len, i - start + 1)
    
    return max_len
```

**Q4: Trapping Rain Water**
```python
# Time: O(n), Space: O(1)
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

**Q5: Merge Intervals**
```python
# Time: O(n log n), Space: O(n)
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged
```

---

### Linked Lists

**Q6: Reverse Linked List**
```python
# Iterative: Time O(n), Space O(1)
def reverse_list(head):
    prev = None
    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev

# Recursive: Time O(n), Space O(n)
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

**Q7: Merge K Sorted Lists**
```python
import heapq

def merge_k_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    curr = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

**Q8: LRU Cache**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

### Trees

**Q9: Validate Binary Search Tree**
```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and 
            is_valid_bst(root.right, root.val, max_val))
```

**Q10: Serialize and Deserialize Binary Tree**
```python
class Codec:
    def serialize(self, root):
        def dfs(node):
            if not node:
                return ['null']
            return [str(node.val)] + dfs(node.left) + dfs(node.right)
        return ','.join(dfs(root))
    
    def deserialize(self, data):
        vals = iter(data.split(','))
        
        def dfs():
            val = next(vals)
            if val == 'null':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()
```

**Q11: Binary Tree Maximum Path Sum**
```python
def max_path_sum(root):
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0
        
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        
        max_sum = max(max_sum, left + right + node.val)
        return max(left, right) + node.val
    
    dfs(root)
    return max_sum
```

---

### Graphs

**Q12: Number of Islands**
```python
def num_islands(grid):
    if not grid:
        return 0
    
    count = 0
    rows, cols = len(grid), len(grid[0])
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'  # Mark visited
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    
    return count
```

**Q13: Course Schedule (Cycle Detection)**
```python
def can_finish(num_courses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0
    
    while queue:
        course = queue.popleft()
        completed += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return completed == num_courses
```

**Q14: Word Ladder (BFS)**
```python
def ladder_length(begin_word, end_word, word_list):
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    
    queue = deque([(begin_word, 1)])
    
    while queue:
        word, length = queue.popleft()
        
        if word == end_word:
            return length
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set:
                    word_set.remove(new_word)
                    queue.append((new_word, length + 1))
    
    return 0
```

---

### Dynamic Programming

**Q15: Longest Increasing Subsequence**
```python
# O(n log n) solution
def length_of_lis(nums):
    tails = []
    
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    
    return len(tails)
```

**Q16: Word Break**
```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]
```

**Q17: Decode Ways**
```python
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]
```

---

## Python Interview Questions

### Q1: What is the difference between `is` and `==`?
```python
# == compares values (equality)
# is compares identity (same object in memory)

a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b  # True (same value)
a is b  # False (different objects)
a is c  # True (same object)

# Small integers (-5 to 256) are cached
x = 256
y = 256
x is y  # True (cached)

x = 257
y = 257
x is y  # False (not cached)
```

### Q2: Explain Python's GIL (Global Interpreter Lock)
```
- GIL is a mutex that protects access to Python objects
- Only one thread can execute Python bytecode at a time
- Prevents true parallelism in CPU-bound multithreaded programs
- I/O-bound tasks can still benefit from threading (GIL released during I/O)
- Use multiprocessing for CPU-bound parallelism
- NumPy and other C extensions can release GIL
```

### Q3: What are decorators? Write a timing decorator.
```python
import functools
import time

def timer(func):
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

### Q4: Explain generators and their benefits
```python
# Generators are iterators that generate values lazily
# Benefits: Memory efficient, lazy evaluation

def fibonacci_gen(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Memory comparison
list_fib = [x for x in fibonacci_gen(1000000)]  # Uses ~8MB
gen_fib = fibonacci_gen(1000000)  # Uses ~100 bytes

# Generator expressions
squares_gen = (x**2 for x in range(1000000))  # Lazy
squares_list = [x**2 for x in range(1000000)]  # Eager
```

### Q5: What is the difference between `deepcopy` and `copy`?
```python
import copy

original = [[1, 2], [3, 4]]

# Shallow copy - copies outer container, shares inner objects
shallow = copy.copy(original)
shallow[0][0] = 'X'
print(original)  # [['X', 2], [3, 4]] - affected!

# Deep copy - recursively copies all nested objects
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)
deep[0][0] = 'X'
print(original)  # [[1, 2], [3, 4]] - unaffected
```

### Q6: Explain `*args` and `**kwargs`
```python
def func(*args, **kwargs):
    print(f"args: {args}")      # Tuple of positional args
    print(f"kwargs: {kwargs}")  # Dict of keyword args

func(1, 2, 3, a=4, b=5)
# args: (1, 2, 3)
# kwargs: {'a': 4, 'b': 5}

# Unpacking
def add(a, b, c):
    return a + b + c

args = [1, 2, 3]
kwargs = {'a': 1, 'b': 2, 'c': 3}
add(*args)      # 6
add(**kwargs)   # 6
```

### Q7: What are context managers?
```python
# Context managers handle setup/cleanup automatically
# Implement __enter__ and __exit__ methods

class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # Don't suppress exceptions

# Using contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()
```

### Q8: Explain Python's MRO (Method Resolution Order)
```python
class A:
    def method(self):
        return 'A'

class B(A):
    def method(self):
        return 'B'

class C(A):
    def method(self):
        return 'C'

class D(B, C):
    pass

# MRO determines method lookup order
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# D -> B -> C -> A -> object (C3 linearization algorithm)
d = D()
d.method()  # 'B' (first in MRO after D)
```

### Q9: What are metaclasses?
```python
# Metaclasses are "classes of classes"
# They control class creation

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"

db1 = Database()
db2 = Database()
print(db1 is db2)  # True - same instance
```

### Q10: Explain `@staticmethod` vs `@classmethod`
```python
class MyClass:
    class_var = "class variable"
    
    def instance_method(self):
        # Has access to instance (self) and class
        return f"instance: {self}"
    
    @classmethod
    def class_method(cls):
        # Has access to class (cls), not instance
        return f"class: {cls.class_var}"
    
    @staticmethod
    def static_method():
        # No access to instance or class
        # Just a regular function in class namespace
        return "static"

# classmethod is often used for alternative constructors
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)

date = Date.from_string("2024-01-15")
```

### Q11: What are descriptors?
```python
# Descriptors customize attribute access
# Implement __get__, __set__, __delete__

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
    
    def __init__(self, age):
        self.age = age

p = Person(25)   # OK
p = Person(-5)   # Raises ValueError
```

### Q12: Explain async/await in Python
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulated I/O
    return f"Data from {url}"

async def main():
    # Sequential (slow)
    result1 = await fetch_data("url1")
    result2 = await fetch_data("url2")
    
    # Concurrent (fast)
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    
    # Create tasks
    task1 = asyncio.create_task(fetch_data("url1"))
    task2 = asyncio.create_task(fetch_data("url2"))
    await task1
    await task2

asyncio.run(main())

# Key points:
# - async def creates a coroutine
# - await pauses execution until awaitable completes
# - Event loop manages coroutine scheduling
# - Good for I/O-bound tasks, not CPU-bound
```

### Q13: Common Python gotchas
```python
# 1. Mutable default arguments
def bad(items=[]):
    items.append(1)
    return items

bad()  # [1]
bad()  # [1, 1] - Same list!

def good(items=None):
    if items is None:
        items = []
    items.append(1)
    return items

# 2. Late binding in closures
funcs = [lambda x: x * i for i in range(3)]
[f(2) for f in funcs]  # [4, 4, 4] - All use i=2!

# Fix: capture value
funcs = [lambda x, i=i: x * i for i in range(3)]
[f(2) for f in funcs]  # [0, 2, 4]

# 3. Modifying list while iterating
lst = [1, 2, 3, 4, 5]
for item in lst:
    if item % 2 == 0:
        lst.remove(item)  # Don't do this!

# Fix: iterate over copy or use comprehension
lst = [x for x in lst if x % 2 != 0]
```

---

## Complexity Analysis Questions

### Q: What's the time/space complexity?

```python
# O(n) time, O(1) space
def find_max(arr):
    max_val = arr[0]
    for x in arr:
        max_val = max(max_val, x)
    return max_val

# O(n²) time, O(1) space
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# O(n log n) time, O(n) space
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# O(2^n) time, O(n) space
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# O(n) time, O(n) space (with memoization)
@lru_cache
def fibonacci_memo(n):
    if n <= 1:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)
```
