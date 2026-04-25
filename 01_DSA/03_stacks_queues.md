# Stacks & Queues - Complete In-Depth Guide

## ⚡ Interview Quick Summary

> **Core insight**: Stack = LIFO (Last In First Out). Queue = FIFO (First In First Out). Monotonic stacks are the key pattern for "next greater/smaller element" problems.

### When to Use Each

```
Stack problems:           Queue problems:
  Balanced parentheses     BFS traversal
  Evaluate expressions     Level-order processing
  Next greater element     Sliding window maximum (deque)
  Function call tracing    Task scheduling
  Undo/redo               Producer-consumer
  Monotonic stack          Monotonic deque
```

### Monotonic Stack — Most Common Interview Pattern

```python
def next_greater_element(nums):
    """
    For each element, find the next element that is greater.
    Monotonic stack maintains elements in DECREASING order.
    When we find a greater element, it's the 'next greater' for stack top.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices, not values
    
    for i in range(n):
        # Pop elements smaller than current (current is their next greater)
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]  # nums[i] is next greater for nums[idx]
        stack.append(i)
    
    return result

# Examples of monotonic stack problems:
# Next Greater Element (above)
# Largest Rectangle in Histogram (increasing stack)
# Daily Temperatures (decreasing stack, find warmer day)
# Trapping Rain Water (can also use stack)
```

### Python Stack and Queue

```python
# STACK: use list (append/pop from same end)
stack = []
stack.append(1)    # push: O(1)
stack.append(2)
top = stack[-1]    # peek: O(1)
stack.pop()        # pop: O(1)

# QUEUE: use collections.deque (O(1) both ends)
from collections import deque
queue = deque()
queue.append(1)    # enqueue right: O(1)
queue.append(2)
queue.popleft()    # dequeue left: O(1)  ← list.pop(0) is O(n)!

# DEQUE (double-ended queue): both ends O(1)
dq = deque(maxlen=3)  # maxlen: auto-drops oldest when full
dq.appendleft(0)      # add to left
dq.popleft()          # remove from left
```

### 🚨 Top Interview Pitfalls
- Using `list.pop(0)` for queue dequeue — it's O(n)! Use `collections.deque.popleft()` which is O(1)
- Forgetting to handle **empty stack** before accessing `stack[-1]` — always check `if stack:` first
- Monotonic stack: decide INCREASING (for "next smaller") vs DECREASING (for "next greater") based on problem
- **Stack overflow** for recursive DFS on large graphs — convert to iterative with an explicit stack

---

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Stack Implementation](#stack-implementation)
3. [Queue Implementation](#queue-implementation)
4. [Monotonic Stack](#monotonic-stack)
5. [Monotonic Queue](#monotonic-queue)
6. [Stack-Based Parsing](#stack-based-parsing)
7. [BFS with Queue](#bfs-with-queue)
8. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Stack (LIFO - Last In First Out)

| Operation | Time | Description |
|-----------|------|-------------|
| push(x) | O(1) | Add element to top |
| pop() | O(1) | Remove and return top element |
| peek()/top() | O(1) | Return top element without removing |
| isEmpty() | O(1) | Check if stack is empty |
| size() | O(1) | Return number of elements |

**Use Cases:**
- Function call stack (recursion)
- Undo operations
- Expression evaluation
- Backtracking algorithms
- Parentheses matching

### Queue (FIFO - First In First Out)

| Operation | Time | Description |
|-----------|------|-------------|
| enqueue(x) | O(1) | Add element to rear |
| dequeue() | O(1) | Remove and return front element |
| front()/peek() | O(1) | Return front element without removing |
| isEmpty() | O(1) | Check if queue is empty |
| size() | O(1) | Return number of elements |

**Use Cases:**
- BFS traversal
- Task scheduling
- Buffering (producer-consumer)
- Print queue, message queue

---

## 2. Stack Implementation

### Using Python List

```python
class Stack:
    """Stack implementation using Python list."""
    
    def __init__(self):
        self._items = []
    
    def push(self, item) -> None:
        """Add item to top. O(1) amortized."""
        self._items.append(item)
    
    def pop(self):
        """Remove and return top item. O(1)."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        """Return top item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty. O(1)."""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Return number of items. O(1)."""
        return len(self._items)
    
    def __len__(self):
        return self.size()
    
    def __bool__(self):
        return not self.is_empty()
    
    def __repr__(self):
        return f"Stack({self._items})"


# Pythonic way - just use list directly
stack = []
stack.append(1)  # push
stack.append(2)
top = stack.pop()  # pop -> 2
top = stack[-1]    # peek
```

### Using Linked List

```python
class StackNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class LinkedStack:
    """Stack implementation using singly linked list.
    
    All operations are O(1) worst case (no amortization).
    """
    
    def __init__(self):
        self._top = None
        self._size = 0
    
    def push(self, item) -> None:
        """Add item to top. O(1)."""
        self._top = StackNode(item, self._top)
        self._size += 1
    
    def pop(self):
        """Remove and return top item. O(1)."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        val = self._top.val
        self._top = self._top.next
        self._size -= 1
        return val
    
    def peek(self):
        """Return top item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._top.val
    
    def is_empty(self) -> bool:
        return self._top is None
    
    def size(self) -> int:
        return self._size
```

### Min Stack

```python
class MinStack:
    """
    LeetCode 155: Stack with O(1) getMin operation.
    
    Key insight: Store (value, current_min) pairs.
    """
    
    def __init__(self):
        self.stack = []  # (value, min_so_far)
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self) -> None:
        self.stack.pop()
    
    def top(self) -> int:
        return self.stack[-1][0]
    
    def getMin(self) -> int:
        return self.stack[-1][1]


class MinStackOptimized:
    """
    Space-optimized: Only push to min_stack when new minimum.
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Stack of minimums
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push if new min or equal to current min
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]


class MaxStack:
    """
    LeetCode 716: Stack with O(1) getMax and O(log n) popMax.
    
    Uses two stacks + heap for efficient popMax.
    """
    
    def __init__(self):
        from collections import defaultdict
        import heapq
        
        self.stack = []  # (value, id)
        self.heap = []   # (-value, -id) for max heap
        self.removed = set()  # Set of removed ids
        self.id = 0
    
    def push(self, x: int) -> None:
        import heapq
        self.stack.append((x, self.id))
        heapq.heappush(self.heap, (-x, -self.id))
        self.id += 1
    
    def pop(self) -> int:
        self._clean_stack()
        val, id = self.stack.pop()
        self.removed.add(id)
        return val
    
    def top(self) -> int:
        self._clean_stack()
        return self.stack[-1][0]
    
    def peekMax(self) -> int:
        import heapq
        self._clean_heap()
        return -self.heap[0][0]
    
    def popMax(self) -> int:
        import heapq
        self._clean_heap()
        val, id = heapq.heappop(self.heap)
        self.removed.add(-id)
        return -val
    
    def _clean_stack(self):
        while self.stack and self.stack[-1][1] in self.removed:
            self.stack.pop()
    
    def _clean_heap(self):
        import heapq
        while self.heap and -self.heap[0][1] in self.removed:
            heapq.heappop(self.heap)
```

---

## 3. Queue Implementation

### Using collections.deque

```python
from collections import deque

class Queue:
    """Queue implementation using deque (recommended)."""
    
    def __init__(self):
        self._items = deque()
    
    def enqueue(self, item) -> None:
        """Add item to rear. O(1)."""
        self._items.append(item)
    
    def dequeue(self):
        """Remove and return front item. O(1)."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self._items.popleft()
    
    def front(self):
        """Return front item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self._items[0]
    
    def rear(self):
        """Return rear item without removing. O(1)."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)


# Pythonic way - use deque directly
queue = deque()
queue.append(1)     # enqueue
queue.append(2)
front = queue.popleft()  # dequeue -> 1
front = queue[0]    # peek front
```

### Circular Queue (Ring Buffer)

```python
class CircularQueue:
    """
    LeetCode 622: Fixed-size circular queue.
    
    Efficient use of fixed memory with wrap-around.
    """
    
    def __init__(self, k: int):
        self.data = [None] * k
        self.capacity = k
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enQueue(self, value: int) -> bool:
        """Insert element. Returns False if full."""
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.data[self.rear] = value
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        """Delete element. Returns False if empty."""
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def Front(self) -> int:
        """Get front element. Returns -1 if empty."""
        return -1 if self.isEmpty() else self.data[self.front]
    
    def Rear(self) -> int:
        """Get rear element. Returns -1 if empty."""
        return -1 if self.isEmpty() else self.data[self.rear]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


class CircularDeque:
    """
    LeetCode 641: Fixed-size double-ended queue.
    """
    
    def __init__(self, k: int):
        self.data = [None] * k
        self.capacity = k
        self.front = 0
        self.rear = k - 1
        self.size = 0
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        self.front = (self.front - 1) % self.capacity
        self.data[self.front] = value
        self.size += 1
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.data[self.rear] = value
        self.size += 1
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        self.rear = (self.rear - 1) % self.capacity
        self.size -= 1
        return True
    
    def getFront(self) -> int:
        return -1 if self.isEmpty() else self.data[self.front]
    
    def getRear(self) -> int:
        return -1 if self.isEmpty() else self.data[self.rear]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity
```

### Queue Using Two Stacks

```python
class QueueUsingStacks:
    """
    LeetCode 232: Implement queue using two stacks.
    
    Amortized O(1) for all operations.
    """
    
    def __init__(self):
        self.in_stack = []   # For push
        self.out_stack = []  # For pop/peek
    
    def push(self, x: int) -> None:
        """Push to rear. O(1)."""
        self.in_stack.append(x)
    
    def pop(self) -> int:
        """Pop from front. Amortized O(1)."""
        self._transfer()
        return self.out_stack.pop()
    
    def peek(self) -> int:
        """Peek front. Amortized O(1)."""
        self._transfer()
        return self.out_stack[-1]
    
    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack
    
    def _transfer(self) -> None:
        """Transfer from in_stack to out_stack if out is empty."""
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())


class StackUsingQueues:
    """
    LeetCode 225: Implement stack using queues.
    
    O(n) push, O(1) pop approach (or vice versa).
    """
    
    def __init__(self):
        from collections import deque
        self.queue = deque()
    
    def push(self, x: int) -> None:
        """Push element. O(n) - rotate queue to maintain stack order."""
        self.queue.append(x)
        # Rotate to put new element at front
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
    
    def pop(self) -> int:
        """Pop top element. O(1)."""
        return self.queue.popleft()
    
    def top(self) -> int:
        """Get top element. O(1)."""
        return self.queue[0]
    
    def empty(self) -> bool:
        return len(self.queue) == 0
```

---

## 4. Monotonic Stack

### Concept

A **monotonic stack** maintains elements in sorted order (increasing or decreasing).
Used for finding **next/previous greater/smaller element** problems.

```python
def next_greater_element(nums: list[int]) -> list[int]:
    """
    For each element, find the next greater element to its right.
    Time: O(n), Space: O(n)
    
    LeetCode 496, 503
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Indices of elements waiting for their next greater
    
    for i in range(n):
        # Pop elements smaller than current (current is their answer)
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result


def next_smaller_element(nums: list[int]) -> list[int]:
    """
    For each element, find the next smaller element to its right.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Decreasing stack
    
    for i in range(n):
        while stack and nums[i] < nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result


def previous_greater_element(nums: list[int]) -> list[int]:
    """
    For each element, find the previous greater element to its left.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Decreasing stack
    
    for i in range(n):
        while stack and nums[stack[-1]] <= nums[i]:
            stack.pop()
        
        if stack:
            result[i] = nums[stack[-1]]
        
        stack.append(i)
    
    return result


def previous_smaller_element(nums: list[int]) -> list[int]:
    """
    For each element, find the previous smaller element to its left.
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Increasing stack
    
    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        
        if stack:
            result[i] = nums[stack[-1]]
        
        stack.append(i)
    
    return result
```

### Classic Monotonic Stack Problems

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    """
    LeetCode 739: Days until warmer temperature.
    Time: O(n), Space: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Indices of days waiting for warmer day
    
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    
    return result


def largest_rectangle_histogram(heights: list[int]) -> int:
    """
    LeetCode 84: Largest rectangle in histogram.
    Time: O(n), Space: O(n)
    
    Key insight: For each bar, find the widest rectangle with that bar's height.
    Width = (next_smaller_index - prev_smaller_index - 1)
    """
    n = len(heights)
    stack = []
    max_area = 0
    
    for i in range(n + 1):
        # Use 0 as sentinel for the end
        h = heights[i] if i < n else 0
        
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            # Width: from previous smaller to current (exclusive)
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area


def maximal_rectangle(matrix: list[list[str]]) -> int:
    """
    LeetCode 85: Maximal rectangle in binary matrix.
    Time: O(m*n), Space: O(n)
    
    Key insight: Build histogram for each row and apply largest_rectangle.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0
    
    for i in range(m):
        # Update heights for current row
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        # Find largest rectangle with current histogram
        max_area = max(max_area, largest_rectangle_histogram(heights))
    
    return max_area


def trapping_rain_water_stack(height: list[int]) -> int:
    """
    LeetCode 42: Trapping rain water using monotonic stack.
    Time: O(n), Space: O(n)
    """
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            
            # Calculate water trapped
            width = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[bottom]
            water += width * bounded_height
        
        stack.append(i)
    
    return water


def sum_subarray_minimums(arr: list[int]) -> int:
    """
    LeetCode 907: Sum of minimum elements of all subarrays.
    Time: O(n), Space: O(n)
    
    For each element, count subarrays where it's the minimum.
    Count = (left_bound) * (right_bound)
    """
    MOD = 10**9 + 7
    n = len(arr)
    
    # left[i] = number of elements to the left that are >= arr[i]
    left = [0] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        left[i] = i - stack[-1] if stack else i + 1
        stack.append(i)
    
    # right[i] = number of elements to the right that are > arr[i]
    right = [0] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        right[i] = stack[-1] - i if stack else n - i
        stack.append(i)
    
    # Sum contribution of each element
    result = 0
    for i in range(n):
        result = (result + arr[i] * left[i] * right[i]) % MOD
    
    return result


def next_greater_element_circular(nums: list[int]) -> list[int]:
    """
    LeetCode 503: Next greater element in circular array.
    Time: O(n), Space: O(n)
    
    Trick: Iterate through array twice (or use modulo).
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Iterate twice to handle circular
    for i in range(2 * n):
        idx = i % n
        while stack and nums[idx] > nums[stack[-1]]:
            result[stack.pop()] = nums[idx]
        
        if i < n:  # Only push indices from first pass
            stack.append(idx)
    
    return result


def remove_k_digits(num: str, k: int) -> str:
    """
    LeetCode 402: Remove k digits to get smallest number.
    Time: O(n), Space: O(n)
    
    Greedy: Remove larger digits when a smaller digit comes.
    """
    stack = []
    
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    stack = stack[:-k] if k else stack
    
    # Remove leading zeros
    return ''.join(stack).lstrip('0') or '0'


def asteroid_collision(asteroids: list[int]) -> list[int]:
    """
    LeetCode 735: Simulate asteroid collisions.
    Time: O(n), Space: O(n)
    
    Positive = moving right, Negative = moving left.
    Collision happens when positive meets negative.
    """
    stack = []
    
    for asteroid in asteroids:
        alive = True
        
        while alive and stack and asteroid < 0 < stack[-1]:
            # Collision: compare sizes
            if stack[-1] < -asteroid:
                stack.pop()
            elif stack[-1] == -asteroid:
                stack.pop()
                alive = False
            else:
                alive = False
        
        if alive:
            stack.append(asteroid)
    
    return stack
```

---

## 5. Monotonic Queue (Deque)

### Sliding Window Maximum/Minimum

```python
def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    LeetCode 239: Maximum in each sliding window of size k.
    Time: O(n), Space: O(k)
    
    Maintain decreasing deque of indices.
    """
    from collections import deque
    
    result = []
    dq = deque()  # Indices, values are decreasing
    
    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (they can never be max)
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is formed
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def min_sliding_window(nums: list[int], k: int) -> list[int]:
    """Minimum in each sliding window of size k."""
    from collections import deque
    
    result = []
    dq = deque()  # Indices, values are increasing
    
    for i, num in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        while dq and nums[dq[-1]] > num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def shortest_subarray_with_sum_at_least_k(nums: list[int], k: int) -> int:
    """
    LeetCode 862: Shortest subarray with sum at least K.
    Time: O(n), Space: O(n)
    
    This is tricky because array can have negative numbers.
    Use monotonic deque on prefix sums.
    """
    from collections import deque
    
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    result = float('inf')
    dq = deque()  # Indices with increasing prefix sums
    
    for i in range(n + 1):
        # Check if current prefix - smallest prefix >= k
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())
        
        # Maintain increasing order
        while dq and prefix[i] <= prefix[dq[-1]]:
            dq.pop()
        
        dq.append(i)
    
    return result if result != float('inf') else -1


def longest_subarray_with_diff_limit(nums: list[int], limit: int) -> int:
    """
    LeetCode 1438: Longest subarray where max - min <= limit.
    Time: O(n), Space: O(n)
    
    Use two deques: one for max, one for min.
    """
    from collections import deque
    
    max_dq = deque()  # Decreasing
    min_dq = deque()  # Increasing
    left = 0
    result = 0
    
    for right, num in enumerate(nums):
        while max_dq and nums[max_dq[-1]] < num:
            max_dq.pop()
        while min_dq and nums[min_dq[-1]] > num:
            min_dq.pop()
        
        max_dq.append(right)
        min_dq.append(right)
        
        # Shrink window if condition violated
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            left += 1
            if max_dq[0] < left:
                max_dq.popleft()
            if min_dq[0] < left:
                min_dq.popleft()
        
        result = max(result, right - left + 1)
    
    return result
```

---

## 6. Stack-Based Parsing

### Expression Evaluation

```python
def valid_parentheses(s: str) -> bool:
    """
    LeetCode 20: Check if parentheses are valid.
    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0


def min_add_to_make_valid(s: str) -> int:
    """
    LeetCode 921: Minimum additions to make parentheses valid.
    Time: O(n), Space: O(1)
    """
    open_count = 0  # Unmatched '('
    close_needed = 0  # Unmatched ')'
    
    for char in s:
        if char == '(':
            open_count += 1
        elif open_count > 0:
            open_count -= 1
        else:
            close_needed += 1
    
    return open_count + close_needed


def longest_valid_parentheses(s: str) -> int:
    """
    LeetCode 32: Longest valid parentheses substring.
    Time: O(n), Space: O(n)
    """
    stack = [-1]  # Base index for calculating length
    max_len = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if stack:
                max_len = max(max_len, i - stack[-1])
            else:
                stack.append(i)  # New base
    
    return max_len


def evaluate_rpn(tokens: list[str]) -> int:
    """
    LeetCode 150: Evaluate Reverse Polish Notation.
    Time: O(n), Space: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b, a = stack.pop(), stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                # Truncate toward zero
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]


def basic_calculator(s: str) -> int:
    """
    LeetCode 224: Basic calculator with +, -, (, ).
    Time: O(n), Space: O(n)
    """
    stack = []
    result = 0
    num = 0
    sign = 1
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            result *= stack.pop()  # Sign before parenthesis
            result += stack.pop()  # Result before parenthesis
    
    return result + sign * num


def basic_calculator_ii(s: str) -> int:
    """
    LeetCode 227: Calculator with +, -, *, / (no parentheses).
    Time: O(n), Space: O(n)
    """
    stack = []
    num = 0
    prev_op = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(s) - 1:
            if prev_op == '+':
                stack.append(num)
            elif prev_op == '-':
                stack.append(-num)
            elif prev_op == '*':
                stack.append(stack.pop() * num)
            else:  # '/'
                # Truncate toward zero
                prev = stack.pop()
                stack.append(int(prev / num))
            
            prev_op = char
            num = 0
    
    return sum(stack)


def decode_string(s: str) -> str:
    """
    LeetCode 394: Decode string like "3[a2[c]]" -> "accaccacc".
    Time: O(n * max_k), Space: O(n)
    """
    stack = []
    current_str = ""
    current_num = 0
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str


def remove_duplicate_letters(s: str) -> str:
    """
    LeetCode 316: Remove duplicate letters, smallest lexicographical result.
    Time: O(n), Space: O(1) - fixed 26 characters
    """
    from collections import Counter
    
    count = Counter(s)
    in_stack = set()
    stack = []
    
    for char in s:
        count[char] -= 1
        
        if char in in_stack:
            continue
        
        # Remove larger characters if they appear later
        while stack and char < stack[-1] and count[stack[-1]] > 0:
            in_stack.remove(stack.pop())
        
        stack.append(char)
        in_stack.add(char)
    
    return ''.join(stack)
```

---

## 7. BFS with Queue

```python
from collections import deque

def level_order_traversal(root) -> list[list[int]]:
    """
    LeetCode 102: Binary tree level order traversal.
    Time: O(n), Space: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


def number_of_islands(grid: list[list[str]]) -> int:
    """
    LeetCode 200: Count islands using BFS.
    Time: O(m*n), Space: O(min(m, n))
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def bfs(i, j):
        queue = deque([(i, j)])
        grid[i][j] = '0'  # Mark visited
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                    grid[nx][ny] = '0'
                    queue.append((nx, ny))
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                bfs(i, j)
                count += 1
    
    return count


def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    """
    LeetCode 1091: Shortest path in binary matrix (8 directions).
    Time: O(n²), Space: O(n²)
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark visited
    
    while queue:
        x, y, dist = queue.popleft()
        
        if x == n - 1 and y == n - 1:
            return dist
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
                grid[nx][ny] = 1
                queue.append((nx, ny, dist + 1))
    
    return -1


def walls_and_gates(rooms: list[list[int]]) -> None:
    """
    LeetCode 286: Fill distance to nearest gate.
    Time: O(m*n), Space: O(m*n)
    
    Multi-source BFS: Start from all gates simultaneously.
    """
    if not rooms:
        return
    
    m, n = len(rooms), len(rooms[0])
    INF = 2147483647
    
    # Find all gates
    queue = deque()
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                queue.append((i, j))
    
    # BFS from all gates
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == INF:
                rooms[nx][ny] = rooms[x][y] + 1
                queue.append((nx, ny))


def rotten_oranges(grid: list[list[int]]) -> int:
    """
    LeetCode 994: Time for all oranges to rot.
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    
    if fresh == 0:
        return 0
    
    minutes = 0
    while queue:
        minutes += 1
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    fresh -= 1
                    queue.append((nx, ny))
    
    return minutes - 1 if fresh == 0 else -1
```

---

## 8. Interview Problems Summary

### Key Patterns

| Pattern | Example Problems | Key Insight |
|---------|------------------|-------------|
| Monotonic Stack | Next Greater, Histogram | Maintain sorted order |
| Monotonic Queue | Sliding Window Max | Deque for O(1) access |
| Expression Parsing | Calculator, Decode String | Nested structure |
| Multi-source BFS | Walls & Gates, Rotten Oranges | Start from all sources |
| Two Stacks | Queue using Stacks | Amortized O(1) |

### Time Complexity Reference

| Problem | Time | Space |
|---------|------|-------|
| Valid Parentheses | O(n) | O(n) |
| Daily Temperatures | O(n) | O(n) |
| Largest Rectangle | O(n) | O(n) |
| Sliding Window Max | O(n) | O(k) |
| Basic Calculator | O(n) | O(n) |
| Level Order BFS | O(n) | O(n) |

### Interview Tips

1. **Monotonic Stack**: When you need next/prev greater/smaller
2. **Monotonic Queue**: When you need sliding window min/max
3. **Two Stacks for Queue**: Push to one, pop transfers to other
4. **Expression Parsing**: Handle operators and parentheses with stack
5. **Multi-source BFS**: Initialize queue with all starting points
