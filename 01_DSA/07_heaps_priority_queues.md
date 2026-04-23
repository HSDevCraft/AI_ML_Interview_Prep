# Heaps & Priority Queues - Complete In-Depth Guide

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Heap Implementation](#heap-implementation)
3. [Python heapq Module](#python-heapq-module)
4. [Top K Problems](#top-k-problems)
5. [Merge Problems](#merge-problems)
6. [Two Heaps Pattern](#two-heaps-pattern)
7. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Heap Properties

```
Min-Heap: Parent ≤ Children (root is minimum)
Max-Heap: Parent ≥ Children (root is maximum)

Complete Binary Tree stored as array:
- Parent of i: (i - 1) // 2
- Left child of i: 2 * i + 1
- Right child of i: 2 * i + 2

Time Complexities:
- Insert (push): O(log n)
- Remove min/max (pop): O(log n)
- Get min/max (peek): O(1)
- Build heap: O(n)
- Heapify single element: O(log n)
```

---

## 2. Heap Implementation

```python
class MinHeap:
    """Min-heap implementation from scratch."""
    
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def push(self, val):
        """Add element and bubble up. O(log n)"""
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)
    
    def pop(self):
        """Remove and return minimum. O(log n)"""
        if not self.heap:
            raise IndexError("Pop from empty heap")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        
        return min_val
    
    def peek(self):
        """Return minimum without removing. O(1)"""
        if not self.heap:
            raise IndexError("Peek from empty heap")
        return self.heap[0]
    
    def _bubble_up(self, i):
        """Move element up to maintain heap property."""
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def _bubble_down(self, i):
        """Move element down to maintain heap property."""
        n = len(self.heap)
        
        while True:
            smallest = i
            left = self.left_child(i)
            right = self.right_child(i)
            
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest == i:
                break
            
            self.swap(i, smallest)
            i = smallest
    
    def __len__(self):
        return len(self.heap)
    
    def __bool__(self):
        return len(self.heap) > 0


class MaxHeap:
    """Max-heap using negation trick."""
    
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        import heapq
        heapq.heappush(self.heap, -val)
    
    def pop(self):
        import heapq
        return -heapq.heappop(self.heap)
    
    def peek(self):
        return -self.heap[0]
    
    def __len__(self):
        return len(self.heap)
```

---

## 3. Python heapq Module

```python
import heapq

# Basic operations
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
print(heap[0])           # Peek: 1
print(heapq.heappop(heap))  # Pop: 1

# Heapify existing list - O(n)
arr = [3, 1, 4, 1, 5, 9]
heapq.heapify(arr)

# Push and pop in single operation
val = heapq.heappushpop(heap, 4)  # Push 4, then pop smallest
val = heapq.heapreplace(heap, 4)  # Pop smallest, then push 4

# Get k smallest/largest - O(n + k log n)
k_smallest = heapq.nsmallest(3, arr)
k_largest = heapq.nlargest(3, arr)

# With custom key
items = [(3, 'a'), (1, 'b'), (2, 'c')]
smallest = heapq.nsmallest(2, items, key=lambda x: x[0])


# Max-heap using negation
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)  # 5


# Heap with custom objects (use tuple with priority first)
heap = []
heapq.heappush(heap, (priority, counter, item))  # counter for tie-breaking


# Merge sorted iterables - O(n log k)
lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
merged = list(heapq.merge(*lists))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 4. Top K Problems

```python
def kth_largest(nums: list[int], k: int) -> int:
    """
    LeetCode 215: Find kth largest element.
    
    Use min-heap of size k.
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    heap = nums[:k]
    heapq.heapify(heap)
    
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return heap[0]


def kth_smallest_matrix(matrix: list[list[int]], k: int) -> int:
    """
    LeetCode 378: Kth smallest in sorted matrix.
    
    Each row and column is sorted.
    Time: O(k log n), Space: O(n)
    """
    import heapq
    
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)
    
    for _ in range(k - 1):
        val, row, col = heapq.heappop(heap)
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    return heap[0][0]


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    LeetCode 347: Top k frequent elements.
    Time: O(n log k), Space: O(n)
    """
    import heapq
    from collections import Counter
    
    count = Counter(nums)
    
    # Use min-heap with frequency as priority
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]


def top_k_frequent_bucket_sort(nums: list[int], k: int) -> list[int]:
    """
    Bucket sort approach - O(n) time.
    """
    from collections import Counter
    
    count = Counter(nums)
    n = len(nums)
    
    # Bucket index = frequency
    buckets = [[] for _ in range(n + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(n, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result


def k_closest_points(points: list[list[int]], k: int) -> list[list[int]]:
    """
    LeetCode 973: K closest points to origin.
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    # Max-heap with negative distances
    heap = []
    
    for x, y in points:
        dist = -(x * x + y * y)
        
        if len(heap) < k:
            heapq.heappush(heap, (dist, x, y))
        elif dist > heap[0][0]:
            heapq.heapreplace(heap, (dist, x, y))
    
    return [[x, y] for _, x, y in heap]


def reorganize_string(s: str) -> str:
    """
    LeetCode 767: Rearrange string so no adjacent chars are same.
    Time: O(n log k), Space: O(k) where k = unique chars
    """
    import heapq
    from collections import Counter
    
    count = Counter(s)
    max_heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(max_heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        
        # Re-add previous character if it still has count
        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))
        
        prev_freq = freq + 1  # Used one
        prev_char = char
    
    result_str = ''.join(result)
    return result_str if len(result_str) == len(s) else ""


class KthLargestStream:
    """
    LeetCode 703: Kth largest element in stream.
    """
    
    def __init__(self, k: int, nums: list[int]):
        import heapq
        
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val: int) -> int:
        import heapq
        
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        
        return self.heap[0]
```

---

## 5. Merge Problems

```python
def merge_k_sorted_lists(lists: list) -> 'ListNode':
    """
    LeetCode 23: Merge k sorted linked lists.
    Time: O(N log k), Space: O(k)
    """
    import heapq
    
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


def smallest_range(nums: list[list[int]]) -> list[int]:
    """
    LeetCode 632: Smallest range covering elements from all lists.
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    k = len(nums)
    heap = []
    max_val = float('-inf')
    
    # Initialize with first element of each list
    for i in range(k):
        heapq.heappush(heap, (nums[i][0], i, 0))
        max_val = max(max_val, nums[i][0])
    
    result = [float('-inf'), float('inf')]
    
    while True:
        min_val, list_idx, elem_idx = heapq.heappop(heap)
        
        # Update result if current range is smaller
        if max_val - min_val < result[1] - result[0]:
            result = [min_val, max_val]
        
        # If any list is exhausted, we're done
        if elem_idx + 1 == len(nums[list_idx]):
            break
        
        # Add next element from same list
        next_val = nums[list_idx][elem_idx + 1]
        heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
        max_val = max(max_val, next_val)
    
    return result


def find_kth_smallest_pair_distance(nums: list[int], k: int) -> int:
    """
    LeetCode 719: Find kth smallest pair distance.
    
    Binary search + counting approach (more efficient than heap).
    Time: O(n log n + n log W), Space: O(1)
    """
    nums.sort()
    n = len(nums)
    
    def count_pairs_with_distance_le(d):
        count = 0
        left = 0
        for right in range(n):
            while nums[right] - nums[left] > d:
                left += 1
            count += right - left
        return count
    
    lo, hi = 0, nums[-1] - nums[0]
    
    while lo < hi:
        mid = (lo + hi) // 2
        if count_pairs_with_distance_le(mid) < k:
            lo = mid + 1
        else:
            hi = mid
    
    return lo
```

---

## 6. Two Heaps Pattern

```python
class MedianFinder:
    """
    LeetCode 295: Find median from data stream.
    
    Use two heaps:
    - max_heap: smaller half (negate for max behavior)
    - min_heap: larger half
    
    Time: O(log n) for add, O(1) for median
    Space: O(n)
    """
    
    def __init__(self):
        import heapq
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap
    
    def addNum(self, num: int) -> None:
        import heapq
        
        # Add to max heap first
        heapq.heappush(self.small, -num)
        
        # Ensure max of small <= min of large
        if self.small and self.large and -self.small[0] > self.large[0]:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2


def sliding_window_median(nums: list[int], k: int) -> list[float]:
    """
    LeetCode 480: Sliding window median.
    
    Use two heaps with lazy removal.
    Time: O(n log k), Space: O(k)
    """
    import heapq
    from collections import defaultdict
    
    small = []  # Max heap (negated)
    large = []  # Min heap
    removed = defaultdict(int)  # Lazy removal counter
    
    def add(num):
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)
    
    def remove(num):
        removed[num] += 1
    
    def balance():
        while small and removed[-small[0]]:
            removed[-small[0]] -= 1
            heapq.heappop(small)
        while large and removed[large[0]]:
            removed[large[0]] -= 1
            heapq.heappop(large)
    
    def rebalance():
        balance()
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
            balance()
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))
            balance()
    
    def get_median():
        if k % 2 == 1:
            return float(-small[0])
        return (-small[0] + large[0]) / 2
    
    result = []
    
    for i in range(len(nums)):
        add(nums[i])
        
        if i >= k:
            remove(nums[i - k])
        
        rebalance()
        
        if i >= k - 1:
            result.append(get_median())
    
    return result


def ipo(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    """
    LeetCode 502: IPO - Maximize capital after k projects.
    
    Two heaps: min-heap for capital, max-heap for profits.
    """
    import heapq
    
    n = len(profits)
    projects = [(capital[i], profits[i]) for i in range(n)]
    projects.sort()
    
    max_profit_heap = []
    i = 0
    
    for _ in range(k):
        # Add all affordable projects to max heap
        while i < n and projects[i][0] <= w:
            heapq.heappush(max_profit_heap, -projects[i][1])
            i += 1
        
        if not max_profit_heap:
            break
        
        w += -heapq.heappop(max_profit_heap)
    
    return w
```

---

## 7. Interview Problems

```python
def task_scheduler(tasks: list[str], n: int) -> int:
    """
    LeetCode 621: Minimum intervals to finish all tasks.
    
    Time: O(tasks), Space: O(1)
    """
    from collections import Counter
    import heapq
    
    count = Counter(tasks)
    max_heap = [-c for c in count.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        cycle = []
        
        for _ in range(n + 1):
            if max_heap:
                cycle.append(heapq.heappop(max_heap))
        
        for c in cycle:
            if c + 1 < 0:
                heapq.heappush(max_heap, c + 1)
        
        time += n + 1 if max_heap else len(cycle)
    
    return time


def meeting_rooms_ii(intervals: list[list[int]]) -> int:
    """
    LeetCode 253: Minimum meeting rooms required.
    """
    import heapq
    
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    rooms = [intervals[0][1]]  # End times
    
    for start, end in intervals[1:]:
        if start >= rooms[0]:
            heapq.heappop(rooms)
        heapq.heappush(rooms, end)
    
    return len(rooms)


def employee_free_time(schedule: list[list[list[int]]]) -> list[list[int]]:
    """
    LeetCode 759: Find common free time for all employees.
    """
    import heapq
    
    # Flatten and sort all intervals
    heap = []
    for i, emp in enumerate(schedule):
        if emp:
            heapq.heappush(heap, (emp[0][0], i, 0))
    
    result = []
    prev_end = -1
    
    while heap:
        start, emp_id, idx = heapq.heappop(heap)
        
        if prev_end != -1 and start > prev_end:
            result.append([prev_end, start])
        
        prev_end = max(prev_end, schedule[emp_id][idx][1])
        
        if idx + 1 < len(schedule[emp_id]):
            heapq.heappush(heap, (schedule[emp_id][idx + 1][0], emp_id, idx + 1))
    
    return result


def ugly_number_ii(n: int) -> int:
    """
    LeetCode 264: Find nth ugly number (factors only 2, 3, 5).
    """
    import heapq
    
    heap = [1]
    seen = {1}
    
    for _ in range(n):
        curr = heapq.heappop(heap)
        
        for factor in [2, 3, 5]:
            next_ugly = curr * factor
            if next_ugly not in seen:
                seen.add(next_ugly)
                heapq.heappush(heap, next_ugly)
    
    return curr


def super_ugly_number(n: int, primes: list[int]) -> int:
    """
    LeetCode 313: nth super ugly number with given prime factors.
    """
    import heapq
    
    heap = [1]
    seen = {1}
    
    for _ in range(n):
        curr = heapq.heappop(heap)
        
        for prime in primes:
            next_ugly = curr * prime
            if next_ugly not in seen:
                seen.add(next_ugly)
                heapq.heappush(heap, next_ugly)
    
    return curr
```

### Summary

| Problem Type | Approach | Time Complexity |
|--------------|----------|-----------------|
| Kth largest/smallest | Min/Max heap of size k | O(n log k) |
| Top K frequent | Heap or bucket sort | O(n log k) or O(n) |
| Merge K sorted | Min heap | O(N log k) |
| Running median | Two heaps | O(log n) per add |
| Task scheduling | Max heap | O(n log n) |
