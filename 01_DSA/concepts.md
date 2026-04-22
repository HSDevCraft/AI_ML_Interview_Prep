# Data Structures & Algorithms - Complete Guide

## 1. Arrays & Strings

### Time Complexities
| Operation | Array | Dynamic Array |
|-----------|-------|---------------|
| Access | O(1) | O(1) |
| Search | O(n) | O(n) |
| Insert (end) | N/A | O(1) amortized |
| Insert (middle) | O(n) | O(n) |
| Delete | O(n) | O(n) |

### Key Patterns

#### Two Pointers
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

#### Sliding Window
```python
def max_sum_subarray(arr, k):
    """Fixed-size sliding window"""
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

def longest_substring_k_distinct(s, k):
    """Variable-size sliding window"""
    char_count = {}
    left = max_len = 0
    
    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    return max_len
```

#### Prefix Sum
```python
def subarray_sum_equals_k(nums, k):
    """Count subarrays with sum = k"""
    prefix_sum = 0
    count = 0
    sum_count = {0: 1}
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    return count
```

---

## 2. Linked Lists

### Implementation
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(val)
    
    def reverse(self):
        prev, curr = None, self.head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        self.head = prev
```

### Key Patterns

#### Fast & Slow Pointers (Floyd's Algorithm)
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

#### Merge Two Sorted Lists
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 or l2
    return dummy.next
```

---

## 3. Stacks & Queues

### Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
```

### Key Patterns

#### Monotonic Stack
```python
def next_greater_element(nums):
    """Find next greater element for each element"""
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices
    
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result

def largest_rectangle_histogram(heights):
    """Classic monotonic stack problem"""
    stack = []
    max_area = 0
    heights.append(0)  # sentinel
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area
```

#### Valid Parentheses
```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0
```

---

## 4. Trees

### Binary Tree Implementation
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Traversals
```python
def preorder(root):
    """Root -> Left -> Right"""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    """Left -> Root -> Right"""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    """Left -> Right -> Root"""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

def level_order(root):
    """BFS - Level by Level"""
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Iterative Traversals
```python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

### Key Tree Problems
```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def is_balanced(root):
    def height(node):
        if not node:
            return 0
        left_h = height(node.left)
        right_h = height(node.right)
        if left_h == -1 or right_h == -1 or abs(left_h - right_h) > 1:
            return -1
        return 1 + max(left_h, right_h)
    return height(root) != -1

def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

### Binary Search Tree Operations
```python
def search_bst(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst(root.left, val)
    return search_bst(root.right, val)

def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

def delete_bst(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # Find inorder successor
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        root.val = min_node.val
        root.right = delete_bst(root.right, min_node.val)
    return root

def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and 
            is_valid_bst(root.right, root.val, max_val))
```

---

## 5. Heaps / Priority Queues

### Implementation with heapq
```python
import heapq

# Min Heap (default)
min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 1)
heapq.heappush(min_heap, 2)
smallest = heapq.heappop(min_heap)  # 1

# Max Heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
largest = -heapq.heappop(max_heap)  # 3

# Heapify existing list - O(n)
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)

# K largest/smallest
k_largest = heapq.nlargest(3, arr)
k_smallest = heapq.nsmallest(3, arr)
```

### Key Heap Problems
```python
def kth_largest(nums, k):
    """Using min-heap of size k"""
    heap = nums[:k]
    heapq.heapify(heap)
    
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

def merge_k_sorted_lists(lists):
    """Merge k sorted linked lists"""
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

def top_k_frequent(nums, k):
    """Top K frequent elements"""
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

## 6. Graphs

### Representations
```python
# Adjacency List
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Adjacency Matrix
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

# Edge List
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
```

### BFS & DFS
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    return result

def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    result = [node]
    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    return result
```

### Shortest Path Algorithms
```python
def dijkstra(graph, start):
    """Single source shortest path for weighted graphs"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        curr_dist, node = heapq.heappop(heap)
        if curr_dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances

def bellman_ford(n, edges, start):
    """Handles negative weights"""
    distances = [float('inf')] * n
    distances[start] = 0
    
    for _ in range(n - 1):
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
    
    # Check for negative cycle
    for u, v, w in edges:
        if distances[u] + w < distances[v]:
            return None  # Negative cycle exists
    return distances
```

### Topological Sort
```python
def topological_sort_kahn(graph, n):
    """Kahn's Algorithm (BFS-based)"""
    in_degree = [0] * n
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []  # Empty if cycle exists

def topological_sort_dfs(graph, n):
    """DFS-based"""
    visited = [False] * n
    stack = []
    
    def dfs(node):
        visited[node] = True
        for neighbor in graph.get(node, []):
            if not visited[neighbor]:
                dfs(neighbor)
        stack.append(node)
    
    for i in range(n):
        if not visited[i]:
            dfs(i)
    
    return stack[::-1]
```

### Union-Find (Disjoint Set Union)
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

## 7. Dynamic Programming

### Patterns

#### 1D DP
```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def climbing_stairs(n):
    """Ways to climb n stairs (1 or 2 steps at a time)"""
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr

def coin_change(coins, amount):
    """Minimum coins to make amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

#### 2D DP
```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def knapsack_01(weights, values, capacity):
    """0/1 Knapsack Problem"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], 
                              values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

#### State Machine DP
```python
def max_profit_with_cooldown(prices):
    """Best time to buy/sell with cooldown"""
    n = len(prices)
    if n < 2:
        return 0
    
    # States: hold, sold, rest
    hold = -prices[0]
    sold = 0
    rest = 0
    
    for i in range(1, n):
        prev_hold = hold
        hold = max(hold, rest - prices[i])
        rest = max(rest, sold)
        sold = prev_hold + prices[i]
    
    return max(sold, rest)
```

---

## 8. Binary Search

### Templates
```python
def binary_search_exact(arr, target):
    """Find exact match"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search_left(arr, target):
    """Find leftmost insertion point"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def binary_search_right(arr, target):
    """Find rightmost insertion point"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
```

### Binary Search on Answer
```python
def koko_eating_bananas(piles, h):
    """Minimum eating speed to finish in h hours"""
    def can_finish(speed):
        return sum((p + speed - 1) // speed for p in piles) <= h
    
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## 9. Backtracking

### Template
```python
def backtrack(candidates, target, start, path, result):
    if target == 0:
        result.append(path[:])
        return
    if target < 0:
        return
    
    for i in range(start, len(candidates)):
        path.append(candidates[i])
        backtrack(candidates, target - candidates[i], i, path, result)
        path.pop()

def combination_sum(candidates, target):
    result = []
    backtrack(candidates, target, 0, [], result)
    return result
```

### Classic Problems
```python
def permutations(nums):
    result = []
    
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i, num in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(num)
            backtrack(path, used)
            path.pop()
            used[i] = False
    
    backtrack([], [False] * len(nums))
    return result

def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

def n_queens(n):
    result = []
    
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or \
               abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def backtrack(row, board):
        if row == n:
            result.append(['.' * i + 'Q' + '.' * (n-i-1) for i in board])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board.append(col)
                backtrack(row + 1, board)
                board.pop()
    
    backtrack(0, [])
    return result
```

---

## 10. Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

---

## Complexity Cheat Sheet

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n) avg | O(log n) |
| Heap Operations | O(log n) | O(1) |
| BFS/DFS | O(V + E) | O(V) |
| Dijkstra | O((V+E) log V) | O(V) |
| Topological Sort | O(V + E) | O(V) |
| Union-Find | O(α(n)) | O(n) |
