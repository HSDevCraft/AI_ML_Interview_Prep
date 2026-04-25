# DSA & Python Interview Questions Bank

> **Coding Interview Strategy**: Talk through your approach BEFORE coding. State the brute force, identify the pattern, then optimize. Always clarify edge cases upfront.

## DSA Problem-Solving Framework (use every time)

```
1. UNDERSTAND (2 min)
   - Repeat problem in your own words
   - Ask: sorted? duplicates allowed? empty input? integer overflow?
   - Ask: optimize for time or space?

2. EXAMPLES (1 min)
   - Walk through given example
   - Create your own: normal case, edge case (empty, single element, all same)

3. BRUTE FORCE (1 min)
   - State it verbally: "naive approach is O(n²) because..."
   - Don't code it; use it to spot the inefficiency

4. OPTIMIZE (3 min)
   - Identify the pattern (see cheat sheet below)
   - State time/space complexity before coding

5. CODE (15-20 min)
   - Write clean, readable code
   - Talk through what you're doing
   - Use meaningful variable names

6. TEST (3 min)
   - Trace through your example
   - Check edge cases explicitly
   - Look for off-by-one errors, empty inputs, single element
```

## Pattern Recognition Cheat Sheet

| If the problem involves... | Use this pattern |
|---------------------------|------------------|
| Sorted array, find pair | Two Pointers |
| Substring / subarray of length k | Sliding Window |
| Shortest path in unweighted graph | BFS |
| All paths, combinations, permutations | Backtracking / DFS |
| Optimal substructure, overlapping subproblems | Dynamic Programming |
| Repeated min/max queries | Heap / Priority Queue |
| Prefix/suffix relationships | Prefix Sum |
| Top-k elements | Heap or QuickSelect |
| Balanced parentheses, evaluate expressions | Stack |
| Trie for prefix matching | Trie |
| Connected components, cycle detection | Union-Find |
| Binary search on the answer (min/max feasibility) | Binary Search |

## Complexity Quick Reference

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Hash Map | O(1) | O(1) | O(1) | O(1) |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | O(1) top | O(n) | O(log n) | O(log n) |
| Sorted Array | O(1) | O(log n) | O(n) | O(n) |

---

## Table of Contents
1. [Arrays & Strings](#arrays--strings)
2. [Sliding Window](#sliding-window)
3. [Two Pointers](#two-pointers)
4. [Linked Lists](#linked-lists)
5. [Trees](#trees)
6. [Graphs](#graphs)
7. [Dynamic Programming](#dynamic-programming)
8. [Binary Search](#binary-search)
9. [Heaps & Priority Queues](#heaps--priority-queues)
10. [Backtracking](#backtracking)
11. [Python Interview Questions](#python-interview-questions)
12. [Python Advanced Topics](#python-advanced-topics)

---

## DSA Questions

### Arrays & Strings

**Q1: Two Sum**

> **Pattern**: Hash map for O(1) complement lookup.

```python
# Time: O(n), Space: O(n)
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:          # found the pair!
            return [seen[complement], i]
        seen[num] = i                   # store for future lookups
    return []

# Edge cases to mention:
# - Empty array: returns []
# - Same element twice: e.g., [3,3] target=6 → [0,1] (works because we check before storing)
# - No solution: problem guarantees one, but return [] as fallback
```

**🚨 Pitfall**: Using `nums.index(complement)` is O(n), making the whole thing O(n²). Hash map is key.

**Q2: Best Time to Buy and Sell Stock**

> **Pattern**: Track minimum seen so far; max profit = max(current - min_so_far).

```python
# Time: O(n), Space: O(1)
def max_profit(prices):
    if not prices:
        return 0
    min_price = float('inf')
    max_profit_val = 0
    for price in prices:
        min_price = min(min_price, price)          # best day to buy (retrospectively)
        max_profit_val = max(max_profit_val, price - min_price)  # profit if sell today
    return max_profit_val

# Intuition: at each price, ask "if I had bought at the lowest price so far, what's my profit?"
# Edge cases: decreasing prices → return 0 (don't trade), single price → 0
```

**🚨 Pitfall**: Off-by-one: you must BUY before SELL (can't sell on same day in basic version). The min tracking handles this correctly since we update min first, then check profit.

**Q3: Longest Substring Without Repeating Characters**

> **Pattern**: Sliding window with hash map tracking last seen index of each character.

```python
# Time: O(n), Space: O(min(m,n)) where m is charset size
def length_of_longest_substring(s):
    char_index = {}   # char -> last seen index
    max_len = start = 0
    
    for i, char in enumerate(s):
        # If char was seen WITHIN current window, shrink window from left
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1   # jump start past the duplicate
        char_index[char] = i               # update last seen index
        max_len = max(max_len, i - start + 1)
    
    return max_len

# Example trace: "abcba"
# i=0 'a': start=0, window="a",   len=1
# i=1 'b': start=0, window="ab",  len=2
# i=2 'c': start=0, window="abc", len=3
# i=3 'b': 'b' at index 1 >= start=0, so start=2, window="cb", len=2
# i=4 'a': 'a' at index 0 < start=2, DON'T move start! window="cba", len=3
```

**🚨 Pitfall**: The `char_index[char] >= start` check is critical! Without it, when a char was seen before the current window, you'd incorrectly shrink the window.

**Q4: Trapping Rain Water**

> **Pattern**: Two pointers. Water at position i = min(max_left, max_right) - height[i].

```python
# Time: O(n), Space: O(1)  [O(n) with prefix arrays is simpler to explain first]
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            # RIGHT side is taller → left side's water is bounded by left_max
            if height[left] >= left_max:
                left_max = height[left]    # new left boundary
            else:
                water += left_max - height[left]  # water trapped here
            left += 1
        else:
            # LEFT side is taller → right side's water is bounded by right_max
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water

# Simpler O(n) space version (explain this first in interview, then optimize):
def trap_simple(height):
    n = len(height)
    left_max  = [0]*n  # left_max[i] = max height from 0..i
    right_max = [0]*n  # right_max[i] = max height from i..n-1
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    right_max[-1] = height[-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    return sum(min(left_max[i], right_max[i]) - height[i] for i in range(n))
```

**🚨 Interview Tip**: Start with `trap_simple` (O(n) space) to demonstrate understanding, then optimize to O(1) space with two pointers. Interviewers appreciate seeing both solutions.

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

**Q5b: Product of Array Except Self**
```python
# Return array where output[i] = product of all elements except nums[i]
# Time: O(n), Space: O(1) excluding output
def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Left pass: result[i] = product of all elements to the left
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # Right pass: multiply by product of all elements to the right
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result
```

**Q5c: Maximum Subarray (Kadane's Algorithm)**
```python
# Find contiguous subarray with largest sum
# Time: O(n), Space: O(1)
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Variant: Return the subarray itself
def max_subarray_with_indices(nums):
    max_sum = current_sum = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start, end = temp_start, i
    
    return nums[start:end + 1], max_sum
```

---

### Sliding Window

**Q5d: Minimum Window Substring**
```python
# Find minimum window in s that contains all characters of t
# Time: O(n), Space: O(k) where k is charset size
from collections import Counter

def min_window(s, t):
    if not t or not s:
        return ""
    
    t_count = Counter(t)
    required = len(t_count)
    
    left = 0
    formed = 0
    window_counts = {}
    
    min_len = float('inf')
    result = (0, 0)
    
    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        while formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = (left, right)
            
            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1
    
    return "" if min_len == float('inf') else s[result[0]:result[1] + 1]
```

**Q5e: Longest Repeating Character Replacement**
```python
# Max length substring with at most k character replacements
# Time: O(n), Space: O(26)
def character_replacement(s, k):
    count = {}
    max_count = 0
    max_len = 0
    left = 0
    
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_count = max(max_count, count[s[right]])
        
        # Window size - most frequent char count > k means too many replacements
        while (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

**Q5f: Sliding Window Maximum**
```python
# Return max element in each sliding window of size k
# Time: O(n), Space: O(k)
from collections import deque

def max_sliding_window(nums, k):
    if not nums:
        return []
    
    result = []
    dq = deque()  # Store indices, decreasing order of values
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Window is full, record max
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

### Two Pointers

**Q5g: Container With Most Water**
```python
# Find two lines that form container with most water
# Time: O(n), Space: O(1)
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        
        # Move the shorter line inward
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

**Q5h: 3Sum**
```python
# Find all unique triplets that sum to zero
# Time: O(n²), Space: O(1) excluding output
def three_sum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
```

---

### Linked Lists

**Q6: Reverse Linked List**

> **Pattern**: Three-pointer iteration (prev, curr, next). Draw it on paper first.

```python
# Iterative: Time O(n), Space O(1)  ← prefer this in interviews
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next  # save next BEFORE breaking the link
        curr.next = prev       # reverse the pointer
        prev = curr            # move prev forward
        curr = next_node       # move curr forward
    return prev  # prev is now the new head

# Trace: 1 -> 2 -> 3 -> None
# After iter 1: None <- 1  2 -> 3
# After iter 2: None <- 1 <- 2  3
# After iter 3: None <- 1 <- 2 <- 3  (prev=3, return 3)

# Recursive: Time O(n), Space O(n) for call stack
def reverse_list_recursive(head):
    if not head or not head.next:
        return head  # base case: empty or single node
    new_head = reverse_list_recursive(head.next)  # reverse rest
    head.next.next = head   # make next node point back to current
    head.next = None        # remove forward link (current becomes tail)
    return new_head
```

**🚨 Pitfalls**: Forgetting to save `next_node` before overwriting `curr.next`. Returning `head` instead of `prev` (head is now the TAIL after reversal).

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

> **Pattern**: Hash map + Doubly Linked List for O(1) get/put. `OrderedDict` is the Pythonic shortcut.

```python
from collections import OrderedDict

class LRUCache:
    """O(1) get and put using Python's OrderedDict (doubly linked list + hash map)."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()   # maintains insertion order
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # mark as recently used
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)   # update and mark recent
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict LRU (oldest = leftmost)

# If asked to implement WITHOUT OrderedDict (manual DLL + HashMap):
class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCacheManual:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}           # key -> Node
        # Sentinel nodes: left=LRU, right=MRU
        self.left = Node()        # dummy LRU end
        self.right = Node()       # dummy MRU end
        self.left.next = self.right
        self.right.prev = self.left
    
    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev
    
    def _insert_right(self, node):  # insert as MRU
        node.prev = self.right.prev
        node.next = self.right
        self.right.prev.next = node
        self.right.prev = node
    
    def get(self, key):
        if key not in self.cache: return -1
        self._remove(self.cache[key])
        self._insert_right(self.cache[key])  # move to MRU
        return self.cache[key].val
    
    def put(self, key, value):
        if key in self.cache: self._remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self._insert_right(self.cache[key])
        if len(self.cache) > self.cap:
            lru = self.left.next
            self._remove(lru)
            del self.cache[lru.key]
```

**🚨 Interview Tip**: Mention OrderedDict first (shows Python knowledge), then offer to implement the manual DLL version if asked. The manual version shows data structure understanding.

---

### Trees

**Q9: Validate Binary Search Tree**

> **Pattern**: Pass valid range (min, max) through recursion. NOT just checking immediate children!

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    Key insight: every node must satisfy ALL ancestor constraints,
    not just be greater than left child and less than right child.
    
    Counter-example of naive approach:
         5
        / \
       1   6
          / \
         3   7   <- 3 is left child of 6 (3 < 6 ✓) BUT 3 < 5 which violates BST!
    """
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:  # violates range from ancestors
        return False
    return (is_valid_bst(root.left,  min_val, root.val) and   # left subtree: max=root.val
            is_valid_bst(root.right, root.val, max_val))       # right subtree: min=root.val

# Alternative: inorder traversal should be strictly increasing
def is_valid_bst_inorder(root):
    prev = float('-inf')
    def inorder(node):
        nonlocal prev
        if not node: return True
        if not inorder(node.left): return False
        if node.val <= prev: return False   # not strictly increasing
        prev = node.val
        return inorder(node.right)
    return inorder(root)
```

**🚨 Classic Pitfall**: Only checking `left.val < root.val < right.val` is WRONG for subtrees. Always use the range propagation approach.

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

**Q8b: Detect Cycle in Linked List (Floyd's Algorithm)**
```python
# Time: O(n), Space: O(1)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find cycle start
def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Reset slow to head, move both at same speed
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

**Q8c: Copy List with Random Pointer**
```python
# Time: O(n), Space: O(1) - interleaving approach
def copy_random_list(head):
    if not head:
        return None
    
    # Step 1: Create copies interleaved with originals
    curr = head
    while curr:
        copy = Node(curr.val)
        copy.next = curr.next
        curr.next = copy
        curr = copy.next
    
    # Step 2: Assign random pointers
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    
    # Step 3: Separate lists
    curr = head
    copy_head = head.next
    while curr:
        copy = curr.next
        curr.next = copy.next
        curr = curr.next
        if curr:
            copy.next = curr.next
    
    return copy_head
```

---

### Binary Search

**Q14b: Search in Rotated Sorted Array**
```python
# Time: O(log n), Space: O(1)
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

**Q14c: Find Minimum in Rotated Sorted Array**
```python
# Time: O(log n), Space: O(1)
def find_min(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]
```

**Q14d: Koko Eating Bananas (Binary Search on Answer)**
```python
# Find minimum eating speed to finish within h hours
# Time: O(n log m) where m is max pile size
def min_eating_speed(piles, h):
    def can_finish(speed):
        hours = sum((pile + speed - 1) // speed for pile in piles)
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

**Q14e: Median of Two Sorted Arrays**
```python
# Time: O(log(min(m,n))), Space: O(1)
def find_median_sorted_arrays(nums1, nums2):
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0.0
```

---

### Heaps & Priority Queues

**Q14f: Top K Frequent Elements**
```python
# Time: O(n log k), Space: O(n)
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Alternative: Bucket sort O(n)
def top_k_frequent_bucket(nums, k):
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result
```

**Q14g: Find Median from Data Stream**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def addNum(self, num):
        # Add to max heap
        heapq.heappush(self.small, -num)
        
        # Balance: ensure small's max <= large's min
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Q14h: Task Scheduler**
```python
# Time: O(n), Space: O(1)
from collections import Counter

def least_interval(tasks, n):
    count = Counter(tasks)
    max_freq = max(count.values())
    max_count = sum(1 for freq in count.values() if freq == max_freq)
    
    # Formula: (max_freq - 1) * (n + 1) + max_count
    # But can't be less than total tasks
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

---

### Backtracking

**Q14i: Subsets**
```python
# Time: O(n * 2^n), Space: O(n)
def subsets(nums):
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result

# Iterative approach
def subsets_iterative(nums):
    result = [[]]
    for num in nums:
        result += [subset + [num] for subset in result]
    return result
```

**Q14j: Permutations**
```python
# Time: O(n * n!), Space: O(n)
def permute(nums):
    result = []
    
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()
    
    backtrack([], nums)
    return result
```

**Q14k: Combination Sum**
```python
# Time: O(2^target), Space: O(target)
def combination_sum(candidates, target):
    result = []
    
    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return
        
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # Can reuse
            current.pop()
    
    backtrack(0, [], target)
    return result
```

**Q14l: N-Queens**
```python
# Time: O(n!), Space: O(n)
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            backtrack(row + 1)
            
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0)
    return result
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

**Q17b: House Robber**
```python
# Can't rob adjacent houses. Time: O(n), Space: O(1)
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    
    return prev1

# House Robber II: Houses in a circle
def rob_circular(nums):
    if len(nums) == 1:
        return nums[0]
    return max(rob(nums[:-1]), rob(nums[1:]))
```

**Q17c: Coin Change**
```python
# Minimum coins to make amount. Time: O(amount * n), Space: O(amount)
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

**Q17d: Longest Common Subsequence**
```python
# Time: O(m*n), Space: O(min(m,n))
def longest_common_subsequence(text1, text2):
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    prev = [0] * (len(text2) + 1)
    
    for i in range(1, len(text1) + 1):
        curr = [0] * (len(text2) + 1)
        for j in range(1, len(text2) + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    
    return prev[-1]
```

**Q17e: Edit Distance**
```python
# Minimum operations to convert word1 to word2
# Time: O(m*n), Space: O(n)
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    
    return prev[n]
```

**Q17f: Unique Paths**
```python
# Robot in grid, can only move right or down
# Time: O(m*n), Space: O(n)
def unique_paths(m, n):
    dp = [1] * n
    
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    
    return dp[-1]

# With obstacles
def unique_paths_with_obstacles(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = 1
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    
    return dp[-1]
```

**Q17g: Partition Equal Subset Sum (0/1 Knapsack variant)**
```python
# Can array be partitioned into two equal sum subsets?
# Time: O(n * sum), Space: O(sum)
def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]
```

---

### Interval Problems

**Q17h: Meeting Rooms II (Minimum Rooms)**
```python
# Time: O(n log n), Space: O(n)
import heapq

def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    heap = []  # End times
    
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    
    return len(heap)
```

**Q17i: Non-overlapping Intervals**
```python
# Minimum intervals to remove to make non-overlapping
# Time: O(n log n), Space: O(1)
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    end = intervals[0][1]
    count = 0
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            count += 1
        else:
            end = intervals[i][1]
    
    return count
```

---

### Trie Problems

**Q17j: Implement Trie**
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
        node = self._traverse(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        return self._traverse(prefix) is not None
    
    def _traverse(self, s):
        node = self.root
        for char in s:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

**Q17k: Word Search II**
```python
# Find all words from dictionary in board
# Time: O(m*n*4^L + W*L), Space: O(W*L)
def find_words(board, words):
    # Build trie
    trie = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = word
    
    result = []
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, node):
        char = board[r][c]
        if char not in node:
            return
        
        next_node = node[char]
        if '$' in next_node:
            result.append(next_node.pop('$'))
        
        board[r][c] = '#'  # Mark visited
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, next_node)
        
        board[r][c] = char
        
        # Prune empty branches
        if not next_node:
            node.pop(char)
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie)
    
    return result
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

## Python Advanced Topics

### Q14: Memory Management and Garbage Collection
```python
# Reference counting + cyclic garbage collector
import sys
import gc

a = [1, 2, 3]
print(sys.getrefcount(a))  # Reference count (includes function arg)

# Cyclic references
class Node:
    def __init__(self):
        self.ref = None

n1 = Node()
n2 = Node()
n1.ref = n2
n2.ref = n1  # Cycle!

del n1, n2  # Won't free immediately - needs gc

gc.collect()  # Force garbage collection

# Weak references (don't increase ref count)
import weakref

class Cache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
```

### Q15: Threading vs Multiprocessing
```python
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Threading: Good for I/O-bound tasks
def io_bound_task(url):
    # Network request, file I/O, etc.
    pass

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(io_bound_task, urls))

# Multiprocessing: Good for CPU-bound tasks
def cpu_bound_task(data):
    # Heavy computation
    return sum(x**2 for x in data)

with ProcessPoolExecutor() as executor:
    results = list(executor.map(cpu_bound_task, data_chunks))

# Sharing data between processes
from multiprocessing import Manager, Queue, Value, Array

manager = Manager()
shared_dict = manager.dict()
shared_list = manager.list()

# Lock for thread safety
lock = threading.Lock()

def thread_safe_increment(counter):
    with lock:
        counter['value'] += 1
```

### Q16: Python Data Model (Dunder Methods)
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representations
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    # Arithmetic operators
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # Comparison
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
    
    # Container behavior
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        return (self.x, self.y)[index]
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    # Hashing (required for set/dict keys)
    def __hash__(self):
        return hash((self.x, self.y))
    
    # Context manager
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    # Callable
    def __call__(self):
        return (self.x**2 + self.y**2) ** 0.5
```

### Q17: Type Hints and Protocols
```python
from typing import (
    List, Dict, Optional, Union, Callable, TypeVar, Generic,
    Protocol, runtime_checkable, Literal, overload
)

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}"

# Generic types
T = TypeVar('T')

def first(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Generic classes
class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Protocols (structural subtyping)
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

# Circle is Drawable without explicit inheritance
def render(obj: Drawable) -> None:
    obj.draw()

# Function overloads
@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...

def process(x):
    if isinstance(x, int):
        return x * 2
    return x.upper()

# Literal types
Mode = Literal["r", "w", "a"]

def open_file(path: str, mode: Mode) -> None:
    pass
```

### Q18: Slots and Memory Optimization
```python
# Regular class uses __dict__ for attributes
class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Slots class uses fixed array
class SlottedPoint:
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Memory comparison
import sys
regular = RegularPoint(1, 2)
slotted = SlottedPoint(1, 2)

print(sys.getsizeof(regular.__dict__))  # ~104 bytes
# slotted has no __dict__, saves ~40-50% memory per instance

# Slots with inheritance
class SlottedPoint3D(SlottedPoint):
    __slots__ = ('z',)  # Only add new slots
    
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z
```

### Q19: Dataclasses and Named Tuples
```python
from dataclasses import dataclass, field, asdict, astuple
from typing import NamedTuple

# Dataclass
@dataclass
class Person:
    name: str
    age: int
    email: str = ""
    tags: list = field(default_factory=list)
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Frozen (immutable) dataclass
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# Named tuple (immutable, memory efficient)
class Coordinate(NamedTuple):
    x: float
    y: float
    z: float = 0.0

coord = Coordinate(1.0, 2.0)
x, y, z = coord  # Unpacking works
coord.x  # Named access works

# Conversion
person = Person("Alice", 30)
asdict(person)   # {'name': 'Alice', 'age': 30, ...}
astuple(person)  # ('Alice', 30, '', [])
```

### Q20: Testing in Python
```python
import pytest
from unittest.mock import Mock, patch, MagicMock

# Basic pytest
def test_addition():
    assert 1 + 1 == 2

# Fixtures
@pytest.fixture
def database():
    db = Database()
    db.connect()
    yield db
    db.disconnect()

def test_query(database):
    result = database.query("SELECT * FROM users")
    assert len(result) > 0

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("", 0),
    ("world", 5),
])
def test_length(input, expected):
    assert len(input) == expected

# Mocking
def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'status': 'ok'}
        result = fetch_data()
        assert result['status'] == 'ok'
        mock_get.assert_called_once()

# Exception testing
def test_raises():
    with pytest.raises(ValueError, match="invalid"):
        raise ValueError("invalid input")
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
