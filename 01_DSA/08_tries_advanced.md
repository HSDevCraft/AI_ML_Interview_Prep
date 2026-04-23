# Tries & Advanced Data Structures - Complete Guide

## Table of Contents
1. [Trie (Prefix Tree)](#trie)
2. [Segment Tree](#segment-tree)
3. [Fenwick Tree (BIT)](#fenwick-tree)
4. [Monotonic Stack/Queue Review](#monotonic-structures)
5. [Interview Problems](#interview-problems)

---

## 1. Trie (Prefix Tree)

### Basic Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Optional: count words with this prefix


class Trie:
    """
    LeetCode 208: Implement Trie.
    
    Time: O(m) for all operations where m = word length
    Space: O(ALPHABET_SIZE * m * n) where n = number of words
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """Check if word exists in trie."""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def delete(self, word: str) -> bool:
        """Delete word from trie."""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0
            
            char = word[depth]
            if char not in node.children:
                return False
            
            should_delete = _delete(node.children[char], word, depth + 1)
            
            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end
            
            return False
        
        return _delete(self.root, word, 0)
    
    def count_prefix(self, prefix: str) -> int:
        """Count words with given prefix."""
        node = self._find_node(prefix)
        return node.count if node else 0
```

### Trie with Wildcard Search

```python
class WordDictionary:
    """
    LeetCode 211: Add and search words with '.' wildcard.
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            
            char = word[i]
            
            if char == '.':
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], i + 1)
        
        return dfs(self.root, 0)
```

### Trie Applications

```python
def word_search_ii(board: list[list[str]], words: list[str]) -> list[str]:
    """
    LeetCode 212: Find all words from dictionary on board.
    
    Build trie from words, then DFS on board.
    """
    # Build trie
    root = {}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = word
    
    m, n = len(board), len(board[0])
    result = []
    
    def dfs(i, j, node):
        char = board[i][j]
        if char not in node:
            return
        
        next_node = node[char]
        
        if '$' in next_node:
            result.append(next_node['$'])
            del next_node['$']  # Avoid duplicates
        
        board[i][j] = '#'  # Mark visited
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#':
                dfs(ni, nj, next_node)
        
        board[i][j] = char  # Restore
        
        # Prune empty branches
        if not next_node:
            del node[char]
    
    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    
    return result


def replace_words(dictionary: list[str], sentence: str) -> str:
    """
    LeetCode 648: Replace words with their shortest root.
    """
    # Build trie from dictionary
    root = {}
    for word in dictionary:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = word
    
    def find_root(word):
        node = root
        for char in word:
            if '$' in node:
                return node['$']
            if char not in node:
                return word
            node = node[char]
        return node.get('$', word)
    
    return ' '.join(find_root(word) for word in sentence.split())


def longest_word_in_dictionary(words: list[str]) -> str:
    """
    LeetCode 720: Longest word that can be built character by character.
    """
    # Build trie
    root = {}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = True
    
    result = ""
    
    def dfs(node, path):
        nonlocal result
        
        if len(path) > len(result) or (len(path) == len(result) and path < result):
            result = path
        
        for char in sorted(node.keys()):
            if char != '$' and '$' in node[char]:
                dfs(node[char], path + char)
    
    dfs(root, "")
    return result


def maximum_xor_two_numbers(nums: list[int]) -> int:
    """
    LeetCode 421: Maximum XOR of two numbers.
    
    Use binary trie to find best XOR pair for each number.
    Time: O(32n) = O(n)
    """
    # Build binary trie
    root = {}
    
    def insert(num):
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    
    def find_max_xor(num):
        node = root
        xor_val = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for max XOR
            opposite = 1 - bit
            if opposite in node:
                xor_val |= (1 << i)
                node = node[opposite]
            else:
                node = node[bit]
        return xor_val
    
    for num in nums:
        insert(num)
    
    return max(find_max_xor(num) for num in nums)


def palindrome_pairs(words: list[str]) -> list[list[int]]:
    """
    LeetCode 336: Find all pairs forming palindromes.
    """
    def is_palindrome(s):
        return s == s[::-1]
    
    # Map word to index
    word_to_idx = {word: i for i, word in enumerate(words)}
    result = []
    
    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix = word[:j]
            suffix = word[j:]
            
            # Case 1: prefix is palindrome, check if reversed suffix exists
            if is_palindrome(prefix):
                reversed_suffix = suffix[::-1]
                if reversed_suffix in word_to_idx and word_to_idx[reversed_suffix] != i:
                    result.append([word_to_idx[reversed_suffix], i])
            
            # Case 2: suffix is palindrome, check if reversed prefix exists
            if j != len(word) and is_palindrome(suffix):
                reversed_prefix = prefix[::-1]
                if reversed_prefix in word_to_idx and word_to_idx[reversed_prefix] != i:
                    result.append([i, word_to_idx[reversed_prefix]])
    
    return result
```

---

## 2. Segment Tree

### Basic Segment Tree (Range Sum)

```python
class SegmentTree:
    """
    Segment Tree for range sum queries and point updates.
    
    Operations:
    - Build: O(n)
    - Query: O(log n)
    - Update: O(log n)
    Space: O(n)
    """
    
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        if nums:
            self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self._build(nums, left_child, start, mid)
            self._build(nums, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update(self, idx: int, val: int) -> None:
        """Update nums[idx] to val."""
        self._update(0, 0, self.n - 1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._update(left_child, start, mid, idx, val)
            else:
                self._update(right_child, mid + 1, end, idx, val)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, left: int, right: int) -> int:
        """Sum of range [left, right]."""
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self._query(left_child, start, mid, left, right)
        right_sum = self._query(right_child, mid + 1, end, left, right)
        
        return left_sum + right_sum


class NumArray:
    """
    LeetCode 307: Range Sum Query - Mutable
    """
    
    def __init__(self, nums: list[int]):
        self.tree = SegmentTree(nums)
        self.nums = nums
    
    def update(self, index: int, val: int) -> None:
        self.tree.update(index, val)
        self.nums[index] = val
    
    def sumRange(self, left: int, right: int) -> int:
        return self.tree.query(left, right)
```

### Segment Tree with Lazy Propagation

```python
class LazySegmentTree:
    """
    Segment Tree with lazy propagation for range updates.
    
    Supports:
    - Range add: O(log n)
    - Range sum query: O(log n)
    """
    
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if nums:
            self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self._build(nums, left_child, start, mid)
            self._build(nums, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def _push_down(self, node, start, end):
        """Propagate lazy value to children."""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.tree[left_child] += self.lazy[node] * (mid - start + 1)
            self.tree[right_child] += self.lazy[node] * (end - mid)
            self.lazy[left_child] += self.lazy[node]
            self.lazy[right_child] += self.lazy[node]
            self.lazy[node] = 0
    
    def range_add(self, left: int, right: int, val: int) -> None:
        """Add val to all elements in range [left, right]."""
        self._range_add(0, 0, self.n - 1, left, right, val)
    
    def _range_add(self, node, start, end, left, right, val):
        if right < start or left > end:
            return
        
        if left <= start and end <= right:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._range_add(left_child, start, mid, left, right, val)
        self._range_add(right_child, mid + 1, end, left, right, val)
        
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0
        
        if left <= start and end <= right:
            return self.tree[node]
        
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        return (self._query(left_child, start, mid, left, right) +
                self._query(right_child, mid + 1, end, left, right))
```

---

## 3. Fenwick Tree (Binary Indexed Tree)

```python
class FenwickTree:
    """
    Binary Indexed Tree for prefix sums and point updates.
    
    Operations:
    - Update: O(log n)
    - Prefix sum query: O(log n)
    - Range sum query: O(log n)
    
    More space-efficient than segment tree but less versatile.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed
    
    @classmethod
    def from_array(cls, nums: list[int]) -> 'FenwickTree':
        """Build from array in O(n)."""
        bit = cls(len(nums))
        for i, num in enumerate(nums):
            bit.add(i, num)
        return bit
    
    def add(self, i: int, delta: int) -> None:
        """Add delta to element at index i (0-indexed)."""
        i += 1  # Convert to 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # Add lowest set bit
    
    def prefix_sum(self, i: int) -> int:
        """Sum of elements [0, i] (0-indexed)."""
        i += 1  # Convert to 1-indexed
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)  # Remove lowest set bit
        return total
    
    def range_sum(self, left: int, right: int) -> int:
        """Sum of elements [left, right] (0-indexed)."""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def update(self, i: int, val: int, old_val: int) -> None:
        """Update element at index i from old_val to val."""
        self.add(i, val - old_val)


def count_smaller_after_self(nums: list[int]) -> list[int]:
    """
    LeetCode 315: Count smaller elements to the right.
    
    Use BIT with coordinate compression.
    Time: O(n log n), Space: O(n)
    """
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_nums)}
    
    n = len(sorted_nums)
    bit = FenwickTree(n)
    result = []
    
    # Process from right to left
    for num in reversed(nums):
        r = rank[num]
        result.append(bit.prefix_sum(r - 1))
        bit.add(r - 1, 1)
    
    return result[::-1]


def range_sum_query_2d(matrix: list[list[int]]) -> None:
    """
    LeetCode 308: 2D BIT for range sum with updates.
    """
    class BIT2D:
        def __init__(self, m: int, n: int):
            self.m = m
            self.n = n
            self.tree = [[0] * (n + 1) for _ in range(m + 1)]
        
        def add(self, row: int, col: int, delta: int):
            i = row + 1
            while i <= self.m:
                j = col + 1
                while j <= self.n:
                    self.tree[i][j] += delta
                    j += j & (-j)
                i += i & (-i)
        
        def prefix_sum(self, row: int, col: int) -> int:
            total = 0
            i = row + 1
            while i > 0:
                j = col + 1
                while j > 0:
                    total += self.tree[i][j]
                    j -= j & (-j)
                i -= i & (-i)
            return total
        
        def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
            return (self.prefix_sum(r2, c2) 
                    - self.prefix_sum(r1 - 1, c2)
                    - self.prefix_sum(r2, c1 - 1)
                    + self.prefix_sum(r1 - 1, c1 - 1))
```

---

## 4. Bit Manipulation

### Basics

```python
# Basic Operations
x & y    # AND - both bits 1
x | y    # OR - either bit 1
x ^ y    # XOR - bits differ
~x       # NOT - flip all bits
x << n   # Left shift - multiply by 2^n
x >> n   # Right shift - divide by 2^n

# Common Tricks
x & 1           # Check if odd
x & (x - 1)     # Clear lowest set bit
x & (-x)        # Get lowest set bit
x | (x - 1)     # Set all bits after lowest set bit

# Check if power of 2
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

# Count set bits (Brian Kernighan)
def count_bits(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# Get/Set/Clear bit at position
def get_bit(n, i):
    return (n >> i) & 1

def set_bit(n, i):
    return n | (1 << i)

def clear_bit(n, i):
    return n & ~(1 << i)

def toggle_bit(n, i):
    return n ^ (1 << i)
```

### Bit Manipulation Problems

```python
def single_number(nums: list[int]) -> int:
    """
    LeetCode 136: Find element appearing once (others twice).
    XOR all numbers - pairs cancel out.
    """
    result = 0
    for num in nums:
        result ^= num
    return result


def single_number_ii(nums: list[int]) -> int:
    """
    LeetCode 137: Find element appearing once (others three times).
    Count bits at each position mod 3.
    """
    result = 0
    for i in range(32):
        bit_sum = sum((num >> i) & 1 for num in nums)
        if bit_sum % 3:
            result |= 1 << i
    
    # Handle negative numbers (Python integers are arbitrary precision)
    if result >= 2**31:
        result -= 2**32
    return result


def single_number_iii(nums: list[int]) -> list[int]:
    """
    LeetCode 260: Find two elements appearing once (others twice).
    """
    # XOR all numbers gives xor of the two unique numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num
    
    # Find rightmost set bit (differs between the two numbers)
    diff_bit = xor_all & (-xor_all)
    
    # Partition numbers based on this bit
    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num
    
    return [a, b]


def missing_number(nums: list[int]) -> int:
    """
    LeetCode 268: Find missing number in [0, n].
    """
    n = len(nums)
    result = n  # Start with n (might be missing)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result


def reverse_bits(n: int) -> int:
    """
    LeetCode 190: Reverse bits of 32-bit integer.
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def hamming_weight(n: int) -> int:
    """
    LeetCode 191: Count number of 1 bits.
    """
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


def hamming_distance(x: int, y: int) -> int:
    """
    LeetCode 461: Hamming distance (differing bits).
    """
    return bin(x ^ y).count('1')


def total_hamming_distance(nums: list[int]) -> int:
    """
    LeetCode 477: Sum of hamming distances between all pairs.
    
    For each bit position: count * (n - count) pairs differ.
    """
    total = 0
    n = len(nums)
    
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        total += count * (n - count)
    
    return total


def subsets_with_bits(nums: list[int]) -> list[list[int]]:
    """
    LeetCode 78: Generate all subsets using bitmask.
    """
    n = len(nums)
    result = []
    
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result


def counting_bits(n: int) -> list[int]:
    """
    LeetCode 338: Count bits for all numbers [0, n].
    
    dp[i] = dp[i >> 1] + (i & 1)
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


def gray_code(n: int) -> list[int]:
    """
    LeetCode 89: Generate n-bit Gray code.
    
    Gray code: g(i) = i ^ (i >> 1)
    """
    return [i ^ (i >> 1) for i in range(1 << n)]
```

---

## 5. Interview Problems

```python
def autocomplete_system():
    """
    LeetCode 642: Design autocomplete system using Trie.
    """
    class AutocompleteSystem:
        def __init__(self, sentences: list[str], times: list[int]):
            self.root = {}
            self.current = self.root
            self.search = ""
            
            for sentence, time in zip(sentences, times):
                self._add(sentence, time)
        
        def _add(self, sentence: str, time: int):
            node = self.root
            for char in sentence:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['#'] = node.get('#', 0) + time
        
        def _dfs(self, node, path, results):
            if '#' in node:
                results.append((-node['#'], path))
            
            for char in node:
                if char != '#':
                    self._dfs(node[char], path + char, results)
        
        def input(self, c: str) -> list[str]:
            if c == '#':
                self._add(self.search, 1)
                self.search = ""
                self.current = self.root
                return []
            
            self.search += c
            
            if self.current is None or c not in self.current:
                self.current = None
                return []
            
            self.current = self.current[c]
            
            results = []
            self._dfs(self.current, self.search, results)
            results.sort()
            
            return [r[1] for r in results[:3]]


def stream_of_characters():
    """
    LeetCode 1032: Check if stream ends with any word.
    
    Use reversed trie for efficient suffix matching.
    """
    class StreamChecker:
        def __init__(self, words: list[str]):
            self.root = {}
            self.stream = []
            
            for word in words:
                node = self.root
                for char in reversed(word):
                    if char not in node:
                        node[char] = {}
                    node = node[char]
                node['$'] = True
        
        def query(self, letter: str) -> bool:
            self.stream.append(letter)
            node = self.root
            
            for char in reversed(self.stream):
                if '$' in node:
                    return True
                if char not in node:
                    return False
                node = node[char]
            
            return '$' in node
```

### Summary Tables

#### Trie Time Complexities
| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(m) | O(m) |
| Search | O(m) | O(1) |
| StartsWith | O(m) | O(1) |
| Delete | O(m) | O(1) |

#### Segment Tree vs BIT
| Feature | Segment Tree | Fenwick Tree |
|---------|--------------|--------------|
| Space | O(4n) | O(n) |
| Point Update | O(log n) | O(log n) |
| Range Query | O(log n) | O(log n) |
| Range Update | O(log n) with lazy | Complex |
| Implementation | Complex | Simple |
| Versatility | High | Low |
