# Arrays & Strings - Complete In-Depth Guide

## ⚡ Interview Quick Summary

> **Core insight**: Arrays and strings account for ~40% of coding interview problems. Master two pointers and sliding window — they convert O(n²) brute-force solutions into O(n).

### Pattern Recognition

```
Problem asks for...               Use this pattern
──────────────────────────────────────────────────
Pair/triplet in sorted array    → Two Pointers (opposite ends)
Remove duplicates in-place       → Two Pointers (fast/slow)
Subarray/substring of size k    → Fixed Sliding Window
Longest subarray satisfying X   → Variable Sliding Window
Subarray sum equals target       → Prefix Sum + HashMap
Contiguous subarray max sum      → Kadane's Algorithm
```

### Edge Cases — Always Mention These

```
Empty array/string:    if not nums: return ...
Single element:        len(nums) == 1 → often trivial
All same elements:     [5,5,5,5] → check for duplicates logic
All negative:          max_sum = nums[0] (not 0) in Kadane's
Integer overflow:      use Python (unlimited int) or check bounds
Sorted vs unsorted:    sorted enables binary search, two pointers
```

### Two Pointers Template

```python
left, right = 0, len(arr) - 1
while left < right:
    if condition_met(arr[left], arr[right]):
        # process result
        left += 1; right -= 1
    elif need_larger:
        left += 1
    else:
        right -= 1
```

### Sliding Window Template (Variable)

```python
left = 0
window_state = {}           # track what's in window
result = 0
for right in range(len(s)):
    # 1. Expand window (add s[right])
    window_state[s[right]] = window_state.get(s[right], 0) + 1
    # 2. Shrink window while invalid
    while window_is_invalid(window_state):
        window_state[s[left]] -= 1
        if window_state[s[left]] == 0:
            del window_state[s[left]]
        left += 1
    # 3. Update result with valid window
    result = max(result, right - left + 1)
```

### 🚨 Top Interview Pitfalls
- **Off-by-one**: sliding window length = `right - left + 1`, not `right - left`
- **Two pointers on unsorted data**: must sort first (or use a hash map instead)
- **Kadane's with all negatives**: initialize `max_sum = nums[0]`, NOT `0` (empty subarray is not allowed)
- **Prefix sum initialization**: `prefix[0] = 0` (empty prefix), actual values start at `prefix[1]`

---

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Two Pointers](#two-pointers)
3. [Sliding Window](#sliding-window)
4. [Prefix Sum](#prefix-sum)
5. [Kadane's Algorithm](#kadanes-algorithm)
6. [Matrix Problems](#matrix-problems)
7. [String Manipulation](#string-manipulation)
8. [Pattern Matching](#pattern-matching)
9. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Time & Space Complexities

| Operation | Static Array | Dynamic Array (List) | Notes |
|-----------|-------------|---------------------|-------|
| Access by Index | O(1) | O(1) | Direct memory access |
| Search (unsorted) | O(n) | O(n) | Linear scan |
| Search (sorted) | O(log n) | O(log n) | Binary search |
| Insert at End | N/A | O(1) amortized | May trigger resize |
| Insert at Beginning | O(n) | O(n) | Shift all elements |
| Insert at Middle | O(n) | O(n) | Shift elements after |
| Delete at End | N/A | O(1) | Simple removal |
| Delete at Beginning | O(n) | O(n) | Shift all elements |
| Delete at Middle | O(n) | O(n) | Shift elements after |

### Python List Internals

```python
import sys

# Python list is a dynamic array
arr = []
print(f"Empty list size: {sys.getsizeof(arr)} bytes")

# Observe capacity growth (over-allocation strategy)
for i in range(20):
    arr.append(i)
    print(f"Length: {len(arr)}, Size: {sys.getsizeof(arr)} bytes")

# List comprehension vs append (list comp is faster)
import timeit

# Slower - function call overhead
def using_append():
    result = []
    for i in range(10000):
        result.append(i * 2)
    return result

# Faster - optimized C implementation
def using_comprehension():
    return [i * 2 for i in range(10000)]

# Memory-efficient for large iterables
def using_generator():
    return (i * 2 for i in range(10000))
```

### Array vs List vs Tuple vs NumPy

```python
import array
import numpy as np

# Python list - heterogeneous, dynamic
py_list = [1, 'hello', 3.14, [1, 2]]

# array.array - homogeneous, more memory efficient
int_array = array.array('i', [1, 2, 3, 4, 5])  # 'i' = signed int
float_array = array.array('d', [1.0, 2.0, 3.0])  # 'd' = double

# NumPy array - homogeneous, vectorized operations
np_array = np.array([1, 2, 3, 4, 5])
# Vectorized operations (no Python loops)
result = np_array * 2 + 1  # [3, 5, 7, 9, 11]

# Tuple - immutable, hashable (can be dict key)
my_tuple = (1, 2, 3)
# my_tuple[0] = 5  # TypeError: 'tuple' object does not support item assignment

# Memory comparison
import sys
py_list = list(range(1000))
np_arr = np.arange(1000, dtype=np.int64)
print(f"Python list: {sys.getsizeof(py_list)} bytes")
print(f"NumPy array: {np_arr.nbytes} bytes")  # Much smaller
```

---

## 2. Two Pointers

### Pattern 1: Opposite Direction (Start & End)

**When to use**: Sorted arrays, palindrome checks, pair sum problems

```python
def two_sum_sorted(arr: list[int], target: int) -> list[int]:
    """
    Find two numbers that sum to target in sorted array.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []  # No solution found


def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Find all unique triplets that sum to zero.
    Time: O(n²), Space: O(1) excluding output
    
    Key insight: Fix one number, use two pointers for remaining two.
    """
    nums.sort()  # Required for two pointers
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Early termination: if smallest is positive, no solution
        if nums[i] > 0:
            break
        
        left, right = i + 1, n - 1
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


def four_sum(nums: list[int], target: int) -> list[list[int]]:
    """
    Find all unique quadruplets that sum to target.
    Time: O(n³), Space: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Pruning: min/max possible sums
        if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
            break
        if nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target:
            continue
        
        for j in range(i + 1, n - 2):
            # Skip duplicates
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            
            left, right = j + 1, n - 1
            remaining = target - nums[i] - nums[j]
            
            while left < right:
                current = nums[left] + nums[right]
                if current == remaining:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current < remaining:
                    left += 1
                else:
                    right -= 1
    
    return result


def container_with_most_water(height: list[int]) -> int:
    """
    Find two lines that together with x-axis forms a container with max water.
    Time: O(n), Space: O(1)
    
    Key insight: Always move the shorter line inward (might find taller one).
    """
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Water level is limited by shorter line
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        
        # Move the shorter line (greedy choice)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water


def trapping_rain_water(height: list[int]) -> int:
    """
    Calculate total water that can be trapped.
    Time: O(n), Space: O(1)
    
    Key insight: Water at any position = min(max_left, max_right) - height
    Two pointer approach tracks max from both sides.
    """
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


def valid_palindrome_ii(s: str) -> bool:
    """
    Check if string can become palindrome by removing at most one character.
    Time: O(n), Space: O(1)
    """
    def is_palindrome(left: int, right: int) -> bool:
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try removing either character
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    
    return True
```

### Pattern 2: Same Direction (Fast & Slow)

**When to use**: Remove duplicates, partition arrays, Dutch National Flag

```python
def remove_duplicates_sorted(nums: list[int]) -> int:
    """
    Remove duplicates in-place from sorted array.
    Time: O(n), Space: O(1)
    
    Slow pointer: position to write next unique element
    Fast pointer: scans through array
    """
    if not nums:
        return 0
    
    slow = 0  # Position of last unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Length of unique portion


def remove_duplicates_allow_two(nums: list[int]) -> int:
    """
    Remove duplicates allowing at most two of each.
    Time: O(n), Space: O(1)
    """
    if len(nums) <= 2:
        return len(nums)
    
    slow = 2  # First two elements always kept
    
    for fast in range(2, len(nums)):
        # Compare with element two positions back
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow


def move_zeroes(nums: list[int]) -> None:
    """
    Move all zeroes to end while maintaining order of non-zero elements.
    Time: O(n), Space: O(1)
    """
    slow = 0  # Position for next non-zero
    
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


def sort_colors(nums: list[int]) -> None:
    """
    Dutch National Flag Problem - Sort array of 0s, 1s, 2s.
    Time: O(n), Space: O(1)
    
    Three pointers: low (next 0), mid (current), high (next 2)
    """
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid - need to check swapped element


def partition_array(nums: list[int], pivot: int) -> None:
    """
    Partition array around pivot (elements < pivot come first).
    Time: O(n), Space: O(1)
    """
    slow = 0
    
    for fast in range(len(nums)):
        if nums[fast] < pivot:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

---

## 3. Sliding Window

### Fixed-Size Window

```python
def max_sum_subarray_size_k(nums: list[int], k: int) -> int:
    """
    Maximum sum of any contiguous subarray of size k.
    Time: O(n), Space: O(1)
    """
    if len(nums) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide window: add new element, remove old element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum


def find_all_anagrams(s: str, p: str) -> list[int]:
    """
    Find all start indices of p's anagrams in s.
    Time: O(n), Space: O(1) - fixed 26 characters
    
    LeetCode 438
    """
    from collections import Counter
    
    if len(p) > len(s):
        return []
    
    p_count = Counter(p)
    s_count = Counter(s[:len(p)])
    result = []
    
    if s_count == p_count:
        result.append(0)
    
    for i in range(len(p), len(s)):
        # Add new character
        s_count[s[i]] += 1
        
        # Remove old character
        old_char = s[i - len(p)]
        s_count[old_char] -= 1
        if s_count[old_char] == 0:
            del s_count[old_char]
        
        if s_count == p_count:
            result.append(i - len(p) + 1)
    
    return result


def max_consecutive_ones_iii(nums: list[int], k: int) -> int:
    """
    Maximum consecutive 1s if you can flip at most k 0s.
    Time: O(n), Space: O(1)
    
    LeetCode 1004
    """
    left = 0
    zeros = 0
    max_len = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        
        # Shrink window if too many zeros
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### Variable-Size Window

```python
def longest_substring_without_repeating(s: str) -> int:
    """
    Longest substring without repeating characters.
    Time: O(n), Space: O(min(m, n)) where m = charset size
    
    LeetCode 3
    """
    char_index = {}  # char -> last seen index
    left = 0
    max_len = 0
    
    for right, char in enumerate(s):
        # If char seen and within current window, shrink window
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len


def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Longest substring with at most k distinct characters.
    Time: O(n), Space: O(k)
    
    LeetCode 340
    """
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_len = 0
    
    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        
        # Shrink window while more than k distinct chars
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len


def minimum_window_substring(s: str, t: str) -> str:
    """
    Minimum window in s containing all characters of t.
    Time: O(n + m), Space: O(m)
    
    LeetCode 76 - HARD but very common!
    """
    from collections import Counter
    
    if not s or not t or len(s) < len(t):
        return ""
    
    t_count = Counter(t)
    required = len(t_count)  # Number of unique characters needed
    
    left = 0
    formed = 0  # Number of unique chars with desired frequency
    window_counts = {}
    
    # (window_length, left, right)
    result = (float('inf'), 0, 0)
    
    for right, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1
        
        # Check if current char satisfies requirement
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Try to shrink window while valid
        while formed == required:
            # Update result if smaller window found
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            # Remove left char from window
            left_char = s[left]
            window_counts[left_char] -= 1
            
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]


def longest_repeating_character_replacement(s: str, k: int) -> int:
    """
    Longest substring with same letter after at most k replacements.
    Time: O(n), Space: O(26) = O(1)
    
    LeetCode 424
    
    Key insight: Valid window if (window_size - max_freq) <= k
    """
    char_count = {}
    left = 0
    max_freq = 0  # Max frequency of any char in current window
    max_len = 0
    
    for right, char in enumerate(s):
        char_count[char] = char_count.get(char, 0) + 1
        max_freq = max(max_freq, char_count[char])
        
        # Window invalid if chars to replace > k
        window_size = right - left + 1
        if window_size - max_freq > k:
            char_count[s[left]] -= 1
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len


def fruit_into_baskets(fruits: list[int]) -> int:
    """
    Maximum fruits you can pick with 2 baskets (each holds one type).
    Time: O(n), Space: O(1)
    
    LeetCode 904 - Essentially longest subarray with at most 2 distinct
    """
    basket = {}
    left = 0
    max_fruits = 0
    
    for right, fruit in enumerate(fruits):
        basket[fruit] = basket.get(fruit, 0) + 1
        
        while len(basket) > 2:
            left_fruit = fruits[left]
            basket[left_fruit] -= 1
            if basket[left_fruit] == 0:
                del basket[left_fruit]
            left += 1
        
        max_fruits = max(max_fruits, right - left + 1)
    
    return max_fruits
```

### Sliding Window Template

```python
def sliding_window_template(s: str, pattern_or_condition) -> any:
    """
    General sliding window template for substring problems.
    """
    from collections import defaultdict
    
    # Initialize window and pattern tracking
    window = defaultdict(int)
    need = defaultdict(int)  # If comparing to pattern
    
    left = 0
    valid = 0  # Condition counter
    result = ...  # Initialize based on problem (min/max length, count, etc.)
    
    for right in range(len(s)):
        # 1. Expand window: add s[right] to window
        char = s[right]
        # Update window state...
        
        # 2. Check if need to shrink (problem-specific condition)
        while "window needs shrink":
            # 3. Update result if valid
            # ...
            
            # 4. Shrink window: remove s[left] from window
            left_char = s[left]
            # Update window state...
            left += 1
    
    return result
```

---

## 4. Prefix Sum

### Basic Prefix Sum

```python
class PrefixSum:
    """
    Prefix sum for O(1) range queries after O(n) preprocessing.
    """
    def __init__(self, nums: list[int]):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def range_sum(self, left: int, right: int) -> int:
        """Sum of elements from index left to right (inclusive)."""
        return self.prefix[right + 1] - self.prefix[left]


def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """
    Count subarrays with sum exactly k.
    Time: O(n), Space: O(n)
    
    LeetCode 560
    
    Key insight: If prefix_sum[j] - prefix_sum[i] = k, 
    then subarray (i, j] sums to k.
    """
    prefix_sum = 0
    count = 0
    sum_count = {0: 1}  # Sum -> frequency
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count


def subarray_sum_divisible_by_k(nums: list[int], k: int) -> int:
    """
    Count subarrays with sum divisible by k.
    Time: O(n), Space: O(k)
    
    LeetCode 974
    
    Key insight: If two prefix sums have same remainder when divided by k,
    their difference is divisible by k.
    """
    prefix_sum = 0
    count = 0
    remainder_count = {0: 1}
    
    for num in nums:
        prefix_sum += num
        remainder = prefix_sum % k
        
        # Handle negative remainders
        if remainder < 0:
            remainder += k
        
        if remainder in remainder_count:
            count += remainder_count[remainder]
        
        remainder_count[remainder] = remainder_count.get(remainder, 0) + 1
    
    return count


def contiguous_array(nums: list[int]) -> int:
    """
    Longest subarray with equal 0s and 1s.
    Time: O(n), Space: O(n)
    
    LeetCode 525
    
    Key insight: Treat 0 as -1, find longest subarray with sum 0.
    """
    prefix_sum = 0
    max_len = 0
    sum_index = {0: -1}  # Sum -> first occurrence index
    
    for i, num in enumerate(nums):
        prefix_sum += 1 if num == 1 else -1
        
        if prefix_sum in sum_index:
            max_len = max(max_len, i - sum_index[prefix_sum])
        else:
            sum_index[prefix_sum] = i
    
    return max_len


def product_except_self(nums: list[int]) -> list[int]:
    """
    Product of all elements except self without division.
    Time: O(n), Space: O(1) excluding output
    
    LeetCode 238
    """
    n = len(nums)
    result = [1] * n
    
    # Left products: result[i] = product of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right products: multiply by product of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result
```

### 2D Prefix Sum

```python
class NumMatrix:
    """
    2D prefix sum for O(1) region queries.
    
    LeetCode 304
    """
    def __init__(self, matrix: list[list[int]]):
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        # prefix[i][j] = sum of rectangle from (0,0) to (i-1, j-1)
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.prefix[i][j] = (
                    matrix[i-1][j-1]
                    + self.prefix[i-1][j]
                    + self.prefix[i][j-1]
                    - self.prefix[i-1][j-1]  # Subtract overlap
                )
    
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Sum of elements in rectangle from (row1, col1) to (row2, col2)."""
        return (
            self.prefix[row2+1][col2+1]
            - self.prefix[row1][col2+1]
            - self.prefix[row2+1][col1]
            + self.prefix[row1][col1]  # Add back the over-subtracted corner
        )


def max_sum_rectangle(matrix: list[list[int]], k: int) -> int:
    """
    Maximum sum rectangle no larger than k.
    Time: O(m² * n * log n), Space: O(n)
    
    LeetCode 363
    """
    import bisect
    
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    result = float('-inf')
    
    # Fix left and right columns
    for left in range(n):
        row_sums = [0] * m
        
        for right in range(left, n):
            # Update row sums for current column range
            for i in range(m):
                row_sums[i] += matrix[i][right]
            
            # Find max subarray sum <= k in row_sums
            prefix_sums = [0]
            curr_sum = 0
            
            for row_sum in row_sums:
                curr_sum += row_sum
                # Find smallest prefix_sum >= curr_sum - k
                idx = bisect.bisect_left(prefix_sums, curr_sum - k)
                if idx < len(prefix_sums):
                    result = max(result, curr_sum - prefix_sums[idx])
                bisect.insort(prefix_sums, curr_sum)
    
    return result
```

---

## 5. Kadane's Algorithm

```python
def max_subarray(nums: list[int]) -> int:
    """
    Maximum sum of contiguous subarray.
    Time: O(n), Space: O(1)
    
    LeetCode 53
    
    DP interpretation: max_ending_here[i] = max(nums[i], max_ending_here[i-1] + nums[i])
    """
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        # Either start fresh or extend previous subarray
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def max_subarray_with_indices(nums: list[int]) -> tuple[int, int, int]:
    """
    Return max sum and the start/end indices of the subarray.
    """
    max_sum = current_sum = nums[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end


def max_subarray_circular(nums: list[int]) -> int:
    """
    Maximum sum of circular subarray.
    Time: O(n), Space: O(1)
    
    LeetCode 918
    
    Key insight: Max circular = Total - Min subarray (if not all negative)
    """
    total = 0
    max_sum = min_sum = nums[0]
    current_max = current_min = nums[0]
    
    for num in nums[1:]:
        total += num
        current_max = max(num, current_max + num)
        current_min = min(num, current_min + num)
        max_sum = max(max_sum, current_max)
        min_sum = min(min_sum, current_min)
    
    total += nums[0]
    
    # If all numbers are negative, max_sum is the answer
    if max_sum < 0:
        return max_sum
    
    return max(max_sum, total - min_sum)


def max_product_subarray(nums: list[int]) -> int:
    """
    Maximum product of contiguous subarray.
    Time: O(n), Space: O(1)
    
    LeetCode 152
    
    Key insight: Track both max and min (negative * negative = positive)
    """
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    
    return result
```

---

## 6. Matrix Problems

```python
def rotate_matrix_90_clockwise(matrix: list[list[int]]) -> None:
    """
    Rotate matrix 90 degrees clockwise in-place.
    Time: O(n²), Space: O(1)
    
    LeetCode 48
    
    Method: Transpose + Reverse each row
    """
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for row in matrix:
        row.reverse()


def spiral_order(matrix: list[list[int]]) -> list[int]:
    """
    Return elements in spiral order.
    Time: O(m*n), Space: O(1) excluding output
    
    LeetCode 54
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Left to right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Top to bottom
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Right to left (if rows remain)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Bottom to top (if columns remain)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result


def set_matrix_zeroes(matrix: list[list[int]]) -> None:
    """
    If element is 0, set entire row and column to 0.
    Time: O(m*n), Space: O(1)
    
    LeetCode 73
    
    Use first row/column as markers.
    """
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    
    # Mark zeros in first row/column
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0


def search_2d_matrix(matrix: list[list[int]], target: int) -> bool:
    """
    Search in row-sorted and column-sorted matrix.
    Time: O(m + n), Space: O(1)
    
    LeetCode 240
    
    Start from top-right (or bottom-left) corner.
    """
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1  # Start top-right
    
    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return False


def diagonal_traverse(matrix: list[list[int]]) -> list[int]:
    """
    Traverse matrix in diagonal order.
    Time: O(m*n), Space: O(1) excluding output
    
    LeetCode 498
    """
    if not matrix:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    row, col = 0, 0
    going_up = True
    
    while len(result) < m * n:
        result.append(matrix[row][col])
        
        if going_up:
            if col == n - 1:  # Hit right boundary
                row += 1
                going_up = False
            elif row == 0:  # Hit top boundary
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == m - 1:  # Hit bottom boundary
                col += 1
                going_up = True
            elif col == 0:  # Hit left boundary
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    
    return result
```

---

## 7. String Manipulation

```python
def reverse_words(s: str) -> str:
    """
    Reverse the order of words in a string.
    Time: O(n), Space: O(n)
    
    LeetCode 151
    """
    return ' '.join(s.split()[::-1])


def reverse_words_in_place(s: list[str]) -> None:
    """
    Reverse words in-place (given as char array).
    Time: O(n), Space: O(1)
    """
    def reverse(start: int, end: int):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
    
    # Reverse entire string
    reverse(0, len(s) - 1)
    
    # Reverse each word
    start = 0
    for i in range(len(s) + 1):
        if i == len(s) or s[i] == ' ':
            reverse(start, i - 1)
            start = i + 1


def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams together.
    Time: O(n * k log k) where k = max string length
    Space: O(n * k)
    
    LeetCode 49
    """
    from collections import defaultdict
    
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())


def group_anagrams_optimal(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using character count as key.
    Time: O(n * k) where k = max string length
    Space: O(n * k)
    """
    from collections import defaultdict
    
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Use character count tuple as key
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        anagram_groups[tuple(count)].append(s)
    
    return list(anagram_groups.values())


def valid_anagram(s: str, t: str) -> bool:
    """
    Check if t is an anagram of s.
    Time: O(n), Space: O(1) - fixed 26 chars
    
    LeetCode 242
    """
    if len(s) != len(t):
        return False
    
    count = [0] * 26
    
    for i in range(len(s)):
        count[ord(s[i]) - ord('a')] += 1
        count[ord(t[i]) - ord('a')] -= 1
    
    return all(c == 0 for c in count)


def longest_palindromic_substring(s: str) -> str:
    """
    Find longest palindromic substring.
    Time: O(n²), Space: O(1)
    
    LeetCode 5
    
    Expand around center approach.
    """
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    result = ""
    for i in range(len(s)):
        # Odd length palindrome
        odd = expand_around_center(i, i)
        if len(odd) > len(result):
            result = odd
        
        # Even length palindrome
        even = expand_around_center(i, i + 1)
        if len(even) > len(result):
            result = even
    
    return result


def count_palindromic_substrings(s: str) -> int:
    """
    Count all palindromic substrings.
    Time: O(n²), Space: O(1)
    
    LeetCode 647
    """
    def count_palindromes(left: int, right: int) -> int:
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total = 0
    for i in range(len(s)):
        total += count_palindromes(i, i)      # Odd length
        total += count_palindromes(i, i + 1)  # Even length
    
    return total


def encode_decode_strings(strs: list[str]) -> str:
    """
    Encode list of strings to single string and decode back.
    
    LeetCode 271
    """
    def encode(strs: list[str]) -> str:
        """Encode: length + '#' + string"""
        return ''.join(f"{len(s)}#{s}" for s in strs)
    
    def decode(s: str) -> list[str]:
        result = []
        i = 0
        while i < len(s):
            # Find delimiter
            j = s.index('#', i)
            length = int(s[i:j])
            result.append(s[j + 1:j + 1 + length])
            i = j + 1 + length
        return result
    
    return encode, decode
```

---

## 8. Pattern Matching

### KMP Algorithm

```python
def kmp_search(text: str, pattern: str) -> list[int]:
    """
    KMP string matching algorithm.
    Time: O(n + m), Space: O(m)
    
    Returns all starting indices of pattern in text.
    """
    def build_lps(pattern: str) -> list[int]:
        """
        Build Longest Proper Prefix which is also Suffix array.
        lps[i] = length of longest proper prefix of pattern[0..i]
                 which is also a suffix.
        """
        m = len(pattern)
        lps = [0] * m
        length = 0  # Length of previous longest prefix suffix
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]  # Try shorter prefix
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    if m == 0:
        return [0]
    if m > n:
        return []
    
    lps = build_lps(pattern)
    result = []
    i = j = 0  # i for text, j for pattern
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            result.append(i - j)
            j = lps[j - 1]  # Continue searching
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result


def rabin_karp(text: str, pattern: str, prime: int = 101) -> list[int]:
    """
    Rabin-Karp string matching using rolling hash.
    Time: O(n + m) average, O(nm) worst case
    Space: O(1)
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    base = 256
    
    # Calculate pattern hash and first window hash
    pattern_hash = 0
    window_hash = 0
    h = pow(base, m - 1, prime)  # base^(m-1) % prime
    
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % prime
        window_hash = (window_hash * base + ord(text[i])) % prime
    
    result = []
    
    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            # Verify match (handle hash collision)
            if text[i:i + m] == pattern:
                result.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            window_hash = (window_hash - ord(text[i]) * h) * base + ord(text[i + m])
            window_hash %= prime
            if window_hash < 0:
                window_hash += prime
    
    return result


def z_algorithm(s: str) -> list[int]:
    """
    Z-algorithm: z[i] = length of longest substring starting at i
                        that is also a prefix of s.
    Time: O(n), Space: O(n)
    
    Useful for pattern matching: concat pattern + '$' + text
    """
    n = len(s)
    z = [0] * n
    z[0] = n
    left = right = 0
    
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        if i + z[i] > right:
            left = i
            right = i + z[i]
    
    return z


def pattern_match_z(text: str, pattern: str) -> list[int]:
    """Use Z-algorithm for pattern matching."""
    combined = pattern + '$' + text
    z = z_algorithm(combined)
    
    result = []
    for i in range(len(pattern) + 1, len(combined)):
        if z[i] == len(pattern):
            result.append(i - len(pattern) - 1)
    
    return result
```

---

## 9. Interview Problems

### Must-Know Problems

```python
def best_time_to_buy_sell_stock(prices: list[int]) -> int:
    """
    LeetCode 121 - One transaction allowed.
    Time: O(n), Space: O(1)
    """
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit


def best_time_multiple_transactions(prices: list[int]) -> int:
    """
    LeetCode 122 - Unlimited transactions.
    Time: O(n), Space: O(1)
    """
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    LeetCode 56
    Time: O(n log n), Space: O(n)
    """
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # Overlapping
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged


def insert_interval(intervals: list[list[int]], new: list[int]) -> list[list[int]]:
    """
    LeetCode 57
    Time: O(n), Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before new interval
    while i < n and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals with new
    while i < n and intervals[i][0] <= new[1]:
        new[0] = min(new[0], intervals[i][0])
        new[1] = max(new[1], intervals[i][1])
        i += 1
    result.append(new)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result


def meeting_rooms_ii(intervals: list[list[int]]) -> int:
    """
    LeetCode 253 - Minimum meeting rooms required.
    Time: O(n log n), Space: O(n)
    """
    import heapq
    
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    rooms = [intervals[0][1]]  # Min-heap of end times
    
    for start, end in intervals[1:]:
        if start >= rooms[0]:  # Can reuse earliest ending room
            heapq.heappop(rooms)
        heapq.heappush(rooms, end)
    
    return len(rooms)


def non_overlapping_intervals(intervals: list[list[int]]) -> int:
    """
    LeetCode 435 - Minimum intervals to remove.
    Time: O(n log n), Space: O(1)
    
    Greedy: Sort by end time, keep non-overlapping.
    """
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])
    count = 0
    prev_end = float('-inf')
    
    for start, end in intervals:
        if start >= prev_end:  # No overlap
            prev_end = end
        else:
            count += 1  # Remove this interval
    
    return count


def first_missing_positive(nums: list[int]) -> int:
    """
    LeetCode 41
    Time: O(n), Space: O(1)
    
    Use array itself as hash table: nums[i] should equal i+1
    """
    n = len(nums)
    
    # Place each number in its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
    
    # Find first position where nums[i] != i+1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1


def longest_consecutive_sequence(nums: list[int]) -> int:
    """
    LeetCode 128
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start counting from sequence start
        if num - 1 not in num_set:
            current = num
            streak = 1
            
            while current + 1 in num_set:
                current += 1
                streak += 1
            
            longest = max(longest, streak)
    
    return longest


def next_permutation(nums: list[int]) -> None:
    """
    LeetCode 31
    Time: O(n), Space: O(1)
    
    1. Find largest i where nums[i] < nums[i+1]
    2. Find largest j > i where nums[j] > nums[i]
    3. Swap nums[i] and nums[j]
    4. Reverse suffix starting at i+1
    """
    n = len(nums)
    i = n - 2
    
    # Step 1: Find first decreasing element from right
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Step 2: Find element just larger than nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        # Step 3: Swap
        nums[i], nums[j] = nums[j], nums[i]
    
    # Step 4: Reverse suffix
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
```

### Complexity Summary

| Problem | Time | Space | Key Technique |
|---------|------|-------|---------------|
| Two Sum | O(n) | O(n) | Hash map |
| Three Sum | O(n²) | O(1) | Sort + two pointers |
| Max Subarray | O(n) | O(1) | Kadane's |
| Merge Intervals | O(n log n) | O(n) | Sort + merge |
| Trapping Rain Water | O(n) | O(1) | Two pointers |
| Min Window Substring | O(n) | O(k) | Sliding window |
| Longest Palindrome | O(n²) | O(1) | Expand around center |
| KMP Pattern Match | O(n+m) | O(m) | Prefix function |
