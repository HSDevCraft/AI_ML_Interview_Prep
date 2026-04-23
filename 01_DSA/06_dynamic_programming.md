# Dynamic Programming - Complete In-Depth Guide

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [1D DP Patterns](#1d-dp-patterns)
3. [2D DP Patterns](#2d-dp-patterns)
4. [String DP](#string-dp)
5. [Knapsack Problems](#knapsack-problems)
6. [State Machine DP](#state-machine-dp)
7. [Interval DP](#interval-dp)
8. [DP on Trees](#dp-on-trees)
9. [Bitmask DP](#bitmask-dp)
10. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### What is Dynamic Programming?

```
DP = Optimal Substructure + Overlapping Subproblems

1. Optimal Substructure: 
   Optimal solution can be constructed from optimal solutions of subproblems.

2. Overlapping Subproblems:
   Same subproblems are solved multiple times.

Approaches:
- Top-Down (Memoization): Recursive with cache
- Bottom-Up (Tabulation): Iterative, build solution from smaller to larger

Steps to Solve DP Problems:
1. Define the state (what does dp[i] represent?)
2. Define the base case
3. Define the transition (recurrence relation)
4. Determine the order of computation
5. Optimize space if possible
```

### Memoization vs Tabulation

```python
# Example: Fibonacci

# Top-Down (Memoization)
def fib_memo(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


# Using @lru_cache decorator (Pythonic way)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo_decorator(n: int) -> int:
    if n <= 1:
        return n
    return fib_memo_decorator(n - 1) + fib_memo_decorator(n - 2)


# Bottom-Up (Tabulation)
def fib_tabulation(n: int) -> int:
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


# Space Optimized
def fib_optimized(n: int) -> int:
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    
    return prev1
```

---

## 2. 1D DP Patterns

### Climbing Stairs Pattern

```python
def climbing_stairs(n: int) -> int:
    """
    LeetCode 70: Ways to climb n stairs (1 or 2 steps).
    
    State: dp[i] = number of ways to reach step i
    Transition: dp[i] = dp[i-1] + dp[i-2]
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    
    return prev1


def min_cost_climbing_stairs(cost: list[int]) -> int:
    """
    LeetCode 746: Minimum cost to climb stairs.
    
    State: dp[i] = minimum cost to reach step i
    Transition: dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    """
    n = len(cost)
    prev2, prev1 = 0, 0
    
    for i in range(2, n + 1):
        curr = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
        prev2 = prev1
        prev1 = curr
    
    return prev1
```

### House Robber Pattern

```python
def house_robber(nums: list[int]) -> int:
    """
    LeetCode 198: Max money without robbing adjacent houses.
    
    State: dp[i] = max money robbing houses 0..i
    Transition: dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    
    return prev1


def house_robber_ii(nums: list[int]) -> int:
    """
    LeetCode 213: Circular houses (first and last are adjacent).
    
    Key insight: Can't rob both first and last.
    Solution: max(rob[0..n-2], rob[1..n-1])
    """
    if len(nums) == 1:
        return nums[0]
    
    def rob(houses):
        prev2, prev1 = 0, 0
        for money in houses:
            curr = max(prev1, prev2 + money)
            prev2 = prev1
            prev1 = curr
        return prev1
    
    return max(rob(nums[:-1]), rob(nums[1:]))


def delete_and_earn(nums: list[int]) -> int:
    """
    LeetCode 740: Pick number, delete all num-1 and num+1.
    
    Key insight: Transform to house robber.
    Picking num[i] means can't pick num[i]-1 or num[i]+1.
    """
    from collections import Counter
    
    count = Counter(nums)
    max_num = max(nums)
    
    # Transform: points[i] = i * count[i]
    prev2, prev1 = 0, count.get(1, 0)
    
    for i in range(2, max_num + 1):
        curr = max(prev1, prev2 + i * count.get(i, 0))
        prev2 = prev1
        prev1 = curr
    
    return prev1
```

### Longest Increasing Subsequence (LIS)

```python
def longest_increasing_subsequence(nums: list[int]) -> int:
    """
    LeetCode 300: Length of longest increasing subsequence.
    
    O(n²) DP approach:
    State: dp[i] = length of LIS ending at index i
    Transition: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def lis_binary_search(nums: list[int]) -> int:
    """
    O(n log n) approach using binary search.
    
    Maintain array 'tails' where tails[i] = smallest tail element
    for LIS of length i+1.
    """
    import bisect
    
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def lis_with_sequence(nums: list[int]) -> list[int]:
    """Return the actual LIS, not just length."""
    import bisect
    
    n = len(nums)
    if n == 0:
        return []
    
    tails = []
    tails_idx = []
    parent = [-1] * n
    
    for i, num in enumerate(nums):
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
            tails_idx.append(i)
        else:
            tails[pos] = num
            tails_idx[pos] = i
        
        if pos > 0:
            parent[i] = tails_idx[pos - 1]
    
    # Reconstruct sequence
    lis = []
    idx = tails_idx[-1]
    while idx != -1:
        lis.append(nums[idx])
        idx = parent[idx]
    
    return lis[::-1]


def number_of_lis(nums: list[int]) -> int:
    """
    LeetCode 673: Count number of longest increasing subsequences.
    """
    if not nums:
        return 0
    
    n = len(nums)
    length = [1] * n  # Length of LIS ending at i
    count = [1] * n   # Count of LIS ending at i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    count[i] = count[j]
                elif length[j] + 1 == length[i]:
                    count[i] += count[j]
    
    max_len = max(length)
    return sum(c for l, c in zip(length, count) if l == max_len)
```

### Coin Change Pattern

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    LeetCode 322: Minimum coins to make amount.
    
    State: dp[i] = minimum coins to make amount i
    Transition: dp[i] = min(dp[i - coin] + 1) for each coin
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ii(amount: int, coins: list[int]) -> int:
    """
    LeetCode 518: Number of combinations to make amount.
    
    Key: Iterate coins first to avoid counting permutations.
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]


def combination_sum_iv(nums: list[int], target: int) -> int:
    """
    LeetCode 377: Number of permutations to make target.
    
    Key: Iterate amounts first to count permutations.
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(1, target + 1):
        for num in nums:
            if num <= i:
                dp[i] += dp[i - num]
    
    return dp[target]


def perfect_squares(n: int) -> int:
    """
    LeetCode 279: Minimum perfect squares summing to n.
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    
    return dp[n]
```

### Maximum Subarray Pattern

```python
def max_subarray(nums: list[int]) -> int:
    """
    LeetCode 53: Kadane's Algorithm.
    
    State: dp[i] = max subarray sum ending at index i
    Transition: dp[i] = max(nums[i], dp[i-1] + nums[i])
    """
    max_sum = curr_sum = nums[0]
    
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    
    return max_sum


def max_product_subarray(nums: list[int]) -> int:
    """
    LeetCode 152: Maximum product subarray.
    
    Key: Track both max and min (negative * negative = positive)
    """
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    
    return result


def maximum_sum_circular_subarray(nums: list[int]) -> int:
    """
    LeetCode 918: Max subarray sum (array is circular).
    
    Answer = max(normal_kadane, total_sum - min_subarray)
    """
    total = sum(nums)
    max_sum = min_sum = nums[0]
    curr_max = curr_min = nums[0]
    
    for num in nums[1:]:
        curr_max = max(num, curr_max + num)
        curr_min = min(num, curr_min + num)
        max_sum = max(max_sum, curr_max)
        min_sum = min(min_sum, curr_min)
    
    # If all negative, min_sum would be total
    if max_sum < 0:
        return max_sum
    
    return max(max_sum, total - min_sum)
```

### Word Break Pattern

```python
def word_break(s: str, wordDict: list[str]) -> bool:
    """
    LeetCode 139: Can string be segmented into dictionary words?
    
    State: dp[i] = True if s[0:i] can be segmented
    Transition: dp[i] = any(dp[j] and s[j:i] in wordSet) for j < i
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]


def word_break_ii(s: str, wordDict: list[str]) -> list[str]:
    """
    LeetCode 140: Return all valid segmentations.
    """
    word_set = set(wordDict)
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return [""]
        
        results = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for suffix in backtrack(end):
                    if suffix:
                        results.append(word + " " + suffix)
                    else:
                        results.append(word)
        
        memo[start] = results
        return results
    
    return backtrack(0)
```

---

## 3. 2D DP Patterns

### Grid Path Problems

```python
def unique_paths(m: int, n: int) -> int:
    """
    LeetCode 62: Count paths from top-left to bottom-right.
    
    State: dp[i][j] = number of paths to reach (i, j)
    Transition: dp[i][j] = dp[i-1][j] + dp[i][j-1]
    """
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]


def unique_paths_optimized(m: int, n: int) -> int:
    """Space optimized to O(n)."""
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]


def unique_paths_with_obstacles(obstacleGrid: list[list[int]]) -> int:
    """
    LeetCode 63: Paths with obstacles (1 = blocked).
    """
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    
    if obstacleGrid[0][0] == 1:
        return 0
    
    dp = [0] * n
    dp[0] = 1
    
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    
    return dp[n - 1]


def min_path_sum(grid: list[list[int]]) -> int:
    """
    LeetCode 64: Minimum path sum from top-left to bottom-right.
    """
    m, n = len(grid), len(grid[0])
    dp = [float('inf')] * n
    dp[0] = 0
    
    for i in range(m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]
    
    return dp[n - 1]


def maximal_square(matrix: list[list[str]]) -> int:
    """
    LeetCode 221: Largest square containing only 1s.
    
    State: dp[i][j] = side length of largest square with bottom-right at (i,j)
    Transition: dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    """
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side
```

### Triangle Problem

```python
def minimum_total(triangle: list[list[int]]) -> int:
    """
    LeetCode 120: Minimum path sum in triangle.
    
    Bottom-up approach, modify in place or use O(n) space.
    """
    n = len(triangle)
    dp = triangle[-1][:]  # Start from bottom row
    
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    
    return dp[0]
```

---

## 4. String DP

### Longest Common Subsequence (LCS)

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LeetCode 1143: Length of LCS.
    
    State: dp[i][j] = LCS of text1[0:i] and text2[0:j]
    Transition:
    - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def lcs_space_optimized(text1: str, text2: str) -> int:
    """O(min(m, n)) space."""
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    m, n = len(text1), len(text2)
    dp = [0] * (n + 1)
    
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    
    return dp[n]


def lcs_with_string(text1: str, text2: str) -> str:
    """Return actual LCS string."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

### Edit Distance

```python
def min_distance(word1: str, word2: str) -> int:
    """
    LeetCode 72: Edit distance (Levenshtein distance).
    
    State: dp[i][j] = min operations to convert word1[0:i] to word2[0:j]
    Operations: insert, delete, replace
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from word1
                    dp[i][j - 1],      # Insert into word1
                    dp[i - 1][j - 1]   # Replace in word1
                )
    
    return dp[m][n]


def one_edit_distance(s: str, t: str) -> bool:
    """
    LeetCode 161: Check if exactly one edit away.
    """
    m, n = len(s), len(t)
    
    if abs(m - n) > 1:
        return False
    
    if m > n:
        s, t = t, s
        m, n = n, m
    
    for i in range(m):
        if s[i] != t[i]:
            if m == n:
                return s[i + 1:] == t[i + 1:]  # Replace
            else:
                return s[i:] == t[i + 1:]  # Delete from t
    
    return m + 1 == n  # Delete last char from t
```

### Palindrome DP

```python
def longest_palindromic_subsequence(s: str) -> int:
    """
    LeetCode 516: Length of longest palindromic subsequence.
    
    Key insight: LPS(s) = LCS(s, reverse(s))
    
    Or direct DP:
    State: dp[i][j] = LPS of s[i:j+1]
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1
    
    # Fill diagonally
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]


def min_insertions_palindrome(s: str) -> int:
    """
    LeetCode 1312: Minimum insertions to make palindrome.
    
    Answer = len(s) - LPS(s)
    """
    return len(s) - longest_palindromic_subsequence(s)


def longest_palindromic_substring(s: str) -> str:
    """
    LeetCode 5: Longest palindromic substring.
    
    State: dp[i][j] = True if s[i:j+1] is palindrome
    """
    n = len(s)
    if n < 2:
        return s
    
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    
    # All single chars are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start, max_len = i, 2
    
    # Check length 3 and above
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start, max_len = i, length
    
    return s[start:start + max_len]


def count_palindromic_substrings(s: str) -> int:
    """
    LeetCode 647: Count all palindromic substrings.
    """
    n = len(s)
    count = 0
    
    # Expand around center approach (more efficient)
    def expand(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    for i in range(n):
        expand(i, i)      # Odd length
        expand(i, i + 1)  # Even length
    
    return count


def palindrome_partitioning_min_cuts(s: str) -> int:
    """
    LeetCode 132: Minimum cuts for palindrome partitioning.
    """
    n = len(s)
    
    # is_palin[i][j] = True if s[i:j+1] is palindrome
    is_palin = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or is_palin[i + 1][j - 1]):
                is_palin[i][j] = True
    
    # dp[i] = minimum cuts for s[0:i+1]
    dp = list(range(n))  # Max cuts = i (all single chars)
    
    for i in range(n):
        if is_palin[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_palin[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n - 1]
```

### Distinct Subsequences

```python
def num_distinct(s: str, t: str) -> int:
    """
    LeetCode 115: Count distinct subsequences of s that equal t.
    
    State: dp[i][j] = number of ways to form t[0:j] from s[0:i]
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Empty t can be formed from any prefix of s in 1 way
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Don't use s[i-1]
            dp[i][j] = dp[i - 1][j]
            
            # Use s[i-1] if it matches t[j-1]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]
```

### Regular Expression / Wildcard Matching

```python
def is_match_regex(s: str, p: str) -> bool:
    """
    LeetCode 10: Regular expression matching.
    '.' matches any single character
    '*' matches zero or more of the preceding element
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc.
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Zero occurrences of preceding char
                dp[i][j] = dp[i][j - 2]
                
                # One or more occurrences
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            
            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]


def is_match_wildcard(s: str, p: str) -> bool:
    """
    LeetCode 44: Wildcard matching.
    '?' matches any single character
    '*' matches any sequence (including empty)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle leading *'s
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # * matches empty or extends match
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]
```

---

## 5. Knapsack Problems

### 0/1 Knapsack

```python
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Classic 0/1 Knapsack: Each item can be used at most once.
    
    State: dp[i][w] = max value using items 0..i-1 with capacity w
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i - 1][w]
            
            # Take item i-1 if possible
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    return dp[n][capacity]


def knapsack_01_optimized(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Space optimized to O(capacity).
    Key: Iterate capacity in reverse to avoid using updated values.
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def can_partition(nums: list[int]) -> bool:
    """
    LeetCode 416: Can partition into two equal sum subsets?
    
    This is a 0/1 knapsack where target = sum/2.
    """
    total = sum(nums)
    if total % 2 == 1:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]


def target_sum(nums: list[int], target: int) -> int:
    """
    LeetCode 494: Count ways to assign +/- to reach target.
    
    Let P = sum of positive, N = sum of negative
    P - N = target, P + N = total
    So P = (target + total) / 2
    
    This becomes counting subsets with sum P.
    """
    total = sum(nums)
    
    if (total + target) % 2 == 1 or abs(target) > total:
        return 0
    
    new_target = (total + target) // 2
    dp = [0] * (new_target + 1)
    dp[0] = 1
    
    for num in nums:
        for j in range(new_target, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[new_target]
```

### Unbounded Knapsack

```python
def unbounded_knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Unbounded Knapsack: Each item can be used unlimited times.
    
    Key: Iterate capacity forward (can reuse same item).
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


def coin_change_min(coins: list[int], amount: int) -> int:
    """
    LeetCode 322: Unbounded knapsack variant.
    Minimize coins to make amount.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## 6. State Machine DP

### Best Time to Buy and Sell Stock

```python
def max_profit_one_transaction(prices: list[int]) -> int:
    """
    LeetCode 121: One transaction allowed.
    """
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    
    return max_profit


def max_profit_unlimited(prices: list[int]) -> int:
    """
    LeetCode 122: Unlimited transactions.
    """
    return sum(max(0, prices[i] - prices[i - 1]) 
               for i in range(1, len(prices)))


def max_profit_k_transactions(k: int, prices: list[int]) -> int:
    """
    LeetCode 188: At most k transactions.
    
    States: hold[i][j] = max profit on day i holding stock, j transactions used
            cash[i][j] = max profit on day i not holding, j transactions used
    """
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # Optimization for large k
    if k >= n // 2:
        return max_profit_unlimited(prices)
    
    # dp[j][0] = max profit with j transactions, not holding
    # dp[j][1] = max profit with j transactions, holding
    dp = [[0, float('-inf')] for _ in range(k + 1)]
    
    for price in prices:
        for j in range(k, 0, -1):
            dp[j][0] = max(dp[j][0], dp[j][1] + price)  # Sell
            dp[j][1] = max(dp[j][1], dp[j - 1][0] - price)  # Buy
    
    return dp[k][0]


def max_profit_with_cooldown(prices: list[int]) -> int:
    """
    LeetCode 309: Cooldown day after selling.
    
    States: hold, sold, rest
    """
    if len(prices) < 2:
        return 0
    
    hold = -prices[0]  # Holding stock
    sold = 0           # Just sold
    rest = 0           # Resting (cooldown or idle)
    
    for price in prices[1:]:
        prev_hold = hold
        hold = max(hold, rest - price)  # Buy after rest
        rest = max(rest, sold)          # Continue resting or after cooldown
        sold = prev_hold + price        # Sell what we're holding
    
    return max(sold, rest)


def max_profit_with_fee(prices: list[int], fee: int) -> int:
    """
    LeetCode 714: Transaction fee on each sale.
    """
    hold = -prices[0]  # Holding stock
    cash = 0           # Not holding
    
    for price in prices[1:]:
        hold = max(hold, cash - price)
        cash = max(cash, hold + price - fee)
    
    return cash
```

---

## 7. Interval DP

```python
def burst_balloons(nums: list[int]) -> int:
    """
    LeetCode 312: Maximum coins from bursting balloons.
    
    Key insight: Think in reverse - which balloon to burst LAST in range.
    
    State: dp[i][j] = max coins from bursting balloons in range (i, j) exclusive
    """
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n):  # length of interval
        for left in range(n - length):
            right = left + length
            
            for k in range(left + 1, right):  # Last balloon to burst
                coins = nums[left] * nums[k] * nums[right]
                coins += dp[left][k] + dp[k][right]
                dp[left][right] = max(dp[left][right], coins)
    
    return dp[0][n - 1]


def min_cost_merge_stones(stones: list[int], k: int) -> int:
    """
    LeetCode 1000: Minimum cost to merge stones.
    
    Can only merge k consecutive piles at a time.
    """
    n = len(stones)
    if (n - 1) % (k - 1) != 0:
        return -1
    
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]
    
    # dp[i][j] = min cost to merge stones[i:j+1] into minimum piles
    dp = [[0] * n for _ in range(n)]
    
    for length in range(k, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for mid in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid + 1][j])
            
            if (length - 1) % (k - 1) == 0:
                dp[i][j] += prefix[j + 1] - prefix[i]
    
    return dp[0][n - 1]


def strange_printer(s: str) -> int:
    """
    LeetCode 664: Minimum turns to print string.
    
    Printer can print same character multiple times in one turn.
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            dp[i][j] = dp[i][j - 1] + 1  # Print s[j] separately
            
            for k in range(i, j):
                if s[k] == s[j]:
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j - 1])
    
    return dp[0][n - 1] if n > 0 else 0
```

---

## 8. DP on Trees

```python
def rob_iii(root) -> int:
    """
    LeetCode 337: House Robber III (binary tree).
    
    Each node returns (rob_this, skip_this).
    """
    def dfs(node):
        if not node:
            return (0, 0)
        
        left = dfs(node.left)
        right = dfs(node.right)
        
        # Rob this node: can't rob children
        rob = node.val + left[1] + right[1]
        
        # Skip this node: take max from children
        skip = max(left) + max(right)
        
        return (rob, skip)
    
    return max(dfs(root))


def longest_path_diff_adjacent(parent: list[int], s: str) -> int:
    """
    LeetCode 2246: Longest path where adjacent chars are different.
    """
    from collections import defaultdict
    
    n = len(parent)
    children = defaultdict(list)
    for i in range(1, n):
        children[parent[i]].append(i)
    
    result = [1]
    
    def dfs(node):
        max_path = 0
        
        for child in children[node]:
            child_path = dfs(child)
            
            if s[child] != s[node]:
                # Update global result with path through node
                result[0] = max(result[0], max_path + child_path + 1)
                max_path = max(max_path, child_path)
        
        return max_path + 1
    
    dfs(0)
    return result[0]
```

---

## 9. Bitmask DP

```python
def count_arrangement(n: int) -> int:
    """
    LeetCode 526: Count beautiful arrangements.
    
    Arrangement is beautiful if: perm[i] divides i or i divides perm[i]
    """
    @lru_cache(maxsize=None)
    def dp(pos, mask):
        if pos > n:
            return 1
        
        count = 0
        for num in range(1, n + 1):
            if not (mask & (1 << num)):
                if num % pos == 0 or pos % num == 0:
                    count += dp(pos + 1, mask | (1 << num))
        
        return count
    
    return dp(1, 0)


def shortest_path_visiting_all(graph: list[list[int]]) -> int:
    """
    LeetCode 847: Shortest path to visit all nodes.
    
    BFS with state = (current_node, visited_mask)
    """
    from collections import deque
    
    n = len(graph)
    target = (1 << n) - 1
    
    # (node, visited_mask)
    queue = deque([(i, 1 << i, 0) for i in range(n)])
    visited = {(i, 1 << i) for i in range(n)}
    
    while queue:
        node, mask, dist = queue.popleft()
        
        if mask == target:
            return dist
        
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            
            if (neighbor, new_mask) not in visited:
                visited.add((neighbor, new_mask))
                queue.append((neighbor, new_mask, dist + 1))
    
    return -1


def traveling_salesman(dist: list[list[int]]) -> int:
    """
    Traveling Salesman Problem (TSP) using bitmask DP.
    
    State: dp[mask][i] = min cost to visit cities in mask, ending at i
    """
    n = len(dist)
    INF = float('inf')
    
    # dp[mask][i] = min cost
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0
    
    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == INF:
                continue
            
            for v in range(n):
                if mask & (1 << v):
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])
    
    # Return to starting city
    final_mask = (1 << n) - 1
    return min(dp[final_mask][i] + dist[i][0] for i in range(n))
```

---

## 10. Interview Problems Summary

### DP Pattern Recognition

| Problem Type | Pattern | Key Insight |
|--------------|---------|-------------|
| Sequences | 1D DP | Current depends on previous states |
| Two sequences | 2D DP | Compare/match positions |
| Grid paths | 2D DP | Each cell depends on neighbors |
| Subsets | Knapsack | Include/exclude decisions |
| Partitioning | Interval DP | Which element to process last |
| State transitions | State Machine | Define explicit states |
| All subsets | Bitmask | Use binary to represent subset |

### Common Transitions

```
1D: dp[i] = f(dp[i-1], dp[i-2], ...)
2D: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
Knapsack: dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i]] + val[i])
Interval: dp[i][j] = f(dp[i][k], dp[k][j]) for i < k < j
```

### Space Optimization Tips

1. If dp[i] only depends on dp[i-1], use two variables
2. If dp[i][j] only depends on dp[i-1][j] and dp[i][j-1], use 1D array
3. For 0/1 knapsack, iterate backwards to avoid reusing
4. For unbounded knapsack, iterate forwards to allow reuse

### Time Complexity

| Pattern | Typical Time | Typical Space |
|---------|-------------|---------------|
| 1D DP | O(n) or O(n²) | O(n) or O(1) |
| 2D DP | O(n²) or O(nm) | O(n²) or O(n) |
| Knapsack | O(nW) | O(W) |
| Interval | O(n³) | O(n²) |
| Bitmask | O(2ⁿ * n) | O(2ⁿ) |
