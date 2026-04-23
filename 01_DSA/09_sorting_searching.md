# Sorting & Searching - Complete In-Depth Guide

## Table of Contents
1. [Sorting Algorithms](#sorting-algorithms)
2. [Binary Search Patterns](#binary-search-patterns)
3. [Quick Select](#quick-select)
4. [Search in Special Arrays](#search-in-special-arrays)
5. [Interview Problems](#interview-problems)

---

## 1. Sorting Algorithms

### Comparison of Sorting Algorithms

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |
| Radix Sort | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes |

### Merge Sort

```python
def merge_sort(arr: list[int]) -> list[int]:
    """
    Divide and conquer: split, sort, merge.
    Time: O(n log n), Space: O(n)
    Stable: Yes
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort_in_place(arr: list[int], left: int, right: int) -> None:
    """In-place merge sort (still O(n) space for merge)."""
    if left < right:
        mid = (left + right) // 2
        merge_sort_in_place(arr, left, mid)
        merge_sort_in_place(arr, mid + 1, right)
        merge_in_place(arr, left, mid, right)


def merge_in_place(arr: list[int], left: int, mid: int, right: int) -> None:
    """Merge arr[left:mid+1] and arr[mid+1:right+1]."""
    left_arr = arr[left:mid + 1]
    right_arr = arr[mid + 1:right + 1]
    
    i = j = 0
    k = left
    
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1
    
    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1
```

### Quick Sort

```python
def quick_sort(arr: list[int]) -> list[int]:
    """
    Partition around pivot, recursively sort partitions.
    Time: O(n log n) average, O(n²) worst
    Space: O(log n) average
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)


def quick_sort_in_place(arr: list[int], low: int, high: int) -> None:
    """In-place quick sort using Lomuto partition."""
    if low < high:
        pivot_idx = partition_lomuto(arr, low, high)
        quick_sort_in_place(arr, low, pivot_idx - 1)
        quick_sort_in_place(arr, pivot_idx + 1, high)


def partition_lomuto(arr: list[int], low: int, high: int) -> int:
    """
    Lomuto partition scheme.
    Choose rightmost element as pivot.
    """
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def partition_hoare(arr: list[int], low: int, high: int) -> int:
    """
    Hoare partition scheme (more efficient).
    Choose middle element as pivot.
    """
    pivot = arr[(low + high) // 2]
    i = low - 1
    j = high + 1
    
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        arr[i], arr[j] = arr[j], arr[i]


def quick_sort_three_way(arr: list[int], low: int, high: int) -> None:
    """
    Three-way partition for arrays with many duplicates.
    Dutch National Flag partitioning.
    """
    if low >= high:
        return
    
    pivot = arr[low]
    lt = low      # arr[low..lt-1] < pivot
    gt = high     # arr[gt+1..high] > pivot
    i = low + 1   # arr[lt..i-1] == pivot
    
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[gt], arr[i] = arr[i], arr[gt]
            gt -= 1
        else:
            i += 1
    
    quick_sort_three_way(arr, low, lt - 1)
    quick_sort_three_way(arr, gt + 1, high)
```

### Heap Sort

```python
def heap_sort(arr: list[int]) -> None:
    """
    Build max heap, repeatedly extract max.
    Time: O(n log n), Space: O(1)
    """
    n = len(arr)
    
    # Build max heap (heapify from bottom up)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)


def heapify(arr: list[int], n: int, i: int) -> None:
    """Heapify subtree rooted at index i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

### Counting Sort & Radix Sort

```python
def counting_sort(arr: list[int]) -> list[int]:
    """
    Count occurrences, compute positions.
    Time: O(n + k), Space: O(k) where k = range of values
    Good for small range of integers.
    """
    if not arr:
        return arr
    
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val + 1
    
    count = [0] * range_size
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num - min_val] += 1
    
    # Cumulative count (positions)
    for i in range(1, range_size):
        count[i] += count[i - 1]
    
    # Build output (reverse for stability)
    for num in reversed(arr):
        count[num - min_val] -= 1
        output[count[num - min_val]] = num
    
    return output


def radix_sort(arr: list[int]) -> list[int]:
    """
    Sort digit by digit using counting sort.
    Time: O(d * (n + k)) where d = digits, k = base
    """
    if not arr:
        return arr
    
    max_val = max(arr)
    exp = 1
    
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr


def counting_sort_by_digit(arr: list[int], exp: int) -> list[int]:
    """Counting sort based on digit at exp position."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    for num in reversed(arr):
        digit = (num // exp) % 10
        count[digit] -= 1
        output[count[digit]] = num
    
    return output
```

---

## 2. Binary Search Patterns

### Basic Templates

```python
def binary_search_exact(arr: list[int], target: int) -> int:
    """
    Find exact match. Returns -1 if not found.
    Template: left <= right
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def binary_search_leftmost(arr: list[int], target: int) -> int:
    """
    Find leftmost position where target could be inserted.
    Returns index of first element >= target.
    Template: left < right
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def binary_search_rightmost(arr: list[int], target: int) -> int:
    """
    Find rightmost position where target could be inserted.
    Returns index of first element > target.
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left


def first_occurrence(arr: list[int], target: int) -> int:
    """Find index of first occurrence of target."""
    idx = binary_search_leftmost(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1


def last_occurrence(arr: list[int], target: int) -> int:
    """Find index of last occurrence of target."""
    idx = binary_search_rightmost(arr, target) - 1
    if idx >= 0 and arr[idx] == target:
        return idx
    return -1


def count_occurrences(arr: list[int], target: int) -> int:
    """Count occurrences of target."""
    left = binary_search_leftmost(arr, target)
    right = binary_search_rightmost(arr, target)
    return right - left
```

### Binary Search on Answer

```python
def koko_eating_bananas(piles: list[int], h: int) -> int:
    """
    LeetCode 875: Minimum eating speed to finish in h hours.
    
    Binary search on answer (eating speed).
    """
    def can_finish(speed: int) -> bool:
        hours = sum((pile + speed - 1) // speed for pile in piles)
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def min_days_bouquets(bloomDay: list[int], m: int, k: int) -> int:
    """
    LeetCode 1482: Minimum days to make m bouquets.
    """
    if m * k > len(bloomDay):
        return -1
    
    def can_make(days: int) -> bool:
        bouquets = flowers = 0
        for bloom in bloomDay:
            if bloom <= days:
                flowers += 1
                if flowers == k:
                    bouquets += 1
                    flowers = 0
            else:
                flowers = 0
        return bouquets >= m
    
    left, right = min(bloomDay), max(bloomDay)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_make(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def ship_within_days(weights: list[int], days: int) -> int:
    """
    LeetCode 1011: Minimum ship capacity to ship in days.
    """
    def can_ship(capacity: int) -> bool:
        day_count = 1
        current_weight = 0
        
        for weight in weights:
            if current_weight + weight > capacity:
                day_count += 1
                current_weight = 0
            current_weight += weight
        
        return day_count <= days
    
    left = max(weights)
    right = sum(weights)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def split_array_largest_sum(nums: list[int], k: int) -> int:
    """
    LeetCode 410: Minimize largest sum when splitting into k subarrays.
    """
    def can_split(max_sum: int) -> bool:
        splits = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                splits += 1
                current_sum = 0
            current_sum += num
        
        return splits <= k
    
    left = max(nums)
    right = sum(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left


def minimize_max_distance(stations: list[int], k: int) -> float:
    """
    LeetCode 774: Minimize max distance between gas stations.
    
    Binary search on floating point answer.
    """
    def can_achieve(max_dist: float) -> bool:
        added = 0
        for i in range(1, len(stations)):
            gap = stations[i] - stations[i - 1]
            added += int(gap / max_dist)
        return added <= k
    
    left, right = 0, stations[-1] - stations[0]
    
    while right - left > 1e-6:
        mid = (left + right) / 2
        
        if can_achieve(mid):
            right = mid
        else:
            left = mid
    
    return left
```

---

## 3. Quick Select

```python
def quick_select(arr: list[int], k: int) -> int:
    """
    Find kth smallest element.
    Time: O(n) average, O(n²) worst
    Space: O(1)
    """
    def partition(left: int, right: int, pivot_idx: int) -> int:
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        
        arr[store_idx], arr[right] = arr[right], arr[store_idx]
        return store_idx
    
    def select(left: int, right: int, k: int) -> int:
        if left == right:
            return arr[left]
        
        import random
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)
        
        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            return select(left, pivot_idx - 1, k)
        else:
            return select(pivot_idx + 1, right, k)
    
    return select(0, len(arr) - 1, k - 1)  # k-1 for 0-indexed


def find_kth_largest(nums: list[int], k: int) -> int:
    """
    LeetCode 215: Kth largest element.
    """
    return quick_select(nums, len(nums) - k + 1)


def top_k_frequent_quick_select(nums: list[int], k: int) -> list[int]:
    """
    LeetCode 347: Top k frequent elements using quick select.
    """
    from collections import Counter
    
    count = Counter(nums)
    unique = list(count.keys())
    
    def partition(left: int, right: int) -> int:
        pivot_freq = count[unique[right]]
        store = left
        
        for i in range(left, right):
            if count[unique[i]] < pivot_freq:
                unique[store], unique[i] = unique[i], unique[store]
                store += 1
        
        unique[store], unique[right] = unique[right], unique[store]
        return store
    
    def quick_select(left: int, right: int, k_smallest: int):
        if left == right:
            return
        
        import random
        pivot_idx = random.randint(left, right)
        unique[pivot_idx], unique[right] = unique[right], unique[pivot_idx]
        
        pivot_idx = partition(left, right)
        
        if k_smallest == pivot_idx:
            return
        elif k_smallest < pivot_idx:
            quick_select(left, pivot_idx - 1, k_smallest)
        else:
            quick_select(pivot_idx + 1, right, k_smallest)
    
    n = len(unique)
    quick_select(0, n - 1, n - k)
    
    return unique[n - k:]
```

---

## 4. Search in Special Arrays

```python
def search_rotated_sorted(nums: list[int], target: int) -> int:
    """
    LeetCode 33: Search in rotated sorted array (no duplicates).
    Time: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
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


def search_rotated_sorted_ii(nums: list[int], target: int) -> bool:
    """
    LeetCode 81: Search in rotated sorted array (with duplicates).
    Time: O(n) worst case due to duplicates
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return True
        
        # Handle duplicates
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return False


def find_minimum_rotated(nums: list[int]) -> int:
    """
    LeetCode 153: Find minimum in rotated sorted array.
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]


def find_peak_element(nums: list[int]) -> int:
    """
    LeetCode 162: Find a peak element.
    Peak: nums[i] > nums[i-1] and nums[i] > nums[i+1]
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    
    return left


def search_2d_matrix(matrix: list[list[int]], target: int) -> bool:
    """
    LeetCode 74: Search in row-wise sorted matrix.
    Treat as flattened sorted array.
    """
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        row, col = divmod(mid, n)
        val = matrix[row][col]
        
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


def search_2d_matrix_ii(matrix: list[list[int]], target: int) -> bool:
    """
    LeetCode 240: Search in row and column sorted matrix.
    Start from top-right corner.
    Time: O(m + n)
    """
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1
    
    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return False


def find_first_and_last(nums: list[int], target: int) -> list[int]:
    """
    LeetCode 34: First and last position of target.
    """
    def find_left(target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
    
    def find_right(target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left
    
    left_idx = find_left(target)
    
    if left_idx == len(nums) or nums[left_idx] != target:
        return [-1, -1]
    
    return [left_idx, find_right(target) - 1]
```

---

## 5. Interview Problems

```python
def median_of_two_sorted(nums1: list[int], nums2: list[int]) -> float:
    """
    LeetCode 4: Median of two sorted arrays.
    Time: O(log(min(m, n)))
    
    Binary search on partition position.
    """
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


def find_duplicate_number(nums: list[int]) -> int:
    """
    LeetCode 287: Find duplicate in array of n+1 integers [1, n].
    
    Floyd's cycle detection (treat as linked list).
    """
    slow = fast = nums[0]
    
    # Find meeting point
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Find cycle start
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow


def count_inversions(arr: list[int]) -> int:
    """
    Count inversions using modified merge sort.
    Inversion: i < j but arr[i] > arr[j]
    """
    def merge_count(arr, left, mid, right):
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        inversions = 0
        
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                inversions += len(left_arr) - i
                j += 1
            k += 1
        
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
        
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
        
        return inversions
    
    def sort_count(arr, left, right):
        if left >= right:
            return 0
        
        mid = (left + right) // 2
        count = sort_count(arr, left, mid)
        count += sort_count(arr, mid + 1, right)
        count += merge_count(arr, left, mid, right)
        
        return count
    
    return sort_count(arr[:], 0, len(arr) - 1)


def h_index(citations: list[int]) -> int:
    """
    LeetCode 274: H-index.
    """
    citations.sort(reverse=True)
    
    for i, c in enumerate(citations):
        if c < i + 1:
            return i
    
    return len(citations)


def h_index_ii(citations: list[int]) -> int:
    """
    LeetCode 275: H-index II (sorted array).
    Binary search approach.
    """
    n = len(citations)
    left, right = 0, n
    
    while left < right:
        mid = (left + right) // 2
        if citations[mid] >= n - mid:
            right = mid
        else:
            left = mid + 1
    
    return n - left
```

### Summary

| Pattern | When to Use | Template |
|---------|-------------|----------|
| Exact match | Find specific value | left <= right |
| Leftmost | First >= target | left < right, right = mid |
| Rightmost | Last <= target | left < right, left = mid + 1 |
| Search on answer | Minimize/maximize satisfying condition | left < right with predicate |
| Rotated array | Array rotated at pivot | Compare with endpoints |
| 2D matrix | Row/column sorted | Treat as 1D or corner start |
