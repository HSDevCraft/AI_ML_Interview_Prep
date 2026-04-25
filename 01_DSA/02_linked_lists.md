# Linked Lists - Complete In-Depth Guide

## ⚡ Interview Quick Summary

> **Core insight**: Linked list problems almost always reduce to pointer manipulation. Draw the list before coding, trace carefully, and always save `next` before breaking a link.

### Pattern Catalog

```
Fast/Slow Pointers (Floyd's):    cycle detection, middle node, kth from end
Dummy Head Node:                 simplifies edge cases (empty list, single node)
In-place Reversal:               reverse entire list, reverse k-group
Merge:                           merge sorted lists, merge k sorted lists
Two Lists:                       intersection, palindrome check
```

### Fast/Slow Pointer Template

```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:    # fast moves 2x speed
        slow = slow.next
        fast = fast.next.next
    return slow                  # slow is at middle

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:         # they meet → cycle!
            return True
    return False                 # fast hit None → no cycle

def kth_from_end(head, k):
    fast = slow = head
    for _ in range(k):           # advance fast by k
        fast = fast.next
    while fast:                  # move both until fast hits end
        slow = slow.next
        fast = fast.next
    return slow                  # slow is k from end
```

### Dummy Node Pattern — Simplifies Edge Cases

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head            # dummy before head handles removing head
    fast = slow = dummy
    for _ in range(n + 1):       # advance fast n+1 steps
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next   # skip the target node
    return dummy.next            # return dummy.next (handles head removal)
```

### 🚨 Top Interview Pitfalls
- **Forgetting to save `next`** before breaking a link during reversal — always `nxt = curr.next` first
- **Not returning `dummy.next`** when using dummy head pattern (returns wrong node)
- For **cycle start**: after slow==fast, reset one pointer to head and move both at speed 1 — they meet at cycle start
- **Modifying input vs copying**: clarify with interviewer whether in-place modification is OK

---

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Singly Linked List](#singly-linked-list)
3. [Doubly Linked List](#doubly-linked-list)
4. [Fast & Slow Pointers](#fast--slow-pointers)
5. [Reversal Techniques](#reversal-techniques)
6. [Merge & Sort Operations](#merge--sort-operations)
7. [Advanced Patterns](#advanced-patterns)
8. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Time & Space Complexities

| Operation | Singly LL | Doubly LL | Array |
|-----------|-----------|-----------|-------|
| Access by Index | O(n) | O(n) | O(1) |
| Search | O(n) | O(n) | O(n) |
| Insert at Head | O(1) | O(1) | O(n) |
| Insert at Tail | O(n)* | O(1) | O(1) |
| Insert at Middle | O(n) | O(n) | O(n) |
| Delete at Head | O(1) | O(1) | O(n) |
| Delete at Tail | O(n) | O(1) | O(1) |
| Delete at Middle | O(n) | O(n) | O(n) |

*O(1) if tail pointer is maintained

### When to Use Linked Lists

**Advantages:**
- Dynamic size (no pre-allocation needed)
- Efficient insertions/deletions at known positions
- No wasted memory from over-allocation
- Can grow without copying entire data structure

**Disadvantages:**
- No random access (must traverse)
- Extra memory for pointers
- Poor cache locality (nodes scattered in memory)
- Not suitable for binary search

---

## 2. Singly Linked List

### Complete Implementation

```python
class ListNode:
    """Node for singly linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList:
    """
    Full implementation of singly linked list with all operations.
    """
    def __init__(self):
        self.head = None
        self.tail = None  # Optional: O(1) append
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        """Allow iteration: for val in linked_list"""
        curr = self.head
        while curr:
            yield curr.val
            curr = curr.next
    
    def __str__(self):
        return " -> ".join(str(val) for val in self) + " -> None"
    
    def is_empty(self) -> bool:
        return self.head is None
    
    # ==================== Insert Operations ====================
    
    def prepend(self, val) -> None:
        """Insert at beginning. O(1)"""
        new_node = ListNode(val, self.head)
        self.head = new_node
        if self.tail is None:
            self.tail = new_node
        self.size += 1
    
    def append(self, val) -> None:
        """Insert at end. O(1) with tail pointer"""
        new_node = ListNode(val)
        if self.tail:
            self.tail.next = new_node
            self.tail = new_node
        else:
            self.head = self.tail = new_node
        self.size += 1
    
    def insert_at_index(self, index: int, val) -> bool:
        """Insert at specific index. O(n)"""
        if index < 0 or index > self.size:
            return False
        
        if index == 0:
            self.prepend(val)
            return True
        
        if index == self.size:
            self.append(val)
            return True
        
        # Find node before insertion point
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        
        new_node = ListNode(val, curr.next)
        curr.next = new_node
        self.size += 1
        return True
    
    # ==================== Delete Operations ====================
    
    def delete_head(self):
        """Delete first node. O(1)"""
        if not self.head:
            return None
        
        val = self.head.val
        self.head = self.head.next
        if not self.head:
            self.tail = None
        self.size -= 1
        return val
    
    def delete_tail(self):
        """Delete last node. O(n) for singly linked list"""
        if not self.head:
            return None
        
        if self.head == self.tail:
            return self.delete_head()
        
        # Find second to last node
        curr = self.head
        while curr.next != self.tail:
            curr = curr.next
        
        val = self.tail.val
        curr.next = None
        self.tail = curr
        self.size -= 1
        return val
    
    def delete_at_index(self, index: int):
        """Delete at specific index. O(n)"""
        if index < 0 or index >= self.size:
            return None
        
        if index == 0:
            return self.delete_head()
        
        # Find node before deletion point
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        
        val = curr.next.val
        curr.next = curr.next.next
        
        if curr.next is None:
            self.tail = curr
        
        self.size -= 1
        return val
    
    def delete_value(self, val) -> bool:
        """Delete first occurrence of value. O(n)"""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.delete_head()
            return True
        
        curr = self.head
        while curr.next and curr.next.val != val:
            curr = curr.next
        
        if curr.next:
            if curr.next == self.tail:
                self.tail = curr
            curr.next = curr.next.next
            self.size -= 1
            return True
        
        return False
    
    # ==================== Search Operations ====================
    
    def get_at_index(self, index: int):
        """Get value at index. O(n)"""
        if index < 0 or index >= self.size:
            return None
        
        curr = self.head
        for _ in range(index):
            curr = curr.next
        return curr.val
    
    def find(self, val) -> int:
        """Find index of first occurrence. O(n). Returns -1 if not found."""
        curr = self.head
        index = 0
        while curr:
            if curr.val == val:
                return index
            curr = curr.next
            index += 1
        return -1
    
    def contains(self, val) -> bool:
        """Check if value exists. O(n)"""
        return self.find(val) != -1
    
    # ==================== Utility Operations ====================
    
    def reverse(self) -> None:
        """Reverse list in-place. O(n) time, O(1) space"""
        self.tail = self.head
        prev = None
        curr = self.head
        
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        self.head = prev
    
    def get_middle(self):
        """Get middle node value using slow/fast pointers. O(n)"""
        if not self.head:
            return None
        
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow.val
    
    def to_list(self) -> list:
        """Convert to Python list. O(n)"""
        return list(self)
    
    @classmethod
    def from_list(cls, arr: list) -> 'SinglyLinkedList':
        """Create linked list from Python list. O(n)"""
        ll = cls()
        for val in arr:
            ll.append(val)
        return ll


# Usage example
ll = SinglyLinkedList.from_list([1, 2, 3, 4, 5])
print(ll)  # 1 -> 2 -> 3 -> 4 -> 5 -> None
ll.reverse()
print(ll)  # 5 -> 4 -> 3 -> 2 -> 1 -> None
```

---

## 3. Doubly Linked List

```python
class DoublyListNode:
    """Node for doubly linked list."""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


class DoublyLinkedList:
    """
    Doubly linked list with sentinel nodes for cleaner code.
    """
    def __init__(self):
        # Sentinel nodes eliminate edge cases
        self.head = DoublyListNode()  # Dummy head
        self.tail = DoublyListNode()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        curr = self.head.next
        while curr != self.tail:
            yield curr.val
            curr = curr.next
    
    def _add_between(self, val, predecessor, successor) -> DoublyListNode:
        """Add node between two nodes. O(1)"""
        new_node = DoublyListNode(val, predecessor, successor)
        predecessor.next = new_node
        successor.prev = new_node
        self.size += 1
        return new_node
    
    def _remove_node(self, node) -> any:
        """Remove specific node. O(1)"""
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self.size -= 1
        return node.val
    
    def prepend(self, val) -> DoublyListNode:
        """Insert at beginning. O(1)"""
        return self._add_between(val, self.head, self.head.next)
    
    def append(self, val) -> DoublyListNode:
        """Insert at end. O(1)"""
        return self._add_between(val, self.tail.prev, self.tail)
    
    def delete_head(self):
        """Delete first actual node. O(1)"""
        if self.size == 0:
            return None
        return self._remove_node(self.head.next)
    
    def delete_tail(self):
        """Delete last actual node. O(1)"""
        if self.size == 0:
            return None
        return self._remove_node(self.tail.prev)
    
    def delete_node(self, node):
        """Delete specific node reference. O(1)"""
        return self._remove_node(node)
    
    def move_to_front(self, node) -> None:
        """Move existing node to front. O(1) - useful for LRU cache"""
        self._remove_node(node)
        self._add_between(node.val, self.head, self.head.next)


class LRUCache:
    """
    LeetCode 146: LRU Cache using Doubly Linked List + HashMap.
    
    - get(key): O(1)
    - put(key, value): O(1)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Doubly linked list with sentinel nodes
        self.head = DoublyListNode()  # Most recently used
        self.tail = DoublyListNode()  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_front(self, node):
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove node from its current position."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_front(self, node):
        """Move existing node to front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._move_to_front(node)
        return node.val[1]  # val is (key, value) tuple
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = (key, value)
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used (before tail)
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.val[0]]
            
            # Add new node
            node = DoublyListNode((key, value))
            self._add_to_front(node)
            self.cache[key] = node


class LFUCache:
    """
    LeetCode 460: LFU Cache - O(1) get and put.
    
    Uses frequency buckets, each containing a doubly linked list.
    """
    def __init__(self, capacity: int):
        from collections import defaultdict
        
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_node = {}  # key -> (value, freq, node)
        self.freq_to_list = defaultdict(DoublyLinkedList)
    
    def _update_freq(self, key):
        """Increment frequency of key and move to new bucket."""
        value, freq, node = self.key_to_node[key]
        
        # Remove from current frequency list
        freq_list = self.freq_to_list[freq]
        freq_list.delete_node(node)
        
        # Update min_freq if necessary
        if freq == self.min_freq and len(freq_list) == 0:
            self.min_freq += 1
        
        # Add to new frequency list
        new_freq = freq + 1
        new_node = self.freq_to_list[new_freq].prepend(key)
        self.key_to_node[key] = (value, new_freq, new_node)
    
    def get(self, key: int) -> int:
        if key not in self.key_to_node:
            return -1
        
        self._update_freq(key)
        return self.key_to_node[key][0]
    
    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        if key in self.key_to_node:
            self._update_freq(key)
            old_val, freq, node = self.key_to_node[key]
            self.key_to_node[key] = (value, freq, node)
        else:
            if len(self.key_to_node) >= self.capacity:
                # Evict LFU (and LRU among LFU)
                lfu_list = self.freq_to_list[self.min_freq]
                evict_key = lfu_list.delete_tail()
                del self.key_to_node[evict_key]
            
            # Add new key with frequency 1
            self.min_freq = 1
            node = self.freq_to_list[1].prepend(key)
            self.key_to_node[key] = (value, 1, node)
```

---

## 4. Fast & Slow Pointers (Floyd's Algorithm)

```python
def has_cycle(head: ListNode) -> bool:
    """
    LeetCode 141: Detect if linked list has a cycle.
    Time: O(n), Space: O(1)
    
    Floyd's Tortoise and Hare: If there's a cycle, fast will eventually
    meet slow inside the cycle.
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next        # Move 1 step
        fast = fast.next.next   # Move 2 steps
        
        if slow == fast:
            return True
    
    return False


def detect_cycle_start(head: ListNode) -> ListNode:
    """
    LeetCode 142: Find the node where cycle begins.
    Time: O(n), Space: O(1)
    
    Mathematical proof:
    - Let distance from head to cycle start = a
    - Let distance from cycle start to meeting point = b
    - Let cycle length = c
    
    When they meet:
    - slow traveled: a + b
    - fast traveled: a + b + n*c (for some n >= 1)
    - Since fast travels 2x speed: 2(a + b) = a + b + n*c
    - Therefore: a + b = n*c, so a = n*c - b = (n-1)*c + (c-b)
    
    Distance from meeting point to cycle start (going forward) = c - b
    This equals 'a' modulo c, so starting from head and meeting point,
    they'll meet at cycle start.
    """
    slow = fast = head
    
    # Phase 1: Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow


def find_cycle_length(head: ListNode) -> int:
    """Find the length of the cycle, or 0 if no cycle."""
    slow = fast = head
    
    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Count cycle length
            length = 1
            curr = slow.next
            while curr != slow:
                length += 1
                curr = curr.next
            return length
    
    return 0


def find_middle(head: ListNode) -> ListNode:
    """
    LeetCode 876: Find middle node.
    Time: O(n), Space: O(1)
    
    For even length, returns second middle.
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow


def find_middle_first(head: ListNode) -> ListNode:
    """
    For even length, returns first middle.
    Useful for splitting list into two halves for merge sort.
    """
    slow = fast = head
    prev = None
    
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    
    return prev if prev else slow


def is_palindrome(head: ListNode) -> bool:
    """
    LeetCode 234: Check if linked list is palindrome.
    Time: O(n), Space: O(1)
    
    1. Find middle
    2. Reverse second half
    3. Compare both halves
    4. (Optional) Restore list
    """
    if not head or not head.next:
        return True
    
    # Find middle (for even length, first middle)
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverse_list(slow.next)
    slow.next = None  # Cut the list
    
    # Compare
    first_half = head
    is_palin = True
    while second_half:
        if first_half.val != second_half.val:
            is_palin = False
            break
        first_half = first_half.next
        second_half = second_half.next
    
    # Restore (optional)
    # slow.next = reverse_list(second_half_head)
    
    return is_palin


def reverse_list(head: ListNode) -> ListNode:
    """Helper: Reverse a linked list."""
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev


def find_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    LeetCode 19 helper: Find nth node from end.
    Time: O(n), Space: O(1)
    
    Use two pointers with n-gap.
    """
    fast = head
    for _ in range(n):
        if not fast:
            return None
        fast = fast.next
    
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow


def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    LeetCode 19: Remove nth node from end.
    Time: O(n), Space: O(1)
    """
    dummy = ListNode(0, head)
    fast = dummy
    
    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both until fast reaches end
    slow = dummy
    while fast:
        slow = slow.next
        fast = fast.next
    
    # Remove the nth node
    slow.next = slow.next.next
    
    return dummy.next


def reorder_list(head: ListNode) -> None:
    """
    LeetCode 143: Reorder L0→L1→...→Ln to L0→Ln→L1→Ln-1→...
    Time: O(n), Space: O(1)
    
    1. Find middle
    2. Reverse second half
    3. Merge alternately
    """
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second = slow.next
    slow.next = None
    prev = None
    while second:
        next_node = second.next
        second.next = prev
        prev = second
        second = next_node
    second = prev
    
    # Merge alternately
    first = head
    while second:
        first_next = first.next
        second_next = second.next
        
        first.next = second
        second.next = first_next
        
        first = first_next
        second = second_next
```

---

## 5. Reversal Techniques

```python
def reverse_list_iterative(head: ListNode) -> ListNode:
    """
    LeetCode 206: Reverse entire list iteratively.
    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next  # Save next
        curr.next = prev       # Reverse link
        prev = curr            # Move prev forward
        curr = next_node       # Move curr forward
    
    return prev


def reverse_list_recursive(head: ListNode) -> ListNode:
    """
    Reverse entire list recursively.
    Time: O(n), Space: O(n) call stack
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Reverse the rest
    new_head = reverse_list_recursive(head.next)
    
    # Put current at the end
    head.next.next = head
    head.next = None
    
    return new_head


def reverse_between(head: ListNode, left: int, right: int) -> ListNode:
    """
    LeetCode 92: Reverse from position left to right (1-indexed).
    Time: O(n), Space: O(1)
    """
    if left == right:
        return head
    
    dummy = ListNode(0, head)
    prev = dummy
    
    # Move prev to node before left
    for _ in range(left - 1):
        prev = prev.next
    
    # Reverse nodes from left to right
    curr = prev.next
    for _ in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
    
    return dummy.next


def reverse_k_group(head: ListNode, k: int) -> ListNode:
    """
    LeetCode 25: Reverse nodes in k-group.
    Time: O(n), Space: O(1)
    """
    def get_kth_node(start: ListNode, k: int) -> ListNode:
        """Get kth node from start, or None if less than k nodes."""
        curr = start
        for _ in range(k):
            if not curr:
                return None
            curr = curr.next
        return curr
    
    def reverse_segment(start: ListNode, end: ListNode) -> ListNode:
        """Reverse segment and return new head."""
        prev = end
        curr = start
        while curr != end:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev
    
    dummy = ListNode(0, head)
    group_prev = dummy
    
    while True:
        # Find kth node
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        
        group_next = kth.next
        
        # Reverse current group
        prev, curr = kth.next, group_prev.next
        while curr != group_next:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        # Connect with previous part
        temp = group_prev.next
        group_prev.next = kth
        group_prev = temp


def swap_pairs(head: ListNode) -> ListNode:
    """
    LeetCode 24: Swap every two adjacent nodes.
    Time: O(n), Space: O(1)
    """
    dummy = ListNode(0, head)
    prev = dummy
    
    while prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next
        
        # Swap
        prev.next = second
        first.next = second.next
        second.next = first
        
        prev = first
    
    return dummy.next


def rotate_right(head: ListNode, k: int) -> ListNode:
    """
    LeetCode 61: Rotate list to the right by k places.
    Time: O(n), Space: O(1)
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length and tail
    length = 1
    tail = head
    while tail.next:
        length += 1
        tail = tail.next
    
    # Normalize k
    k = k % length
    if k == 0:
        return head
    
    # Find new tail (length - k - 1 steps from head)
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    
    # Rotate
    new_head = new_tail.next
    new_tail.next = None
    tail.next = head
    
    return new_head
```

---

## 6. Merge & Sort Operations

```python
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    LeetCode 21: Merge two sorted lists.
    Time: O(n + m), Space: O(1)
    """
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


def merge_two_lists_recursive(l1: ListNode, l2: ListNode) -> ListNode:
    """Merge two sorted lists recursively."""
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val <= l2.val:
        l1.next = merge_two_lists_recursive(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_lists_recursive(l1, l2.next)
        return l2


def merge_k_lists(lists: list[ListNode]) -> ListNode:
    """
    LeetCode 23: Merge k sorted lists.
    Time: O(N log k) where N = total nodes, k = number of lists
    Space: O(k) for heap
    """
    import heapq
    
    # Custom comparison for heap
    # Python 3 doesn't allow comparing ListNode, so use (val, index, node)
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


def merge_k_lists_divide_conquer(lists: list[ListNode]) -> ListNode:
    """
    Merge k sorted lists using divide and conquer.
    Time: O(N log k), Space: O(log k) call stack
    """
    if not lists:
        return None
    
    def merge_range(start: int, end: int) -> ListNode:
        if start == end:
            return lists[start]
        
        mid = (start + end) // 2
        left = merge_range(start, mid)
        right = merge_range(mid + 1, end)
        return merge_two_lists(left, right)
    
    return merge_range(0, len(lists) - 1)


def sort_list(head: ListNode) -> ListNode:
    """
    LeetCode 148: Sort linked list using merge sort.
    Time: O(n log n), Space: O(log n) for recursion
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Find middle (use slow for even-length to get first middle)
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Split list
    mid = slow.next
    slow.next = None
    
    # Sort both halves
    left = sort_list(head)
    right = sort_list(mid)
    
    # Merge sorted halves
    return merge_two_lists(left, right)


def sort_list_bottom_up(head: ListNode) -> ListNode:
    """
    Sort linked list using bottom-up merge sort.
    Time: O(n log n), Space: O(1) - truly iterative
    """
    if not head or not head.next:
        return head
    
    # Get length
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next
    
    dummy = ListNode(0, head)
    size = 1
    
    while size < length:
        prev = dummy
        curr = dummy.next
        
        while curr:
            # Get left list of size 'size'
            left = curr
            right = split(left, size)
            curr = split(right, size)
            
            # Merge and connect
            prev = merge(left, right, prev)
        
        size *= 2
    
    return dummy.next


def split(head: ListNode, size: int) -> ListNode:
    """Split list after 'size' nodes, return head of remaining."""
    for _ in range(size - 1):
        if not head:
            return None
        head = head.next
    
    if not head:
        return None
    
    next_head = head.next
    head.next = None
    return next_head


def merge(l1: ListNode, l2: ListNode, prev: ListNode) -> ListNode:
    """Merge two lists and attach to prev. Return tail."""
    curr = prev
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 or l2
    
    while curr.next:
        curr = curr.next
    
    return curr


def insertion_sort_list(head: ListNode) -> ListNode:
    """
    LeetCode 147: Sort using insertion sort.
    Time: O(n²), Space: O(1)
    """
    dummy = ListNode(0)
    curr = head
    
    while curr:
        # Find insertion position
        prev = dummy
        while prev.next and prev.next.val < curr.val:
            prev = prev.next
        
        # Insert curr after prev
        next_node = curr.next
        curr.next = prev.next
        prev.next = curr
        curr = next_node
    
    return dummy.next
```

---

## 7. Advanced Patterns

```python
def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    LeetCode 2: Add two numbers (digits in reverse order).
    Time: O(max(m, n)), Space: O(max(m, n))
    """
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        curr.next = ListNode(total % 10)
        curr = curr.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next


def add_two_numbers_ii(l1: ListNode, l2: ListNode) -> ListNode:
    """
    LeetCode 445: Add two numbers (most significant digit first).
    Time: O(m + n), Space: O(m + n)
    """
    # Use stacks to reverse order
    stack1, stack2 = [], []
    
    while l1:
        stack1.append(l1.val)
        l1 = l1.next
    while l2:
        stack2.append(l2.val)
        l2 = l2.next
    
    carry = 0
    head = None
    
    while stack1 or stack2 or carry:
        val1 = stack1.pop() if stack1 else 0
        val2 = stack2.pop() if stack2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        
        # Build list from tail to head
        node = ListNode(total % 10)
        node.next = head
        head = node
    
    return head


def partition_list(head: ListNode, x: int) -> ListNode:
    """
    LeetCode 86: Partition list around value x.
    Time: O(n), Space: O(1)
    """
    # Two dummy heads for two partitions
    before = before_head = ListNode(0)
    after = after_head = ListNode(0)
    
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    
    # Connect partitions
    after.next = None  # Important: prevent cycle
    before.next = after_head.next
    
    return before_head.next


def copy_random_list(head: 'Node') -> 'Node':
    """
    LeetCode 138: Copy list with random pointer.
    Time: O(n), Space: O(n)
    """
    if not head:
        return None
    
    # First pass: create all nodes and store mapping
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.val)
        curr = curr.next
    
    # Second pass: set next and random pointers
    curr = head
    while curr:
        new_node = old_to_new[curr]
        new_node.next = old_to_new.get(curr.next)
        new_node.random = old_to_new.get(curr.random)
        curr = curr.next
    
    return old_to_new[head]


def copy_random_list_o1_space(head: 'Node') -> 'Node':
    """
    Copy list with random pointer using O(1) extra space.
    
    Technique: Interweave original and copy nodes.
    """
    if not head:
        return None
    
    # Step 1: Create copy nodes interweaved
    # Original: A -> B -> C
    # After: A -> A' -> B -> B' -> C -> C'
    curr = head
    while curr:
        copy = Node(curr.val)
        copy.next = curr.next
        curr.next = copy
        curr = copy.next
    
    # Step 2: Set random pointers for copies
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    
    # Step 3: Separate the lists
    dummy = Node(0)
    copy_curr = dummy
    curr = head
    
    while curr:
        copy_curr.next = curr.next
        copy_curr = copy_curr.next
        curr.next = curr.next.next
        curr = curr.next
    
    return dummy.next


def flatten_multilevel_list(head: 'Node') -> 'Node':
    """
    LeetCode 430: Flatten doubly linked list with child pointers.
    Time: O(n), Space: O(depth)
    """
    if not head:
        return None
    
    dummy = Node(0, None, head, None)
    prev = dummy
    stack = [head]
    
    while stack:
        curr = stack.pop()
        prev.next = curr
        curr.prev = prev
        
        if curr.next:
            stack.append(curr.next)
        
        if curr.child:
            stack.append(curr.child)
            curr.child = None
        
        prev = curr
    
    dummy.next.prev = None
    return dummy.next


def remove_duplicates_sorted(head: ListNode) -> ListNode:
    """
    LeetCode 83: Remove duplicates from sorted list (keep one of each).
    Time: O(n), Space: O(1)
    """
    curr = head
    
    while curr and curr.next:
        if curr.val == curr.next.val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    
    return head


def remove_duplicates_sorted_ii(head: ListNode) -> ListNode:
    """
    LeetCode 82: Remove all nodes that have duplicates.
    Time: O(n), Space: O(1)
    """
    dummy = ListNode(0, head)
    prev = dummy
    
    while prev.next:
        curr = prev.next
        
        # Check if current value is duplicated
        if curr.next and curr.val == curr.next.val:
            # Skip all nodes with this value
            while curr.next and curr.val == curr.next.val:
                curr = curr.next
            prev.next = curr.next
        else:
            prev = prev.next
    
    return dummy.next


def odd_even_list(head: ListNode) -> ListNode:
    """
    LeetCode 328: Group odd-indexed nodes before even-indexed.
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return head
    
    odd = head
    even = head.next
    even_head = even
    
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    
    odd.next = even_head
    return head
```

---

## 8. Interview Problems Summary

### Common Patterns

| Pattern | Problems | Key Technique |
|---------|----------|---------------|
| Two Pointers | Cycle detection, Middle, Nth from end | Fast/Slow, Gap |
| Reversal | Reverse list, K-group, Palindrome | Prev/Curr/Next |
| Merge | Merge 2/K lists, Sort list | Dummy node, Heap |
| Partition | Partition list, Odd-even | Two dummy heads |
| Copy | Random pointer, Clone | HashMap or Interweave |

### Interview Tips

1. **Always use dummy node** for operations that might modify head
2. **Draw the pointers** before writing code
3. **Handle edge cases**: empty list, single node, two nodes
4. **Watch for null pointer** exceptions
5. **Consider both iterative and recursive** solutions
6. **For space optimization**, consider in-place modifications

### Time Complexity Summary

| Operation | Typical Time |
|-----------|-------------|
| Find middle | O(n) |
| Reverse | O(n) |
| Detect cycle | O(n) |
| Merge two sorted | O(n + m) |
| Merge k sorted | O(N log k) |
| Sort (merge sort) | O(n log n) |
