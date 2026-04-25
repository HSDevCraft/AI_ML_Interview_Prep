# Trees - Complete In-Depth Guide

## ⚡ Interview Quick Summary

> **Core insight**: Trees are recursion made visual. Master the four traversals and the general DFS pattern, and 90% of tree problems follow from those building blocks.

### Tree Traversals — Know All 4

```python
def inorder(root):    # Left → Root → Right  (BST: sorted order!)
    if root:
        inorder(root.left)
        process(root.val)
        inorder(root.right)

def preorder(root):   # Root → Left → Right  (copy/serialize tree)
    if root:
        process(root.val)
        preorder(root.left)
        preorder(root.right)

def postorder(root):  # Left → Right → Root  (delete tree, evaluate expressions)
    if root:
        postorder(root.left)
        postorder(root.right)
        process(root.val)

from collections import deque
def level_order(root): # BFS  (level-by-level, shortest path in unweighted tree)
    if not root: return
    q = deque([root])
    while q:
        for _ in range(len(q)):  # process one level at a time
            node = q.popleft()
            process(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
```

### General DFS Pattern — Template for Most Tree Problems

```python
def solve(root):
    # Base case
    if not root:
        return base_value  # 0, None, True, float('inf'), etc.
    
    # Recurse on children
    left_result  = solve(root.left)
    right_result = solve(root.right)
    
    # Combine results
    return combine(root.val, left_result, right_result)

# Examples:
# Height:     return 1 + max(left, right)
# Min depth:  return 1 + min(left, right) [handle None children!]
# Path sum:   return root.val + max(left, right)  [if negative, take 0]
# Is same:    return root1.val==root2.val and left and right
```

### BST Properties — Critical for Interviews

```
BST Invariant: left subtree values < node.val < right subtree values
  → This applies to ALL ancestors, not just immediate parent!
  (Common pitfall: only checking root.left.val < root.val)

BST operations: O(h) where h = height
  Balanced BST (AVL, Red-Black): h = O(log n)
  Degenerate (sorted input): h = O(n)  ← list, not tree!

BST Search:
  if target < node.val: go left
  if target > node.val: go right
  if target == node.val: found!

BST Successor:
  If right subtree exists: leftmost node in right subtree
  Else: first ancestor where current is in left subtree
```

### 🚨 Top Interview Pitfalls
- For **BST validation**: never check only `left.val < root.val < right.val` — must propagate valid range through entire subtree
- For **tree height**: base case should return 0 for `None` (not -1 or 1); height = 1 + max(left, right)
- For **min-depth**: when one child is None, don't return 0 (that's wrong) — return depth of the existing child
- For **LCA**: track whether each subtree contains p/q; when a node has both p and q in different subtrees, it's the LCA

---

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Binary Tree Traversals](#binary-tree-traversals)
3. [Binary Search Tree (BST)](#binary-search-tree)
4. [Tree Construction](#tree-construction)
5. [Path Problems](#path-problems)
6. [Lowest Common Ancestor](#lowest-common-ancestor)
7. [Serialization](#serialization)
8. [Advanced Tree Types](#advanced-tree-types)
9. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Tree Terminology

```
        1          <- Root (level 0, depth 0)
       / \
      2   3        <- Level 1, depth 1
     / \   \
    4   5   6      <- Level 2, depth 2 (leaves)
   /
  7                <- Level 3, depth 3 (leaf)

- Node: Element in tree (1, 2, 3, etc.)
- Root: Topmost node (1)
- Parent: Node above (2 is parent of 4, 5)
- Child: Node below (4, 5 are children of 2)
- Siblings: Nodes with same parent (4, 5 are siblings)
- Leaf: Node with no children (4, 5, 6, 7)
- Internal Node: Node with at least one child (1, 2, 3)
- Edge: Connection between nodes
- Path: Sequence of nodes and edges
- Height: Longest path from node to leaf (height of 1 = 3)
- Depth: Path length from root to node (depth of 4 = 2)
- Level: All nodes at same depth
- Subtree: Tree formed by a node and its descendants
```

### Binary Tree Properties

```python
# Full Binary Tree: Every node has 0 or 2 children
# Complete Binary Tree: All levels filled except possibly last (filled left to right)
# Perfect Binary Tree: All internal nodes have 2 children, all leaves at same level
# Balanced Binary Tree: Height difference of left and right subtrees <= 1

# For Perfect Binary Tree with height h:
# - Number of nodes = 2^(h+1) - 1
# - Number of leaves = 2^h
# - Number of internal nodes = 2^h - 1

# For any binary tree with n nodes:
# - Minimum height = floor(log2(n))
# - Maximum height = n - 1 (skewed tree)
```

### Node Implementation

```python
class TreeNode:
    """Binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"


class NaryTreeNode:
    """N-ary tree node."""
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children else []
```

---

## 2. Binary Tree Traversals

### Recursive Traversals

```python
def preorder_recursive(root: TreeNode) -> list[int]:
    """
    Preorder: Root -> Left -> Right
    Use: Create copy of tree, prefix expression
    """
    if not root:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)


def inorder_recursive(root: TreeNode) -> list[int]:
    """
    Inorder: Left -> Root -> Right
    Use: BST sorted order, infix expression
    """
    if not root:
        return []
    return inorder_recursive(root.left) + [root.val] + inorder_recursive(root.right)


def postorder_recursive(root: TreeNode) -> list[int]:
    """
    Postorder: Left -> Right -> Root
    Use: Delete tree, postfix expression, calculate size
    """
    if not root:
        return []
    return postorder_recursive(root.left) + postorder_recursive(root.right) + [root.val]
```

### Iterative Traversals

```python
def preorder_iterative(root: TreeNode) -> list[int]:
    """
    LeetCode 144: Preorder using stack.
    Time: O(n), Space: O(h)
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result


def inorder_iterative(root: TreeNode) -> list[int]:
    """
    LeetCode 94: Inorder using stack.
    Time: O(n), Space: O(h)
    """
    result = []
    stack = []
    curr = root
    
    while curr or stack:
        # Go to leftmost node
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process current node
        curr = stack.pop()
        result.append(curr.val)
        
        # Move to right subtree
        curr = curr.right
    
    return result


def postorder_iterative(root: TreeNode) -> list[int]:
    """
    LeetCode 145: Postorder using stack.
    Time: O(n), Space: O(h)
    
    Trick: Modified preorder (Root -> Right -> Left) then reverse.
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push left first, then right
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return result[::-1]


def postorder_iterative_single_stack(root: TreeNode) -> list[int]:
    """Postorder with single stack, no reversal."""
    result = []
    stack = []
    curr = root
    last_visited = None
    
    while curr or stack:
        # Go to leftmost
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Peek the top
        peek_node = stack[-1]
        
        # If right child exists and not visited, go right
        if peek_node.right and peek_node.right != last_visited:
            curr = peek_node.right
        else:
            # Process current node
            result.append(peek_node.val)
            last_visited = stack.pop()
    
    return result
```

### Level Order Traversal (BFS)

```python
from collections import deque

def level_order(root: TreeNode) -> list[list[int]]:
    """
    LeetCode 102: Level order traversal.
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


def level_order_bottom(root: TreeNode) -> list[list[int]]:
    """LeetCode 107: Bottom-up level order."""
    result = level_order(root)
    return result[::-1]


def zigzag_level_order(root: TreeNode) -> list[list[int]]:
    """
    LeetCode 103: Zigzag level order.
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level))
        left_to_right = not left_to_right
    
    return result


def right_side_view(root: TreeNode) -> list[int]:
    """
    LeetCode 199: Right side view (last node of each level).
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            if i == level_size - 1:  # Last node in level
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

### Morris Traversal (O(1) Space)

```python
def morris_inorder(root: TreeNode) -> list[int]:
    """
    Morris Inorder Traversal: O(n) time, O(1) space.
    
    Idea: Use threaded binary tree - make right pointer of
    rightmost node in left subtree point to current node.
    """
    result = []
    curr = root
    
    while curr:
        if not curr.left:
            # No left subtree, visit current and go right
            result.append(curr.val)
            curr = curr.right
        else:
            # Find inorder predecessor (rightmost in left subtree)
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create thread
                predecessor.right = curr
                curr = curr.left
            else:
                # Thread exists, remove it and visit current
                predecessor.right = None
                result.append(curr.val)
                curr = curr.right
    
    return result


def morris_preorder(root: TreeNode) -> list[int]:
    """Morris Preorder: O(n) time, O(1) space."""
    result = []
    curr = root
    
    while curr:
        if not curr.left:
            result.append(curr.val)
            curr = curr.right
        else:
            predecessor = curr.left
            while predecessor.right and predecessor.right != curr:
                predecessor = predecessor.right
            
            if not predecessor.right:
                result.append(curr.val)  # Visit before going left
                predecessor.right = curr
                curr = curr.left
            else:
                predecessor.right = None
                curr = curr.right
    
    return result
```

---

## 3. Binary Search Tree (BST)

### BST Properties

```
BST Property: For every node:
- All values in left subtree < node's value
- All values in right subtree > node's value

Time Complexities (balanced BST):
- Search: O(log n)
- Insert: O(log n)
- Delete: O(log n)

Worst case (skewed): O(n) for all operations
```

### BST Operations

```python
def search_bst(root: TreeNode, val: int) -> TreeNode:
    """
    LeetCode 700: Search in BST.
    Time: O(h), Space: O(1) iterative
    """
    while root and root.val != val:
        root = root.left if val < root.val else root.right
    return root


def insert_bst(root: TreeNode, val: int) -> TreeNode:
    """
    LeetCode 701: Insert into BST.
    Time: O(h), Space: O(h) recursive
    """
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    
    return root


def delete_bst(root: TreeNode, key: int) -> TreeNode:
    """
    LeetCode 450: Delete node from BST.
    Time: O(h), Space: O(h)
    
    Three cases:
    1. Node is leaf: simply remove
    2. Node has one child: replace with child
    3. Node has two children: replace with inorder successor/predecessor
    """
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        # Node found
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Two children: find inorder successor (smallest in right subtree)
        successor = root.right
        while successor.left:
            successor = successor.left
        
        root.val = successor.val
        root.right = delete_bst(root.right, successor.val)
    
    return root


def is_valid_bst(root: TreeNode) -> bool:
    """
    LeetCode 98: Validate BST.
    Time: O(n), Space: O(h)
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


def is_valid_bst_inorder(root: TreeNode) -> bool:
    """Validate BST using inorder traversal (should be sorted)."""
    prev = float('-inf')
    stack = []
    curr = root
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack.pop()
        if curr.val <= prev:
            return False
        prev = curr.val
        curr = curr.right
    
    return True
```

### BST Problems

```python
def kth_smallest(root: TreeNode, k: int) -> int:
    """
    LeetCode 230: Kth smallest element in BST.
    Time: O(h + k), Space: O(h)
    """
    stack = []
    curr = root
    count = 0
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack.pop()
        count += 1
        if count == k:
            return curr.val
        
        curr = curr.right
    
    return -1


def inorder_successor(root: TreeNode, p: TreeNode) -> TreeNode:
    """
    LeetCode 285: Inorder successor in BST.
    Time: O(h), Space: O(1)
    """
    successor = None
    
    while root:
        if p.val < root.val:
            successor = root  # Potential successor
            root = root.left
        else:
            root = root.right
    
    return successor


def inorder_predecessor(root: TreeNode, p: TreeNode) -> TreeNode:
    """Inorder predecessor in BST."""
    predecessor = None
    
    while root:
        if p.val > root.val:
            predecessor = root  # Potential predecessor
            root = root.right
        else:
            root = root.left
    
    return predecessor


def closest_value(root: TreeNode, target: float) -> int:
    """
    LeetCode 270: Closest BST value.
    Time: O(h), Space: O(1)
    """
    closest = root.val
    
    while root:
        if abs(root.val - target) < abs(closest - target):
            closest = root.val
        
        root = root.left if target < root.val else root.right
    
    return closest


def range_sum_bst(root: TreeNode, low: int, high: int) -> int:
    """
    LeetCode 938: Sum of values in range [low, high].
    Time: O(n), Space: O(h)
    """
    if not root:
        return 0
    
    total = 0
    if low <= root.val <= high:
        total += root.val
    
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total


def convert_bst_to_sorted_dll(root: TreeNode) -> TreeNode:
    """
    LeetCode 426: Convert BST to sorted circular doubly-linked list.
    Time: O(n), Space: O(h)
    """
    if not root:
        return None
    
    first = None  # Smallest node
    last = None   # Previous node in inorder
    
    def inorder(node):
        nonlocal first, last
        
        if not node:
            return
        
        inorder(node.left)
        
        if last:
            last.right = node
            node.left = last
        else:
            first = node
        last = node
        
        inorder(node.right)
    
    inorder(root)
    
    # Make circular
    first.left = last
    last.right = first
    
    return first


def recover_bst(root: TreeNode) -> None:
    """
    LeetCode 99: Two nodes are swapped, recover BST.
    Time: O(n), Space: O(h)
    
    Find two nodes that violate BST property during inorder.
    """
    first = second = prev = None
    
    def inorder(node):
        nonlocal first, second, prev
        
        if not node:
            return
        
        inorder(node.left)
        
        if prev and prev.val > node.val:
            if not first:
                first = prev  # First violation
            second = node  # Second violation (might be same swap)
        prev = node
        
        inorder(node.right)
    
    inorder(root)
    
    # Swap values
    first.val, second.val = second.val, first.val
```

---

## 4. Tree Construction

```python
def build_tree_preorder_inorder(preorder: list[int], inorder: list[int]) -> TreeNode:
    """
    LeetCode 105: Construct from preorder and inorder.
    Time: O(n), Space: O(n)
    
    Preorder: [root, ...left..., ...right...]
    Inorder:  [...left..., root, ...right...]
    """
    if not preorder or not inorder:
        return None
    
    # Build index map for inorder (O(1) lookup)
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build(pre_start, pre_end, in_start, in_end):
        if pre_start > pre_end:
            return None
        
        root_val = preorder[pre_start]
        root = TreeNode(root_val)
        
        # Find root in inorder
        root_idx = inorder_map[root_val]
        left_size = root_idx - in_start
        
        root.left = build(pre_start + 1, pre_start + left_size,
                         in_start, root_idx - 1)
        root.right = build(pre_start + left_size + 1, pre_end,
                          root_idx + 1, in_end)
        
        return root
    
    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


def build_tree_inorder_postorder(inorder: list[int], postorder: list[int]) -> TreeNode:
    """
    LeetCode 106: Construct from inorder and postorder.
    Time: O(n), Space: O(n)
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build(in_start, in_end, post_start, post_end):
        if in_start > in_end:
            return None
        
        root_val = postorder[post_end]
        root = TreeNode(root_val)
        
        root_idx = inorder_map[root_val]
        left_size = root_idx - in_start
        
        root.left = build(in_start, root_idx - 1,
                         post_start, post_start + left_size - 1)
        root.right = build(root_idx + 1, in_end,
                          post_start + left_size, post_end - 1)
        
        return root
    
    return build(0, len(inorder) - 1, 0, len(postorder) - 1)


def build_tree_preorder_postorder(preorder: list[int], postorder: list[int]) -> TreeNode:
    """
    LeetCode 889: Construct from preorder and postorder.
    Note: Result may not be unique if nodes have only one child.
    """
    if not preorder:
        return None
    
    root = TreeNode(preorder[0])
    if len(preorder) == 1:
        return root
    
    # preorder[1] is root of left subtree
    # Find it in postorder to determine left subtree size
    left_root = preorder[1]
    left_size = postorder.index(left_root) + 1
    
    root.left = build_tree_preorder_postorder(preorder[1:left_size + 1],
                                               postorder[:left_size])
    root.right = build_tree_preorder_postorder(preorder[left_size + 1:],
                                                postorder[left_size:-1])
    
    return root


def sorted_array_to_bst(nums: list[int]) -> TreeNode:
    """
    LeetCode 108: Convert sorted array to height-balanced BST.
    Time: O(n), Space: O(log n)
    """
    def build(left, right):
        if left > right:
            return None
        
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)
        
        return root
    
    return build(0, len(nums) - 1)


def sorted_list_to_bst(head: 'ListNode') -> TreeNode:
    """
    LeetCode 109: Convert sorted linked list to BST.
    Time: O(n), Space: O(log n)
    
    Key: Build tree in inorder to avoid finding middle repeatedly.
    """
    # Find length
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next
    
    self_head = [head]  # Use list to allow modification in nested function
    
    def build(left, right):
        if left > right:
            return None
        
        mid = (left + right) // 2
        
        # Build left subtree first (inorder)
        left_child = build(left, mid - 1)
        
        # Current node
        root = TreeNode(self_head[0].val)
        root.left = left_child
        self_head[0] = self_head[0].next
        
        # Build right subtree
        root.right = build(mid + 1, right)
        
        return root
    
    return build(0, length - 1)
```

---

## 5. Path Problems

```python
def max_depth(root: TreeNode) -> int:
    """
    LeetCode 104: Maximum depth of binary tree.
    Time: O(n), Space: O(h)
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def min_depth(root: TreeNode) -> int:
    """
    LeetCode 111: Minimum depth (to nearest leaf).
    Time: O(n), Space: O(h)
    """
    if not root:
        return 0
    
    if not root.left:
        return 1 + min_depth(root.right)
    if not root.right:
        return 1 + min_depth(root.left)
    
    return 1 + min(min_depth(root.left), min_depth(root.right))


def has_path_sum(root: TreeNode, target_sum: int) -> bool:
    """
    LeetCode 112: Root-to-leaf path with target sum.
    Time: O(n), Space: O(h)
    """
    if not root:
        return False
    
    # Check if leaf with matching sum
    if not root.left and not root.right:
        return root.val == target_sum
    
    remaining = target_sum - root.val
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)


def path_sum_ii(root: TreeNode, target_sum: int) -> list[list[int]]:
    """
    LeetCode 113: Find all root-to-leaf paths with target sum.
    Time: O(n²), Space: O(n)
    """
    result = []
    
    def backtrack(node, remaining, path):
        if not node:
            return
        
        path.append(node.val)
        
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])
        else:
            backtrack(node.left, remaining - node.val, path)
            backtrack(node.right, remaining - node.val, path)
        
        path.pop()
    
    backtrack(root, target_sum, [])
    return result


def path_sum_iii(root: TreeNode, target_sum: int) -> int:
    """
    LeetCode 437: Count paths summing to target (any start/end).
    Time: O(n), Space: O(h)
    
    Use prefix sum with hashmap.
    """
    def dfs(node, curr_sum, prefix_sums):
        if not node:
            return 0
        
        curr_sum += node.val
        # Count paths ending at current node
        count = prefix_sums.get(curr_sum - target_sum, 0)
        
        # Add current sum to prefix sums
        prefix_sums[curr_sum] = prefix_sums.get(curr_sum, 0) + 1
        
        # Recurse
        count += dfs(node.left, curr_sum, prefix_sums)
        count += dfs(node.right, curr_sum, prefix_sums)
        
        # Backtrack
        prefix_sums[curr_sum] -= 1
        
        return count
    
    return dfs(root, 0, {0: 1})


def binary_tree_paths(root: TreeNode) -> list[str]:
    """
    LeetCode 257: All root-to-leaf paths as strings.
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node, path):
        if not node.left and not node.right:
            result.append(path + str(node.val))
            return
        
        if node.left:
            dfs(node.left, path + str(node.val) + "->")
        if node.right:
            dfs(node.right, path + str(node.val) + "->")
    
    dfs(root, "")
    return result


def max_path_sum(root: TreeNode) -> int:
    """
    LeetCode 124: Maximum path sum (any node to any node).
    Time: O(n), Space: O(h)
    
    Path can go through any node, not just root-to-leaf.
    """
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Max sum of left/right paths (ignore negative paths)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))
        
        # Path through current node
        current_path = node.val + left_max + right_max
        max_sum = max(max_sum, current_path)
        
        # Return max path starting from current node (can only go one direction up)
        return node.val + max(left_max, right_max)
    
    dfs(root)
    return max_sum


def diameter_of_binary_tree(root: TreeNode) -> int:
    """
    LeetCode 543: Diameter (longest path between any two nodes).
    Time: O(n), Space: O(h)
    """
    diameter = 0
    
    def depth(node):
        nonlocal diameter
        
        if not node:
            return 0
        
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        
        # Path through current node
        diameter = max(diameter, left_depth + right_depth)
        
        return 1 + max(left_depth, right_depth)
    
    depth(root)
    return diameter


def longest_univalue_path(root: TreeNode) -> int:
    """
    LeetCode 687: Longest path where all nodes have same value.
    Time: O(n), Space: O(h)
    """
    longest = 0
    
    def dfs(node):
        nonlocal longest
        
        if not node:
            return 0
        
        left_len = dfs(node.left)
        right_len = dfs(node.right)
        
        # Extend path if values match
        left_path = left_len + 1 if node.left and node.left.val == node.val else 0
        right_path = right_len + 1 if node.right and node.right.val == node.val else 0
        
        longest = max(longest, left_path + right_path)
        
        return max(left_path, right_path)
    
    dfs(root)
    return longest
```

---

## 6. Lowest Common Ancestor

```python
def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    LeetCode 236: LCA of two nodes in binary tree.
    Time: O(n), Space: O(h)
    """
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root  # p and q are in different subtrees
    
    return left or right


def lca_bst(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    LeetCode 235: LCA in BST.
    Time: O(h), Space: O(1)
    """
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    
    return None


def lca_with_parent(p: 'Node', q: 'Node') -> 'Node':
    """
    LeetCode 1650: LCA with parent pointers.
    Time: O(h), Space: O(1)
    
    Similar to finding intersection of two linked lists.
    """
    a, b = p, q
    
    while a != b:
        a = a.parent if a else q
        b = b.parent if b else p
    
    return a


def lca_deepest_leaves(root: TreeNode) -> TreeNode:
    """
    LeetCode 1123: LCA of deepest leaves.
    Time: O(n), Space: O(h)
    """
    def dfs(node):
        if not node:
            return (None, 0)  # (LCA, depth)
        
        left_lca, left_depth = dfs(node.left)
        right_lca, right_depth = dfs(node.right)
        
        if left_depth > right_depth:
            return (left_lca, left_depth + 1)
        elif right_depth > left_depth:
            return (right_lca, right_depth + 1)
        else:
            return (node, left_depth + 1)  # Current node is LCA
    
    return dfs(root)[0]


def distance_between_nodes(root: TreeNode, p: int, q: int) -> int:
    """
    Distance between two nodes = depth(p) + depth(q) - 2*depth(LCA).
    """
    def find_lca_and_depths(node, target1, target2, depth):
        if not node:
            return None, -1, -1
        
        if node.val == target1:
            d1 = depth
        else:
            d1 = -1
        
        if node.val == target2:
            d2 = depth
        else:
            d2 = -1
        
        left_lca, left_d1, left_d2 = find_lca_and_depths(node.left, target1, target2, depth + 1)
        right_lca, right_d1, right_d2 = find_lca_and_depths(node.right, target1, target2, depth + 1)
        
        d1 = max(d1, left_d1, right_d1)
        d2 = max(d2, left_d2, right_d2)
        
        if left_lca:
            return left_lca, d1, d2
        if right_lca:
            return right_lca, d1, d2
        
        if d1 != -1 and d2 != -1:
            return node, d1, d2
        
        return None, d1, d2
    
    lca, d1, d2 = find_lca_and_depths(root, p, q, 0)
    
    # Find depth of LCA
    lca_depth = 0
    curr = root
    while curr and curr.val != lca.val:
        if p < curr.val and q < curr.val:
            curr = curr.left
        elif p > curr.val and q > curr.val:
            curr = curr.right
        else:
            break
        lca_depth += 1
    
    return d1 + d2 - 2 * lca_depth
```

---

## 7. Serialization

```python
def serialize(root: TreeNode) -> str:
    """
    LeetCode 297: Serialize binary tree.
    
    Use preorder with null markers.
    """
    if not root:
        return "null"
    
    result = []
    
    def preorder(node):
        if not node:
            result.append("null")
            return
        
        result.append(str(node.val))
        preorder(node.left)
        preorder(node.right)
    
    preorder(root)
    return ",".join(result)


def deserialize(data: str) -> TreeNode:
    """Deserialize binary tree."""
    values = iter(data.split(","))
    
    def build():
        val = next(values)
        if val == "null":
            return None
        
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    
    return build()


def serialize_bst(root: TreeNode) -> str:
    """
    LeetCode 449: Serialize BST (more compact, no null markers).
    
    Use preorder - BST structure can be reconstructed from preorder alone.
    """
    if not root:
        return ""
    
    result = []
    
    def preorder(node):
        if node:
            result.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
    
    preorder(root)
    return ",".join(result)


def deserialize_bst(data: str) -> TreeNode:
    """Deserialize BST from preorder."""
    if not data:
        return None
    
    values = [int(x) for x in data.split(",")]
    
    def build(min_val, max_val):
        if not values or values[0] < min_val or values[0] > max_val:
            return None
        
        val = values.pop(0)
        node = TreeNode(val)
        node.left = build(min_val, val)
        node.right = build(val, max_val)
        return node
    
    return build(float('-inf'), float('inf'))


def codec_level_order(root: TreeNode) -> str:
    """Serialize using level order."""
    if not root:
        return ""
    
    from collections import deque
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")
    
    # Remove trailing nulls
    while result and result[-1] == "null":
        result.pop()
    
    return ",".join(result)
```

---

## 8. Advanced Tree Types

### Balanced Tree Check

```python
def is_balanced(root: TreeNode) -> bool:
    """
    LeetCode 110: Check if height-balanced.
    Time: O(n), Space: O(h)
    """
    def height(node):
        if not node:
            return 0
        
        left_h = height(node.left)
        if left_h == -1:
            return -1
        
        right_h = height(node.right)
        if right_h == -1:
            return -1
        
        if abs(left_h - right_h) > 1:
            return -1
        
        return 1 + max(left_h, right_h)
    
    return height(root) != -1
```

### Symmetric Tree

```python
def is_symmetric(root: TreeNode) -> bool:
    """
    LeetCode 101: Check if tree is symmetric.
    Time: O(n), Space: O(h)
    """
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        
        return (t1.val == t2.val and
                is_mirror(t1.left, t2.right) and
                is_mirror(t1.right, t2.left))
    
    return is_mirror(root, root)


def is_symmetric_iterative(root: TreeNode) -> bool:
    """Iterative approach using queue."""
    from collections import deque
    
    queue = deque([(root, root)])
    
    while queue:
        t1, t2 = queue.popleft()
        
        if not t1 and not t2:
            continue
        if not t1 or not t2 or t1.val != t2.val:
            return False
        
        queue.append((t1.left, t2.right))
        queue.append((t1.right, t2.left))
    
    return True
```

### Subtree Check

```python
def is_subtree(root: TreeNode, subRoot: TreeNode) -> bool:
    """
    LeetCode 572: Check if subRoot is a subtree of root.
    Time: O(m * n), Space: O(h)
    """
    def is_same(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1.val == t2.val and
                is_same(t1.left, t2.left) and
                is_same(t1.right, t2.right))
    
    if not root:
        return False
    
    if is_same(root, subRoot):
        return True
    
    return is_subtree(root.left, subRoot) or is_subtree(root.right, subRoot)
```

### Invert/Flip Tree

```python
def invert_tree(root: TreeNode) -> TreeNode:
    """
    LeetCode 226: Invert binary tree.
    Time: O(n), Space: O(h)
    """
    if not root:
        return None
    
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


def flip_equivalent(root1: TreeNode, root2: TreeNode) -> bool:
    """
    LeetCode 951: Check if trees are flip equivalent.
    """
    if not root1 and not root2:
        return True
    if not root1 or not root2 or root1.val != root2.val:
        return False
    
    # Either no flip or flip
    return ((flip_equivalent(root1.left, root2.left) and
             flip_equivalent(root1.right, root2.right)) or
            (flip_equivalent(root1.left, root2.right) and
             flip_equivalent(root1.right, root2.left)))
```

### Count Nodes

```python
def count_nodes_complete(root: TreeNode) -> int:
    """
    LeetCode 222: Count nodes in complete binary tree.
    Time: O(log²n), Space: O(log n)
    """
    if not root:
        return 0
    
    def get_height(node, go_left):
        height = 0
        while node:
            height += 1
            node = node.left if go_left else node.right
        return height
    
    left_height = get_height(root, True)
    right_height = get_height(root, False)
    
    if left_height == right_height:
        # Perfect tree
        return 2 ** left_height - 1
    else:
        # Recurse on subtrees
        return 1 + count_nodes_complete(root.left) + count_nodes_complete(root.right)
```

---

## 9. Interview Problems Summary

### Common Patterns

| Pattern | Example Problems | Key Technique |
|---------|------------------|---------------|
| DFS Recursion | Max Depth, Path Sum | Return value from children |
| BFS Level | Level Order, Right View | Queue with level size |
| BST Property | Validate, Kth Smallest | Inorder is sorted |
| Tree Construction | From Traversals | Divide and conquer |
| Path Problems | Max Path Sum, Diameter | Track global max |
| LCA | LCA Binary Tree/BST | Recursive search |

### Key Insights

1. **Recursion**: Think about what each node should return to its parent
2. **BST**: Inorder traversal gives sorted order
3. **Complete Binary Tree**: Can use array indexing (parent i, children 2i+1, 2i+2)
4. **Path Problems**: Often need to track both "path through node" and "path to node"
5. **Construction**: Need two traversals (or one for BST) to uniquely determine tree

### Time Complexity

| Operation | Balanced Tree | Skewed Tree |
|-----------|--------------|-------------|
| Traversal | O(n) | O(n) |
| Search BST | O(log n) | O(n) |
| Insert BST | O(log n) | O(n) |
| Find LCA | O(n) | O(n) |
| Serialize | O(n) | O(n) |
