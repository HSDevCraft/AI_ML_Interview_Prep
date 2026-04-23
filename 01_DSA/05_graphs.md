# Graphs - Complete In-Depth Guide

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Graph Representations](#graph-representations)
3. [BFS & DFS](#bfs--dfs)
4. [Shortest Path Algorithms](#shortest-path-algorithms)
5. [Topological Sort](#topological-sort)
6. [Union-Find (Disjoint Set)](#union-find)
7. [Minimum Spanning Tree](#minimum-spanning-tree)
8. [Advanced Algorithms](#advanced-algorithms)
9. [Interview Problems](#interview-problems)

---

## 1. Fundamentals

### Graph Terminology

```
Graph G = (V, E) where V = vertices, E = edges

Types:
- Directed (digraph): Edges have direction (u → v)
- Undirected: Edges are bidirectional (u — v)
- Weighted: Edges have weights/costs
- Unweighted: All edges have equal weight (or weight 1)

Properties:
- Degree: Number of edges connected to a vertex
  - In-degree: Incoming edges (directed)
  - Out-degree: Outgoing edges (directed)
- Path: Sequence of vertices connected by edges
- Cycle: Path that starts and ends at same vertex
- Connected: Path exists between every pair of vertices (undirected)
- Strongly Connected: Path exists in both directions (directed)
- DAG: Directed Acyclic Graph (no cycles)

Special Graphs:
- Tree: Connected graph with n-1 edges, no cycles
- Bipartite: Vertices can be divided into two disjoint sets
- Complete: Edge between every pair of vertices
```

### Graph Complexities

| Representation | Space | Add Edge | Remove Edge | Query Edge | Iterate Neighbors |
|---------------|-------|----------|-------------|------------|-------------------|
| Adjacency Matrix | O(V²) | O(1) | O(1) | O(1) | O(V) |
| Adjacency List | O(V+E) | O(1) | O(E) | O(degree) | O(degree) |
| Edge List | O(E) | O(1) | O(E) | O(E) | O(E) |

---

## 2. Graph Representations

```python
from collections import defaultdict

# ==================== Adjacency List ====================

# Using dictionary of lists (most common)
graph = defaultdict(list)
graph[0].append(1)  # Edge 0 -> 1
graph[0].append(2)  # Edge 0 -> 2
graph[1].append(2)  # Edge 1 -> 2

# For weighted graphs: store (neighbor, weight) tuples
weighted_graph = defaultdict(list)
weighted_graph[0].append((1, 5))  # Edge 0 -> 1 with weight 5
weighted_graph[0].append((2, 3))  # Edge 0 -> 2 with weight 3

# Using dictionary of sets (for fast edge lookup)
graph_set = defaultdict(set)
graph_set[0].add(1)
print(1 in graph_set[0])  # O(1) edge check


# ==================== Adjacency Matrix ====================

# For dense graphs or when need O(1) edge query
n = 5  # Number of vertices
adj_matrix = [[0] * n for _ in range(n)]
adj_matrix[0][1] = 1  # Edge 0 -> 1
adj_matrix[0][2] = 1  # Edge 0 -> 2

# For weighted graphs, store weight instead of 1
weighted_matrix = [[float('inf')] * n for _ in range(n)]
for i in range(n):
    weighted_matrix[i][i] = 0  # Distance to self is 0
weighted_matrix[0][1] = 5  # Edge 0 -> 1 with weight 5


# ==================== Edge List ====================

# Useful for algorithms like Kruskal's MST
edges = [
    (0, 1, 5),   # (source, destination, weight)
    (0, 2, 3),
    (1, 2, 2),
]

# Sort by weight for MST algorithms
edges.sort(key=lambda x: x[2])


# ==================== Building Graph from Input ====================

def build_adjacency_list(n: int, edges: list[list[int]], directed: bool = True) -> dict:
    """Build adjacency list from edge list."""
    graph = defaultdict(list)
    
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    
    return graph


def build_adjacency_list_weighted(n: int, edges: list[list[int]], directed: bool = True) -> dict:
    """Build weighted adjacency list from edge list with weights."""
    graph = defaultdict(list)
    
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    
    return graph
```

---

## 3. BFS & DFS

### Breadth-First Search (BFS)

```python
from collections import deque

def bfs(graph: dict, start: int) -> list[int]:
    """
    BFS traversal from start vertex.
    Time: O(V + E), Space: O(V)
    
    Properties:
    - Visits nodes level by level
    - Finds shortest path in unweighted graphs
    - Uses queue (FIFO)
    """
    visited = {start}
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


def bfs_shortest_path(graph: dict, start: int, end: int) -> list[int]:
    """
    Find shortest path in unweighted graph using BFS.
    Time: O(V + E), Space: O(V)
    """
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path exists


def bfs_level_order(graph: dict, start: int) -> list[list[int]]:
    """BFS with level tracking."""
    visited = {start}
    queue = deque([start])
    levels = []
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        levels.append(current_level)
    
    return levels


def bfs_distance(graph: dict, start: int) -> dict[int, int]:
    """Find distance from start to all reachable nodes."""
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances
```

### Depth-First Search (DFS)

```python
def dfs_recursive(graph: dict, start: int, visited: set = None) -> list[int]:
    """
    DFS traversal using recursion.
    Time: O(V + E), Space: O(V) for call stack
    
    Properties:
    - Goes as deep as possible before backtracking
    - Uses stack (implicit via recursion)
    - Good for: path finding, cycle detection, topological sort
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result


def dfs_iterative(graph: dict, start: int) -> list[int]:
    """
    DFS traversal using explicit stack.
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        result.append(node)
        
        # Add neighbors in reverse order for consistent ordering
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return result


def dfs_all_paths(graph: dict, start: int, end: int) -> list[list[int]]:
    """
    Find all paths from start to end.
    Time: O(V! * V) worst case, Space: O(V * number of paths)
    
    LeetCode 797
    """
    all_paths = []
    
    def backtrack(node, path):
        if node == end:
            all_paths.append(path[:])
            return
        
        for neighbor in graph[node]:
            path.append(neighbor)
            backtrack(neighbor, path)
            path.pop()
    
    backtrack(start, [start])
    return all_paths


def has_cycle_undirected(graph: dict, n: int) -> bool:
    """
    Detect cycle in undirected graph using DFS.
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Back edge found
        
        return False
    
    # Check all components
    for node in range(n):
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False


def has_cycle_directed(graph: dict, n: int) -> bool:
    """
    Detect cycle in directed graph using DFS with coloring.
    Time: O(V + E), Space: O(V)
    
    Colors: 0 = white (unvisited), 1 = gray (in progress), 2 = black (done)
    """
    color = [0] * n
    
    def dfs(node):
        color[node] = 1  # Mark in progress
        
        for neighbor in graph[node]:
            if color[neighbor] == 1:  # Back edge to ancestor
                return True
            if color[neighbor] == 0 and dfs(neighbor):
                return True
        
        color[node] = 2  # Mark completed
        return False
    
    for node in range(n):
        if color[node] == 0:
            if dfs(node):
                return True
    
    return False
```

### Grid-Based BFS/DFS

```python
def number_of_islands(grid: list[list[str]]) -> int:
    """
    LeetCode 200: Count number of islands.
    Time: O(m * n), Space: O(m * n)
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        
        grid[i][j] = '0'  # Mark visited
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
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
    queue = deque([(0, 0, 1)])
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
    LeetCode 286: Multi-source BFS from all gates.
    Time: O(m * n), Space: O(m * n)
    """
    if not rooms:
        return
    
    m, n = len(rooms), len(rooms[0])
    INF = 2147483647
    
    queue = deque()
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:  # Gate
                queue.append((i, j))
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == INF:
                rooms[nx][ny] = rooms[x][y] + 1
                queue.append((nx, ny))


def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    LeetCode 417: Water flow to both oceans.
    Time: O(m * n), Space: O(m * n)
    
    Key insight: DFS/BFS from oceans, find intersection.
    """
    if not heights:
        return []
    
    m, n = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(i, j, visited):
        visited.add((i, j))
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < m and 0 <= nj < n and 
                (ni, nj) not in visited and 
                heights[ni][nj] >= heights[i][j]):
                dfs(ni, nj, visited)
    
    # Start from Pacific (top and left edges)
    for i in range(m):
        dfs(i, 0, pacific)
    for j in range(n):
        dfs(0, j, pacific)
    
    # Start from Atlantic (bottom and right edges)
    for i in range(m):
        dfs(i, n - 1, atlantic)
    for j in range(n):
        dfs(m - 1, j, atlantic)
    
    return list(pacific & atlantic)
```

---

## 4. Shortest Path Algorithms

### Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph: dict, start: int) -> dict[int, int]:
    """
    Single-source shortest path for non-negative weights.
    Time: O((V + E) log V), Space: O(V)
    
    graph format: {node: [(neighbor, weight), ...]}
    """
    distances = {start: 0}
    heap = [(0, start)]  # (distance, node)
    
    while heap:
        dist, node = heapq.heappop(heap)
        
        # Skip if we've found a better path
        if dist > distances.get(node, float('inf')):
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return distances


def dijkstra_with_path(graph: dict, start: int, end: int) -> tuple[int, list[int]]:
    """Dijkstra's with path reconstruction."""
    distances = {start: 0}
    parents = {start: None}
    heap = [(0, start)]
    
    while heap:
        dist, node = heapq.heappop(heap)
        
        if node == end:
            break
        
        if dist > distances.get(node, float('inf')):
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                parents[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))
    
    # Reconstruct path
    if end not in distances:
        return float('inf'), []
    
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = parents[curr]
    
    return distances[end], path[::-1]


def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """
    LeetCode 743: Time for signal to reach all nodes.
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    distances = dijkstra(graph, k)
    
    if len(distances) != n:
        return -1
    
    return max(distances.values())
```

### Bellman-Ford Algorithm

```python
def bellman_ford(n: int, edges: list[tuple], start: int) -> list[int]:
    """
    Single-source shortest path, handles negative weights.
    Time: O(V * E), Space: O(V)
    
    Can detect negative cycles.
    """
    distances = [float('inf')] * n
    distances[start] = 0
    
    # Relax all edges V-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                updated = True
        
        if not updated:  # Early termination
            break
    
    # Check for negative cycles
    for u, v, w in edges:
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            return None  # Negative cycle detected
    
    return distances


def cheapest_flights_k_stops(n: int, flights: list[list[int]], 
                             src: int, dst: int, k: int) -> int:
    """
    LeetCode 787: Cheapest flight with at most k stops.
    Time: O(k * E), Space: O(V)
    
    Modified Bellman-Ford with limited iterations.
    """
    distances = [float('inf')] * n
    distances[src] = 0
    
    for _ in range(k + 1):
        # Use temp to avoid using updated values in same iteration
        temp = distances[:]
        
        for u, v, w in flights:
            if distances[u] != float('inf'):
                temp[v] = min(temp[v], distances[u] + w)
        
        distances = temp
    
    return distances[dst] if distances[dst] != float('inf') else -1
```

### Floyd-Warshall Algorithm

```python
def floyd_warshall(n: int, edges: list[tuple]) -> list[list[int]]:
    """
    All-pairs shortest path.
    Time: O(V³), Space: O(V²)
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    # Initialize
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
    
    # Main algorithm
    for k in range(n):  # Intermediate vertex
        for i in range(n):  # Source
            for j in range(n):  # Destination
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist


def find_the_city(n: int, edges: list[list[int]], distanceThreshold: int) -> int:
    """
    LeetCode 1334: Find city with smallest number of reachable cities.
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Find city with minimum reachable cities
    min_count = n
    result = 0
    
    for i in range(n):
        count = sum(1 for j in range(n) if dist[i][j] <= distanceThreshold)
        if count <= min_count:
            min_count = count
            result = i
    
    return result
```

### 0-1 BFS

```python
def zero_one_bfs(graph: dict, start: int, n: int) -> list[int]:
    """
    Shortest path when edge weights are only 0 or 1.
    Time: O(V + E), Space: O(V)
    
    Use deque: add 0-weight edges to front, 1-weight to back.
    """
    distances = [float('inf')] * n
    distances[start] = 0
    dq = deque([start])
    
    while dq:
        node = dq.popleft()
        
        for neighbor, weight in graph[node]:
            new_dist = distances[node] + weight
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                
                if weight == 0:
                    dq.appendleft(neighbor)
                else:
                    dq.append(neighbor)
    
    return distances


def minimum_obstacles(grid: list[list[int]]) -> int:
    """
    LeetCode 2290: Minimum obstacles to remove.
    
    0-1 BFS where obstacle = weight 1, empty = weight 0.
    """
    m, n = len(grid), len(grid[0])
    distances = [[float('inf')] * n for _ in range(m)]
    distances[0][0] = grid[0][0]
    dq = deque([(0, 0)])
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while dq:
        x, y = dq.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < m and 0 <= ny < n:
                new_dist = distances[x][y] + grid[nx][ny]
                
                if new_dist < distances[nx][ny]:
                    distances[nx][ny] = new_dist
                    
                    if grid[nx][ny] == 0:
                        dq.appendleft((nx, ny))
                    else:
                        dq.append((nx, ny))
    
    return distances[m-1][n-1]
```

---

## 5. Topological Sort

```python
def topological_sort_kahn(graph: dict, n: int) -> list[int]:
    """
    Kahn's Algorithm (BFS-based).
    Time: O(V + E), Space: O(V)
    
    Process nodes with in-degree 0 first.
    """
    in_degree = [0] * n
    
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []  # Empty if cycle exists


def topological_sort_dfs(graph: dict, n: int) -> list[int]:
    """
    DFS-based topological sort.
    Time: O(V + E), Space: O(V)
    """
    visited = [0] * n  # 0: unvisited, 1: in progress, 2: done
    result = []
    has_cycle = [False]
    
    def dfs(node):
        if has_cycle[0]:
            return
        
        visited[node] = 1
        
        for neighbor in graph[node]:
            if visited[neighbor] == 1:  # Cycle
                has_cycle[0] = True
                return
            if visited[neighbor] == 0:
                dfs(neighbor)
        
        visited[node] = 2
        result.append(node)
    
    for i in range(n):
        if visited[i] == 0:
            dfs(i)
    
    return result[::-1] if not has_cycle[0] else []


def course_schedule(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    LeetCode 207: Can finish all courses?
    """
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        node = queue.popleft()
        count += 1
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses


def course_schedule_ii(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    LeetCode 210: Return course order.
    """
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return order if len(order) == numCourses else []


def alien_dictionary(words: list[str]) -> str:
    """
    LeetCode 269: Derive alien alphabet order.
    """
    # Build graph
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in graph}
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        
        # Check invalid case: prefix comes after full word
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""
        
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
    
    # Topological sort
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        c = queue.popleft()
        result.append(c)
        
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return ''.join(result) if len(result) == len(graph) else ""
```

---

## 6. Union-Find (Disjoint Set)

```python
class UnionFind:
    """
    Union-Find with path compression and union by rank.
    
    Operations:
    - find(x): O(α(n)) ≈ O(1) amortized
    - union(x, y): O(α(n)) ≈ O(1) amortized
    
    α(n) is inverse Ackermann function, effectively constant.
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of components
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union by rank. Returns True if merged, False if already connected.
        """
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        # Union by rank: attach smaller tree under larger
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in same component."""
        return self.find(x) == self.find(y)
    
    def get_count(self) -> int:
        """Get number of connected components."""
        return self.count


def number_of_connected_components(n: int, edges: list[list[int]]) -> int:
    """LeetCode 323: Count connected components."""
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.get_count()


def redundant_connection(edges: list[list[int]]) -> list[int]:
    """
    LeetCode 684: Find edge that creates a cycle.
    """
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates cycle
    
    return []


def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    LeetCode 721: Merge accounts with common emails.
    """
    email_to_id = {}
    email_to_name = {}
    
    # Map emails to indices
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = len(email_to_id)
            email_to_name[email] = name
    
    # Union emails in same account
    uf = UnionFind(len(email_to_id))
    
    for account in accounts:
        first_id = email_to_id[account[1]]
        for email in account[2:]:
            uf.union(first_id, email_to_id[email])
    
    # Group emails by root
    components = defaultdict(list)
    for email, idx in email_to_id.items():
        root = uf.find(idx)
        components[root].append(email)
    
    # Format result
    result = []
    for emails in components.values():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))
    
    return result


def smallest_string_with_swaps(s: str, pairs: list[list[int]]) -> str:
    """
    LeetCode 1202: Smallest string by swapping characters at given pairs.
    """
    n = len(s)
    uf = UnionFind(n)
    
    for i, j in pairs:
        uf.union(i, j)
    
    # Group indices by component
    components = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)
    
    # Sort characters within each component
    result = list(s)
    for indices in components.values():
        chars = sorted(result[i] for i in indices)
        for i, idx in enumerate(sorted(indices)):
            result[idx] = chars[i]
    
    return ''.join(result)
```

---

## 7. Minimum Spanning Tree

### Kruskal's Algorithm

```python
def kruskal_mst(n: int, edges: list[tuple]) -> tuple[int, list[tuple]]:
    """
    Kruskal's MST using Union-Find.
    Time: O(E log E), Space: O(V)
    
    edges: [(u, v, weight), ...]
    """
    # Sort edges by weight
    edges = sorted(edges, key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []
    
    for u, v, w in edges:
        if uf.union(u, v):  # No cycle
            mst_weight += w
            mst_edges.append((u, v, w))
            
            if len(mst_edges) == n - 1:
                break
    
    return mst_weight, mst_edges


def min_cost_connect_points(points: list[list[int]]) -> int:
    """
    LeetCode 1584: Minimum cost to connect all points.
    """
    n = len(points)
    
    # Build all edges with Manhattan distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((i, j, dist))
    
    # Kruskal's
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0
    
    for u, v, w in edges:
        if uf.union(u, v):
            total_cost += w
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return total_cost
```

### Prim's Algorithm

```python
def prim_mst(graph: dict, n: int) -> int:
    """
    Prim's MST using min-heap.
    Time: O((V + E) log V), Space: O(V)
    
    graph: {node: [(neighbor, weight), ...]}
    """
    visited = set()
    mst_weight = 0
    heap = [(0, 0)]  # (weight, node), start from node 0
    
    while heap and len(visited) < n:
        weight, node = heapq.heappop(heap)
        
        if node in visited:
            continue
        
        visited.add(node)
        mst_weight += weight
        
        for neighbor, edge_weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(heap, (edge_weight, neighbor))
    
    return mst_weight if len(visited) == n else -1


def min_cost_connect_points_prim(points: list[list[int]]) -> int:
    """Prim's approach for connecting points."""
    n = len(points)
    
    def manhattan(i, j):
        return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
    
    visited = set()
    heap = [(0, 0)]  # Start from point 0
    total_cost = 0
    
    while heap and len(visited) < n:
        cost, node = heapq.heappop(heap)
        
        if node in visited:
            continue
        
        visited.add(node)
        total_cost += cost
        
        for neighbor in range(n):
            if neighbor not in visited:
                heapq.heappush(heap, (manhattan(node, neighbor), neighbor))
    
    return total_cost
```

---

## 8. Advanced Algorithms

### Strongly Connected Components (Kosaraju's)

```python
def kosaraju_scc(graph: dict, n: int) -> list[list[int]]:
    """
    Find strongly connected components in directed graph.
    Time: O(V + E), Space: O(V)
    """
    # Step 1: DFS and record finish order
    visited = set()
    finish_order = []
    
    def dfs1(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)
    
    for node in range(n):
        if node not in visited:
            dfs1(node)
    
    # Step 2: Build reverse graph
    reverse_graph = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            reverse_graph[neighbor].append(node)
    
    # Step 3: DFS on reverse graph in reverse finish order
    visited = set()
    sccs = []
    
    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in reverse_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)
    
    for node in reversed(finish_order):
        if node not in visited:
            component = []
            dfs2(node, component)
            sccs.append(component)
    
    return sccs
```

### Tarjan's Algorithm (SCC)

```python
def tarjan_scc(graph: dict, n: int) -> list[list[int]]:
    """
    Tarjan's algorithm for strongly connected components.
    Time: O(V + E), Space: O(V)
    """
    index_counter = [0]
    index = [0] * n
    lowlink = [0] * n
    on_stack = [False] * n
    stack = []
    sccs = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in graph[node]:
            if index[neighbor] == 0 and neighbor != node:  # Unvisited
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif on_stack[neighbor]:
                lowlink[node] = min(lowlink[node], index[neighbor])
        
        # Root of SCC
        if lowlink[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)
    
    for node in range(n):
        if index[node] == 0:
            strongconnect(node)
    
    return sccs
```

### Bridges and Articulation Points

```python
def find_bridges(graph: dict, n: int) -> list[tuple]:
    """
    Find bridges (edges whose removal disconnects the graph).
    Time: O(V + E), Space: O(V)
    """
    disc = [0] * n  # Discovery time
    low = [0] * n   # Lowest reachable discovery time
    visited = [False] * n
    bridges = []
    timer = [0]
    
    def dfs(node, parent):
        visited[node] = True
        disc[node] = low[node] = timer[0]
        timer[0] += 1
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                
                # Bridge condition
                if low[neighbor] > disc[node]:
                    bridges.append((node, neighbor))
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])
    
    for node in range(n):
        if not visited[node]:
            dfs(node, -1)
    
    return bridges


def critical_connections(n: int, connections: list[list[int]]) -> list[list[int]]:
    """LeetCode 1192: Find all bridges."""
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    return find_bridges(graph, n)
```

### Bipartite Check

```python
def is_bipartite(graph: list[list[int]]) -> bool:
    """
    LeetCode 785: Check if graph is bipartite.
    Time: O(V + E), Space: O(V)
    
    Try to 2-color the graph.
    """
    n = len(graph)
    colors = [0] * n  # 0: uncolored, 1: color A, -1: color B
    
    def bfs(start):
        queue = deque([start])
        colors[start] = 1
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if colors[neighbor] == 0:
                    colors[neighbor] = -colors[node]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return False
        
        return True
    
    for node in range(n):
        if colors[node] == 0:
            if not bfs(node):
                return False
    
    return True
```

---

## 9. Interview Problems Summary

### Algorithm Selection Guide

| Problem Type | Algorithm | Time Complexity |
|--------------|-----------|-----------------|
| Shortest path (unweighted) | BFS | O(V + E) |
| Shortest path (non-negative weights) | Dijkstra | O((V+E) log V) |
| Shortest path (negative weights) | Bellman-Ford | O(V * E) |
| All-pairs shortest path | Floyd-Warshall | O(V³) |
| Dependency ordering | Topological Sort | O(V + E) |
| Connected components | Union-Find / DFS | O(V + E) |
| Minimum spanning tree | Kruskal / Prim | O(E log E) |
| Cycle detection | DFS with coloring | O(V + E) |
| Bridges/Articulation points | Tarjan's | O(V + E) |

### Key Patterns

1. **BFS**: Shortest path in unweighted, level-order, multi-source
2. **DFS**: Path finding, cycle detection, topological sort
3. **Dijkstra**: Weighted shortest path with non-negative weights
4. **Union-Find**: Dynamic connectivity, MST (Kruskal's)
5. **Topological Sort**: Dependency resolution, course scheduling

### Interview Tips

1. **Clarify**: Directed vs undirected, weighted vs unweighted, connected vs disconnected
2. **Representation**: Choose adjacency list for sparse, matrix for dense
3. **Multi-source BFS**: Start with all sources in queue
4. **Cycle detection**: Use colors (white/gray/black) for directed graphs
5. **Grid problems**: Convert to graph or process directly with directions array
