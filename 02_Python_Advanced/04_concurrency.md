# Concurrency & Parallelism - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Python has three concurrency models. Choose based on what the bottleneck is: IO-bound → `asyncio` or `threading`; CPU-bound → `multiprocessing`. The GIL prevents true thread parallelism for CPU tasks.

### When to Use Each Model

```
                Threading     Multiprocessing    Asyncio
GIL impact       Limited       None (own GIL)     N/A (single thread)
CPU-bound        ✗ No          ✓ Yes              ✗ No
IO-bound         ✓ Yes         ✓ Overkill         ✓ Best
Memory           Shared        Separate           Shared
Overhead         Low           High (fork/spawn)  Very low
Best for         IO + simple   CPU computation    High concurrency IO

Typical use cases:
  Threading:       File IO, HTTP requests, database calls (simple cases)
  Multiprocessing: Data processing, ML training, image processing
  Asyncio:         Web servers, API clients, websockets, many concurrent IO
```

### Asyncio — The Modern Way for IO-Bound

```python
import asyncio
import aiohttp

# async/await: cooperative multitasking (only one coroutine runs at a time)
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.json()   # yields control while waiting

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        # asyncio.gather: run coroutines concurrently (not in parallel!)
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # all run concurrently
    return results

# Run the event loop
asyncio.run(fetch_all(["http://api1.com", "http://api2.com"]))
# With asyncio: 100 concurrent requests take ~= time of 1 request!
# With sequential: 100 requests take 100x time of 1 request
```

### ThreadPoolExecutor — IO Parallelism Made Simple

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests

urls = ["http://api1.com", "http://api2.com", "http://api3.com"]

# Threading for IO-bound
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(requests.get, urls))  # concurrent requests

# Multiprocessing for CPU-bound
def cpu_heavy(n):
    return sum(i*i for i in range(n))

with ProcessPoolExecutor(max_workers=4) as executor:  # 4 CPU cores
    results = list(executor.map(cpu_heavy, [10**6, 10**6, 10**6, 10**6]))
```

### Thread Safety — Common Issues

```python
import threading

# Race condition example
counter = 0
lock = threading.Lock()

def increment():
    global counter
    # BAD (race condition): counter += 1  (read-modify-write is NOT atomic!)
    with lock:        # GOOD: mutex protects the critical section
        counter += 1

threads = [threading.Thread(target=increment) for _ in range(1000)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # 1000 (correct with lock)
```

### 🚨 Top Interview Pitfalls
- Saying "threads make Python faster for CPU tasks" — they don't due to the GIL; use `multiprocessing`
- Forgetting `t.join()` after spawning threads — main thread may exit before workers finish
- Not knowing that `asyncio` is **single-threaded** — it's concurrency via cooperative yielding, not parallelism
- Using `time.sleep()` in async code — blocks the event loop; use `asyncio.sleep()` instead
- Deadlock: acquiring locks in different orders in different threads — always acquire in consistent order

---

## Table of Contents
1. [Concurrency Concepts](#concurrency-concepts)
2. [Threading](#threading)
3. [Multiprocessing](#multiprocessing)
4. [Asyncio](#asyncio)
5. [Concurrent.futures](#concurrent-futures)
6. [Synchronization](#synchronization)
7. [Interview Questions](#interview-questions)

---

## 1. Concurrency Concepts

### Concurrency vs Parallelism

```python
"""
CONCURRENCY: Managing multiple tasks (may not run simultaneously)
- Tasks can start, run, and complete in overlapping time periods
- Single CPU can achieve concurrency via context switching
- Good for I/O-bound tasks

PARALLELISM: Running multiple tasks simultaneously
- Requires multiple CPUs/cores
- Tasks execute at the exact same time
- Good for CPU-bound tasks

Python's GIL (Global Interpreter Lock):
- Only one thread executes Python bytecode at a time
- Threading: Concurrent but not parallel (for CPU-bound)
- Multiprocessing: True parallelism (separate processes)
- Asyncio: Concurrent single-threaded (for I/O-bound)
"""

# I/O-bound: Waiting for external resources
# - Network requests
# - File operations
# - Database queries
# Best: asyncio or threading

# CPU-bound: Heavy computation
# - Mathematical calculations
# - Image processing
# - Data crunching
# Best: multiprocessing
```

### GIL Deep Dive

```python
"""
GIL (Global Interpreter Lock):
- Mutex that protects Python objects
- Prevents race conditions in CPython's memory management
- Only one thread executes Python bytecode at a time

When is GIL released?
- I/O operations (file, network, sleep)
- Some C extensions (NumPy, etc.)
- Every 5ms (Python 3.2+) to allow thread switching

Implications:
- CPU-bound multithreading: No speedup (may be slower!)
- I/O-bound multithreading: Works well
- For CPU parallelism: Use multiprocessing
"""

import threading
import time

# Demonstrating GIL impact on CPU-bound work
def cpu_bound(n):
    count = 0
    for i in range(n):
        count += i
    return count

# Sequential
start = time.time()
cpu_bound(10**7)
cpu_bound(10**7)
print(f"Sequential: {time.time() - start:.2f}s")

# Threaded (not faster due to GIL!)
start = time.time()
t1 = threading.Thread(target=cpu_bound, args=(10**7,))
t2 = threading.Thread(target=cpu_bound, args=(10**7,))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Threaded: {time.time() - start:.2f}s")
# Threaded is similar or slower than sequential!
```

---

## 2. Threading

### Basic Threading

```python
import threading
import time

# Method 1: Function target
def worker(name, delay):
    print(f"Worker {name} starting")
    time.sleep(delay)
    print(f"Worker {name} done")
    return f"Result from {name}"

# Create threads
t1 = threading.Thread(target=worker, args=("A", 1))
t2 = threading.Thread(target=worker, args=("B", 2))

# Start threads
t1.start()
t2.start()

# Wait for completion
t1.join()
t2.join()
print("All workers done")


# Method 2: Subclass Thread
class WorkerThread(threading.Thread):
    def __init__(self, name, delay):
        super().__init__()
        self.name = name
        self.delay = delay
        self.result = None
    
    def run(self):
        print(f"Worker {self.name} starting")
        time.sleep(self.delay)
        self.result = f"Result from {self.name}"
        print(f"Worker {self.name} done")

t = WorkerThread("C", 1)
t.start()
t.join()
print(t.result)


# Thread properties
t = threading.Thread(target=worker, args=("D", 1), daemon=True)
t.daemon = True  # Dies when main thread exits
t.name = "CustomName"
print(t.is_alive())
print(t.ident)  # Thread ID (after start)


# Current thread info
print(threading.current_thread().name)
print(threading.active_count())
print(threading.enumerate())  # List all threads
```

### Thread Communication with Queue

```python
import threading
import queue
import time

# Producer-Consumer pattern
def producer(q, items):
    for item in items:
        print(f"Producing {item}")
        q.put(item)
        time.sleep(0.5)
    q.put(None)  # Sentinel to signal completion

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consuming {item}")
        q.task_done()

# Thread-safe queue
q = queue.Queue(maxsize=10)

# Start threads
producer_thread = threading.Thread(target=producer, args=(q, range(5)))
consumer_thread = threading.Thread(target=consumer, args=(q,))

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()


# Queue types
q1 = queue.Queue()       # FIFO
q2 = queue.LifoQueue()   # LIFO (stack)
q3 = queue.PriorityQueue()  # Sorted by priority

# Priority queue usage
q3.put((2, "medium priority"))
q3.put((1, "high priority"))
q3.put((3, "low priority"))

while not q3.empty():
    print(q3.get())  # (1, high), (2, medium), (3, low)


# Queue methods
q = queue.Queue()
q.put(item)              # Add item (blocks if full)
q.put(item, block=False) # Raise queue.Full if full
q.put(item, timeout=1)   # Wait up to 1 second

item = q.get()           # Get item (blocks if empty)
item = q.get(block=False)# Raise queue.Empty if empty
item = q.get(timeout=1)  # Wait up to 1 second

q.task_done()            # Mark item as processed
q.join()                 # Wait until all items processed
q.qsize()                # Approximate size
q.empty()                # Is empty?
q.full()                 # Is full?
```

### Thread Pool

```python
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

# Using ThreadPoolExecutor (preferred)
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit individual tasks
    future = executor.submit(task, 5)
    print(future.result())  # 25
    
    # Map over iterable
    results = executor.map(task, range(10))
    print(list(results))


# Manual thread pool
class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)
    
    def _worker(self):
        while True:
            func, args, future = self.tasks.get()
            try:
                result = func(*args)
                future['result'] = result
                future['done'] = True
            except Exception as e:
                future['error'] = e
                future['done'] = True
            self.tasks.task_done()
    
    def submit(self, func, *args):
        future = {'done': False, 'result': None, 'error': None}
        self.tasks.put((func, args, future))
        return future
    
    def wait(self):
        self.tasks.join()
```

---

## 3. Multiprocessing

### Basic Multiprocessing

```python
import multiprocessing as mp
import time
import os

def worker(name):
    print(f"Worker {name} in process {os.getpid()}")
    time.sleep(1)
    return name * 2

# Create process
p = mp.Process(target=worker, args=("A",))
p.start()
p.join()


# Multiple processes
if __name__ == '__main__':  # Required on Windows!
    processes = []
    for i in range(4):
        p = mp.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()


# Getting return values with Queue
def worker_with_result(n, result_queue):
    result = n * n
    result_queue.put(result)

if __name__ == '__main__':
    result_queue = mp.Queue()
    processes = []
    
    for i in range(4):
        p = mp.Process(target=worker_with_result, args=(i, result_queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    results = [result_queue.get() for _ in range(4)]
    print(results)  # [0, 1, 4, 9]
```

### Process Pool

```python
from multiprocessing import Pool
import time

def cpu_bound_task(n):
    """Simulates CPU-intensive work."""
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == '__main__':
    # Create pool with 4 workers
    with Pool(processes=4) as pool:
        # Method 1: map (blocking, ordered results)
        results = pool.map(cpu_bound_task, [10**6] * 8)
        print(f"Map results: {results}")
        
        # Method 2: map_async (non-blocking)
        async_result = pool.map_async(cpu_bound_task, [10**6] * 8)
        # Do other work...
        results = async_result.get(timeout=10)
        
        # Method 3: apply (single task, blocking)
        result = pool.apply(cpu_bound_task, (10**6,))
        
        # Method 4: apply_async (single task, non-blocking)
        async_result = pool.apply_async(cpu_bound_task, (10**6,))
        result = async_result.get()
        
        # Method 5: imap (lazy iterator)
        for result in pool.imap(cpu_bound_task, [10**6] * 8):
            print(result)
        
        # Method 6: imap_unordered (results as completed)
        for result in pool.imap_unordered(cpu_bound_task, [10**6] * 8):
            print(result)
        
        # Method 7: starmap (for multiple arguments)
        def func(a, b):
            return a + b
        results = pool.starmap(func, [(1, 2), (3, 4), (5, 6)])
```

### Shared State

```python
import multiprocessing as mp
from multiprocessing import Value, Array, Manager

# Value and Array for simple shared data
def increment_counter(counter, lock):
    for _ in range(1000):
        with lock:
            counter.value += 1

if __name__ == '__main__':
    counter = Value('i', 0)  # 'i' = integer
    lock = mp.Lock()
    
    processes = [
        mp.Process(target=increment_counter, args=(counter, lock))
        for _ in range(4)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(counter.value)  # 4000


# Manager for complex shared data
def worker_with_manager(shared_list, shared_dict):
    shared_list.append(os.getpid())
    shared_dict[os.getpid()] = "done"

if __name__ == '__main__':
    with Manager() as manager:
        shared_list = manager.list()
        shared_dict = manager.dict()
        
        processes = [
            mp.Process(target=worker_with_manager, args=(shared_list, shared_dict))
            for _ in range(4)
        ]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        print(list(shared_list))
        print(dict(shared_dict))
```

### Inter-Process Communication

```python
import multiprocessing as mp

# Pipe (two-way communication)
def pipe_worker(conn):
    conn.send("Hello from child")
    print(conn.recv())
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=pipe_worker, args=(child_conn,))
    p.start()
    
    print(parent_conn.recv())  # Hello from child
    parent_conn.send("Hello from parent")
    
    p.join()


# Queue (thread/process safe)
def queue_worker(q):
    q.put("Item from worker")

if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=queue_worker, args=(q,))
    p.start()
    print(q.get())
    p.join()
```

---

## 4. Asyncio

### Basic Asyncio

```python
import asyncio

# Coroutine definition
async def greet(name, delay):
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"Hello, {name}!")
    return f"Greeted {name}"

# Running coroutines
async def main():
    # Sequential execution
    await greet("Alice", 1)
    await greet("Bob", 1)
    # Total: 2 seconds
    
    # Concurrent execution with gather
    results = await asyncio.gather(
        greet("Alice", 1),
        greet("Bob", 1),
        greet("Charlie", 1)
    )
    # Total: 1 second (concurrent)
    print(results)

# Run the event loop
asyncio.run(main())


# Creating tasks explicitly
async def main():
    # Create tasks (starts immediately)
    task1 = asyncio.create_task(greet("Alice", 2))
    task2 = asyncio.create_task(greet("Bob", 1))
    
    # Do other work...
    print("Tasks created")
    
    # Await results
    result1 = await task1
    result2 = await task2
    
    # Or wait for all
    # await asyncio.gather(task1, task2)

asyncio.run(main())
```

### Async Patterns

```python
import asyncio
import aiohttp  # pip install aiohttp

# Async HTTP requests
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def main():
    urls = [
        "http://example.com",
        "http://example.org",
        "http://example.net"
    ]
    results = await fetch_all(urls)
    for url, content in zip(urls, results):
        print(f"{url}: {len(content)} bytes")

# asyncio.run(main())


# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)
        return False

async def main():
    async with AsyncResource() as resource:
        print("Using resource")


# Async iterator
class AsyncRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __aiter__(self):
        self.current = self.start
        return self
    
    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        self.current += 1
        return self.current - 1

async def main():
    async for i in AsyncRange(0, 5):
        print(i)


# Async generator
async def async_generator(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for value in async_generator(5):
        print(value)
```

### Asyncio Synchronization

```python
import asyncio

# Lock
async def worker(lock, name):
    async with lock:
        print(f"{name} acquired lock")
        await asyncio.sleep(1)
        print(f"{name} releasing lock")

async def main():
    lock = asyncio.Lock()
    await asyncio.gather(
        worker(lock, "A"),
        worker(lock, "B"),
        worker(lock, "C")
    )


# Semaphore (limit concurrent access)
async def limited_worker(sem, name):
    async with sem:
        print(f"{name} acquired semaphore")
        await asyncio.sleep(1)
        print(f"{name} releasing semaphore")

async def main():
    sem = asyncio.Semaphore(2)  # Max 2 concurrent
    await asyncio.gather(
        *[limited_worker(sem, f"Worker-{i}") for i in range(5)]
    )


# Event (signaling between coroutines)
async def waiter(event, name):
    print(f"{name} waiting for event")
    await event.wait()
    print(f"{name} got event!")

async def setter(event):
    await asyncio.sleep(2)
    print("Setting event")
    event.set()

async def main():
    event = asyncio.Event()
    await asyncio.gather(
        waiter(event, "A"),
        waiter(event, "B"),
        setter(event)
    )


# Queue (async producer-consumer)
async def producer(queue):
    for i in range(5):
        await queue.put(i)
        print(f"Produced {i}")
        await asyncio.sleep(0.5)

async def consumer(queue):
    while True:
        item = await queue.get()
        print(f"Consumed {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    
    producers = asyncio.create_task(producer(queue))
    consumers = asyncio.create_task(consumer(queue))
    
    await producers
    await queue.join()
    consumers.cancel()
```

### Timeouts and Cancellation

```python
import asyncio

# Timeout
async def long_operation():
    await asyncio.sleep(10)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(long_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("Operation timed out!")


# Cancellation
async def cancellable_task():
    try:
        while True:
            print("Working...")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Task was cancelled!")
        raise  # Re-raise to properly cancel

async def main():
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(3)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Task cancelled successfully")


# Shield (protect from cancellation)
async def main():
    task = asyncio.create_task(long_operation())
    shielded = asyncio.shield(task)
    
    try:
        result = await asyncio.wait_for(shielded, timeout=2)
    except asyncio.TimeoutError:
        print("Shield timed out, but task continues")
        result = await task  # Original task still running
```

---

## 5. Concurrent.futures

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed, wait
import time

def task(n):
    time.sleep(1)
    return n * n

# ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit returns Future object
    future = executor.submit(task, 5)
    print(future.result())  # Blocks until done
    
    # Map over iterable
    results = list(executor.map(task, range(10)))
    print(results)


# ProcessPoolExecutor
if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(task, range(10)))
        print(results)


# Future object methods
with ThreadPoolExecutor() as executor:
    future = executor.submit(task, 5)
    
    future.done()       # Is complete?
    future.running()    # Is running?
    future.cancelled()  # Was cancelled?
    future.cancel()     # Attempt to cancel
    future.result(timeout=5)  # Get result
    future.exception(timeout=5)  # Get exception if any
    future.add_done_callback(lambda f: print(f.result()))


# as_completed (process results as they complete)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(task, n): n for n in range(10)}
    
    for future in as_completed(futures):
        n = futures[future]
        try:
            result = future.result()
            print(f"Task {n} completed: {result}")
        except Exception as e:
            print(f"Task {n} failed: {e}")


# wait (wait for specific conditions)
from concurrent.futures import FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(task, n) for n in range(10)]
    
    # Wait until first completes
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    print(f"First completed: {done.pop().result()}")
    
    # Wait for all
    done, pending = wait(futures, return_when=ALL_COMPLETED)
```

---

## 6. Synchronization

### Threading Synchronization

```python
import threading

# Lock (basic mutual exclusion)
lock = threading.Lock()

def with_lock():
    with lock:  # Acquires and releases
        # Critical section
        pass

def manual_lock():
    lock.acquire()
    try:
        # Critical section
        pass
    finally:
        lock.release()


# RLock (reentrant lock - same thread can acquire multiple times)
rlock = threading.RLock()

def recursive_func(n):
    with rlock:
        if n > 0:
            recursive_func(n - 1)  # Same thread can re-acquire


# Semaphore (limit concurrent access)
semaphore = threading.Semaphore(3)  # Max 3 concurrent

def limited_resource():
    with semaphore:
        # Only 3 threads here at once
        pass


# BoundedSemaphore (error if release without acquire)
bounded_sem = threading.BoundedSemaphore(3)


# Event (signaling between threads)
event = threading.Event()

def waiter():
    event.wait()  # Blocks until set
    print("Event received!")

def setter():
    time.sleep(2)
    event.set()  # Wake all waiters


# Condition (wait for condition with lock)
condition = threading.Condition()
data = []

def consumer():
    with condition:
        while not data:
            condition.wait()  # Release lock and wait
        item = data.pop(0)
        print(f"Consumed: {item}")

def producer():
    with condition:
        data.append("item")
        condition.notify()  # Wake one waiter
        # condition.notify_all()  # Wake all waiters


# Barrier (synchronize multiple threads at a point)
barrier = threading.Barrier(3)

def worker(name):
    print(f"{name} before barrier")
    barrier.wait()  # All threads must reach here
    print(f"{name} after barrier")

threads = [threading.Thread(target=worker, args=(f"T{i}",)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()


# Timer (delayed execution)
def delayed_task():
    print("Delayed task executed!")

timer = threading.Timer(5.0, delayed_task)
timer.start()
# timer.cancel()  # Cancel if needed
```

### Thread-Local Data

```python
import threading

# Thread-local storage
local_data = threading.local()

def worker(name):
    local_data.name = name  # Each thread has its own copy
    print(f"Thread {threading.current_thread().name}: {local_data.name}")

threads = [
    threading.Thread(target=worker, args=(f"Worker-{i}",))
    for i in range(3)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## 7. Interview Questions

```python
# Q1: Explain GIL and its implications
"""
GIL = Global Interpreter Lock
- Only one thread executes Python bytecode at a time
- Prevents race conditions in CPython's memory management
- CPU-bound multithreading: No speedup
- I/O-bound multithreading: Works well
- For CPU parallelism: Use multiprocessing
"""


# Q2: Threading vs Multiprocessing vs Asyncio?
"""
Threading:
- Shared memory space
- Limited by GIL for CPU-bound
- Good for I/O-bound tasks
- Simpler than multiprocessing

Multiprocessing:
- Separate memory space
- True parallelism (no GIL)
- Higher overhead (process creation)
- Good for CPU-bound tasks

Asyncio:
- Single-threaded
- Cooperative multitasking
- Best for I/O-bound with many connections
- Requires async/await syntax
"""


# Q3: How to share data between processes?
"""
1. Queue/Pipe - For message passing
2. Value/Array - For simple shared data
3. Manager - For complex shared objects
4. Shared memory - For raw memory sharing (Python 3.8+)
"""

# Q4: What is a race condition and how to prevent it?
"""
Race condition: Multiple threads/processes access shared data
and at least one modifies it, leading to unpredictable results.

Prevention:
- Locks/Mutexes
- Semaphores
- Atomic operations
- Thread-safe data structures
- Avoiding shared mutable state
"""
counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    with lock:
        counter += 1


# Q5: Explain deadlock and how to avoid it
"""
Deadlock: Two or more threads waiting for each other to release locks.

Prevention:
- Lock ordering (always acquire in same order)
- Timeout on lock acquisition
- Use RLock for recursive calls
- Avoid holding multiple locks
"""


# Q6: async/await vs threads?
"""
Async/await:
- Single thread, cooperative
- Lower overhead
- Requires async libraries
- Better for many I/O connections

Threads:
- Multiple threads, preemptive
- Higher overhead
- Works with any code
- Better for blocking operations
"""


# Q7: How to handle exceptions in threads?
"""
Exceptions in threads don't propagate to main thread.
Solutions:
1. Use concurrent.futures (exceptions stored in Future)
2. Use Queue to pass exceptions
3. Use threading.excepthook (Python 3.8+)
"""
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    future = executor.submit(risky_function)
    try:
        result = future.result()
    except Exception as e:
        print(f"Thread raised: {e}")


# Q8: What is a daemon thread?
"""
Daemon thread:
- Background thread that doesn't prevent program exit
- Automatically killed when main thread exits
- Good for: background tasks, heartbeats, cleanup
- Don't use for: tasks that must complete
"""
t = threading.Thread(target=func, daemon=True)


# Q9: ProcessPoolExecutor vs Pool?
"""
ProcessPoolExecutor:
- Part of concurrent.futures
- Consistent API with ThreadPoolExecutor
- Returns Future objects
- Recommended for new code

multiprocessing.Pool:
- More methods (map_async, starmap, etc.)
- More flexible
- Older API
"""


# Q10: How to limit concurrent connections?
"""
Use Semaphore or BoundedSemaphore:
"""
import asyncio

async def limited_fetch(url, sem):
    async with sem:
        # Only N concurrent connections
        return await fetch(url)

async def main():
    sem = asyncio.Semaphore(10)  # Max 10 concurrent
    urls = [...]
    tasks = [limited_fetch(url, sem) for url in urls]
    results = await asyncio.gather(*tasks)
```
