# OOP, Metaclasses & Descriptors - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Python OOP is built on dunder methods. Every operator, comparison, container behavior, and context manager is a dunder method call. Understanding this unlocks the entire object model.

### OOP Pillars — Python Implementation

```python
# ENCAPSULATION: control access via naming conventions
class BankAccount:
    def __init__(self, balance):
        self._balance = balance      # protected (convention: don't access directly)
        self.__secret = "hidden"     # private (name-mangled to _BankAccount__secret)
    
    @property                         # encapsulate attribute access
    def balance(self): return self._balance
    
    @balance.setter
    def balance(self, value):
        if value < 0: raise ValueError("Balance cannot be negative")
        self._balance = value

# INHERITANCE: isinstance() and MRO (Method Resolution Order)
class Animal:
    def speak(self): return "..."

class Dog(Animal):
    def speak(self): return "Woof!"  # override

class GuideDog(Dog):
    def speak(self): return super().speak() + " (I'm a guide dog)"

# MRO: Python uses C3 linearization for diamond inheritance
# print(GuideDog.__mro__) to see the resolution order

# POLYMORPHISM: duck typing
def make_speak(animal):  # works for ANY object with .speak()
    return animal.speak()
```

### Essential Dunder Methods — Must Know

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __repr__(self): return f"Vector({self.x}, {self.y})"   # unambiguous repr for devs
    def __str__(self):  return f"({self.x}, {self.y})"         # readable repr for users
    def __eq__(self, other): return self.x == other.x and self.y == other.y  # ==
    def __hash__(self): return hash((self.x, self.y))  # needed if __eq__ defined!
    def __add__(self, other): return Vector(self.x+other.x, self.y+other.y)  # +
    def __len__(self): return 2                        # len()
    def __getitem__(self, i): return (self.x, self.y)[i]  # v[0], v[1]
    def __iter__(self): return iter((self.x, self.y))    # for x in v
    def __contains__(self, val): return val in (self.x, self.y)  # in operator
    
    # Context manager protocol
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return False  # don't suppress exceptions
```

### Abstract Classes vs Interfaces

```python
from abc import ABC, abstractmethod

class Shape(ABC):       # Abstract Base Class
    @abstractmethod
    def area(self) -> float:    # MUST be implemented by subclasses
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass
    
    def describe(self):         # concrete method (can be inherited)
        return f"Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return 3.14159 * self.r ** 2
    def perimeter(self): return 2 * 3.14159 * self.r

# Shape()  # TypeError: Can't instantiate abstract class
c = Circle(5)  # OK, all abstractmethods implemented
```

### 🚨 Top Interview Pitfalls
- Defining `__eq__` without defining `__hash__` — Python sets `__hash__ = None`, making objects unhashable (can't use in sets/dicts)
- Forgetting `__repr__` — always implement it; `__str__` falls back to `__repr__` if not defined, but not vice versa
- `super()` in multiple inheritance: always use `super()`, not `ParentClass.method(self)`, to respect MRO
- Mutable class attributes vs instance attributes: `class Foo: items = []` creates ONE list shared by ALL instances

---

## Table of Contents
1. [OOP Fundamentals](#oop-fundamentals)
2. [Inheritance & MRO](#inheritance--mro)
3. [Magic Methods](#magic-methods)
4. [Descriptors](#descriptors)
5. [Metaclasses](#metaclasses)
6. [Design Patterns](#design-patterns)
7. [Interview Questions](#interview-questions)

---

## 1. OOP Fundamentals

### Class Definition

```python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    _instance_count = 0
    
    def __init__(self, name: str, age: int):
        """Initialize instance attributes."""
        self.name = name        # Public attribute
        self._age = age         # Convention: "protected"
        self.__id = id(self)    # Name mangling: becomes _Dog__id
        Dog._instance_count += 1
    
    # Instance method
    def bark(self) -> str:
        return f"{self.name} says Woof!"
    
    # Class method - receives class as first argument
    @classmethod
    def get_instance_count(cls) -> int:
        return cls._instance_count
    
    @classmethod
    def from_birth_year(cls, name: str, birth_year: int) -> 'Dog':
        """Alternative constructor."""
        import datetime
        age = datetime.date.today().year - birth_year
        return cls(name, age)
    
    # Static method - no implicit first argument
    @staticmethod
    def is_adult(age: int) -> bool:
        return age >= 2
    
    # Property - computed attribute
    @property
    def age(self) -> int:
        return self._age
    
    @age.setter
    def age(self, value: int) -> None:
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value
    
    @age.deleter
    def age(self) -> None:
        del self._age
    
    # String representations
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.name}, {self._age} years old"
    
    def __repr__(self) -> str:
        """Developer representation (should be unambiguous)."""
        return f"Dog(name={self.name!r}, age={self._age!r})"


# Usage
dog = Dog("Buddy", 3)
print(dog.name)              # Buddy
print(dog.age)               # 3 (via property)
print(dog.bark())            # Buddy says Woof!
print(Dog.species)           # Canis familiaris
print(Dog.get_instance_count())  # 1

# Alternative constructor
dog2 = Dog.from_birth_year("Max", 2020)

# Static method
print(Dog.is_adult(3))       # True
```

### Encapsulation Conventions

```python
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance      # Protected (convention)
        self.__pin = "1234"          # Private (name mangling)
    
    def get_balance(self) -> float:
        return self._balance
    
    def _internal_method(self):
        """Convention: Don't call from outside class."""
        pass
    
    def __private_method(self):
        """Name mangled to _BankAccount__private_method."""
        pass


account = BankAccount(1000)
print(account._balance)              # Works (convention only)
# print(account.__pin)               # AttributeError
print(account._BankAccount__pin)     # "1234" (name mangling workaround)


# Better: Use properties
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance
    
    @property
    def balance(self) -> float:
        return self._balance
    
    @balance.setter
    def balance(self, value: float) -> None:
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value
```

### Slots

```python
import sys

# Regular class - uses __dict__ for attributes
class PointRegular:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With __slots__ - fixed set of attributes, less memory
class PointSlots:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Memory comparison
regular = PointRegular(1, 2)
slots = PointSlots(1, 2)

print(sys.getsizeof(regular.__dict__))  # ~104 bytes
# print(slots.__dict__)  # AttributeError - no __dict__

# Can't add new attributes with __slots__
regular.z = 3  # Works
# slots.z = 3  # AttributeError


# Slots with inheritance
class Point3DSlots(PointSlots):
    __slots__ = ['z']  # Add to parent's slots
    
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z
```

---

## 2. Inheritance & MRO

### Single Inheritance

```python
class Animal:
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        raise NotImplementedError("Subclass must implement")
    
    def move(self) -> str:
        return f"{self.name} is moving"


class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)  # Call parent __init__
        self.breed = breed
    
    def speak(self) -> str:
        return f"{self.name} says Woof!"
    
    def fetch(self) -> str:
        return f"{self.name} is fetching"


class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"


dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())    # Buddy says Woof!
print(dog.move())     # Buddy is moving (inherited)
print(dog.fetch())    # Buddy is fetching


# isinstance and issubclass
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

### Multiple Inheritance & MRO

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B -> " + super().method()

class C(A):
    def method(self):
        return "C -> " + super().method()

class D(B, C):
    def method(self):
        return "D -> " + super().method()


# Method Resolution Order (MRO) - C3 Linearization
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
print(d.method())  # D -> B -> C -> A


# Diamond problem resolved by MRO
class Base:
    def __init__(self):
        print("Base.__init__")

class Left(Base):
    def __init__(self):
        print("Left.__init__")
        super().__init__()

class Right(Base):
    def __init__(self):
        print("Right.__init__")
        super().__init__()

class Child(Left, Right):
    def __init__(self):
        print("Child.__init__")
        super().__init__()

c = Child()
# Child.__init__
# Left.__init__
# Right.__init__
# Base.__init__  (called only once!)
```

### Mixins

```python
# Mixins provide specific functionality to be mixed into classes
class JsonMixin:
    """Mixin to add JSON serialization."""
    def to_json(self) -> str:
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_str: str):
        import json
        data = json.loads(json_str)
        return cls(**data)


class LoggingMixin:
    """Mixin to add logging."""
    def log(self, message: str) -> None:
        print(f"[{self.__class__.__name__}] {message}")


class ComparableMixin:
    """Mixin for comparison operations."""
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        return not self.__eq__(other)


# Use mixins
class User(JsonMixin, LoggingMixin, ComparableMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email


user = User("Alice", "alice@example.com")
print(user.to_json())  # {"name": "Alice", "email": "alice@example.com"}
user.log("Created")    # [User] Created

user2 = User.from_json('{"name": "Alice", "email": "alice@example.com"}')
print(user == user2)   # True
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""
    
    @abstractmethod
    def area(self) -> float:
        """Calculate area. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter. Must be implemented by subclasses."""
        pass
    
    # Concrete method (optional)
    def description(self) -> str:
        return f"A shape with area {self.area():.2f}"


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        import math
        return 2 * math.pi * self.radius


# Can't instantiate abstract class
# shape = Shape()  # TypeError

rect = Rectangle(3, 4)
print(rect.area())        # 12
print(rect.description()) # A shape with area 12.00


# Abstract properties
class Vehicle(ABC):
    @property
    @abstractmethod
    def num_wheels(self) -> int:
        pass

class Car(Vehicle):
    @property
    def num_wheels(self) -> int:
        return 4
```

---

## 3. Magic Methods

### Object Lifecycle

```python
class Resource:
    def __new__(cls, *args, **kwargs):
        """Called before __init__ to create instance."""
        print("__new__ called")
        instance = super().__new__(cls)
        return instance
    
    def __init__(self, name: str):
        """Initialize instance."""
        print("__init__ called")
        self.name = name
    
    def __del__(self):
        """Called when object is garbage collected (avoid using)."""
        print("__del__ called")
    
    def __repr__(self) -> str:
        return f"Resource({self.name!r})"


r = Resource("test")
# __new__ called
# __init__ called
del r
# __del__ called (timing not guaranteed)
```

### Comparison Methods

```python
from functools import total_ordering

@total_ordering  # Generates missing comparison methods
class Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))


v1 = Version(1, 2, 3)
v2 = Version(1, 2, 4)
print(v1 < v2)   # True
print(v1 <= v2)  # True (generated by @total_ordering)
print(v1 == v2)  # False
```

### Container Methods

```python
class Playlist:
    def __init__(self, songs: list = None):
        self._songs = songs or []
    
    def __len__(self) -> int:
        return len(self._songs)
    
    def __getitem__(self, index):
        return self._songs[index]
    
    def __setitem__(self, index, value):
        self._songs[index] = value
    
    def __delitem__(self, index):
        del self._songs[index]
    
    def __contains__(self, item) -> bool:
        return item in self._songs
    
    def __iter__(self):
        return iter(self._songs)
    
    def __reversed__(self):
        return reversed(self._songs)


playlist = Playlist(["Song1", "Song2", "Song3"])
print(len(playlist))        # 3
print(playlist[0])          # Song1
print("Song2" in playlist)  # True
for song in playlist:
    print(song)
```

### Numeric Methods

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    # Unary operators
    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y)
    
    def __pos__(self) -> 'Vector':
        return Vector(+self.x, +self.y)
    
    def __abs__(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    # Binary operators
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        """Right multiplication: scalar * vector."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self.x / scalar, self.y / scalar)
    
    # In-place operators
    def __iadd__(self, other: 'Vector') -> 'Vector':
        self.x += other.x
        self.y += other.y
        return self
    
    # Dot product
    def __matmul__(self, other: 'Vector') -> float:
        return self.x * other.x + self.y * other.y
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"


v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(v1 + v2)      # Vector(4, 6)
print(v1 * 2)       # Vector(6, 8)
print(2 * v1)       # Vector(6, 8) via __rmul__
print(abs(v1))      # 5.0
print(v1 @ v2)      # 11 (dot product)
```

### Attribute Access

```python
class DynamicObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name: str):
        """Called when attribute not found normally."""
        return f"Attribute '{name}' not found"
    
    def __setattr__(self, name: str, value):
        """Called for every attribute assignment."""
        print(f"Setting {name} = {value}")
        super().__setattr__(name, value)
    
    def __delattr__(self, name: str):
        """Called when deleting attribute."""
        print(f"Deleting {name}")
        super().__delattr__(name)
    
    def __getattribute__(self, name: str):
        """Called for EVERY attribute access (be careful!)."""
        print(f"Accessing {name}")
        return super().__getattribute__(name)


obj = DynamicObject(x=10)
# Setting x = 10
print(obj.x)        # Accessing x, 10
print(obj.unknown)  # Accessing unknown, Attribute 'unknown' not found
```

### Context Manager Methods

```python
class DatabaseConnection:
    def __init__(self, host: str):
        self.host = host
        self.connection = None
    
    def __enter__(self):
        """Enter context: setup."""
        print(f"Connecting to {self.host}")
        self.connection = f"Connection to {self.host}"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: cleanup."""
        print(f"Closing connection to {self.host}")
        self.connection = None
        
        # Return True to suppress exception, False to propagate
        if exc_type is ValueError:
            print("Suppressing ValueError")
            return True
        return False


with DatabaseConnection("localhost") as db:
    print(db.connection)
# Connecting to localhost
# Connection to localhost
# Closing connection to localhost
```

### Callable Objects

```python
class Multiplier:
    def __init__(self, factor: int):
        self.factor = factor
    
    def __call__(self, value: int) -> int:
        """Make instance callable like a function."""
        return value * self.factor


double = Multiplier(2)
triple = Multiplier(3)

print(double(5))    # 10
print(triple(5))    # 15
print(callable(double))  # True


# Use case: Stateful functions
class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
```

---

## 4. Descriptors

### Descriptor Protocol

```python
class Descriptor:
    """
    Descriptor protocol methods:
    - __get__(self, obj, objtype=None): Called on attribute access
    - __set__(self, obj, value): Called on attribute assignment
    - __delete__(self, obj): Called on attribute deletion
    - __set_name__(self, owner, name): Called when descriptor assigned to class
    """
    
    def __set_name__(self, owner, name):
        """Called when assigned to class attribute."""
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        """Called on attribute access."""
        if obj is None:
            return self  # Accessed via class
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj, value):
        """Called on attribute assignment."""
        setattr(obj, self.private_name, value)
    
    def __delete__(self, obj):
        """Called on attribute deletion."""
        delattr(obj, self.private_name)


class MyClass:
    attr = Descriptor()

obj = MyClass()
obj.attr = 42       # Calls Descriptor.__set__
print(obj.attr)     # Calls Descriptor.__get__ -> 42
print(obj._attr)    # 42 (stored in instance __dict__)
```

### Validation Descriptors

```python
class Validator:
    """Base validator descriptor."""
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)
    
    def validate(self, value):
        pass  # Override in subclasses


class PositiveNumber(Validator):
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if value < 0:
            raise ValueError(f"{self.name} must be positive")


class NonEmptyString(Validator):
    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if not value.strip():
            raise ValueError(f"{self.name} cannot be empty")


class RangeValidator(Validator):
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value):
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"{self.name} must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"{self.name} must be <= {self.max_val}")


class Person:
    name = NonEmptyString()
    age = PositiveNumber()
    score = RangeValidator(0, 100)
    
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score


person = Person("Alice", 30, 85)
# person.age = -5    # ValueError: age must be positive
# person.name = ""   # ValueError: name cannot be empty
# person.score = 150 # ValueError: score must be <= 100
```

### Data vs Non-Data Descriptors

```python
"""
Data Descriptor: Has __get__ AND __set__ (or __delete__)
Non-Data Descriptor: Has only __get__

Lookup order:
1. Data descriptor from class
2. Instance __dict__
3. Non-data descriptor from class
4. Class __dict__
5. __getattr__
"""

class DataDescriptor:
    def __get__(self, obj, objtype=None):
        return "data descriptor"
    
    def __set__(self, obj, value):
        pass  # Makes it a data descriptor

class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        return "non-data descriptor"

class MyClass:
    data = DataDescriptor()
    non_data = NonDataDescriptor()

obj = MyClass()
obj.__dict__['data'] = "instance"
obj.__dict__['non_data'] = "instance"

print(obj.data)      # "data descriptor" (descriptor wins)
print(obj.non_data)  # "instance" (instance __dict__ wins)
```

### Property as Descriptor

```python
# @property is implemented as a descriptor!
class Property:
    """Simplified property implementation."""
    
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)
    
    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)
    
    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)
    
    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)
    
    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)
    
    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

---

## 5. Metaclasses

### Understanding Metaclasses

```python
"""
Metaclass: A class whose instances are classes.
- Regular objects are instances of classes
- Classes are instances of metaclasses
- The default metaclass is 'type'
"""

# Classes are instances of type
class MyClass:
    pass

print(type(MyClass))        # <class 'type'>
print(isinstance(MyClass, type))  # True

# Creating class with type()
MyClass2 = type('MyClass2', (object,), {'x': 10, 'method': lambda self: 'hello'})
print(MyClass2.x)           # 10
print(MyClass2().method())  # hello
```

### Custom Metaclass

```python
class MyMeta(type):
    """Custom metaclass."""
    
    def __new__(mcs, name, bases, namespace):
        """Called when a new class is created."""
        print(f"Creating class: {name}")
        
        # Modify namespace before class creation
        namespace['created_by'] = 'MyMeta'
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        return cls
    
    def __init__(cls, name, bases, namespace):
        """Called after class is created."""
        print(f"Initializing class: {name}")
        super().__init__(name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        """Called when class is instantiated."""
        print(f"Creating instance of {cls.__name__}")
        return super().__call__(*args, **kwargs)


class MyClass(metaclass=MyMeta):
    pass

# Creating class: MyClass
# Initializing class: MyClass

obj = MyClass()
# Creating instance of MyClass

print(MyClass.created_by)  # MyMeta
```

### Practical Metaclass Examples

```python
# 1. Singleton metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

db1 = Database()
db2 = Database()
print(db1 is db2)  # True


# 2. Registry metaclass
class PluginMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Don't register base class
            mcs.registry[name] = cls
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class AudioPlugin(Plugin):
    pass

class VideoPlugin(Plugin):
    pass

print(PluginMeta.registry)
# {'AudioPlugin': <class 'AudioPlugin'>, 'VideoPlugin': <class 'VideoPlugin'>}


# 3. Validation metaclass
class ValidatedMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Ensure all methods have docstrings
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_'):
                if not value.__doc__:
                    raise TypeError(f"Method {key} must have a docstring")
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=ValidatedMeta):
    def method(self):
        """This method has a docstring."""
        pass
    
    # def bad_method(self):  # Would raise TypeError
    #     pass


# 4. Auto-property metaclass
class AutoPropertyMeta(type):
    def __new__(mcs, name, bases, namespace):
        for key in list(namespace.keys()):
            if key.startswith('_') and not key.startswith('__'):
                public_name = key[1:]
                namespace[public_name] = property(
                    lambda self, k=key: getattr(self, k),
                    lambda self, v, k=key: setattr(self, k, v)
                )
        return super().__new__(mcs, name, bases, namespace)

class Person(metaclass=AutoPropertyMeta):
    def __init__(self, name, age):
        self._name = name
        self._age = age

p = Person("Alice", 30)
print(p.name)  # Alice (via auto-generated property)
```

### `__init_subclass__` (Alternative to Metaclass)

```python
# Python 3.6+ - Simpler than metaclass for many use cases
class Plugin:
    registry = []
    
    def __init_subclass__(cls, category=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.category = category
        Plugin.registry.append(cls)

class AudioPlugin(Plugin, category="audio"):
    pass

class VideoPlugin(Plugin, category="video"):
    pass

print(Plugin.registry)        # [AudioPlugin, VideoPlugin]
print(AudioPlugin.category)   # audio
print(VideoPlugin.category)   # video
```

---

## 6. Design Patterns

```python
# Factory Pattern
class Animal:
    @staticmethod
    def factory(animal_type: str):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal: {animal_type}")


# Builder Pattern
class QueryBuilder:
    def __init__(self):
        self._select = "*"
        self._from = None
        self._where = []
    
    def select(self, fields):
        self._select = fields
        return self
    
    def from_table(self, table):
        self._from = table
        return self
    
    def where(self, condition):
        self._where.append(condition)
        return self
    
    def build(self):
        query = f"SELECT {self._select} FROM {self._from}"
        if self._where:
            query += " WHERE " + " AND ".join(self._where)
        return query

query = QueryBuilder().select("name, age").from_table("users").where("age > 18").build()


# Observer Pattern
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


# Strategy Pattern
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # Simplified

class BubbleSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # Simplified

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data):
        return self._strategy.sort(data)
```

---

## 7. Interview Questions

```python
# Q1: What is the difference between @classmethod and @staticmethod?
"""
@classmethod: Receives class as first argument (cls)
- Can access class attributes
- Can create instances
- Used for alternative constructors

@staticmethod: No implicit first argument
- Can't access class or instance
- Just a regular function in class namespace
- Used for utility functions
"""


# Q2: Explain MRO (Method Resolution Order)
"""
MRO determines the order in which base classes are searched for methods.
Python uses C3 linearization algorithm.
"""
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass
print(D.__mro__)  # D -> B -> C -> A -> object


# Q3: What are descriptors?
"""
Objects that define __get__, __set__, or __delete__ methods.
Used to customize attribute access.
Properties are implemented using descriptors.
"""


# Q4: What is a metaclass?
"""
A metaclass is a class whose instances are classes.
'type' is the default metaclass.
Used to customize class creation.
"""


# Q5: Difference between __new__ and __init__?
"""
__new__: Creates and returns the instance (class method)
__init__: Initializes the instance (instance method)
__new__ is called before __init__
"""


# Q6: What is __slots__?
"""
__slots__ defines a fixed set of attributes.
- Saves memory (no __dict__)
- Faster attribute access
- Can't add new attributes dynamically
"""


# Q7: Explain super() in multiple inheritance
"""
super() follows MRO, not just parent class.
Enables cooperative multiple inheritance.
"""
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")
        super().method()  # Calls next in MRO, not necessarily A

class C(A):
    def method(self):
        print("C")
        super().method()

class D(B, C):
    def method(self):
        print("D")
        super().method()

D().method()  # D, B, C, A


# Q8: What is duck typing?
"""
"If it walks like a duck and quacks like a duck, it's a duck."
Python focuses on object behavior, not type.
Check for methods/attributes, not specific types.
"""


# Q9: Abstract base classes vs duck typing?
"""
ABC: Explicit interface definition
Duck typing: Implicit interface

Use ABC when:
- Want to enforce interface
- Need isinstance() checks
- Framework/library development
"""


# Q10: How does property work internally?
"""
@property is a descriptor that:
- Stores getter, setter, deleter functions
- __get__ calls getter
- __set__ calls setter
- __delete__ calls deleter
"""
```
