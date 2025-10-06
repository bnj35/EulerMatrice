# Eulerian Graph Analyzer

This module provides a complete `Graph` class to analyze the Eulerian properties of a graph and find Eulerian paths/cycles.

## 🎯 Features

- **Eulerian detection**: Determines if a graph has an Eulerian path or cycle
- **Classification**: Distinguishes between Eulerian trail and Eulerian circuit
- **Adjacency matrix**: Generates the matrix representation of the graph
- **Hierholzer's algorithm**: Finds a complete Eulerian path if it exists
- **Support for directed/undirected graphs**

## 📊 Recommended input structures

### 1. Edge list (Recommended)
```python
# Undirected graph
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]

# Directed graph  
edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
```

**Advantages**:
- Simple and intuitive
- Easy to build from tabular data
- Automatic vertex management

### 2. Adjacency dictionary
```python
# Example: X->Y, X->Z, Y->Z
adjacency = {
    'X': ['Y', 'Z'],
    'Y': ['Z'], 
    'Z': []
}
```

**Advantages**:
- Precise control of structure
- Efficient for sparse graphs
- Allows multiple edges

## 🚀 Usage

### Basic example
```python
from graph import Graph

# Create a square graph (cycle)
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
g = Graph(edges=edges, directed=False)

# Complete analysis
g.print_analysis()

# Specific checks
is_eulerian, euler_type = g.is_eulerian()
path = g.find_eulerian_path()
matrix, vertices = g.get_adjacency_matrix()
```

### Directed graph
```python
# Directed Eulerian circuit
edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
g = Graph(edges=edges, directed=True)
g.print_analysis()
```

### From adjacency dictionary
```python
adj = {'A': ['B'], 'B': ['C'], 'C': ['A']}
g = Graph(adjacency_dict=adj, directed=True)
```

## 📋 Graph class API

### Constructor
```python
Graph(edges=None, adjacency_dict=None, directed=False)
```

### Main methods

#### `is_eulerian() -> Tuple[bool, str]`
Returns `(True/False, "cycle"/"chain"/"none")`

#### `find_eulerian_path() -> Optional[List]`
Finds a complete Eulerian path using Hierholzer's algorithm

#### `get_adjacency_matrix() -> Tuple[List[List[int]], List]`
Returns `(matrix, vertex_order)`

#### `print_analysis()`
Displays a complete graph analysis

## 🔍 Types of Eulerian graphs

### Undirected graph
- **Eulerian cycle**: All vertices have even degree
- **Eulerian trail**: Exactly 2 vertices have odd degree
- **Non-Eulerian**: More than 2 vertices with odd degree

### Directed graph  
- **Eulerian circuit**: in_degree = out_degree for all vertices
- **Eulerian path**: 1 vertex with out_degree - in_degree = 1, 1 vertex with in_degree - out_degree = 1
- **Non-Eulerian**: Other cases

## ⚡ Complexity

- **Eulerian verification**: O(V + E)
- **Path search**: O(E) 
- **Adjacency matrix**: O(V²)

Where V = number of vertices, E = number of edges

## 🧪 Tests and examples

Run the built-in examples:
```bash
# In the Docker container
docker exec graph python graph.py
```

### Example output

**Eulerian cycle (square)**:
```
Undirected graph
Vertices: ['A', 'B', 'C', 'D']
Eulerian type: Eulerian cycle (circuit)
Eulerian path: B -> C -> D -> A -> B
```

**Eulerian trail**:
```
Eulerian type: Eulerian trail
Eulerian path: B -> D -> C -> B -> A
```

**Non-Eulerian (star)**:
```
Eulerian type: Non-Eulerian
No Eulerian path found
```

## 🎓 Typical use cases

1. **Chinese postman problem**: Optimize a delivery route
2. **Drawing puzzle**: Draw a figure without lifting the pencil
3. **Electronic circuits**: Verify connection continuity
4. **Transportation networks**: Analyze optimal routes

## 🐳 Docker

The project includes a Docker configuration with Jupyter Lab:

```bash
docker-compose up -d  # Start
docker exec graph python graph.py  # Test
```

Jupyter Lab available at http://localhost:8888

