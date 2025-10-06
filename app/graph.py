"""
Module for Eulerian graph analysis.

This module provides a Graph class that allows to:
- Check if a graph is Eulerian
- Determine if it's an Eulerian chain or cycle
- Generate an adjacency matrix
- Find an Eulerian path if it exists

Recommended input structure:
1. Edge list: [(u1, v1), (u2, v2), ...] for undirected graph
2. Arc list: [(u1, v1), (u2, v2), ...] for directed graph
3. Adjacency dictionary: {vertex: [neighbors], ...}

Complexity: O(V + E) for verification, O(E) to find the path
"""

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional, Union


class Graph:
    """
    Class to represent and analyze a graph.
    
    Supports directed and undirected graphs.
    Allows analysis of Eulerian properties.
    """
    
    def __init__(self, edges: List[Tuple] = None, adjacency_dict: Dict = None, directed: bool = False):
        """
        Initialize the graph.
        
        Args:
            edges: List of edges/arcs as tuples (u, v)
            adjacency_dict: Adjacency dictionary {vertex: [neighbors]}
            directed: True if the graph is directed, False otherwise
        """
        self.directed = directed
        self.adjacency_list = defaultdict(list)
        self.vertices = set()
        
        if edges:
            self._build_from_edges(edges)
        elif adjacency_dict:
            self._build_from_dict(adjacency_dict)
    
    def _build_from_edges(self, edges: List[Tuple]):
        """Build the graph from an edge list."""
        for edge in edges:
            u, v = edge[0], edge[1]
            self.vertices.add(u)
            self.vertices.add(v)
            self.adjacency_list[u].append(v)
            
            if not self.directed:
                self.adjacency_list[v].append(u)
    
    def _build_from_dict(self, adjacency_dict: Dict):
        """Build the graph from an adjacency dictionary."""
        for vertex, neighbors in adjacency_dict.items():
            self.vertices.add(vertex)
            for neighbor in neighbors:
                self.vertices.add(neighbor)
                self.adjacency_list[vertex].append(neighbor)
    
    def get_degree(self, vertex) -> int:
        """Return the degree of a vertex (undirected graph)."""
        if self.directed:
            raise ValueError("Use get_in_degree() and get_out_degree() for a directed graph")
        return len(self.adjacency_list[vertex])
    
    def get_in_degree(self, vertex) -> int:
        """Return the in-degree of a vertex (directed graph)."""
        if not self.directed:
            raise ValueError("Use get_degree() for an undirected graph")
        
        in_degree = 0
        for v in self.vertices:
            if vertex in self.adjacency_list[v]:
                in_degree += 1
        return in_degree
    
    def get_out_degree(self, vertex) -> int:
        """Return the out-degree of a vertex (directed graph)."""
        if not self.directed:
            raise ValueError("Use get_degree() for an undirected graph")
        return len(self.adjacency_list[vertex])
    
    def is_connected(self) -> bool:
        """Check if the graph is connected (undirected) or weakly connected (directed)."""
        if not self.vertices:
            return True
        
        # For a directed graph, check weak connectivity
        if self.directed:
            # Create an equivalent undirected graph
            undirected_adj = defaultdict(list)
            for u in self.adjacency_list:
                for v in self.adjacency_list[u]:
                    undirected_adj[u].append(v)
                    undirected_adj[v].append(u)
            
            # DFS on the undirected graph
            visited = set()
            start = next(iter(self.vertices))
            stack = [start]
            
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    for neighbor in undirected_adj[vertex]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            return len(visited) == len(self.vertices)
        
        else:
            # DFS for undirected graph
            visited = set()
            start = next(iter(self.vertices))
            stack = [start]
            
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    for neighbor in self.adjacency_list[vertex]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            return len(visited) == len(self.vertices)
    
    def is_eulerian(self) -> Tuple[bool, str]:
        """
        Check if the graph is Eulerian.
        
        Returns:
            Tuple (is_eulerian, type) where type can be:
            - "cycle": Eulerian cycle (all vertices have even degree)
            - "chain": Eulerian chain (exactly 2 vertices have odd degree)
            - "none": not Eulerian
        """
        if not self.is_connected():
            return False, "none"
        
        if self.directed:
            return self._is_eulerian_directed()
        else:
            return self._is_eulerian_undirected()
    
    def _is_eulerian_undirected(self) -> Tuple[bool, str]:
        """Check Eulerian properties for an undirected graph."""
        odd_degree_vertices = []
        
        for vertex in self.vertices:
            if self.get_degree(vertex) % 2 == 1:
                odd_degree_vertices.append(vertex)
        
        if len(odd_degree_vertices) == 0:
            return True, "cycle"
        elif len(odd_degree_vertices) == 2:
            return True, "chain"
        else:
            return False, "none"
    
    def _is_eulerian_directed(self) -> Tuple[bool, str]:
        """Check Eulerian properties for a directed graph."""
        # Check if all vertices have in-degree = out-degree
        all_balanced = True
        start_vertices = []  # Vertices with out_degree - in_degree = 1
        end_vertices = []    # Vertices with in_degree - out_degree = 1
        
        for vertex in self.vertices:
            in_deg = self.get_in_degree(vertex)
            out_deg = self.get_out_degree(vertex)
            
            if out_deg - in_deg == 1:
                start_vertices.append(vertex)
                all_balanced = False
            elif in_deg - out_deg == 1:
                end_vertices.append(vertex)
                all_balanced = False
            elif in_deg != out_deg:
                return False, "none"
        
        if all_balanced:
            return True, "cycle"
        elif len(start_vertices) == 1 and len(end_vertices) == 1:
            return True, "chain"
        else:
            return False, "none"
    
    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Generate the adjacency matrix of the graph.
        
        Returns:
            Adjacency matrix as a list of lists
        """
        vertices_list = sorted(list(self.vertices))
        n = len(vertices_list)
        vertex_to_index = {v: i for i, v in enumerate(vertices_list)}
        
        matrix = [[0] * n for _ in range(n)]
        
        for vertex in self.adjacency_list:
            i = vertex_to_index[vertex]
            for neighbor in self.adjacency_list[vertex]:
                j = vertex_to_index[neighbor]
                matrix[i][j] += 1  # +1 to handle multiple edges
        
        return matrix, vertices_list
    
    def find_eulerian_path(self) -> Optional[List]:
        """
        Find an Eulerian path if it exists (Hierholzer's algorithm).
        
        Returns:
            List of vertices in the Eulerian path, or None if non-existent
        """
        is_eul, eul_type = self.is_eulerian()
        if not is_eul:
            return None
        
        if self.directed:
            return self._hierholzer_directed(eul_type)
        else:
            return self._hierholzer_undirected(eul_type)
    
    def _hierholzer_undirected(self, eul_type: str) -> List:
        """Hierholzer's algorithm for undirected graph."""
        # Copy of adjacency list for modification
        adj_copy = defaultdict(list)
        for u in self.adjacency_list:
            adj_copy[u] = self.adjacency_list[u].copy()
        
        # Choose starting vertex
        if eul_type == "chain":
            # Start with a vertex of odd degree
            start = None
            for vertex in self.vertices:
                if len(adj_copy[vertex]) % 2 == 1:
                    start = vertex
                    break
        else:
            # Cycle: start with any vertex
            start = next(iter(self.vertices))
        
        # Hierholzer's algorithm
        circuit = []
        path = [start]
        
        while path:
            curr = path[-1]
            if adj_copy[curr]:
                next_vertex = adj_copy[curr].pop()
                adj_copy[next_vertex].remove(curr)  # Remove reverse edge
                path.append(next_vertex)
            else:
                circuit.append(path.pop())
        
        return circuit[::-1]
    
    def _hierholzer_directed(self, eul_type: str) -> List:
        """Hierholzer's algorithm for directed graph."""
        # Copy of adjacency list for modification
        adj_copy = defaultdict(list)
        for u in self.adjacency_list:
            adj_copy[u] = self.adjacency_list[u].copy()
        
        # Choose starting vertex
        if eul_type == "chain":
            # Start with vertex where out_degree - in_degree = 1
            start = None
            for vertex in self.vertices:
                if self.get_out_degree(vertex) - self.get_in_degree(vertex) == 1:
                    start = vertex
                    break
        else:
            # Circuit: start with any vertex
            start = next(iter(self.vertices))
        
        # Hierholzer's algorithm
        circuit = []
        path = [start]
        
        while path:
            curr = path[-1]
            if adj_copy[curr]:
                next_vertex = adj_copy[curr].pop()
                path.append(next_vertex)
            else:
                circuit.append(path.pop())
        
        return circuit[::-1]
    
    def print_analysis(self):
        """Display a complete analysis of the graph."""
        print(f"Graph {'directed' if self.directed else 'undirected'}")
        print(f"Vertices: {sorted(list(self.vertices))}")
        print(f"Number of vertices: {len(self.vertices)}")
        
        # Number of edges
        edge_count = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        if not self.directed:
            edge_count //= 2
        print(f"Number of edges: {edge_count}")
        
        print(f"Connected: {'Yes' if self.is_connected() else 'No'}")
        
        is_eul, eul_type = self.is_eulerian()
        if is_eul:
            if eul_type == "cycle":
                print("Eulerian type: Eulerian cycle (circuit)")
            else:
                print("Eulerian type: Eulerian chain (trail)")
        else:
            print("Eulerian type: Non-Eulerian")
        
        # Adjacency matrix
        matrix, vertices_order = self.get_adjacency_matrix()
        print(f"\nAdjacency matrix (order: {vertices_order}):")
        for row in matrix:
            print(row)
        
        # Eulerian path
        path = self.find_eulerian_path()
        if path:
            print(f"\nEulerian path: {' -> '.join(map(str, path))}")
        else:
            print("\nNo Eulerian path found")


# Usage examples
if __name__ == "__main__":
    print("=== Example 1: Eulerian cycle (square) ===")
    edges_square = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('B', 'D'), ('C', 'D'), ('C', 'D'), ('C', 'A')]
    g1 = Graph(edges=edges_square, directed=False)
    g1.print_analysis()
    
    print("\n" + "="*50)
    print("=== Example 2: Eulerian chain ===")
    edges_path = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'B')]
    g2 = Graph(edges=edges_path, directed=False)
    g2.print_analysis()
    
    print("\n" + "="*50)
    print("=== Example 3: Non-Eulerian ===")
    edges_star = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    g3 = Graph(edges=edges_star, directed=False)
    g3.print_analysis()
