"""
Module pour l'analyse de graphes eulériens.

Ce module fournit une classe Graph qui permet de :
- Vérifier si un graphe est eulérien
- Déterminer s'il s'agit d'une chaîne ou d'un cycle eulérien
- Générer une matrice d'adjacence
- Trouver un chemin eulérien s'il existe

Structure d'entrée recommandée :
1. Liste d'arêtes : [(u1, v1), (u2, v2), ...] pour graphe non orienté
2. Liste d'arcs : [(u1, v1), (u2, v2), ...] pour graphe orienté
3. Dictionnaire d'adjacence : {sommet: [voisins], ...}

Complexité : O(V + E) pour la vérification, O(E) pour trouver le chemin
"""

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional, Union


class Graph:
    """
    Classe pour représenter et analyser un graphe.
    
    Supporte les graphes orientés et non orientés.
    Permet l'analyse des propriétés eulériennes.
    """
    
    def __init__(self, edges: List[Tuple] = None, adjacency_dict: Dict = None, directed: bool = False):
        """
        Initialise le graphe.
        
        Args:
            edges: Liste d'arêtes/arcs sous forme de tuples (u, v)
            adjacency_dict: Dictionnaire d'adjacence {sommet: [voisins]}
            directed: True si le graphe est orienté, False sinon
        """
        self.directed = directed
        self.adjacency_list = defaultdict(list)
        self.vertices = set()
        
        if edges:
            self._build_from_edges(edges)
        elif adjacency_dict:
            self._build_from_dict(adjacency_dict)
    
    def _build_from_edges(self, edges: List[Tuple]):
        """Construit le graphe à partir d'une liste d'arêtes."""
        for edge in edges:
            u, v = edge[0], edge[1]
            self.vertices.add(u)
            self.vertices.add(v)
            self.adjacency_list[u].append(v)
            
            if not self.directed:
                self.adjacency_list[v].append(u)
    
    def _build_from_dict(self, adjacency_dict: Dict):
        """Construit le graphe à partir d'un dictionnaire d'adjacence."""
        for vertex, neighbors in adjacency_dict.items():
            self.vertices.add(vertex)
            for neighbor in neighbors:
                self.vertices.add(neighbor)
                self.adjacency_list[vertex].append(neighbor)
    
    def get_degree(self, vertex) -> int:
        """Retourne le degré d'un sommet (graphe non orienté)."""
        if self.directed:
            raise ValueError("Utilisez get_in_degree() et get_out_degree() pour un graphe orienté")
        return len(self.adjacency_list[vertex])
    
    def get_in_degree(self, vertex) -> int:
        """Retourne le degré entrant d'un sommet (graphe orienté)."""
        if not self.directed:
            raise ValueError("Utilisez get_degree() pour un graphe non orienté")
        
        in_degree = 0
        for v in self.vertices:
            if vertex in self.adjacency_list[v]:
                in_degree += 1
        return in_degree
    
    def get_out_degree(self, vertex) -> int:
        """Retourne le degré sortant d'un sommet (graphe orienté)."""
        if not self.directed:
            raise ValueError("Utilisez get_degree() pour un graphe non orienté")
        return len(self.adjacency_list[vertex])
    
    def is_connected(self) -> bool:
        """Vérifie si le graphe est connexe (non orienté) ou faiblement connexe (orienté)."""
        if not self.vertices:
            return True
        
        # Pour un graphe orienté, on vérifie la connexité faible
        if self.directed:
            # Créer un graphe non orienté équivalent
            undirected_adj = defaultdict(list)
            for u in self.adjacency_list:
                for v in self.adjacency_list[u]:
                    undirected_adj[u].append(v)
                    undirected_adj[v].append(u)
            
            # DFS sur le graphe non orienté
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
            # DFS pour graphe non orienté
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
        Vérifie si le graphe est eulérien.
        
        Returns:
            Tuple (is_eulerian, type) où type peut être :
            - "cycle" : cycle eulérien (tous les sommets de degré pair)
            - "chain" : chaîne eulérienne (exactement 2 sommets de degré impair)
            - "none" : pas eulérien
        """
        if not self.is_connected():
            return False, "none"
        
        if self.directed:
            return self._is_eulerian_directed()
        else:
            return self._is_eulerian_undirected()
    
    def _is_eulerian_undirected(self) -> Tuple[bool, str]:
        """Vérifie les propriétés eulériennes pour un graphe non orienté."""
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
        """Vérifie les propriétés eulériennes pour un graphe orienté."""
        # Vérifier si tous les sommets ont degré entrant = degré sortant
        all_balanced = True
        start_vertices = []  # Sommets avec out_degree - in_degree = 1
        end_vertices = []    # Sommets avec in_degree - out_degree = 1
        
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
        Génère la matrice d'adjacence du graphe.
        
        Returns:
            Matrice d'adjacence sous forme de liste de listes
        """
        vertices_list = sorted(list(self.vertices))
        n = len(vertices_list)
        vertex_to_index = {v: i for i, v in enumerate(vertices_list)}
        
        matrix = [[0] * n for _ in range(n)]
        
        for vertex in self.adjacency_list:
            i = vertex_to_index[vertex]
            for neighbor in self.adjacency_list[vertex]:
                j = vertex_to_index[neighbor]
                matrix[i][j] += 1  # +1 pour gérer les arêtes multiples
        
        return matrix, vertices_list
    
    def find_eulerian_path(self) -> Optional[List]:
        """
        Trouve un chemin eulérien s'il existe (algorithme de Hierholzer).
        
        Returns:
            Liste des sommets du chemin eulérien, ou None si inexistant
        """
        is_eul, eul_type = self.is_eulerian()
        if not is_eul:
            return None
        
        if self.directed:
            return self._hierholzer_directed(eul_type)
        else:
            return self._hierholzer_undirected(eul_type)
    
    def _hierholzer_undirected(self, eul_type: str) -> List:
        """Algorithme de Hierholzer pour graphe non orienté."""
        # Copie de la liste d'adjacence pour modification
        adj_copy = defaultdict(list)
        for u in self.adjacency_list:
            adj_copy[u] = self.adjacency_list[u].copy()
        
        # Choisir le sommet de départ
        if eul_type == "chain":
            # Commencer par un sommet de degré impair
            start = None
            for vertex in self.vertices:
                if len(adj_copy[vertex]) % 2 == 1:
                    start = vertex
                    break
        else:
            # Cycle : commencer par n'importe quel sommet
            start = next(iter(self.vertices))
        
        # Algorithme de Hierholzer
        circuit = []
        path = [start]
        
        while path:
            curr = path[-1]
            if adj_copy[curr]:
                next_vertex = adj_copy[curr].pop()
                adj_copy[next_vertex].remove(curr)  # Supprimer l'arête inverse
                path.append(next_vertex)
            else:
                circuit.append(path.pop())
        
        return circuit[::-1]
    
    def _hierholzer_directed(self, eul_type: str) -> List:
        """Algorithme de Hierholzer pour graphe orienté."""
        # Copie de la liste d'adjacence pour modification
        adj_copy = defaultdict(list)
        for u in self.adjacency_list:
            adj_copy[u] = self.adjacency_list[u].copy()
        
        # Choisir le sommet de départ
        if eul_type == "chain":
            # Commencer par le sommet avec out_degree - in_degree = 1
            start = None
            for vertex in self.vertices:
                if self.get_out_degree(vertex) - self.get_in_degree(vertex) == 1:
                    start = vertex
                    break
        else:
            # Circuit : commencer par n'importe quel sommet
            start = next(iter(self.vertices))
        
        # Algorithme de Hierholzer
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
        """Affiche une analyse complète du graphe."""
        print(f"Graphe {'orienté' if self.directed else 'non orienté'}")
        print(f"Sommets: {sorted(list(self.vertices))}")
        print(f"Nombre de sommets: {len(self.vertices)}")
        
        # Nombre d'arêtes
        edge_count = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        if not self.directed:
            edge_count //= 2
        print(f"Nombre d'arêtes: {edge_count}")
        
        print(f"Connexe: {'Oui' if self.is_connected() else 'Non'}")
        
        is_eul, eul_type = self.is_eulerian()
        if is_eul:
            if eul_type == "cycle":
                print("Type eulérien: Cycle eulérien (circuit)")
            else:
                print("Type eulérien: Chaîne eulérienne (trail)")
        else:
            print("Type eulérien: Non eulérien")
        
        # Matrice d'adjacence
        matrix, vertices_order = self.get_adjacency_matrix()
        print(f"\nMatrice d'adjacence (ordre: {vertices_order}):")
        for row in matrix:
            print(row)
        
        # Chemin eulérien
        path = self.find_eulerian_path()
        if path:
            print(f"\nChemin eulérien: {' -> '.join(map(str, path))}")
        else:
            print("\nAucun chemin eulérien trouvé")


# Exemples d'utilisation
if __name__ == "__main__":
    print("=== Exemple 1: Cycle eulérien (carré) ===")
    edges_square = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('B', 'D'), ('C', 'D'), ('C', 'D'), ('C', 'A')]
    g1 = Graph(edges=edges_square, directed=False)
    g1.print_analysis()
    
    print("\n" + "="*50)
    print("=== Exemple 2: Chaîne eulérienne ===")
    edges_path = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'B')]
    g2 = Graph(edges=edges_path, directed=False)
    g2.print_analysis()
    
    print("\n" + "="*50)
    print("=== Exemple 3: Non eulérien ===")
    edges_star = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    g3 = Graph(edges=edges_star, directed=False)
    g3.print_analysis()
