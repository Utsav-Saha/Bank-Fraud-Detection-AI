"""
Graph-Based Fraud Ring & Investigation Service for SecureBank AI.
Features a self-contained pure-Python directed graph implementation with
optional NetworkX acceleration.
Implements Breadth-First Search (BFS), Depth-First Search (DFS),
and cycle detection for money laundering rings.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from collections import deque

class SimpleDiGraph:
    """Pure-Python directed graph implementation for zero-dependency portability."""
    def __init__(self):
        self._adj = {}  # {node: {neighbor: edge_data}}
        self._nodes = {}  # {node: node_data}

    def add_node(self, node: str, **attrs):
        if node not in self._nodes:
            self._nodes[node] = attrs
            self._adj[node] = {}

    def add_edge(self, u: str, v: str, **attrs):
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = attrs

    def neighbors(self, node: str):
        return iter(self._adj.get(node, {}).keys())

    def get_edge_data(self, u: str, v: str) -> Optional[Dict[str, Any]]:
        return self._adj.get(u, {}).get(v, None)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self._adj.values())

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        edges_list = []
        for u, neighbors in self._adj.items():
            for v, data in neighbors.items():
                edges_list.append((u, v, data))
        return edges_list


class TransactionGraphAnalyzer:
    def __init__(self):
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
            self._is_nx = True
        except ImportError:
            self.graph = SimpleDiGraph()
            self._is_nx = False
        
    def add_transaction(
        self,
        sender_acc: str,
        receiver_acc: str,
        amount: float,
        txn_id: str,
        is_fraud: bool = False,
        timestamp: str = ""
    ):
        """Adds a directed financial transfer edge between accounts."""
        self.graph.add_node(sender_acc, type="account")
        self.graph.add_node(receiver_acc, type="account")
        self.graph.add_edge(
            sender_acc,
            receiver_acc,
            amount=amount,
            txn_id=txn_id,
            is_fraud=is_fraud,
            timestamp=timestamp
        )

    def bfs_connected_accounts(self, start_account: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        BFS Traversal to find all directly and indirectly connected accounts up to max_hops.
        Returns list of reached accounts with their shortest hop distance.
        """
        if start_account not in self.graph.nodes:
            return []
            
        visited = {start_account: 0}
        queue = deque([(start_account, 0)])
        results = []
        
        while queue:
            curr_node, hops = queue.popleft()
            if hops >= max_hops:
                continue
                
            for neighbor in self.graph.neighbors(curr_node):
                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    edge_data = self.graph.get_edge_data(curr_node, neighbor) or {}
                    results.append({
                        "from_account": curr_node,
                        "to_account": neighbor,
                        "hops": hops + 1,
                        "amount": edge_data.get("amount", 0.0),
                        "txn_id": edge_data.get("txn_id", "N/A"),
                        "is_fraud": edge_data.get("is_fraud", False)
                    })
                    queue.append((neighbor, hops + 1))
        return results

    def dfs_find_paths(
        self,
        start_account: str,
        end_account: str,
        max_depth: int = 4
    ) -> List[List[str]]:
        """
        DFS Traversal to identify all financial pathways between two accounts.
        """
        if start_account not in self.graph.nodes or end_account not in self.graph.nodes:
            return []
            
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target and len(path) > 1:
                all_paths.append(list(path))
                return
            for neighbor in self.graph.neighbors(current):
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, target, path, depth + 1)
                    path.pop()
                    
        dfs(start_account, end_account, [start_account], 0)
        return all_paths

    def detect_circular_money_rings(self) -> List[Dict[str, Any]]:
        """
        Detects circular transactions (A -> B -> C -> A) commonly seen in
        money laundering and structuring rings.
        """
        rings = []
        if self._is_nx:
            try:
                import networkx as nx
                cycles = list(nx.simple_cycles(self.graph))
                for cycle in cycles:
                    if len(cycle) >= 3:
                        total_ring_amount = 0.0
                        fraud_count = 0
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            data = self.graph.get_edge_data(u, v) or {}
                            total_ring_amount += data.get("amount", 0.0)
                            if data.get("is_fraud", False):
                                fraud_count += 1
                        rings.append({
                            "ring_path": " ➔ ".join(cycle + [cycle[0]]),
                            "accounts": cycle,
                            "cycle_length": len(cycle),
                            "total_volume_usd": round(total_ring_amount, 2),
                            "suspicious_edges": fraud_count
                        })
                return rings
            except Exception:
                pass
                
        # Pure-Python elementary cycle finding fallback (DFS based)
        visited_nodes: Set[str] = set()
        
        def find_cycles_from(start: str, current: str, path: List[str], depth: int):
            if depth > 6:
                return
            for neighbor in self.graph.neighbors(current):
                if neighbor == start and len(path) >= 3:
                    # Found a cycle
                    cycle = list(path)
                    total_amount = sum((self.graph.get_edge_data(cycle[i], cycle[(i+1)%len(cycle)]) or {}).get("amount", 0.0) for i in range(len(cycle)))
                    rings.append({
                        "ring_path": " ➔ ".join(cycle + [cycle[0]]),
                        "accounts": cycle,
                        "cycle_length": len(cycle),
                        "total_volume_usd": round(total_amount, 2),
                        "suspicious_edges": 1
                    })
                elif neighbor not in path and neighbor not in visited_nodes:
                    path.append(neighbor)
                    find_cycles_from(start, neighbor, path, depth + 1)
                    path.pop()
                    
        for node in list(self.graph.nodes.keys()) if hasattr(self.graph.nodes, "keys") else list(self.graph.nodes):
            find_cycles_from(node, node, [node], 0)
            visited_nodes.add(node)
            
        return rings

    def get_graph_summary(self) -> Dict[str, Any]:
        """Summary statistics of current transaction graph."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        return {
            "total_accounts": n_nodes,
            "total_transfers": n_edges,
            "is_directed": True,
            "density": round(density, 4),
            "backend": "networkx" if self._is_nx else "pure_python_dag"
        }

def generate_sample_banking_network() -> TransactionGraphAnalyzer:
    """
    Constructs a realistic synthetic multi-account transaction network
    for the Investigation Dashboard.
    """
    analyzer = TransactionGraphAnalyzer()
    
    # 1. Laundering Ring Alpha (Circular 4-hop ring)
    analyzer.add_transaction("ACC-8801 (Mule A)", "ACC-8802 (Shell Co)", 4500.0, "TXN-RING-01", is_fraud=True)
    analyzer.add_transaction("ACC-8802 (Shell Co)", "ACC-8803 (Offshore)", 4450.0, "TXN-RING-02", is_fraud=True)
    analyzer.add_transaction("ACC-8803 (Offshore)", "ACC-8804 (Crypto Broker)", 4400.0, "TXN-RING-03", is_fraud=True)
    analyzer.add_transaction("ACC-8804 (Crypto Broker)", "ACC-8801 (Mule A)", 4350.0, "TXN-RING-04", is_fraud=True)
    
    # 2. Fan-out Smurfing (Structuring) Network
    analyzer.add_transaction("ACC-1009 (Master)", "ACC-2001 (Smurf 1)", 950.0, "TXN-SMURF-1", is_fraud=True)
    analyzer.add_transaction("ACC-1009 (Master)", "ACC-2002 (Smurf 2)", 980.0, "TXN-SMURF-2", is_fraud=True)
    analyzer.add_transaction("ACC-1009 (Master)", "ACC-2003 (Smurf 3)", 920.0, "TXN-SMURF-3", is_fraud=True)
    analyzer.add_transaction("ACC-2001 (Smurf 1)", "ACC-9999 (Collector)", 940.0, "TXN-CONSOLIDATE-1", is_fraud=True)
    analyzer.add_transaction("ACC-2002 (Smurf 2)", "ACC-9999 (Collector)", 970.0, "TXN-CONSOLIDATE-2", is_fraud=True)
    analyzer.add_transaction("ACC-2003 (Smurf 3)", "ACC-9999 (Collector)", 910.0, "TXN-CONSOLIDATE-3", is_fraud=True)
    
    # 3. Legitimate Commercial Transactions
    analyzer.add_transaction("ACC-5001 (Alice)", "ACC-MERCHANT (Amazon)", 129.50, "TXN-LEGIT-1", is_fraud=False)
    analyzer.add_transaction("ACC-5002 (Bob)", "ACC-MERCHANT (Amazon)", 74.20, "TXN-LEGIT-2", is_fraud=False)
    analyzer.add_transaction("ACC-5003 (Charlie)", "ACC-UTILITY (Power Co)", 210.00, "TXN-LEGIT-3", is_fraud=False)
    analyzer.add_transaction("ACC-5001 (Alice)", "ACC-5002 (Bob)", 45.00, "TXN-LEGIT-4", is_fraud=False)
    
    return analyzer
