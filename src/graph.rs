use indexmap::IndexMap;
use num_traits::Zero;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Add;
use std::rc::Rc;

// We use this wrapper over W to implement Ord from types that only implement PartialOrd. We simply assume that we
// it doesn't make sense to have edges with nan weights so we will panic if we encounter this while trying to do a
// comparison.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
struct NotNan<W> {
    value: W,
}

impl<W: PartialOrd> Eq for NotNan<W> {}

impl<W: PartialOrd> Ord for NotNan<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

//We use an alias for this type for convenience, but is just a hashmap
type AdjacencyList<N, W> = IndexMap<Rc<N>, Vec<Edge<N, W>>>;

/// Represents a weighted edge from a parent node to `to` Node
#[derive(Debug)]
struct Edge<N, W> {
    to: Rc<N>,
    weight: W,
}

/// Remove an edge from the adjacency list. If the edge doesn't exist, nothing is changed.
fn _remove_edge<N, W>(adj_list: &mut AdjacencyList<N, W>, from: &N, to: &N)
where
    N: Eq + Hash,
{
    // We are guaranteed to have only one directed edge between two nodes.
    // Otherwise we would need to use edges.retain(|edge| edge.to != to)
    // We first get a mutable reference of the edges for `from` node and map over it if the node exists.
    // We then look for the index of the edge we're looking fore and remove it if we find it.
    adj_list.get_mut(from).map(|edges| {
        if let Some(index) = edges.iter().position(|edge| edge.to.as_ref() == to) {
            edges.swap_remove(index);
        }
    });
}

/// Represents a graph stored using an adjacency list. It is implemented using a hashmap where nodes are used as keys
/// and a `Vec<Edge>` as values for the edges from the key node. If the graph is undirected, all edges are duplicated.
/// It is generic over the nodes and the edge weights.
// TODO: make it generic over the nodes and edges
#[derive(Debug)]
pub struct Graph<N, W>
where
    N: Eq + Hash,
{
    adj_list: AdjacencyList<N, W>,
    directed: bool,
}

impl Default for Graph<u32, i32> {
    /// Undirected graph with nodes encoded as 32 bits unsigned integers and weights as 32 bits integers.
    fn default() -> Self {
        Self {
            adj_list: IndexMap::new(),
            directed: false,
        }
    }
}

// TODO: we can do without the Clone trait
impl<N, W> Graph<N, W>
where
    N: Hash + Eq + Ord + Clone + Debug,
    W: Zero + Add + PartialOrd + Copy,
{
    pub fn new(directed: bool) -> Self {
        Self {
            adj_list: IndexMap::new(),
            directed,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.adj_list.len() == 0
    }

    pub fn is_directed(&self) -> bool {
        self.directed
    }

    pub fn num_nodes(&self) -> usize {
        self.adj_list.len()
    }

    pub fn num_edges(&self) -> usize {
        let mut cnt: usize = 0;
        for edges in self.adj_list.values() {
            cnt += edges.len()
        }

        cnt
    }

    /// Add a new node to the graph. Return an error if the node already exists, otherwise return a Rc to the node
    /// for further usage.  This is useful to avoid copies when N is not a primitive type.
    pub fn add_node(&mut self, node: N) -> Result<Rc<N>, String> {
        let node = Rc::new(node);

        if self.has_node(&node) {
            return Err(format!("Node {:?} already exists", node));
        }

        self.adj_list.insert(Rc::clone(&node), Vec::new());

        Ok(node)
    }

    /// Add a new  weighted edge between `from` and `to` nodes. Return an error if the edge already exists.
    pub fn add_edge(&mut self, from: N, to: N, weight: W) -> Result<(Rc<N>, Rc<N>), String> {
        // Use Rc to manage shared ownership and avoid move issues with `from` and `to`
        let from = Rc::new(from);
        let to = Rc::new(to);

        if self.has_edge(&from, &to) {
            return Err(format!("Edge from {:?} to {:?} already exists", from, to));
        }

        self.adj_list.entry(Rc::clone(&from)).or_default();
        self.adj_list.entry(Rc::clone(&to)).or_default();

        if let Some(edge_list) = self.adj_list.get_mut(&from) {
            edge_list.push(Edge {
                to: Rc::clone(&to),
                weight,
            });
        }

        if !self.directed {
            if let Some(reverse_edge_list) = self.adj_list.get_mut(&to) {
                reverse_edge_list.push(Edge {
                    to: Rc::clone(&from),
                    weight,
                });
            }
        }

        Ok((from, to))
    }

    /// Drop a node and any edge pointing to it.
    pub fn remove_node(&mut self, node: &N) {
        self.adj_list.swap_remove(node);

        // look for occurrences of the node in some edge from another node and drop the edge if found.
        for edges in self.adj_list.values_mut() {
            if let Some(index) = edges.iter().position(|edge| edge.to.as_ref() == node) {
                edges.swap_remove(index);
            }
        }
    }

    /// Drop an edge if it exists. This can result in nodes disconnected from the rest of the graph.
    pub fn remove_edge(&mut self, from: &N, to: &N) {
        _remove_edge(&mut self.adj_list, from, to);

        if !self.directed {
            _remove_edge(&mut self.adj_list, to, from);
        }
    }

    pub fn has_node(&self, node: &N) -> bool {
        self.adj_list.contains_key(node)
    }

    /// Check if two nodes are neighbors
    pub fn has_edge(&self, from: &N, to: &N) -> bool {
        // we first look for the node, if it doesn't exist we return false, otherwise we iterate over the edges
        // and try to find the one going to `to`. If we find it, we return true, otherwise false.
        self.adj_list.get(from).map_or(false, |edges| {
            edges.iter().find(|&edge| edge.to.as_ref() == to).is_some()
        })
    }

    /// Return references to all the nodes in the graph.
    pub fn get_nodes(&self) -> Vec<&N> {
        self.adj_list.keys().map(|node| node.as_ref()).collect()
    }

    /// Return the number of outgoing edges from a given node. Return None if the node is not in the graph.
    pub fn degree(&self, node: &N) -> Option<usize> {
        self.adj_list.get(node).map(|edges| edges.len())
    }

    /// Return references to all neighbors of `nodes`. Return None if the node is not in the graph.
    pub fn neighbors(&self, node: &N) -> Option<Vec<&N>> {
        self.adj_list
            .get(node)
            .map(|edges| edges.iter().map(|edge| edge.to.as_ref()).collect())
    }

    pub fn edge_weight(&self, from: &N, to: &N) -> Option<W> {
        // we first get a reference to the values if the key exists (hence its an option)
        // then we iterate over the edges corresponding to the node if we got Some(edges) until we find the edge
        // if we find it, we extract the weight on Some(edge) and flatten the nested options
        // otherwise returns None
        self.adj_list
            .get(from)
            .map(|edges| {
                edges
                    .iter()
                    .find(|&edge| edge.to.as_ref() == to)
                    .map(|edge| edge.weight)
            })
            .flatten()
    }

    /// Check if a path exists between two nodes
    pub fn has_path(&self, from: &N, to: &N) -> bool {
        self.dijkstra(from, to).is_some()
    }

    /// Return visited nodes in BFS (iterative) order. Nodes returned are copies and not references to graph nodes.
    /// If the node `from` is not in the graph, returns an empty option, otherwise returns a Vec<N>.
    pub fn dfs(&self, from: &N) -> Option<Vec<N>> {
        if !self.has_node(from) {
            return None;
        }

        let mut out: Vec<N> = Vec::new();
        let mut stack = Vec::from([from]);
        let mut visited: HashSet<&N> = HashSet::new();

        while let Some(node) = stack.pop() {
            if !visited.contains(node) {
                visited.insert(node);
                stack.extend(self.neighbors(node).unwrap());
                out.push(node.clone());
            }
        }

        Some(out)
    }

    /// Return visited nodes in BFS order. Nodes returned are copies and not references to graph nodes.
    /// If the node `from` is not in the graph, returns an empty option, otherwise returns a Vec<N>.
    pub fn bfs(&self, from: &N) -> Option<Vec<N>> {
        if !self.has_node(from) {
            return None;
        }

        let mut out: Vec<N> = Vec::new();
        let mut queue = VecDeque::from([from]);
        let mut visited = HashSet::from([from]);

        while let Some(node) = queue.pop_front() {
            for neighbor in self.neighbors(node).unwrap() {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
            out.push(node.clone());
        }

        Some(out)
    }

    pub fn dijkstra(&self, from: &N, to: &N) -> Option<(Vec<N>, W)> {
        // use an option to mean "not reachable" instead of infinity (as it would require another trait for W)
        let mut distances: IndexMap<&N, Option<W>> = self
            .adj_list
            .keys()
            .map(|node| (node.as_ref(), None))
            .collect();
        distances.insert(from, Some(W::zero()));

        // because the std binary heap only works on elements that implement Ord, we have to wrap our type W
        // in order to support float weights (that could potentially be nan and so only implement PartialOrd).
        let mut heap = BinaryHeap::new();
        heap.push((NotNan { value: W::zero() }, from));

        let mut predecessors: HashMap<&N, &N> = HashMap::new();

        while let Some((ref cost, node)) = heap.pop() {
            // found the destination, let's build the path and return it
            if node == to {
                let mut path = vec![to.clone()];
                let mut current_node = to;
                while let Some(&previous_node) = predecessors.get(current_node) {
                    path.push(previous_node.clone());
                    current_node = previous_node;
                }
                path.reverse();

                return Some((path, cost.value));
            }

            // we already found a better path from `from` to node so let's skip it
            if let Some(Some(distance)) = distances.get(node) {
                if &cost.value > distance {
                    continue;
                }
            }

            // update distances and predecessors for each neighbor
            if let Some(edges) = self.adj_list.get(node) {
                for edge in edges {
                    let next_node = edge.to.as_ref();
                    let new_cost = cost.value + edge.weight;

                    match distances.get(next_node) {
                        Some(Some(current_cost)) if &new_cost >= current_cost => continue,
                        _ => {
                            distances.insert(next_node, Some(new_cost));
                            predecessors.insert(next_node, node);
                            heap.push((
                                NotNan {
                                    value: Reverse(new_cost).0,
                                },
                                next_node,
                            ));
                        }
                    }
                }
            }
        }

        None
    }

    /// Returns the unique adjacency matrix with nodes ordered by insertion.
    pub fn adjacency_matrix(&self) -> Option<Vec<Vec<W>>> {
        if self.is_empty() {
            return None;
        }

        let n = self.num_nodes();
        let mut out = vec![vec![W::zero(); n]; n];

        for (from_idx, (_, edges)) in self.adj_list.iter().enumerate() {
            for Edge { to, weight } in edges.iter() {
                let (to_idx, ..) = self.adj_list.get_full(to.as_ref()).unwrap();
                out[from_idx][to_idx] = *weight;
            }
        }

        Some(out)
    }
}
