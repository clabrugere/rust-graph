#![allow(dead_code)]

use itertools::Itertools;
use num_traits:: Num;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::fmt::Debug;
use std::hash::Hash;
use std::cmp::Reverse;

// We use aliases for those traits for convenience.
pub trait Node: Eq + Hash + Ord + Copy + Debug {}
impl<T: Eq + Hash + Ord + Copy + Debug> Node for T {}

// Ord is required to use a heap, but that means that we can't work with floats (they have nan...)
pub trait Weight: Num + Copy + Ord {}
impl<T: Num + Copy + Ord> Weight for T {}

// Same for this type, which is just a hashmap
type AdjacencyList<N, W>
where
    N: Node,
    W: Weight,
= HashMap<N, Vec<Edge<N, W>>>;

/// Represents a weighted edge from a parent node to `to` Node
#[derive(Debug)]
pub struct Edge<N, W>
where
    N: Node,
    W: Weight,
{
    to: N,
    weight: W,
}

/// Add a new edge to the adjacency list. If nodes `from` and `to` don't exist, insert them first.
/// It assumes that the caller first check that the edge doesn't already exist.
fn _add_edge<N, W>(adj_list: &mut AdjacencyList<N, W>, from: N, to: N, weight: W)
where
    N: Node,
    W: Weight,
{
    adj_list
        .entry(from)
        .or_default()
        .push(Edge { to: to, weight });

    adj_list.entry(to).or_default();
}

/// Remove an edge from the adjacency list. If the edge doesn't exist, nothing is changed.
fn _remove_edge<N, W>(adj_list: &mut AdjacencyList<N, W>, from: &N, to: &N)
where
    N: Node,
    W: Weight,
{
    // We are guaranteed to have only one directed edge between two nodes.
    // Otherwise we would need to use edges.retain(|edge| edge.to != to)
    // We first get a mutable reference of the edges for `from` node and map over it if the node exists.
    // We then look for the index of the edge we're looking fore and remove it if we find it.
    adj_list.get_mut(from).map(|edges| {
        if let Some(index) = edges.iter().position(|edge| &edge.to == to) {
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
    N: Node,
    W: Weight,
{
    adj_list: AdjacencyList<N, W>,
    directed: bool,
}

impl Default for Graph<u32, i32> {
    /// Defaults to `Graph<u32, i32>` of nodes encoded as 32 bits unsigned integers and weights as 32 bits integers.
    fn default() -> Self {
        Self {
            adj_list: HashMap::new(),
            directed: false,
        }
    }
}

impl<N, W> Graph<N, W>
where
    N: Node,
    W: Weight,
{
    pub fn new(directed: bool) -> Self {
        Self {
            adj_list: HashMap::new(),
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

    /// Takes ownership of node for simplicity sake (I don't quite get lifetimes lol)
    pub fn add_node(&mut self, node: N) -> Result<(), String> {
        if self.has_node(&node) {
            return Err(format!("Node {:?} already exists", node));
        }

        self.adj_list.insert(node, Vec::new());

        Ok(())
    }

    /// Takes ownership of nodes for the same reason: I don't really get lifetimes
    pub fn add_edge(&mut self, from: N, to: N, weight: W) -> Result<(), String> {
        if self.has_edge(&from, &to) {
            return Err(format!("Edge from {:?} to {:?} already exists", from, to));
        }

        _add_edge(&mut self.adj_list, from, to, weight);

        if !self.directed {
            _add_edge(&mut self.adj_list, to, from, weight);
        }

        Ok(())
    }

    /// Drop a node and any edge pointing to it.
    pub fn remove_node(&mut self, node: &N) {
        self.adj_list.remove(node);

        // look for occurrences of the node in some edge from another node
        for edges in self.adj_list.values_mut() {
            if let Some(index) = edges.iter().position(|edge| &edge.to == node) {
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
            edges.iter().find(|&edge| &edge.to == to).is_some()
        })
    }

    pub fn get_nodes(&self) -> Vec<&N> {
        self.adj_list.keys().collect()
    }

    /// Returns the number of outgoing edges from a given node.
    pub fn degree(&self, node: &N) -> Option<usize> {
        self.adj_list.get(node).map(|edges| edges.len())
    }

    pub fn neighbors(&self, node: &N) -> Option<&Vec<Edge<N, W>>> {
        self.adj_list.get(node)
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
                    .find(|&edge| &edge.to == to)
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

                let neighbors = self
                    .neighbors(node)
                    .unwrap()
                    .iter()
                    .map(|edge| &edge.to);

                stack.extend(neighbors);
                out.push(*node);
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
            // it's a bit hairy but it simply iterates over current node's edges, extract the Node attribute while
            // filtering out nodes we already visited, and finally appending the next nodes to visit to the queue.
            let neighbors: Vec<&N> = self
                .neighbors(node)
                .unwrap()
                .iter()
                .map(|edge| &edge.to)
                .filter(|&node| !visited.contains(node))
                .collect();

            queue.extend(&neighbors);
            visited.extend(&neighbors);
            out.push(*node);
        }

        Some(out)
    }

    pub fn dijkstra(&self, from: &N, to: &N) -> Option<(Vec<N>, W)> {
        // use an option to mean "not reachable" instead of infinity (as it would require another trait for W)
        let mut distances: HashMap<&N, Option<W>> = self.adj_list.keys().map(|node| (node, None)).collect();
        distances.insert(from, Some(W::zero()));

        let mut heap = BinaryHeap::new();
        heap.push((W::zero(), from));

        let mut predecessors: HashMap<&N, &N> = HashMap::new();

        while let Some((ref cost, node)) = heap.pop() {
            // found the destination, let's build the path and return it
            if node == to {
                let mut path = vec![*to];
                let mut current_node = to;
                while let Some(&previous_node) = predecessors.get(current_node) {
                    path.push(*previous_node);
                    current_node = previous_node;
                }
                path.reverse();

                return Some((path, *cost));
            }
            
            // we already found a better path from `from` to node so let's skip it
            if let Some(Some(distance)) = distances.get(node) {
                if cost > distance {
                    continue;
                }
            }

            // update distances and predecessors for each neighbor
            if let Some(edges) = self.adj_list.get(node) {
                for edge in edges {
                    let next_node = &edge.to;
                    let new_cost = *cost + edge.weight;

                    match distances.get(next_node) {
                        Some(Some(current_cost)) if &new_cost >= current_cost => continue,
                        _ => {
                            distances.insert(next_node, Some(new_cost));
                            predecessors.insert(next_node, node);
                            heap.push((Reverse(new_cost).0, next_node));
                        }
                    }
                }
            }
        }

        None
    }

    /// Returns the unique adjacency matrix with nodes ordered (depends on how the type Node implements the trait Ord).
    pub fn adjacency_matrix(&self) -> Option<Vec<Vec<W>>> {
        if self.is_empty() {
            return None;
        }

        let n = self.num_nodes();
        let mut out = vec![vec![W::zero(); n]; n];
        let node_to_index: HashMap<&N, usize> = self
            .adj_list
            .keys()
            .sorted()
            .enumerate()
            .map(|(index, node)| (node, index))
            .collect();

        for (from, edges) in self.adj_list.iter() {
            for Edge { to, weight } in edges.iter() {
                let from = *node_to_index.get(from).unwrap();
                let to = *node_to_index.get(to).unwrap();
                out[from][to] = *weight;
            }
        }

        Some(out)
    }
}
