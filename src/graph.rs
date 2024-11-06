use indexmap::IndexMap;
use num_traits::{One, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::seq::SliceRandom;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::ops::AddAssign;

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

/// Represents a weighted edge from a parent node to `to` Node
#[derive(Debug)]
struct Edge<W> {
    to_idx: usize,
    weight: W,
}

fn _add_edge<W>(
    edge_list: &mut HashMap<usize, Vec<Edge<W>>>,
    from_idx: usize,
    to_idx: usize,
    weight: W,
) {
    edge_list
        .get_mut(&from_idx)
        .map(|edges| edges.push(Edge { to_idx, weight }));
}

/// Remove an edge from the adjacency list. If the edge doesn't exist, nothing is changed.
fn _remove_edge<W>(edge_list: &mut HashMap<usize, Vec<Edge<W>>>, from_idx: usize, to_idx: usize) {
    // We are guaranteed to have only one directed edge between two nodes.
    // Otherwise we would need to use edges.retain(|edge| edge.to != to)
    // We first get a mutable reference of the edges for `from` node and map over it if the node exists.
    // We then look for the index of the edge we're looking fore and remove it if we find it.
    edge_list.get_mut(&from_idx).map(|edges| {
        if let Some(index) = edges.iter().position(|edge| edge.to_idx == to_idx) {
            edges.swap_remove(index);
        }
    });
}

/// Represents a graph stored using an adjacency list. It is implemented using a hashmap where the keys are unique indices
/// and values are the nodes. Another hashmap is used to map node indices to their list of edges.
/// If the graph is undirected, all edges are duplicated.
/// It is generic over the nodes and the edge weights.
#[derive(Debug)]
pub struct Graph<N, W> {
    nodes: IndexMap<usize, N>,
    edge_list: HashMap<usize, Vec<Edge<W>>>,
    directed: bool,
    next_node_idx: usize,
}

impl Default for Graph<u32, i32> {
    /// Undirected graph with nodes encoded as 32 bits unsigned integers and weights as 32 bits integers.
    fn default() -> Self {
        Self {
            nodes: IndexMap::new(),
            edge_list: HashMap::new(),
            directed: false,
            next_node_idx: 0,
        }
    }
}

impl<N, W> Graph<N, W>
where
    N: Debug,
    W: Zero + Copy + Debug,
{
    pub fn new(directed: bool) -> Self {
        Self {
            nodes: IndexMap::new(),
            edge_list: HashMap::new(),
            directed,
            next_node_idx: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_nodes() == 0
    }

    pub fn is_directed(&self) -> bool {
        self.directed
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edge_list.values().map(Vec::len).sum()
    }

    /// Add a new node to the graph. Returns the unique id of the node for further usage.
    pub fn add_node(&mut self, node: N) -> usize {
        let node_idx = self.next_node_idx;
        self.nodes.insert(node_idx, node);
        self.edge_list.insert(node_idx, Vec::new());
        self.next_node_idx += 1;

        node_idx
    }

    /// Add nodes from an iterator of nodes and return a vec of their respective indices.
    pub fn add_nodes_from_iterator<I>(&mut self, nodes: I) -> Vec<usize>
    where
        I: Iterator<Item = N>,
    {
        nodes.map(|node| self.add_node(node)).collect()
    }

    /// Add the two nodes before creating an edge between them. Return their unique indices.
    pub fn add_nodes_then_edge(&mut self, from: N, to: N, weight: W) -> (usize, usize) {
        let (from_idx, to_idx) = (self.add_node(from), self.add_node(to));
        self.add_edge(from_idx, to_idx, weight).unwrap();

        (from_idx, to_idx)
    }

    /// Add a new  weighted edge between `from` and `to` nodes. Return an error if the edge already exists.
    pub fn add_edge(&mut self, from_idx: usize, to_idx: usize, weight: W) -> Result<(), String> {
        if self.has_edge(from_idx, to_idx) {
            return Err(format!(
                "Edge from {:?} to {:?} already exists",
                from_idx, to_idx
            ));
        }

        _add_edge(&mut self.edge_list, from_idx, to_idx, weight);

        if !self.directed {
            _add_edge(&mut self.edge_list, to_idx, from_idx, weight);
        }

        Ok(())
    }

    /// Apply a closure over edges between `node` and all its neighbors. This can be useful for message passing logic
    pub fn aggregate_edges<F>(&self, node_idx: usize, f: F) -> Option<W>
    where
        F: Fn(Vec<W>) -> W,
    {
        self.edge_list
            .get(&node_idx)
            .map(|edges| f(edges.iter().map(|edge| edge.weight).collect()))
    }

    /// Drop a node and any edge pointing to it.
    pub fn remove_node(&mut self, node_idx: usize) {
        // look for occurrences of the node in some edge from another node and drop the edge if found.
        for edges in self.edge_list.values_mut() {
            if let Some(index) = edges.iter().position(|edge| edge.to_idx == node_idx) {
                edges.swap_remove(index);
            }
        }
        self.nodes.swap_remove(&node_idx);
    }

    /// Drop an edge if it exists. This can result in nodes disconnected from the rest of the graph.
    pub fn remove_edge(&mut self, from_idx: usize, to_idx: usize) {
        _remove_edge(&mut self.edge_list, from_idx, to_idx);

        if !self.directed {
            _remove_edge(&mut self.edge_list, to_idx, from_idx);
        }
    }

    /// Check is a node exists.
    pub fn has_node(&self, node_idx: usize) -> bool {
        self.get_node(node_idx).is_some()
    }

    /// Check if two nodes are neighbors
    pub fn has_edge(&self, from_idx: usize, to_idx: usize) -> bool {
        // we first look for the node, if it doesn't exist we return false, otherwise we iterate over the edges
        // and try to find the one going to `to`. If we find it, we return true, otherwise false.
        self.edge_list.get(&from_idx).map_or(false, |edges| {
            edges.iter().find(|&edge| edge.to_idx == to_idx).is_some()
        })
    }

    /// Return a reference of a node if it exists, None otherwise.
    pub fn get_node(&self, node_idx: usize) -> Option<&N> {
        self.nodes.get(&node_idx)
    }

    /// Return references to all the nodes in the graph.
    pub fn get_nodes(&self) -> Vec<&N> {
        self.nodes.values().collect()
    }

    /// Return the number of outgoing edges from a given node. Return None if the node is not in the graph.
    pub fn degree(&self, node_idx: usize) -> Option<usize> {
        self.edge_list.get(&node_idx).map(|edges| edges.len())
    }

    /// Return reference to the list of outgoing edges from a node. Return None if the node is not in the graph.
    fn outgoing_edges(&self, node_idx: &usize) -> Option<&Vec<Edge<W>>> {
        self.edge_list.get(node_idx)
    }

    /// Return indices of all neighbors of `nodes`. Return None if the node is not in the graph.
    pub fn neighbors_inds(&self, node_idx: usize) -> Option<Vec<usize>> {
        self.outgoing_edges(&node_idx)
            .map(|edges| edges.iter().map(|edge| edge.to_idx).collect())
    }

    /// Return references of all neighbors of `nodes`. Return None if the node is not in the graph.
    pub fn neighbors(&self, node_idx: usize) -> Option<Vec<&N>> {
        self.neighbors_inds(node_idx).map(|neighbors_inds| {
            neighbors_inds
                .iter()
                .map(|neighor_idx| self.nodes.get(neighor_idx).unwrap())
                .collect()
        })
    }

    /// Return the first seen edge between to nodes (it should return a collection actually)
    pub fn get_edge_weight(&self, from_idx: usize, to_idx: usize) -> Option<W> {
        // we first get a reference to the values if the key exists (hence its an option)
        // then we iterate over the edges corresponding to the node if we got Some(edges) until we find the edge
        // if we find it, we extract the weight on Some(edge) and flatten the nested options
        // otherwise returns None
        self.outgoing_edges(&from_idx)
            .map(|edges| {
                edges
                    .iter()
                    .find(|&edge| edge.to_idx == to_idx)
                    .map(|edge| edge.weight)
            })
            .flatten()
    }

    /// Returns the unique adjacency matrix with nodes ordered by insertion.
    pub fn adjacency_matrix(&self) -> Option<Vec<Vec<W>>> {
        if self.is_empty() {
            return None;
        }

        let n = self.num_nodes();
        let mut out = vec![vec![W::zero(); n]; n];

        for (row_idx, (from_idx, _)) in self.nodes.iter().enumerate() {
            for Edge { to_idx, weight } in self.edge_list.get(&from_idx).unwrap() {
                let (col_idx, ..) = self.nodes.get_full(to_idx).unwrap();
                out[row_idx][col_idx] = *weight;
            }
        }

        Some(out)
    }

    /// Return references to visited nodes in BFS (iterative) order.
    /// If the node `from` is not in the graph, returns an empty option, otherwise returns a Vec<&N>.
    pub fn traverse_dfs(&self, from_idx: usize) -> Option<Vec<&N>> {
        if !self.has_node(from_idx) {
            return None;
        }

        let mut out: Vec<&N> = Vec::new();
        let mut stack = Vec::from([from_idx]);
        let mut visited: HashSet<usize> = HashSet::new();

        while let Some(node_idx) = stack.pop() {
            if !visited.contains(&node_idx) {
                visited.insert(node_idx);
                stack.extend(self.neighbors_inds(node_idx).unwrap());

                out.push(self.nodes.get(&node_idx).unwrap());
            }
        }

        Some(out)
    }

    /// Return references to visited nodes in BFS order.
    /// If the node `from` is not in the graph, returns an empty option, otherwise returns a Vec<N>.
    pub fn traverse_bfs(&self, from_idx: usize) -> Option<Vec<&N>> {
        if !self.has_node(from_idx) {
            return None;
        }

        let mut out: Vec<&N> = Vec::new();
        let mut queue = VecDeque::from([from_idx]);
        let mut visited = HashSet::from([from_idx]);

        while let Some(node_idx) = queue.pop_front() {
            for neighbor_idx in self.neighbors_inds(node_idx).unwrap() {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    queue.push_back(neighbor_idx);
                }
            }
            out.push(self.nodes.get(&node_idx).unwrap());
        }

        Some(out)
    }
}

pub trait ShortestPath<N, W> {
    fn has_path(&self, from_idx: usize, to_idx: usize) -> bool;
    fn dijkstra(&self, from_idx: usize, to_idx: usize) -> Option<(Vec<&N>, W)>;
}

impl<N, W> ShortestPath<N, W> for Graph<N, W>
where
    W: Zero + for<'a> AddAssign<&'a W> + PartialOrd + Copy,
{
    /// Check if a path exists between two nodes
    fn has_path(&self, from_idx: usize, to_idx: usize) -> bool {
        self.dijkstra(from_idx, to_idx).is_some()
    }

    /// Return the shorted path between two nodes using Dijkstra algorithm. Edge weights must be positive.
    fn dijkstra(&self, from_idx: usize, to_idx: usize) -> Option<(Vec<&N>, W)> {
        // use an option to mean "not reachable" instead of infinity (as it would require another trait for W)
        let mut distances: IndexMap<&usize, Option<W>> =
            self.nodes.keys().map(|node| (node, None)).collect();
        distances.insert(&from_idx, Some(W::zero()));

        // because the std binary heap only works on elements that implement Ord, we have to wrap our type W
        // in order to support float weights (that could potentially be nan and so only implement PartialOrd).
        let mut heap = BinaryHeap::new();
        heap.push((NotNan { value: W::zero() }, &from_idx));

        let mut predecessors: HashMap<&usize, &usize> = HashMap::new();

        while let Some((ref cost, node_idx)) = heap.pop() {
            // found the destination, let's build the path and return it
            if node_idx == &to_idx {
                let mut path = vec![self.nodes.get(&to_idx).unwrap()];
                let mut current_node_idx = &to_idx;
                while let Some(&previous_node_idx) = predecessors.get(current_node_idx) {
                    path.push(self.nodes.get(previous_node_idx).unwrap());
                    current_node_idx = previous_node_idx;
                }
                path.reverse();

                return Some((path, cost.value));
            }

            // we already found a better path from `from` to node so let's skip it
            if let Some(Some(distance)) = distances.get(&node_idx) {
                if &cost.value > distance {
                    continue;
                }
            }

            // update distances and predecessors for each neighbor
            if let Some(edges) = self.edge_list.get(&node_idx) {
                for edge in edges {
                    let next_node_idx = &edge.to_idx;
                    let new_cost = cost.value + edge.weight;

                    match distances.get(next_node_idx) {
                        Some(Some(current_cost)) if &new_cost >= current_cost => continue,
                        _ => {
                            distances.insert(next_node_idx, Some(new_cost));
                            predecessors.insert(next_node_idx, &node_idx);
                            heap.push((
                                NotNan {
                                    value: Reverse(new_cost).0,
                                },
                                next_node_idx,
                            ));
                        }
                    }
                }
            }
        }

        None
    }
}

pub trait RandomWalk<N, W> {
    fn random_walk(
        &self,
        starting_node_idx: usize,
        num_steps: usize,
        biased: bool,
    ) -> Option<Vec<&N>>;
}

impl<N, W> RandomWalk<N, W> for Graph<N, W>
where
    N: Debug,
    W: Default + Zero + One + for<'a> AddAssign<&'a W> + PartialOrd + SampleUniform + Copy + Debug,
{
    /// Return references of nodes by randomly traversing the graph for `num_steps`, from a starting node. The walk is
    /// either unbiased (each neighbor has the same probabilities of being sampled) or biased (transition probabilities
    /// are proportional to the edges weights).
    /// Note that if a node has no neighbors, the walk will be stuck there until we reach the desired number of steps.
    fn random_walk(
        &self,
        starting_node_idx: usize,
        num_steps: usize,
        biased: bool,
    ) -> Option<Vec<&N>> {
        if !self.has_node(starting_node_idx) {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut out: Vec<&N> = Vec::new();
        let mut current_node_idx = starting_node_idx;

        for _ in 0..num_steps {
            out.push(self.get_node(current_node_idx).unwrap());

            if let Ok(edge) = self
                .outgoing_edges(&current_node_idx)
                .unwrap()
                .choose_weighted(&mut rng, |edge| if biased { edge.weight } else { W::one() })
            {
                current_node_idx = edge.to_idx;
            }
        }
        Some(out)
    }
}
