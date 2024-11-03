# rust-graph

## About

This is a pet project I'm developing to learn Rust. It’s a small library for creating weighted, directed, or undirected graphs, with support for common operations such as:
- Creating a new graph
- Inserting/deleting nodes
- Inserting/deleting edges
- Checking if a node is in the graph
- Checking if an edge connects two nodes
- Getting the degree of a node
- Getting the direct neighbors of a node
- Testing if a path exists between two nodes
- Traversing the graph from a source node in BFS or DFS order
- Finding the shortest path between two nodes
- Generating a unique adjacency matrix from sorted nodes
- Apply a closure `F(Vec<W>) -> W` to a given node outgoing edges for weight aggregation

## Design choices

The core of the library is the `Graph` type, which represents a weighted, directed, or undirected graph. The graph is stored as an adjacency list, mapping each node to a list of edges that store references to neighboring nodes along with weights. This adjacency list is implemented as a hash map, where nodes are the keys and each key's value is a variable-length array of `Edge` structs:

```rust
type AdjacencyList<N, W> = IndexMap<Rc<N>, Vec<Edge<N, W>>>;

struct Edge<N, W> {
    to: Rc<N>,
    weight: W,
}
```

This structure is generic over both nodes and edge weights, allowing it to work with a wide range of types. To ensure efficiency and functionality, certain constraints (traits) are placed on node `N` and weight `W` types:

```rust
struct Graph<N, W>
where
    N: Eq + Hash,
{
    adj_list: AdjacencyList<N, W>,
    directed: bool,
}

impl<N, W> Graph<N, W>
where
    N: Hash + Eq + Ord + Clone + Debug,
    W: Zero + Add + PartialOrd + Copy,
{
    ...
}
```

### Nodes

- **Hashing and Equality**: while using a hash map as adjacency list provides $O(1)$ neighbor lookup, it does require that the node type `N` implements the traits `Hash` and `Eq`. Nevertheless, I think it's not too restrictive.
- **Ordering**: shortest path algorithm is implemented using a binary heap from the standard library whose elements are tuples of `(W, &N)`. Ordering is done on both elements of the tuple if ordering on the first element results in a tie. As a consequence, node type `N` needs to implement the `Ord` trait.

### Weights

- **Addition Identity**: for shortest-path computations (using Dijkstra algorithm), `W` need to support addition with an identity element (a zero element). As a consequence, `W` needs to implement the `Zero` and `Add` traits. Note that the trait `Zero` comes from the external crate `num-traits` but it simply means that `w + W::zero() = w` for any `w` of type `W`.
- **Partial Ordering**: because we use a binary heap in the Dijkstra algorithm - allowing us to have an efficient priority queue - `W` also need to implement the `PartialOrd` trait. Well, in fact, rust's binary heap requires elements to implement the more restrictive `Ord` trait, but that would prevent us from using floating point numbers as weights, which would be quite limiting! This is because floats can also have a special `nan` value, whose behavior is not really defined when it comes to ordering. So it requires an arbitrary choice: is a `nan` bigger than every number, or smaller when it's not even a number? Rust didn't make any choice, so we have to. To circumvent the issue, we implement a simple generic `NotNan` wrapper and decide that for a graph, `nan` as a weight for an edge between two nodes doesn't make sense and that we will not be encountering that. We can then implement `Ord` by simply unwrapping the value from a partial comparison: if we get a `nan` while comparing two elements of type `W`, the program will panic and terminate.
- **Copy**: in this implementation, we assume that edge weights remain simple types whose copy is inexpensive and done implicitly when required.

### Adjacency list

Instead of using a regular hash map, we use a special implementation `IndexMap` from the `indexmap` crate that provide a nice property: iteration ordering independent of the hashed key but rather based on insertion order. This is especially useful to provide a unique adjacency matrix based on node insertion order. Indeed, using the hash map from the standard library would require to sort keys first to generate a unique matrix per graph. Otherwise, we would only get isomorphic matrices at each invocation of the `adjacency_matrix` method.

In addition, for adjacency lists to be memory efficient, it's better if we only store references to other nodes in the edge list: otherwise we would end up with a much worse space complexity. The problem is that you can't directly use references to keys as values or you end up with all sorts of ownership issues all around. For example, rust's hash maps own the keys - so the nodes here - but we also need to store references to those as values from other nodes, and so ownership becomes multiple. 

This is not really an issue if we restrict ourself to primitive types, but it could become a problem for nodes that store large amounts of data. One solution I found to circumvent the issue is to use reference-counted smart pointers `Rc` from the standard library. This way we only store nodes one time, and use multiple references for the edges, saving memory!

As a side note, most of the design choices were done along the way. I first started with a rather basic implementation restricted to integers for nodes and edges, mostly to get used to rust syntax and get the basics. At this point, I thought it would be nice to be able to use more complex types for nodes, and allow edges to be floating point numbers. Then I wanted to avoid to have to clone nodes all over the place, because, well, it's not elegant.

## Prerequisite

You need `rustc` installed on your machine to compile rust code.

## Getting Started

Clone the repo

```
git clone git@github.com:clabrugere/rust-graph.git
```

Check if everything is alright

```
cargo check
```

## Usage

To create a new empty graph, defaulting to an undirected `Graph<u32, i32>` with unsigned integer nodes and signed integer edges, and add some edges:

```rust
use graph::Graph;

let mut g = Graph::default();
let (node0, node1, node2) = (0, 1, 2);

g.add_edge(node0, node1, 1).unwrap();
g.add_edge(node0, node2, 2).unwrap();
g.add_edge(node1, node2, 1).unwrap();
```

To create an empty graph with a user defined node type `Node` (you can specify if the graph is directed or not) and add some edges:

```rust
use std::cmp::{Eq, Ord, PartialEq};
use std::collections::HashSet;
use std::hash::Hash;
use graph::Graph;

#[derive(Debug, Clone)]
struct Node {
    id: String,
    attribute1: i32,
    attribute2: f64,
}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Node {
    pub fn new(id: String, attribute1: i32, attribute2: f64) -> Node {
        Node {
            id,
            attribute1,
            attribute2,
        }
    }
}

let mut g: Graph<Node, i32> = Graph::new(false);
let node0 = Node::new(String::from("node0"), 42, 1.5);
let node1 = Node::new(String::from("node1"), 64, 0.0);

let (node0, node1) = g.add_edge(node0, node1, 1).unwrap();
```

The `add_edge` method returns reference-counted smart pointers (`Rc`) to the created nodes, allowing you to use them in subsequent operations.

## Improvements

### Core
- [x] Methods should always return reference to nodes and not copies (`bfs`, `dfs`, `dijkstra`)
- [x] Use `IndexMap` instead of `HashMap` for the adjacency list in order to have predictable iteration order based on insertion and thus avoid sorting
- [ ] Parallel/multi-threaded operations (`bfs`, `dfs`)
- [ ] Implementation of `Clone` trait on the whole graph

### Features
- [ ] Support shortest-path for negative edge weights (Bellman-Ford algorithm)
- [ ] Insert nodes/edges from iterators
- [ ] Create graph from file
- [ ] Serialize/deserialize graphs
- [ ] Heuristic path finding (A*)
- [ ] Bidirectional search for shortest-path finding
- [ ] Random walks on graphs (unbiased & biased using edge weights)
- [x] Edges weights aggregation functions
- [ ] PageRank, Betweenness Centrality, Closeness Centrality
- [ ] Get sub-graph based on predicates over the edge weights