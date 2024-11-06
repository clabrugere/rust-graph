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
- Generate random graph, complete graph and perfect binary tree
- Random walks (unbiased/biased) on the graph from a starting node

## Design choices

The core of the library is the `Graph` type, which represents a weighted, directed, or undirected graph. The graph is stored as an adjacency list, mapping each node index to a list of edges along with weights. It is implemented using two hashmaps: a first `IndexMap<usize, N>` mapping a unique node index to a node, and a second `HashMap<usize, Vec<Edge<W>>>` mapping the node index to its list of outgoing edges. The `Edge` struct stores a node index and a weight:

```rust
struct Edge<W> {
    to_idx: usize,
    weight: W,
}
```

Ths `Graph` type is generic over the weights and nodes, allowing it to work with a wide range of types. To ensure efficiency and functionality, certain constraints (traits) are placed on weight `W` and node `N` types:

```rust
struct Graph<N, W> {
    nodes: IndexMap<usize, N>,
    edge_list: HashMap<usize, Vec<Edge<W>>>,
    directed: bool,
    next_node_idx: usize,
}

impl<N, W> Graph<N, W>
where
    N: Debug,
    W: Zero + Copy + Debug,
{
    ...
}
```

While the core implementation is quite permissive on `N` and `W` types, we implement two additional traits `ShortestPath` and `RandomWalk` for more specific algorithms that require other traits, due to some specific data structures we use for those.

### Nodes

- **Debug**: to print the struct, we need at least this trait, but ideally we would need to implement `Display`.

### Weights
- **Default**: to provide default value to struct, mainly for the `RandomWalk` trait that uses `choose_weighted` method from `rand`. It is equivalent to `Zero` trait in our use case, so we could probably get rid of `Zero`, but I prefer to keep as it is more explicit for some methods where we think about identity element for addition.
- **Addition Identity**: for shortest-path computations (using Dijkstra algorithm), `W` need to support addition with an identity element (a zero element). As a consequence, `W` needs to implement the `Zero` and `Add` traits. Note that the trait `Zero` comes from the external crate `num-traits` but it simply means that `w + W::zero() = w` for any `w` of type `W`. Moreover, we require `AddAssign` instead of `Add` simply to allow for `x += y` syntax.
- **Multiplication Identity**: for unbiased random walk, we use the identity element `W::one()` from the trait `One` to get equiprobable sampling of neighboring nodes.
- **Partial Ordering**: because we use a binary heap in the Dijkstra algorithm - allowing us to have an efficient priority queue - `W` also need to implement the `PartialOrd` trait. Well, in fact, rust's binary heap requires elements to implement the more restrictive `Ord` trait, but that would prevent us from using floating point numbers as weights, which would be quite limiting! This is because floats can also have a special `nan` value, whose behavior is not really defined when it comes to ordering. So it requires an arbitrary choice: is a `nan` bigger than every number, or smaller when it's not even a number? Rust didn't make any choice, so we have to. To circumvent the issue, we implement a simple generic `NotNan` wrapper and decide that for a graph, `nan` as a weight for an edge between two nodes doesn't make sense and that we will not be encountering that. We can then implement `Ord` by simply unwrapping the value from a partial comparison: if we get a `nan` while comparing two elements of type `W`, the program will panic and terminate.
- **Sampling**: `SampleUniform` trait is required by `rand` crate to sample from collections.
- **Copy**: in this implementation, we assume that edge weights remain simple types whose copy is inexpensive and done implicitly when required.
- **Debug**: for the same reason as above.

### Adjacency list

Instead of using a regular hash map, we use a `IndexMap` from the `indexmap` crate that provides a nice property: iteration ordering independent of the hashed key but rather based on insertion order. This is especially useful to provide a unique adjacency matrix based on node insertion order. Indeed, using the hash map from the standard library would require to sort keys first to generate a unique matrix per graph. Otherwise, we would only get isomorphic matrices at each invocation of the `adjacency_matrix` method.

In addition, for adjacency lists to be memory efficient, it's better if we only store references to other nodes in the edge list: otherwise we would end up with a much worse space complexity. The problem is that you can't directly use references to keys as values or you end up with all sorts of ownership issues all around. For example, rust's hash maps own the keys - so the nodes here - but we also need to store references to those as values from other nodes, and so ownership becomes multiple. 

This is not really an issue if we restrict ourself to primitive types, but it could become a problem for nodes that store large amounts of data. One solution I initially found to circumvent the issue was to use reference-counted smart pointers `Rc` and weak references `Weak` from the standard library. This way we only store nodes one time, and use multiple references for the edges. But this solution was introducing potential memory leaks and made the implementation mode complex. Instead we use unique indices for each node and use that as pointers to other nodes in the edge list.

### Side notes

Most of the design choices were done along the way. I first started with a rather basic implementation restricted to integers for nodes and edges, mostly to get used to rust syntax and get the basics. At this point, I thought it would be nice to be able to use more complex types for nodes, and allow edges to be floating point numbers. 

Then I wanted to avoid to have to clone nodes all over the place, because, well, it's not elegant, so I introduced counted references `Rc` over nodes in the `Edge` struct. But because some graph could have nodes referencing each others, I created reference cycles. This is well documented as a bad pattern: counted references are only destroyed when the counter of references drop to zero, which never happen in the case of reference cycles, creating memory leaks.

So I refactored the whole `Graph` implementation to avoid storing references to nodes in edges in a single container, but instead split that in two: one `IndexMap` mapping unique `usize` ids to nodes, and a `HashMap` mapping the same ids to an edge list.

Another limitation of storing the graph in a single hashmap of nodes to edge list was in the hashing of the nodes directly: it required to implement `Eq` and `Hash` traits for the node type `N` but also that required nodes to be immutable (otherwise their hash would change and the hashmap would not be stable). In addition, hashing nodes directly could cause collisions for large graphs. 

On the code structure, I initially put everything under the same `Graph` implementation. But as I started to add more specific algorithms that required additional traits, I started to split it into traits. This allow to have a general no-too-restrictive base implementation, augmented with specific stuff that might not be useful for every use case and that require more constraints on our generics only if we need to use them.

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
let node0 = 0;
let node1 = 1;

let node0_idx = g.add_node(node0);
let node1_idx = g.add_node(node1);
```

To create an empty graph with a user defined node type `Node` (you can specify if the graph is directed or not) and add some edges:

```rust
use std::cmp::{Eq, Ord, PartialEq};
use graph::Graph;

#[derive(Debug)]
struct Node {
    id: String,
    attribute1: i32,
    attribute2: f64,
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

let (node0_idx, node1_idx) = g.add_nodes_then_edge(node0, node1, 1);
```

The `add_edge` and `add_nodes_then_edge` methods return the unique indices to the created nodes, allowing you to use them in subsequent operations.

## Improvements

### Core
- [x] Use `IndexMap` instead of `HashMap` for the adjacency list in order to have predictable iteration order based on insertion and thus avoid sorting
- [ ] Parallel/multi-threaded operations (`bfs`, `dfs`)
- [ ] Implementation of `Clone` trait on the whole graph

### Features
- [ ] Support shortest-path for negative edge weights (Bellman-Ford algorithm)
- [x] Insert nodes
- [ ] Create graph from file
- [ ] Serialize/deserialize graphs
- [ ] Heuristic path finding (A*)
- [ ] Bidirectional search for shortest-path finding
- [x] Random walks on graphs (unbiased & biased using edge weights)
- [x] Edges weights aggregation functions
- [ ] PageRank, Betweenness Centrality, Closeness Centrality
- [ ] Get sub-graph based on predicates over the edge weights
- [x] Graph generators (random, complete, etc.)