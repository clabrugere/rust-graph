#[allow(dead_code)]
use crate::graph::Graph;
use rand::{
    distributions::{Distribution, Uniform},
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use std::collections::VecDeque;

/// Generate a random graph with `num_nodes` nodes and `num_edges` edges. It is built by iterating until reaching the
/// desired number of edges, at each step sampling a random source node and a random destination node that is not the
/// source node and not already a neighbor of it. Edge weights are sampled from U(0,1)
pub fn random_graph<R: Rng>(
    num_nodes: usize,
    num_edges: usize,
    directed: bool,
    rng: &mut R,
) -> Graph<u32, f64> {
    let num_edges = if directed { num_edges } else { 2 * num_edges };
    let mut g: Graph<u32, f64> = Graph::new(directed);
    let node_inds: Vec<usize> = (0..num_nodes as u32).map(|node| g.add_node(node)).collect();

    let weight_sampler = Uniform::new(0.0, 1.0);

    while g.num_edges() < num_edges {
        let weight = weight_sampler.sample(rng);
        let from_idx = node_inds.choose(rng).unwrap();

        // sample another node that is not `from` and that is not already a neighbor of it
        if let Some(to_idx) = node_inds
            .iter()
            .filter(|&node| {
                node != from_idx
                    && !g
                        .neighbors_inds(*from_idx)
                        .map_or(false, |neighbors| neighbors.contains(node))
            })
            .choose(rng)
        {
            g.add_edge(*from_idx, *to_idx, weight).unwrap();
        }
    }

    g
}

/// Generate a complete graph, where each node is a neighbor of every other node. Edge weights are sampled from U(0,1)
pub fn complete_graph<R: Rng>(num_nodes: usize, rng: &mut R) -> Graph<u32, f64> {
    let mut g: Graph<u32, f64> = Graph::new(false);
    let weight_sampler = Uniform::new(0.0, 1.0);

    let node_inds = g.add_nodes_from_iterator(0..num_nodes as u32);

    for from_idx in node_inds {
        for to_idx in from_idx + 1..num_nodes {
            let weight = weight_sampler.sample(rng);
            g.add_edge(from_idx, to_idx, weight).unwrap();
        }
    }

    g
}

pub fn perfect_binary_tree(depth: usize) -> Graph<u32, i32> {
    let mut g: Graph<u32, i32> = Graph::new(true);

    let root = g.add_node(0);
    let mut queue = VecDeque::from([root]);
    let mut level = 0;
    let mut next_node = 1_u32;

    while !queue.is_empty() {
        level += 1;

        if level == depth + 1 {
            break;
        }

        for _ in 0..queue.len() {
            let parent = queue.pop_front().unwrap();
            let left_child = g.add_node(next_node);
            let right_child = g.add_node(next_node + 1);

            g.add_edge(parent, left_child, 1).unwrap();
            g.add_edge(parent, right_child, 1).unwrap();

            queue.push_back(left_child);
            queue.push_back(right_child);

            next_node += 2;
        }
    }

    g
}
