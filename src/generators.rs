#[allow(dead_code)]
use crate::Graph;
use rand::{
    distributions::{Distribution, Uniform},
    seq::{IteratorRandom, SliceRandom},
};
use std::collections::VecDeque;
use std::rc::Rc;

/// Generate a random graph with `num_nodes` nodes and `num_edges` edges. It is built by iterating until reaching the
/// desired number of edges, at each step sampling a random source node and a random destination node that is not the
/// source node and not already a neighbor of it. Edge weights are sampled from U(0,1)
pub fn random_graph(num_nodes: usize, num_edges: usize, directed: bool) -> Graph<u32, f64> {
    let num_edges = if directed { num_edges } else { 2 * num_edges };
    let mut g: Graph<u32, f64> = Graph::new(directed);
    let nodes: Vec<Rc<u32>> = (0..num_nodes as u32)
        .map(|node| g.add_node(node).unwrap())
        .collect();

    let mut rng = rand::thread_rng();
    let weight_sampler = Uniform::new(0.0, 1.0);

    while g.num_edges() < num_edges {
        let weight = weight_sampler.sample(&mut rng);
        let from = nodes.choose(&mut rng).unwrap();

        // sample another node that is not `from` and that is not already a neighbor of it
        if let Some(to) = nodes
            .iter()
            .filter(|&node| {
                node != from
                    && !g
                        .neighbors(from.as_ref())
                        .map_or(false, |neighbors| neighbors.contains(&node.as_ref()))
            })
            .choose(&mut rng)
        {
            g.add_edge_from_refs(from, to, weight).unwrap();
        }
    }

    g
}

/// Generate a complete graph, where each node is a neighbor of every other node. Edge weights are sampled from U(0,1)
pub fn complete_graph(num_nodes: usize, directed: bool) -> Graph<u32, f64> {
    let mut g: Graph<u32, f64> = Graph::new(directed);
    let mut rng = rand::thread_rng();
    let weight_sampler = Uniform::new(0.0, 1.0);

    for from in 0..num_nodes as u32 {
        for to in from + 1..num_nodes as u32 {
            let weight = weight_sampler.sample(&mut rng);
            g.add_edge(from, to, weight).unwrap();
        }
    }

    g
}

pub fn perfect_binary_tree(depth: usize) -> Graph<u32, f64> {
    let mut g: Graph<u32, f64> = Graph::new(true);

    let root = g.add_node(0).unwrap();
    let mut queue = VecDeque::from([root]);
    let mut level = 0;
    let mut next_node = 1_u32;

    while queue.len() > 0 {
        level += 1;

        if level == depth + 1 {
            break;
        }

        for _ in 0..queue.len() {
            let parent = queue.pop_front().unwrap();
            let left_child = g.add_node(next_node).unwrap();
            let right_child = g.add_node(next_node + 1).unwrap();

            g.add_edge_from_refs(&parent, &left_child, 1.0).unwrap();
            g.add_edge_from_refs(&parent, &right_child, 1.0).unwrap();

            queue.push_back(left_child);
            queue.push_back(right_child);

            next_node += 2;
        }
    }

    g
}
