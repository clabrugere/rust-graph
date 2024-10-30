use std::{collections::HashMap, iter};

#[derive(Debug)]
struct Edge {
    to: u32,
    weight: f32,
}

#[derive(Debug)]
struct Graph {
    adj_list: HashMap<u32, Vec<Edge>>,
    directed: bool,
}

impl Graph {
    fn add_edge(&mut self, from: u32, to: u32, weight: f32) {
        self.adj_list
            .entry(from)
            .or_insert_with(Vec::new)
            .push(Edge { to, weight });
    }

    fn remove_edge(&mut self, from: u32, to: u32) {
        self.adj_list
            .entry(from)
            .and_modify(|edges| edges.retain(|e| e.to != to));
    }

    fn has_node(self, node: u32) -> bool {
        self.adj_list.contains_key(&node)
    }

    fn has_edge(self, from: u32, to: u32) -> bool {
        let node = self.adj_list.get(&from);
        match node {
            Some(neighbors) => {
                neighbors
                    .iter()
                    .filter(|n| n.to == to)
                    .peekable()
                    .peek()
                    .is_some()
            },
            None => false
        }
    }

    fn neighbors(&self, node: u32) -> Option<&Vec<Edge>> {
        self.adj_list.get(&node)
    }
}