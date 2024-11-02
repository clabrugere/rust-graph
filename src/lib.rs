pub mod graph;

#[cfg(test)]
mod tests {
    use std::cmp::{Eq, Ord, PartialEq};
    use std::collections::HashSet;
    use std::hash::Hash;

    use super::*;
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

    fn generate_graph(directed: bool) -> Graph<i32, i32> {
        let mut g = Graph::new(directed);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_edge(1, 3, 1).unwrap();
        g.add_edge(0, 4, 1).unwrap();
        g.add_edge(1, 5, 1).unwrap();
        g.add_edge(2, 6, 1).unwrap();
        g.add_edge(4, 5, 1).unwrap();

        g
    }

    #[test]
    fn create_graph() {
        let g = Graph::default();

        assert!(!g.is_directed());
        assert!(g.is_empty());
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn create_graph_of_structs() {
        let mut g: Graph<Node, i32> = Graph::new(false);
        let node0 = Node::new(String::from("node0"), 42, 1.5);
        let node1 = Node::new(String::from("node1"), 64, 0.0);

        let (node0, node1) = g.add_edge(node0, node1, 1).unwrap();

        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.num_edges(), 2);
        assert!(g.has_path(node0.as_ref(), node1.as_ref()));
    }

    #[test]
    fn add_existing_struct_node() {
        let mut g: Graph<Node, i32> = Graph::new(false);
        let node0 = Node::new(String::from("node0"), 42, 1.5);
        let node0_clone = node0.clone();
        g.add_node(node0).unwrap();

        assert!(g.add_node(node0_clone).is_err());
    }

    #[test]
    fn add_existing_struct_edge() {
        let mut g: Graph<Node, i32> = Graph::new(false);
        let node0 = Node::new(String::from("node0"), 42, 1.5);
        let node1 = Node::new(String::from("node1"), 64, 0.0);
        let node0_clone = node0.clone();
        let node1_clone = node1.clone();
        g.add_edge(node0, node1, 1).unwrap();

        assert!(g.add_edge(node0_clone, node1_clone, 1).is_err());
    }

    #[test]
    fn add_new_nodes() {
        let mut g = Graph::default();
        let node0 = 0;
        let node1 = 1;

        g.add_node(node0).unwrap();
        g.add_node(node1).unwrap();

        assert_eq!(g.num_nodes(), 2);
        assert!(g.has_node(&node0));
        assert!(g.has_node(&node1));
    }

    #[test]
    fn add_existing_nodes() {
        let mut g = Graph::default();
        let node = 0;

        g.add_node(node).unwrap();

        let result = g.add_node(node);
        assert!(result.is_err());
    }

    #[test]
    fn remove_nodes() {
        let mut g = Graph::default();
        let node0 = 0;
        let node1 = 1;
        let (from, to, weight) = (node0, node1, 1);

        g.add_node(node0).unwrap();
        assert!(g.has_node(&node0));

        g.add_edge(from, to, weight).unwrap();
        assert!(g.has_node(&node1));
        assert!(g.has_edge(&from, &to));

        g.remove_node(&node1);
        assert!(!g.has_node(&node1));
        assert!(g.has_node(&node0));
        assert!(!g.has_edge(&from, &to))
    }

    #[test]
    fn add_new_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (node0, node1, node2) = (0, 1, 2);

        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 2).unwrap();
        g.add_edge(node1, node2, 1).unwrap();

        assert_eq!(g.num_edges(), 3, "number of edges");
        assert_eq!(g.num_nodes(), 3, "number of nodes");
        assert!(g.has_edge(&node0, &node1));
        assert!(g.has_edge(&node0, &node2));
        assert!(g.has_edge(&node1, &node2));
    }

    #[test]
    fn add_existing_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (from, to, weight) = (0, 1, 1);
        g.add_edge(from, to, weight).unwrap();

        let result = g.add_edge(from, to, weight);
        assert!(result.is_err());
    }

    #[test]
    fn remove_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (from, to, weight) = (0, 1, 1);

        g.add_edge(from, to, weight).unwrap();
        assert!(g.has_edge(&from, &to));

        g.remove_edge(&from, &to);
        assert!(!g.has_edge(&from, &to));
    }

    #[test]
    fn add_new_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (node0, node1, node2) = (0, 1, 2);

        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 2).unwrap();
        g.add_edge(node1, node2, 1).unwrap();

        assert_eq!(g.num_edges(), 6, "number of edges");
        assert_eq!(g.num_nodes(), 3, "number of nodes");
        assert!(g.has_edge(&node0, &node1));
        assert!(g.has_edge(&node1, &node0));
        assert!(g.has_edge(&node0, &node2));
        assert!(g.has_edge(&node2, &node0));
        assert!(g.has_edge(&node1, &node2));
        assert!(g.has_edge(&node2, &node1));
    }

    #[test]
    fn add_existing_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (from, to, weight) = (0, 1, 1);
        g.add_edge(from, to, weight).unwrap();

        let res = g.add_edge(from, to, weight);
        assert!(res.is_err());

        let res = g.add_edge(to, from, weight);
        assert!(res.is_err());
    }

    #[test]
    fn remove_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (from, to, weight) = (0, 1, 1);

        g.add_edge(from, to, weight).unwrap();
        assert!(g.has_edge(&from, &to));
        assert!(g.has_edge(&to, &from));

        g.remove_edge(&from, &to);
        assert!(!g.has_edge(&from, &to));
        assert!(!g.has_edge(&to, &from));
    }

    #[test]
    fn neighbors() {
        let mut g = Graph::default();
        let (node0, node1, node2, node3, node4) = (0, 1, 2, 3, 4);
        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 1).unwrap();
        g.add_node(node3).unwrap();

        assert_eq!(g.neighbors(&node0).unwrap().len(), 2);
        assert_eq!(g.neighbors(&node3).unwrap().len(), 0);
        assert!(g.neighbors(&node4).is_none());
    }

    #[test]
    fn get_nodes() {
        let mut g = Graph::default();
        let (node0, node1) = (0, 1);
        g.add_node(node0).unwrap();
        g.add_node(node1).unwrap();

        let result = HashSet::from_iter(g.get_nodes());
        let expected = HashSet::from([&node0, &node1]);
        assert_eq!(result, expected);
    }

    #[test]
    fn degree() {
        let mut g = Graph::default();
        let (node0, node1, node2, node3) = (0, 1, 2, 3);
        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 1).unwrap();
        g.add_node(node3).unwrap();

        assert_eq!(g.degree(&node0).unwrap(), 2);
        assert_eq!(g.degree(&node1).unwrap(), 1);
        assert_eq!(g.degree(&node3).unwrap(), 0);
    }

    #[test]
    fn bfs() {
        let g = generate_graph(true);
        let node = 0;

        let result = g.bfs(&node).unwrap();
        let expected = vec![0, 1, 2, 4, 3, 5, 6];

        assert_eq!(result, expected);
    }

    #[test]
    fn dfs() {
        let g = generate_graph(true);
        let node = 0;

        let result = g.dfs(&node).unwrap();
        let expected = vec![0, 4, 5, 2, 6, 1, 3];

        assert_eq!(result, expected);
    }

    #[test]
    fn adjacency_matrix_undirected() {
        let g = generate_graph(false);

        let result = g.adjacency_matrix().unwrap();
        let expected = vec![
            vec![0, 1, 1, 0, 1, 0, 0],
            vec![1, 0, 0, 1, 0, 1, 0],
            vec![1, 0, 0, 0, 0, 0, 1],
            vec![0, 1, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 1, 0],
            vec![0, 1, 0, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 0, 0, 0],
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn adjacency_matrix_directed() {
        let g = generate_graph(true);

        let result = g.adjacency_matrix().unwrap();
        let expected = vec![
            vec![0, 1, 1, 0, 1, 0, 0],
            vec![0, 0, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0],
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn has_path() {
        let g = generate_graph(true);

        let (node0, node2, node3, node5) = (0, 2, 3, 5);

        assert!(g.has_path(&node0, &node3));
        assert!(!g.has_path(&node2, &node5));
    }

    #[test]
    fn dijkstra_integer_weights() {
        let mut g = Graph::new(true);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_edge(1, 3, 1).unwrap();
        g.add_edge(0, 4, 1).unwrap();
        g.add_edge(1, 5, 3).unwrap();
        g.add_edge(2, 6, 1).unwrap();
        g.add_edge(4, 5, 1).unwrap();
        let (node0, node2, node3, node5) = (0, 2, 3, 5);

        assert_eq!(g.dijkstra(&node0, &node5), Some((vec![0, 4, 5], 2)));
        assert!(g.dijkstra(&node2, &node3).is_none());
    }

    #[test]
    fn dijkstra_float_weights() {
        let mut g = Graph::new(true);
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(0, 2, 1.0).unwrap();
        g.add_edge(1, 3, 1.0).unwrap();
        g.add_edge(0, 4, 1.0).unwrap();
        g.add_edge(1, 5, 3.0).unwrap();
        g.add_edge(2, 6, 1.0).unwrap();
        g.add_edge(4, 5, 1.0).unwrap();
        let (node0, node2, node3, node5) = (0, 2, 3, 5);

        assert_eq!(g.dijkstra(&node0, &node5), Some((vec![0, 4, 5], 2.0)));
        assert!(g.dijkstra(&node2, &node3).is_none());
    }
}
