mod generators;
mod graph;
pub use generators::{complete_graph, perfect_binary_tree, random_graph};
pub use graph::Graph;

#[cfg(test)]
mod tests {
    #[allow(dead_code)]
    use std::cmp::{Eq, Ord, PartialEq};
    use std::collections::HashSet;

    use super::*;
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

    fn generate_graph(directed: bool) -> Graph<i32, i32> {
        let mut g = Graph::new(directed);

        let node0 = g.add_node(0);
        let node1 = g.add_node(1);
        let node2 = g.add_node(2);
        let node3 = g.add_node(3);
        let node4 = g.add_node(4);
        let node5 = g.add_node(5);
        let node6 = g.add_node(6);

        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 1).unwrap();
        g.add_edge(node1, node3, 1).unwrap();
        g.add_edge(node0, node4, 1).unwrap();
        g.add_edge(node1, node5, 1).unwrap();
        g.add_edge(node2, node6, 1).unwrap();
        g.add_edge(node4, node5, 1).unwrap();

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

        let (node0, node1) = g.add_nodes_then_edge(node0, node1, 1);

        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.num_edges(), 2);
        assert!(g.has_path(node0, node1));
        assert_eq!(g.get_node(node0).unwrap().attribute1, 42);
        assert_eq!(g.get_node(node1).unwrap().attribute2, 0.0);
    }

    #[test]
    fn add_existing_struct_edge() {
        let mut g: Graph<Node, i32> = Graph::new(false);
        let node0 = Node::new(String::from("node0"), 42, 1.5);
        let node1 = Node::new(String::from("node1"), 64, 0.0);
        let (node0_idx, node1_idx) = g.add_nodes_then_edge(node0, node1, 1);

        assert!(g.add_edge(node0_idx, node1_idx, 1).is_err());
    }

    #[test]
    fn generate_random_directed_graph() {
        let g = random_graph(10, 5, true);

        assert_eq!(g.num_nodes(), 10);
        assert_eq!(g.num_edges(), 5);
    }

    #[test]
    fn generate_random_undirected_graph() {
        let g = random_graph(10, 5, false);

        assert_eq!(g.num_nodes(), 10);
        assert_eq!(g.num_edges(), 2 * 5);
    }

    #[test]
    fn generate_complete_undirected_graph() {
        let g = complete_graph(6);

        assert_eq!(g.num_nodes(), 6);
        assert_eq!(g.num_edges(), 6 * (6 - 1))
    }

    #[test]
    fn generate_binary_tree() {
        let g = perfect_binary_tree(2);

        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(0, 2));
        assert!(g.has_edge(1, 3));
        assert!(g.has_edge(1, 4));
        assert!(g.has_edge(2, 5));
        assert!(g.has_edge(2, 6));
        assert_eq!(g.neighbors(3).unwrap().len(), 0);
        assert_eq!(g.neighbors(4).unwrap().len(), 0);
        assert_eq!(g.neighbors(5).unwrap().len(), 0);
        assert_eq!(g.neighbors(6).unwrap().len(), 0);
    }

    #[test]
    fn add_new_nodes() {
        let mut g = Graph::default();
        let node0 = 0;
        let node1 = 1;

        let node0 = g.add_node(node0);
        let node1 = g.add_node(node1);

        assert_eq!(g.num_nodes(), 2);
        assert!(g.has_node(node0));
        assert!(g.has_node(node1));
    }

    #[test]
    fn add_nodes_from_iterator() {
        let mut g = Graph::default();
        let nodes = 0..5;
        let nodes_inds = g.add_nodes_from_iterator(nodes);

        assert_eq!(nodes_inds, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn remove_nodes() {
        let mut g = Graph::default();
        let node0 = 0;
        let node1 = 1;

        let node0 = g.add_node(node0);
        let node1 = g.add_node(node1);
        assert!(g.has_node(node0));
        assert!(g.has_node(node1));

        g.add_edge(node0, node1, 1).unwrap();
        assert!(g.has_node(node1));
        assert!(g.has_edge(node0, node1));

        g.remove_node(node1);
        assert!(!g.has_node(node1));
        assert!(g.has_node(node0));
        assert!(!g.has_edge(node0, node1));
    }

    #[test]
    fn add_new_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (node0, node1, node2) = (g.add_node(0), g.add_node(1), g.add_node(2));

        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 2).unwrap();
        g.add_edge(node1, node2, 1).unwrap();

        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.num_nodes(), 3);
        assert!(g.has_edge(node0, node1));
        assert!(g.has_edge(node0, node2));
        assert!(g.has_edge(node1, node2));
    }

    #[test]
    fn add_existing_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (node0, node1) = (g.add_node(0), g.add_node(1));
        g.add_edge(node0, node1, 1).unwrap();

        assert!(g.add_edge(node0, node1, 2).is_err());
    }

    #[test]
    fn remove_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (node0, node1) = g.add_nodes_then_edge(0, 1, 1);

        assert!(g.has_edge(node0, node1));

        g.remove_edge(node0, node1);
        assert!(!g.has_edge(node0, node1));
    }

    #[test]
    fn add_new_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (node0, node1, node2) = (g.add_node(0), g.add_node(1), g.add_node(2));

        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 2).unwrap();
        g.add_edge(node1, node2, 1).unwrap();

        assert_eq!(g.num_edges(), 6);
        assert_eq!(g.num_nodes(), 3);
        assert!(g.has_edge(node0, node1));
        assert!(g.has_edge(node1, node0));
        assert!(g.has_edge(node0, node2));
        assert!(g.has_edge(node2, node0));
        assert!(g.has_edge(node1, node2));
        assert!(g.has_edge(node2, node1));
    }

    #[test]
    fn add_existing_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (node0, node1) = (g.add_node(0), g.add_node(1));
        g.add_edge(node0, node1, 1).unwrap();

        assert!(g.add_edge(node0, node1, 2).is_err());
        assert!(g.add_edge(node1, node0, 2).is_err());
    }

    #[test]
    fn remove_edges_undirected_graph() {
        let mut g = Graph::new(false);
        let (node0, node1) = (g.add_node(0), g.add_node(1));

        g.add_edge(node0, node1, 1).unwrap();
        assert!(g.has_edge(node0, node1));
        assert!(g.has_edge(node1, node0));

        g.remove_edge(node0, node1);
        assert!(!g.has_edge(node0, node1));
        assert!(!g.has_edge(node1, node0));
    }

    #[test]
    fn aggregate_edges() {
        let g = generate_graph(true);
        let result = g.aggregate_edges(0, |nodes| nodes.iter().sum());

        assert_eq!(result, Some(3));
    }

    #[test]
    fn neighbors() {
        let mut g = Graph::default();
        let (node0, node1, node2, node3) =
            (g.add_node(0), g.add_node(1), g.add_node(2), g.add_node(3));
        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 1).unwrap();

        assert_eq!(g.neighbors(node0).unwrap().len(), 2);
        assert_eq!(g.neighbors(node3).unwrap().len(), 0);
        assert!(g.neighbors(4).is_none());
    }

    #[test]
    fn get_node() {
        let mut g = Graph::default();
        let (node0, node1) = (g.add_node(0), g.add_node(1));

        assert_eq!(g.get_node(node0), Some(&0));
        assert_eq!(g.get_node(node1), Some(&1));
    }

    #[test]
    fn get_nodes() {
        let mut g = Graph::default();
        g.add_node(0);
        g.add_node(1);

        let result: HashSet<&u32> = HashSet::from_iter(g.get_nodes());
        let expected = HashSet::from([&0_u32, &1_u32]);
        assert_eq!(result, expected);
    }

    #[test]
    fn degree() {
        let mut g = Graph::default();
        let (node0, node1, node2, node3) =
            (g.add_node(0), g.add_node(1), g.add_node(2), g.add_node(3));
        g.add_edge(node0, node1, 1).unwrap();
        g.add_edge(node0, node2, 1).unwrap();

        assert_eq!(g.degree(node0).unwrap(), 2);
        assert_eq!(g.degree(node1).unwrap(), 1);
        assert_eq!(g.degree(node3).unwrap(), 0);
    }

    #[test]
    fn bfs() {
        let g = generate_graph(true);

        let result = g.bfs(0).unwrap();
        let expected = vec![&0, &1, &2, &4, &3, &5, &6];

        assert_eq!(result, expected);
    }

    #[test]
    fn dfs() {
        let g = generate_graph(true);

        let result = g.dfs(0).unwrap();
        let expected = vec![&0, &4, &5, &2, &6, &1, &3];

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

        assert!(g.has_path(0, 3));
        assert!(!g.has_path(2, 5));
    }

    #[test]
    fn dijkstra_integer_weights() {
        let g = generate_graph(true);

        assert_eq!(g.dijkstra(0, 5), Some((vec![&0, &4, &5], 2)));
        assert!(g.dijkstra(2, 3).is_none());
    }
}
