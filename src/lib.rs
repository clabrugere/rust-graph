mod graph;

#[cfg(test)]
mod tests {
    use super::*;
    use graph::Graph;

    fn generate_graph(directed: bool) -> Graph<i32, i32>{
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

        let res = g.add_node(node);
        assert!(res.is_err());
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

        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 2).unwrap();
        g.add_edge(1, 2, 1).unwrap();

        assert_eq!(g.num_edges(), 3, "number of edges");
        assert_eq!(g.num_nodes(), 3, "number of nodes");
        assert!(g.has_edge(&0, &1));
        assert!(g.has_edge(&0, &2));
        assert!(g.has_edge(&1, &2));
    }

    #[test]
    fn add_existing_edges_directed_graph() {
        let mut g = Graph::new(true);
        let (from, to, weight) = (0, 1, 1);
        g.add_edge(from, to, weight).unwrap();

        let res = g.add_edge(from, to, weight);
        assert!(res.is_err());
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

        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 2).unwrap();
        g.add_edge(1, 2, 1).unwrap();

        assert_eq!(g.num_edges(), 6, "number of edges");
        assert_eq!(g.num_nodes(), 3, "number of nodes");
        assert!(g.has_edge(&0, &1));
        assert!(g.has_edge(&1, &0));
        assert!(g.has_edge(&0, &2));
        assert!(g.has_edge(&2, &0));
        assert!(g.has_edge(&1, &2));
        assert!(g.has_edge(&2, &1));
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
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_node(3).unwrap();

        assert_eq!(g.neighbors(&0).unwrap().len(), 2);
        assert_eq!(g.neighbors(&3).unwrap().len(), 0);
        assert!(g.neighbors(&4).is_none());
    }

    #[test]
    fn get_nodes() {
        let mut g = Graph::default();
        g.add_node(0).unwrap();
        g.add_node(1).unwrap();

        assert_eq!(g.get_nodes().len(), 2)
    }

    #[test]
    fn degree() {
        let mut g = Graph::default();
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_node(3).unwrap();

        assert_eq!(g.degree(&0).unwrap(), 2);
        assert_eq!(g.degree(&1).unwrap(), 1);
        assert_eq!(g.degree(&3).unwrap(), 0);
    }

    #[test]
    fn bfs() {
        let g = generate_graph(true);

        let result = g.bfs(&0).unwrap();
        let expected = vec![0, 1, 2, 4, 3, 5, 6];

        assert_eq!(result, expected);
    }

    #[test]
    fn dfs() {
        let g = generate_graph(true);

        let result = g.dfs(&0).unwrap();
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

        assert!(g.has_path(&0, &3));
        assert!(!g.has_path(&2, &5));
    }

    #[test]
    fn dijkstra() {
        let mut g = Graph::new(true);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_edge(1, 3, 1).unwrap();
        g.add_edge(0, 4, 1).unwrap();
        g.add_edge(1, 5, 3).unwrap();
        g.add_edge(2, 6, 1).unwrap();
        g.add_edge(4, 5, 1).unwrap();

        assert_eq!(g.dijkstra(&0, &5), Some((vec![0, 4, 5], 2)));
        assert!(g.dijkstra(&2, &3).is_none());
    }
}
