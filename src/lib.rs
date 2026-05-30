use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod pyo3_example {
    use pyo3::prelude::*;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, Side, Col};
    use faer::linalg::solvers::EvdError;
    use linfa::prelude::*;
    use linfa::dataset::DatasetBase;
    use linfa_clustering::KMeans;
    use ndarray::Array2;
    use rand_xoshiro::Xoshiro256Plus;
    use rand::SeedableRng;

    /// Given edges and node weights, returns Python Result with eigenvectors and eigenvalues
    #[pyfunction]
    fn compute_laplacian(
        edges: Vec<(u32, u32)>,
        node_weights: Vec<f64>
    ) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
        /* let g = UnGraph:: <i32, ()>::from_edges(&edges); */
        /* println!("Graph: {:?}", Dot::with_config(&g, &[Config::EdgeNoLabel])); */

        let _node_count = node_weights.len();

        let mut triplets: Vec<Triplet<usize,usize,f64>> = Vec::new();

        // Build degree map using averaged edge weights
        let mut degrees = vec![0.0_f64; _node_count];
        for (a, b) in &edges {
            let _edge_weight = (node_weights[*a as usize] + node_weights[*b as usize]) / 2.0;
            degrees[*a as usize] += _edge_weight;
            degrees[*b as usize] += _edge_weight;
        };

        for i in 0.._node_count{
            triplets.push(Triplet::new(i, i, degrees[i]));
        };

        for (a, b) in &edges{
            let edge_weight = (node_weights[*a as usize] + node_weights[*b as usize]) / 2.0;
            triplets.push(Triplet::new(*a as usize, *b as usize, -edge_weight));
            triplets.push(Triplet::new(*b as usize, *a as usize, -edge_weight));
        }
        /* for (a, b) in &edges {
            let i  = node.index();
            let degree = g.neighbors(node).count() as f64;

            triplets.push(Triplet::new(i, i, degree));

            for edge in g.edges(node) {
                let j = edge.target().index();
                triplets.push(Triplet::new(i,j, -1.0 as f64));
            };
        }; */

        let _laplacian = SparseColMat::<usize, f64>::try_new_from_triplets(
            _node_count,
            _node_count,
            &triplets
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?; 

        println!("Laplacian Matrix created: {}x{} with {} non-zero entries", _node_count, _node_count, triplets.len());

        let (_eigenvalues, _eigenvectors, fiedler_vector) = get_fiedler_vector(_node_count, &triplets).unwrap();

        //println!("The first few values of the fielder_vector are {:?}", fiedler_vector);

        let result = fiedler_vector_check(&fiedler_vector).unwrap();
        if !result {
            println!("Something wrong with fiedler_vector");
        }
        else{
            println!("fiedler vector is valid");
        }

        let eigenvectors: Vec<Vec<f64>> = (0.._eigenvectors.ncols())
        .map(|i| _eigenvectors.col(i).iter().copied().collect())
        .collect();

        Ok((eigenvectors, _eigenvalues.iter().copied().collect()))
    }

    fn get_fiedler_vector(
        _node_count: usize, 
        triplets: &[Triplet<usize, usize, f64>],
    ) -> Result<(Col<f64>, Mat<f64>, Col<f64>),EvdError> {
        // First we'll convert to dense for the solver, since the eigenvectors are always dense
        let mut dense_laplacian = Mat::<f64>::zeros(_node_count, _node_count);
        for triplet in triplets{
            dense_laplacian[(triplet.row, triplet.col)] += triplet.val;
        }
        println!("Dense Laplacian Matrix has been created!");

        let decompisition = dense_laplacian.self_adjoint_eigen(Side::Lower)?;

        let _eigenvalues = decompisition.S().column_vector().to_owned();

        let mut index: usize = 0;
        // Print the first few eigenvalues to confirm index 1 is the Fiedler value
        for i in 0.._eigenvalues.nrows() {
            if _eigenvalues[i] > 1e-4_f64{
                println!("λ{}: {:?}", i, _eigenvalues[i]);
                index = i; 
                break;
            }
        }

        let _eigenvectors = decompisition.U().to_owned();
        let mut fiedler_vector = _eigenvectors.col(index).to_owned();
        fiedler_vector = fiedler_vector.iter().map(|&x| if x.abs() < 1e-4_f64 {0.0} else {x}).collect();

        Ok((_eigenvalues, _eigenvectors, fiedler_vector))
    }

    fn fiedler_vector_check(_fiedler_vector: &Col<f64>) -> Result<bool, String> {
        let result = true;
        let sum: f64 = _fiedler_vector.iter().sum();
        println!("L1 norm should be approximately 0: {}", sum);

        let norm: f64 = _fiedler_vector.iter().map(|x| x*x).sum::<f64>().sqrt();
        println!("L2 norm should be approximately 1: {}", norm);
        Ok(result)
    }

    #[pyfunction]
    /// Given a list of eigenvectors, returns a list of clusters
    fn run_spectral_kmeans(
        eigenvectors: Vec<Vec<f64>>,
        k: usize,
    ) -> PyResult<Vec<usize>> {
        let n_nodes = eigenvectors[0].len();
        let n_vecs = (eigenvectors.len() - 1).min(k);

        // Start by creating a spectral embedding matrix
        // Every row is a node, while every column is an eigenvector
        // Skip the trivial eigenvector (eigenvector[0])
        let mut flat : Vec<f64> = Vec::with_capacity(n_nodes * (n_vecs));
        for node_idx in 0..n_nodes {
            for vec_idx in 1..n_vecs {
                flat.push(eigenvectors[vec_idx][node_idx]);
            }
        }

        let observations = Array2::from_shape_vec(
            (n_nodes, n_vecs -1),
            flat
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _> (
            format!("Failed to build obesrvation matrix: {:?}", e)))?;

        // Build the linfa dataSet
        let dataset = DatasetBase::from(observations.clone());

        // Run K-Means clustering
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let model = KMeans::params_with_rng(k, rng)
        .max_n_iterations(100)
        .tolerance(1e-3)
        .fit(&dataset)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fit K-Means model: {:?}", e)))?;

        let result = model.predict(observations);
        let labels: Vec<usize> = result.targets.to_vec();
        println!("K-Means clustering completed with {} clusters", k);

        Ok(labels)
    }
}
