//! Sampling from and decoding a Gaussian HMM.
//!
//! Run with: `cargo run --example gaussian_hmm`

use hmmlearn::covariance::{CovarStore, CovarianceType};
use hmmlearn::models::GaussianHmm;
use hmmlearn::ndarray::{array, Array3};
use hmmlearn::Result;

fn main() -> Result<()> {
    // A 4-state model in 2-D. Note the transition matrix forbids moves between
    // components 0 and 2 (both directions have probability 0).
    let startprob = array![0.6, 0.3, 0.1, 0.0];
    let transmat = array![
        [0.7, 0.2, 0.0, 0.1],
        [0.3, 0.5, 0.2, 0.0],
        [0.0, 0.3, 0.5, 0.2],
        [0.2, 0.0, 0.2, 0.6],
    ];
    let means = array![[0.0, 0.0], [0.0, 11.0], [9.0, 10.0], [11.0, -1.0]];
    // 0.5 * I for every state: a full (4, 2, 2) covariance stack.
    let covars = CovarStore::Full(Array3::from_shape_fn((4, 2, 2), |(_, i, j)| {
        if i == j {
            0.5
        } else {
            0.0
        }
    }));

    // Set the parameters directly instead of fitting them from data.
    let gen_model = GaussianHmm::new(4)
        .covariance_type(CovarianceType::Full)
        .start_prob(startprob)
        .trans_mat(transmat)
        .means(means)
        .covars(covars)
        .into_fitted()?;

    // Draw a sequence: observations `x` and the true hidden states `z`.
    let (x, z) = gen_model.sample(500, Some(42), None);
    println!("Sampled {} two-dimensional observations.\n", x.nrows());

    // Recover the most likely state sequence with Viterbi decoding.
    let (log_prob, states) = gen_model.decode(x.view(), None, None)?;

    let correct = states.iter().zip(z.iter()).filter(|(a, b)| a == b).count();
    println!("Viterbi log-probability: {log_prob:.1}");
    println!(
        "Recovered {correct}/{} hidden states ({:.1}%).",
        z.len(),
        100.0 * correct as f64 / z.len() as f64
    );

    // How often the model visited each state in the drawn sequence.
    let mut counts = [0usize; 4];
    for &s in z.iter() {
        counts[s] += 1;
    }
    println!("\nState occupancy in the sample: {counts:?}");

    Ok(())
}
