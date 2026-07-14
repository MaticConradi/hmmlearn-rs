//! The "occasionally dishonest casino": a categorical HMM over die rolls.
//!
//! Two hidden states — a fair die and a die loaded toward 6 — emit face values
//! 1..6. We sample a run of rolls from a known casino, then use Viterbi decoding
//! to recover *which* rolls came from the loaded die.
//!
//! Run with: `cargo run --example casino`

use hmmlearn::models::CategoricalHmm;
use hmmlearn::ndarray::array;
use hmmlearn::Result;

fn main() -> Result<()> {
    // State 0 = fair die, state 1 = loaded die. Symbols 0..=5 are faces 1..6.
    // The casino rarely switches to the loaded die and, once loaded, tends to
    // stay that way — then it slips a 6 half the time.
    let casino = CategoricalHmm::new(2)
        .start_prob(array![0.5, 0.5])
        .trans_mat(array![[0.95, 0.05], [0.10, 0.90],])
        .emissionprob(array![
            [
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0
            ],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.50],
        ])
        .into_fitted()?;

    let (rolls, truth) = casino.sample(300, Some(7), None);

    // Viterbi-decode the most likely fair/loaded sequence.
    let (log_prob, decoded) = casino.decode(rolls.view(), None, None)?;

    // Labels align (we decode with the true model), so accuracy is meaningful.
    let correct = decoded
        .iter()
        .zip(truth.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = correct as f64 / decoded.len() as f64;

    println!(
        "Rolled {} times; Viterbi log-probability {:.1}.",
        rolls.nrows(),
        log_prob
    );
    println!(
        "Recovered the loaded stretches with {:.1}% accuracy.\n",
        accuracy * 100.0
    );

    // Show the first 60 rolls with fair (.) vs loaded (#) annotations.
    println!("roll:    {}", digits(&rolls, 60));
    println!("truth:   {}", marks(&truth, 60));
    println!("decoded: {}", marks(&decoded, 60));

    Ok(())
}

fn digits(rolls: &hmmlearn::ndarray::Array2<f64>, n: usize) -> String {
    rolls
        .column(0)
        .iter()
        .take(n)
        .map(|&f| ((f as u8) + 1 + b'0') as char)
        .collect()
}

fn marks(states: &hmmlearn::ndarray::Array1<usize>, n: usize) -> String {
    states
        .iter()
        .take(n)
        .map(|&s| if s == 1 { '#' } else { '.' })
        .collect()
}
