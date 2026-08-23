use std::{convert::Infallible, hint::black_box, time::Instant};

use anyhow::{ensure, Context, Result};
use ark_bn254::Fr;
use batch::bucketed_ip::{
    packing_stride_for_bit_width, BvaI8Source, BvaLayout, PackedEoLiteBulletProof,
    StreamingEoProfile, TensorBasisVector,
};

const BIT_WIDTH: usize = 8;

struct IndexedBuckets {
    layout: BvaLayout,
    vectors: Vec<Vec<i8>>,
}

impl BvaI8Source for IndexedBuckets {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
        Some(self.vectors.iter().map(Vec::as_slice).collect())
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error> {
        for (index, values) in self.vectors.iter().enumerate() {
            visit(index, values);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Sample {
    total: f64,
    profile: StreamingEoProfile,
}

fn main() -> Result<()> {
    let mut bucket_count = 1024_usize;
    let mut bucket_log = 14_usize;
    let mut repetitions = 5_usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--bench" {
            continue;
        }
        let value = arguments
            .next()
            .with_context(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--bucket-count" => bucket_count = value.parse()?,
            "--bucket-log" => bucket_log = value.parse()?,
            "--repetitions" => repetitions = value.parse()?,
            _ => anyhow::bail!("unknown argument {argument}"),
        }
    }
    ensure!(
        bucket_count.is_power_of_two(),
        "bucket count must be a power of two"
    );
    ensure!(bucket_log < usize::BITS as usize, "bucket log is too large");
    ensure!(repetitions != 0, "repetitions must be non-zero");

    let bucket_len = 1_usize << bucket_log;
    let vectors = (0..bucket_count)
        .map(|bucket| {
            (0..bucket_len)
                .map(|index| {
                    (((bucket.wrapping_mul(17) + index.wrapping_mul(29)) % 251) as i16 - 125) as i8
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let lengths = vec![bucket_len; bucket_count];
    let layout = BvaLayout::from_lengths(&lengths)?;
    let point = (0..layout.logical_len().ilog2())
        .map(|round| Fr::from(u64::from(round) + 2))
        .collect::<Vec<_>>();
    let tbv = TensorBasisVector::global_tensor(&layout, &point)?;
    let mut source = IndexedBuckets { layout, vectors };
    let claimed = tbv.inner_product_i8_streaming(&mut source)?;
    let new_stride = packing_stride_for_bit_width::<Fr>(source.layout.logical_len(), BIT_WIDTH)?;
    ensure!(
        new_stride % 2 == 0,
        "the new packing stride must be twice the legacy stride"
    );
    let old_stride = new_stride / 2;

    let (old_proof, _, old_warm_profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_with_stride_bucketed_reference(
            &mut source,
            &tbv,
            claimed,
            BIT_WIDTH,
            old_stride,
        )?;
    let (new_proof, _, new_warm_profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
        &mut source,
        &tbv,
        claimed,
        BIT_WIDTH,
        new_stride,
    )?;
    let (same_stride_bucketed_proof, _, _) =
        PackedEoLiteBulletProof::prove_i8_streaming_with_stride_bucketed_reference(
            &mut source,
            &tbv,
            claimed,
            BIT_WIDTH,
            new_stride,
        )?;
    ensure!(
        same_stride_bucketed_proof == new_proof,
        "bucketed and flat implementations produced different proofs at the same stride"
    );
    ensure!(
        old_proof.verify_with_tbv(&tbv, claimed).accepted,
        "legacy proof failed verification"
    );
    ensure!(
        new_proof.verify_with_tbv(&tbv, claimed).accepted,
        "new proof failed verification"
    );
    let old_proof_size_bytes = old_proof.to_bytes()?.len();
    let new_proof_size_bytes = new_proof.to_bytes()?.len();

    let mut before = Vec::with_capacity(repetitions);
    let mut after = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        if repetition % 2 == 0 {
            before.push(run_before(&mut source, &tbv, claimed, old_stride)?);
            after.push(run_after(&mut source, &tbv, claimed, new_stride)?);
        } else {
            after.push(run_after(&mut source, &tbv, claimed, new_stride)?);
            before.push(run_before(&mut source, &tbv, claimed, old_stride)?);
        }
    }

    let before_total = median(before.iter().map(|sample| sample.total));
    let after_total = median(after.iter().map(|sample| sample.total));
    println!("benchmark=packed-eo-flat-ab");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("bucket_count={bucket_count}");
    println!("bucket_log={bucket_log}");
    println!("bucket_entries={bucket_len}");
    println!("logical_entries={}", source.layout.logical_len());
    println!("before_packing_stride={old_stride}");
    println!("after_packing_stride={new_stride}");
    println!("repetitions={repetitions}");
    println!("same_stride_proofs_identical=true");
    println!("before_proof_verified=true");
    println!("after_proof_verified=true");
    println!("before_proof_size_bytes={old_proof_size_bytes}");
    println!("after_proof_size_bytes={new_proof_size_bytes}");
    print_samples("before", &before);
    print_samples("after", &after);
    println!("total_speedup={:.6}", before_total / after_total);
    println!(
        "warm_max_live_before_field_entries={}",
        old_warm_profile.maximum_live_vector_entries
    );
    println!(
        "warm_max_live_after_field_entries={}",
        new_warm_profile.maximum_live_vector_entries
    );
    Ok(())
}

fn run_before(
    source: &mut IndexedBuckets,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    stride: usize,
) -> Result<Sample> {
    let started = Instant::now();
    let (proof, _, profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_with_stride_bucketed_reference(
            source, tbv, claimed, BIT_WIDTH, stride,
        )?;
    black_box(proof);
    Ok(Sample {
        total: started.elapsed().as_secs_f64(),
        profile,
    })
}

fn run_after(
    source: &mut IndexedBuckets,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    stride: usize,
) -> Result<Sample> {
    let started = Instant::now();
    let (proof, _, profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
        source, tbv, claimed, BIT_WIDTH, stride,
    )?;
    black_box(proof);
    Ok(Sample {
        total: started.elapsed().as_secs_f64(),
        profile,
    })
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn print_samples(label: &str, samples: &[Sample]) {
    println!(
        "{label}_total_samples_seconds={}",
        format_samples(samples.iter().map(|sample| sample.total))
    );
    println!(
        "{label}_packed_cross_terms_samples_seconds={}",
        format_samples(
            samples
                .iter()
                .map(|sample| sample.profile.timings.packed_cross_terms_seconds)
        )
    );
    println!(
        "{label}_total_seconds={:.6}",
        median(samples.iter().map(|sample| sample.total))
    );
    println!(
        "{label}_packed_cross_terms_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.packed_cross_terms_seconds)
        )
    );
    println!(
        "{label}_first_fold_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.first_fold_seconds)
        )
    );
    println!(
        "{label}_reduced_check_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.reduced_check_seconds)
        )
    );
    println!(
        "{label}_binary_cross_terms_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.binary_cross_terms_seconds)
        )
    );
    println!(
        "{label}_binary_challenge_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.binary_challenge_seconds)
        )
    );
    println!(
        "{label}_binary_fold_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.binary_fold_seconds)
        )
    );
    println!(
        "{label}_binary_rounds_seconds={:.6}",
        median(
            samples
                .iter()
                .map(|sample| sample.profile.timings.binary_rounds_seconds())
        )
    );
    println!(
        "{label}_max_live_field_entries={}",
        samples[0].profile.maximum_live_vector_entries
    );
}

fn format_samples(values: impl Iterator<Item = f64>) -> String {
    values
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}
