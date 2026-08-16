use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fr};
use ark_poly_commit::smart_pc::SmartPC;
use zksmart::hdr1cs::gpt2::{
    Gpt2GlobalBvaWitnessSource, Gpt2GlobalLayout, Gpt2PublicStatement, Gpt2Q12Checkpoint,
};
use zksmart::protocols::virtualpcs::commit_bva_i32_streaming;

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    let (path, layout_only, commit_bva, rewrite_reference) = arguments();
    println!("experiment=gpt2_q12_streaming_reference");
    println!("checkpoint={}", path.display());
    println!("proof_generated=false");
    println!("purpose=bit_exact_frontend_validation_before_hd_r1cs_encoding");
    println!("peak_rss_before_bytes={}", peak_rss_bytes());

    let file = File::open(&path).expect("open GPT-2 Q12 checkpoint bundle");
    let reader = BufReader::with_capacity(8 << 20, file);
    let load_start = Instant::now();
    let mut checkpoint = Gpt2Q12Checkpoint::from_reader(reader).expect("parse checkpoint bundle");
    println!(
        "checkpoint_format_version={}",
        checkpoint.metadata().format_version
    );
    println!(
        "metadata_load_seconds={:.6}",
        load_start.elapsed().as_secs_f64()
    );
    println!("token_ids={:?}", checkpoint.token_ids());
    println!(
        "parameter_entries={}",
        checkpoint.metadata().parameter_entries
    );
    if checkpoint.metadata().format_version == 2 {
        let statement = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )
        .expect("compile public GPT-2 statement");
        let relation_shape = statement
            .relation_shape()
            .expect("compile primitive relation manifest");
        println!(
            "hd_r1cs_top_level_equations={}",
            relation_shape.top_level_equation_count()
        );
        println!(
            "hd_r1cs_linear_relations={}",
            relation_shape.linear_relations
        );
        println!(
            "hd_r1cs_bilinear_relations={}",
            relation_shape.bilinear_relations
        );
        println!(
            "hd_r1cs_tagged_lookup_relations={}",
            relation_shape.tagged_lookup_relations
        );
        println!(
            "hd_r1cs_io_equation_scalar_rows={}",
            relation_shape.io_equation_scalar_rows
        );
        println!(
            "hd_r1cs_linear_equation_scalar_rows={}",
            relation_shape.linear_equation_scalar_rows
        );
        println!(
            "hd_r1cs_bilinear_equation_scalar_rows={}",
            relation_shape.bilinear_equation_scalar_rows
        );
        println!(
            "hd_r1cs_tagged_lookup_equation_scalar_rows={}",
            relation_shape.tagged_lookup_equation_scalar_rows
        );
        println!(
            "hd_r1cs_equation_bva_logical_rows={}",
            relation_shape.equation_bva_logical_rows
        );
        println!(
            "hd_r1cs_maximum_live_equation_row_entries={}",
            relation_shape.maximum_live_equation_row_entries
        );
        println!(
            "hd_r1cs_equation_materialized_padding_entries={}",
            relation_shape.equation_materialized_padding_entries
        );
        println!(
            "hd_r1cs_auxiliary_blocks={}",
            relation_shape.auxiliary_blocks
        );
        println!(
            "hd_r1cs_auxiliary_entries={}",
            relation_shape.auxiliary_entries
        );
        println!(
            "hd_r1cs_resident_auxiliary_blocks={}",
            relation_shape.resident_auxiliary_blocks
        );
        println!(
            "hd_r1cs_resident_auxiliary_entries={}",
            relation_shape.resident_auxiliary_entries
        );
        println!(
            "hd_r1cs_streamed_range_limb_blocks={}",
            relation_shape.streamed_range_limb_blocks
        );
        println!(
            "hd_r1cs_streamed_range_limb_entries={}",
            relation_shape.streamed_range_limb_entries
        );
        println!(
            "hd_r1cs_maximum_streamed_range_limb_column_entries={}",
            relation_shape.maximum_streamed_range_limb_column_entries
        );
        println!(
            "hd_r1cs_global_witness_blocks={}",
            relation_shape.global_witness_blocks
        );
        println!(
            "hd_r1cs_global_physical_entries={}",
            relation_shape.global_physical_entries
        );
        println!(
            "hd_r1cs_global_bva_logical_entries={}",
            relation_shape.global_bva_logical_entries
        );
        println!(
            "hd_r1cs_materialized_padding_entries={}",
            relation_shape.materialized_padding_entries
        );
    }
    let layout = Gpt2GlobalLayout::compile(checkpoint.metadata())
        .expect("compile public global block layout");
    println!("global_witness_blocks={}", layout.witness_block_count());
    println!("global_model_blocks={}", layout.model_blocks().len());
    println!(
        "global_activation_entries={}",
        layout.activation_entries().expect("activation entry count")
    );
    println!(
        "global_physical_entries={}",
        layout
            .physical_entries(checkpoint.metadata())
            .expect("physical entry count")
    );
    println!(
        "global_logical_padded_entries={}",
        layout
            .logical_padded_entries()
            .expect("logical padded entry count")
    );
    let bva = layout
        .compile_bva()
        .expect("compile non-materialized BVA layout");
    let bva_log = bva.logical_entries().ilog2() as usize;
    let bva_packed_rows = 1_usize << (bva_log - bva_log / 2);
    println!("bva_vectors={}", bva.vector_count());
    println!("bva_physical_entries={}", bva.physical_entries());
    println!("bva_logical_entries={}", bva.logical_entries());
    println!(
        "bva_logical_padding_entries={}",
        bva.logical_padding_entries()
    );
    println!("bva_prepared_base_entries_bit32={}", bva_packed_rows * 32);
    println!("bva_smart_pc_q={}", (bva_log + 1) / 2);
    println!("materialized_model_parameter_entries=0");
    println!("materialized_padding_entries=0");
    println!("peak_rss_after_metadata_bytes={}", peak_rss_bytes());
    if layout_only {
        println!("layout_only=true");
        return;
    }

    let execute_start = Instant::now();
    let (execution, activation_witness) = checkpoint
        .execute_activation_witness(&layout)
        .expect("execute Q12 GPT-2 and build activation witness");
    let operation_events = layout.operations().len();
    let parameter_backed_events = layout
        .operations()
        .iter()
        .filter(|operation| operation.parameter_blocks.iter().any(Option::is_some))
        .count();
    println!(
        "execute_seconds={:.6}",
        execute_start.elapsed().as_secs_f64()
    );
    println!("operation_events={operation_events}");
    println!("parameter_backed_events={parameter_backed_events}");
    println!("operation_manifest_match=true");
    println!("activation_i32_checked=true");
    println!(
        "materialized_activation_entries={}",
        activation_witness.stored_entries()
    );
    println!("logits={}", execution.logits.len());
    println!("argmax={}", execution.argmax);
    println!("checksum={}", execution.checksum);
    let reference_match = execution.logits == checkpoint.reference_logits();
    println!("reference_match={reference_match}");
    if let Some(output_path) = &rewrite_reference {
        assert_ne!(output_path, &path, "rewrite output must differ from input");
        let output = File::create(output_path).expect("create migrated GPT-2 Q12 bundle");
        let mut writer = BufWriter::with_capacity(8 << 20, output);
        checkpoint
            .write_bundle_with_reference(&mut writer, &execution.logits)
            .expect("write exact-Softmax v2 GPT-2 Q12 bundle");
        writer.flush().expect("flush migrated GPT-2 Q12 bundle");
        println!("rewritten_reference_bundle={}", output_path.display());
        println!("rewritten_checkpoint_format_version=2");
    }
    if commit_bva {
        let setup_start = Instant::now();
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup((bva_log + 1) / 2, &mut rng)
            .expect("setup BVA-sized SmartPC parameters");
        println!(
            "bva_commit_setup_seconds={:.6}",
            setup_start.elapsed().as_secs_f64()
        );
        println!("peak_rss_after_bva_setup_bytes={}", peak_rss_bytes());
        let commit_start = Instant::now();
        let mut source =
            Gpt2GlobalBvaWitnessSource::new(&mut checkpoint, &layout, &bva, &activation_witness)
                .expect("build one global BVA witness source");
        let (_commitment, tier_one, profile) =
            commit_bva_i32_streaming::<Bls12_381, _>(&pp, &mut source, Fr::from(73_u64), 32)
                .expect("commit one global BVA witness");
        println!(
            "bva_commit_seconds={:.6}",
            commit_start.elapsed().as_secs_f64()
        );
        println!("bva_commitment_generated=true");
        println!("bva_commit_physical_entries={}", profile.physical_entries);
        println!("bva_commit_nonzero_entries={}", profile.nonzero_entries);
        println!("bva_commit_tier_one_entries={}", tier_one.len());
        println!(
            "bva_commit_prepared_base_entries={}",
            profile.prepared_base_entries
        );
        println!("bva_commit_parallel_vectors={}", profile.parallel_vectors);
        println!(
            "bva_commit_max_live_group_contributions={}",
            profile.max_live_group_contributions
        );
        println!(
            "bva_commit_materialized_padding_entries={}",
            profile.materialized_padding_entries
        );
        println!(
            "bva_commit_padding_group_operations={}",
            profile.padding_group_operations
        );
        println!("peak_rss_after_bva_commit_bytes={}", peak_rss_bytes());
    }
    let model_stream_start = Instant::now();
    let mut model_checksum = 0_i128;
    let model_stream = checkpoint
        .for_each_model_block_column(&layout, |_, _, column| {
            model_checksum += column.iter().map(|&value| i128::from(value)).sum::<i128>();
        })
        .expect("stream all model blocks in global algebraic orientation");
    println!(
        "model_stream_seconds={:.6}",
        model_stream_start.elapsed().as_secs_f64()
    );
    println!(
        "model_stream_physical_entries={}",
        model_stream.physical_entries
    );
    println!(
        "model_stream_max_live_entries={}",
        model_stream.max_live_entries
    );
    println!("model_stream_checksum={model_checksum}");
    println!("peak_rss_final_bytes={}", peak_rss_bytes());
    assert!(
        reference_match || rewrite_reference.is_some(),
        "reference mismatch: computed checksum={} stored checksum={}",
        execution.checksum,
        checkpoint
            .reference_logits()
            .iter()
            .map(|&value| i128::from(value))
            .sum::<i128>()
    );
}

fn arguments() -> (PathBuf, bool, bool, Option<PathBuf>) {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut layout_only = false;
    let mut commit_bva = false;
    let mut rewrite_reference = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--bench" => {}
            "--checkpoint" => path = Some(PathBuf::from(args.next().expect("checkpoint path"))),
            "--layout-only" => layout_only = true,
            "--commit-bva" => commit_bva = true,
            "--rewrite-reference" => {
                rewrite_reference = Some(PathBuf::from(args.next().expect("rewrite path")))
            }
            other => panic!("unknown argument: {}", other),
        }
    }
    let path = path
        .or_else(|| std::env::var_os("ZKSMART_GPT2_Q12_CHECKPOINT").map(PathBuf::from))
        .expect("pass --checkpoint or set ZKSMART_GPT2_Q12_CHECKPOINT");
    (path, layout_only, commit_bva, rewrite_reference)
}

fn peak_rss_bytes() -> u64 {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        rss * 1024
    }
}
