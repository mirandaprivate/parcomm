use std::{fs, path::Path};

use batch::{
    matrix_collection::{DenseBvaDescriptor, DenseMatrixBva},
    model_bva::{ModelBvaManifest, ModelPreset},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const BUCKET_ORDER: &str = "entry_count_desc_then_tensor_index_then_row_offset_then_col_offset";

#[test]
fn converted_i8_bva_is_loaded_in_place_from_validated_manifest_order() {
    let directory = TempDir::new().unwrap();
    let manifest_path = directory.path().join("tiny.manifest.json");
    let bva_path = directory.path().join("tiny.bva.i8");
    let shapes = [(3, 2)];
    let generated =
        DenseMatrixBva::from_fn(&shapes, |_, row, col| (1 + row * 3 + col) as i8).unwrap();
    let bytes = bytemuck::cast_slice::<i8, u8>(generated.values());
    fs::write(&bva_path, bytes).unwrap();
    write_manifest(
        &manifest_path,
        &bva_path,
        generated.descriptor(),
        hex_digest(bytes),
        "i8",
        8,
        bytes.len(),
    );

    let manifest = ModelBvaManifest::load(
        ModelPreset::Llama8bI8,
        directory.path(),
        Some(&manifest_path),
        None,
    )
    .unwrap();
    assert_eq!(manifest.matrix_shapes(), shapes);
    assert_eq!(manifest.bva_path(), bva_path);

    let loaded = manifest.load_i8(true).unwrap();
    assert_eq!(loaded.bva.values(), generated.values());
    assert!(loaded.profile.sha256_verified);
    assert_eq!(loaded.profile.bytes_read, bytes.len() as u64);
    let classified_seconds = loaded.profile.allocation_seconds
        + loaded.profile.read_seconds
        + loaded.profile.sha256_seconds
        + loaded.profile.adopt_seconds;
    assert!(loaded.profile.total_seconds >= classified_seconds);
}

#[test]
fn native_i16_manifest_loads_without_reinterpretation_or_widening() {
    let directory = TempDir::new().unwrap();
    let manifest_path = directory.path().join("tiny-i16.manifest.json");
    let bva_path = directory.path().join("tiny-i16.bva.i16");
    let shapes = [(3, 2)];
    let words = [0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xffff];
    let generated = DenseMatrixBva::<i16>::from_i16_fn(&shapes, |_, row, col| {
        i16::from_le_bytes(words[row * shapes[0].1 + col].to_le_bytes())
    })
    .unwrap();
    let bytes = bytemuck::cast_slice::<i16, u8>(generated.values()).to_vec();
    fs::write(&bva_path, &bytes).unwrap();
    write_manifest(
        &manifest_path,
        &bva_path,
        generated.descriptor(),
        hex_digest(&bytes),
        "i16",
        16,
        bytes.len(),
    );

    let manifest = ModelBvaManifest::load(
        ModelPreset::Llama8bI16,
        directory.path(),
        Some(&manifest_path),
        None,
    )
    .unwrap();
    let error = match manifest.clone().load_i8(false) {
        Err(error) => error,
        Ok(_) => panic!("native i16 was reinterpreted by the i8 loader"),
    };
    assert!(error.to_string().contains("i8 loader does not convert"));
    let loaded = manifest.load_i16(true).unwrap();
    assert_eq!(loaded.bva.values(), generated.values());
    assert!(loaded.profile.sha256_verified);
    assert_eq!(loaded.profile.bytes_read, bytes.len() as u64);
}

#[test]
fn missing_llama70b_conversion_does_not_fall_back_to_synthetic_data() {
    let directory = TempDir::new().unwrap();
    let error =
        ModelBvaManifest::load(ModelPreset::Llama70bI8, directory.path(), None, None).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("Llama-3.1-70B-W8A8"));
    assert!(message.contains("llama-3.1-70b-w8a8.manifest.json"));
}

fn write_manifest(
    path: &Path,
    bva_path: &Path,
    descriptor: &DenseBvaDescriptor,
    sha256: String,
    storage_dtype: &str,
    bit_width: usize,
    bva_size_bytes: usize,
) {
    let bucket_indices = descriptor
        .buckets()
        .iter()
        .enumerate()
        .filter_map(|(index, bucket)| (bucket.matrix() == 0).then_some(index))
        .collect::<Vec<_>>();
    let buckets = descriptor
        .buckets()
        .iter()
        .enumerate()
        .map(|(index, bucket)| {
            json!({
                "index": index,
                "tensor_index": bucket.matrix(),
                "row_offset": bucket.row_offset(),
                "row_len": bucket.row_len(),
                "physical_row_len": bucket.physical_row_len(),
                "col_offset": bucket.col_offset(),
                "col_len": bucket.col_len(),
                "physical_col_len": bucket.physical_col_len(),
                "dense_offset": bucket.dense_offset(),
                "source_offset": bucket.source_offset(),
                "entry_count": bucket.entry_count(),
                "physical_entry_count": bucket.physical_entry_count(),
            })
        })
        .collect::<Vec<_>>();
    let mut manifest = json!({
        "format": format!("parcomm-dense-{storage_dtype}-bva"),
        "version": if bit_width == 16 { 2 } else { 1 },
        "quantization": {
            "storage_dtype": storage_dtype,
            "storage_encoding": if bit_width == 8 {
                "signed_twos_complement_byte"
            } else {
                "signed_twos_complement_little_endian"
            },
            "storage_item_bytes": bit_width / 8,
            "bit_width": bit_width,
        },
        "layout": {
            "tensor_count": 1,
            "bucket_count": descriptor.bucket_count(),
            "physical_entries": descriptor.physical_len(),
            "active_entries": descriptor.layout().active_len(),
            "logical_entries": descriptor.logical_len(),
            "boundary_padding_entries": descriptor.conceptual_boundary_zeros(),
            "final_padding_entries": descriptor.conceptual_final_zeros(),
            "splitting_budget": DenseBvaDescriptor::SPLITTING_BUDGET,
            "bucket_order": BUCKET_ORDER,
            "within_bucket_order": "row_major",
            "rank1_matrix_mapping": "shape_n_to_n_by_1",
            "element_dtype": storage_dtype,
            "element_bytes": bit_width / 8,
            "bva_path": format!("/converter-host/{}", bva_path.file_name().unwrap().to_string_lossy()),
            "bva_size_bytes": bva_size_bytes,
            "bva_sha256": sha256,
        },
        "tensors": [{
            "index": 0,
            "matrix_shape": [3, 2],
            "element_count": 6,
            "bucket_indices": bucket_indices,
        }],
        "buckets": buckets,
    });
    if bit_width == 16 {
        manifest["source"] = json!({ "model_family": "llama-bf16" });
        manifest["quantization"]["scheme"] = json!("bf16_bitcast_i16");
        manifest["quantization"]["native_i8_tensor_count"] = json!(0);
        manifest["quantization"]["converted_float_tensor_count"] = json!(0);
        manifest["quantization"]["bitcast_bf16_tensor_count"] = json!(1);
        manifest["quantization"]["bit_preserving"] = json!(true);
        manifest["quantization"]["numeric_conversion"] = json!(false);
        manifest["quantization"]["source_to_storage"] = json!("bitcast_bf16_u16_to_i16");
        manifest["quantization"]["source_storage_encoding"] =
            json!("bf16_bit_pattern_little_endian");
        manifest["quantization"]["field_interpretation"] = json!("signed_twos_complement_i16");
        manifest["tensors"][0]["source_dtype"] = json!("BF16");
        manifest["tensors"][0]["conversion"] = json!("bitcast_bf16_to_i16");
        manifest["tensors"][0]["max_abs"] = serde_json::Value::Null;
        manifest["tensors"][0]["scale"] = serde_json::Value::Null;
    }
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
