//! Loader for preconverted model BVA artifacts.
//!
//! Converter manifests contain absolute source paths from the machine that
//! produced them. The BVA itself is deliberately resolved by basename next to
//! the manifest, unless the caller supplies an explicit override.

use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    bucketed_ip::BvaInteger,
    matrix_collection::{DenseBvaDescriptor, DenseMatrixBva, MatrixCollectionError},
};

const LEGACY_MANIFEST_VERSION: u64 = 1;
const MANIFEST_VERSION: u64 = 2;
const BF16_BITCAST_SCHEME: &str = "bf16_bitcast_i16";
const BF16_BITCAST_CONVERSION: &str = "bitcast_bf16_to_i16";
const BF16_SOURCE_TO_STORAGE: &str = "bitcast_bf16_u16_to_i16";
const BUCKET_ORDER: &str = "entry_count_desc_then_tensor_index_then_row_offset_then_col_offset";
const WITHIN_BUCKET_ORDER: &str = "row_major";
const RANK1_MATRIX_MAPPING: &str = "shape_n_to_n_by_1";

/// A preconverted model artifact understood by the experiment benchmark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelPreset {
    Llama8bI8,
    Llama8bI16,
    Llama70bI8,
}

impl ModelPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama8bI8 => "llama8b-i8",
            Self::Llama8bI16 => "llama8b-i16",
            Self::Llama70bI8 => "llama70b-i8",
        }
    }

    pub fn manifest_relative_path(self) -> &'static Path {
        match self {
            Self::Llama8bI8 => {
                Path::new("Llama-3.1-8B-W8A8/parcomm-i8/llama-3.1-8b-w8a8.manifest.json")
            }
            Self::Llama8bI16 => Path::new(
                "Llama-3.1-8B-BF16/parcomm-bf16-bits/llama-3.1-8b-bf16-bits.manifest.json",
            ),
            Self::Llama70bI8 => {
                Path::new("Llama-3.1-70B-W8A8/parcomm-i8/llama-3.1-70b-w8a8.manifest.json")
            }
        }
    }

    fn storage_dtype(self) -> &'static str {
        match self {
            Self::Llama8bI8 | Self::Llama70bI8 => "i8",
            Self::Llama8bI16 => "i16",
        }
    }

    fn bit_width(self) -> usize {
        match self {
            Self::Llama8bI8 | Self::Llama70bI8 => 8,
            Self::Llama8bI16 => 16,
        }
    }
}

impl Display for ModelPreset {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelPreset {
    type Err = ModelBvaError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "llama8b-i8" | "llama8b-8bit" | "llama-8b-i8" | "llama-8b-8bit" => Ok(Self::Llama8bI8),
            "llama8b-i16" | "llama8b-16bit" | "llama-8b-i16" | "llama-8b-16bit" => {
                Ok(Self::Llama8bI16)
            }
            "llama70b-i8" | "llama70b-8bit" | "llama-70b-i8" | "llama-70b-8bit" => {
                Ok(Self::Llama70bI8)
            }
            _ => Err(ModelBvaError::UnknownPreset(value.to_owned())),
        }
    }
}

/// Resolve the model-data directory independently of the process working directory.
pub fn default_model_data_root() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = crate_dir
        .parent()
        .expect("the batch crate must have a workspace parent");
    workspace_dir
        .parent()
        .unwrap_or(workspace_dir)
        .join("model-data")
}

/// Validated metadata for one converted BVA.
#[derive(Clone, Debug)]
pub struct ModelBvaManifest {
    preset: ModelPreset,
    manifest_path: PathBuf,
    bva_path: PathBuf,
    storage_dtype: String,
    bit_width: usize,
    element_bytes: usize,
    matrix_shapes: Vec<(usize, usize)>,
    descriptor: DenseBvaDescriptor,
    physical_entries: usize,
    active_entries: usize,
    logical_entries: usize,
    bva_size_bytes: u64,
    expected_sha256: String,
}

impl ModelBvaManifest {
    /// Parse and validate a converter manifest and its BVA file.
    ///
    /// Relative manifest overrides are resolved below `model_data_root`.
    /// Relative BVA overrides are resolved next to the selected manifest.
    pub fn load(
        preset: ModelPreset,
        model_data_root: &Path,
        manifest_override: Option<&Path>,
        bva_override: Option<&Path>,
    ) -> Result<Self, ModelBvaError> {
        let manifest_path = resolve_manifest_path(preset, model_data_root, manifest_override);
        let manifest_file = File::open(&manifest_path)
            .map_err(|source| ModelBvaError::io(&manifest_path, source))?;
        let raw: RawManifest =
            serde_json::from_reader(manifest_file).map_err(|source| ModelBvaError::Json {
                path: manifest_path.clone(),
                source,
            })?;
        validate_scalar_metadata(preset, &raw)?;

        let matrix_shapes = validate_tensor_metadata(&raw.tensors)?;
        let descriptor = DenseBvaDescriptor::from_shapes(&matrix_shapes)?;
        validate_layout(&raw.layout, &descriptor, raw.tensors.len(), preset)?;
        validate_bucket_metadata(&raw, &descriptor)?;

        let bva_path = resolve_bva_path(&manifest_path, &raw.layout.bva_path, bva_override)?;
        let bva_size_bytes = raw.layout.bva_size_bytes;
        let actual_size = bva_path
            .metadata()
            .map_err(|source| ModelBvaError::io(&bva_path, source))?
            .len();
        if actual_size != bva_size_bytes {
            return Err(ModelBvaError::FileSizeMismatch {
                path: bva_path,
                expected: bva_size_bytes,
                actual: actual_size,
            });
        }

        Ok(Self {
            preset,
            manifest_path,
            bva_path,
            storage_dtype: raw.quantization.storage_dtype,
            bit_width: usize_field("quantization.bit_width", raw.quantization.bit_width)?,
            element_bytes: preset.bit_width() / 8,
            matrix_shapes,
            descriptor,
            physical_entries: usize_field("layout.physical_entries", raw.layout.physical_entries)?,
            active_entries: usize_field("layout.active_entries", raw.layout.active_entries)?,
            logical_entries: usize_field("layout.logical_entries", raw.layout.logical_entries)?,
            bva_size_bytes,
            expected_sha256: raw.layout.bva_sha256.to_ascii_lowercase(),
        })
    }

    pub fn load_default(
        preset: ModelPreset,
        manifest_override: Option<&Path>,
        bva_override: Option<&Path>,
    ) -> Result<Self, ModelBvaError> {
        Self::load(
            preset,
            &default_model_data_root(),
            manifest_override,
            bva_override,
        )
    }

    pub fn preset(&self) -> ModelPreset {
        self.preset
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn bva_path(&self) -> &Path {
        &self.bva_path
    }

    pub fn storage_dtype(&self) -> &str {
        &self.storage_dtype
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    pub fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    pub fn matrix_shapes(&self) -> &[(usize, usize)] {
        &self.matrix_shapes
    }

    pub fn descriptor(&self) -> &DenseBvaDescriptor {
        &self.descriptor
    }

    pub fn physical_entries(&self) -> usize {
        self.physical_entries
    }

    pub fn active_entries(&self) -> usize {
        self.active_entries
    }

    pub fn logical_entries(&self) -> usize {
        self.logical_entries
    }

    pub fn bva_size_bytes(&self) -> u64 {
        self.bva_size_bytes
    }

    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }

    /// Read an i8 BVA into its single final allocation and adopt it directly.
    pub fn load_i8(self, verify_sha256: bool) -> Result<LoadedI8ModelBva, ModelBvaError> {
        self.load_native(verify_sha256, "i8", 8, 1, |shapes, values| {
            DenseMatrixBva::from_prepacked_i8(shapes, values)
        })
    }

    /// Read an i16 BVA into its single final allocation without widening it.
    pub fn load_i16(self, verify_sha256: bool) -> Result<LoadedI16ModelBva, ModelBvaError> {
        self.load_native(verify_sha256, "i16", 16, 2, |shapes, values| {
            DenseMatrixBva::from_prepacked_i16(shapes, values)
        })
    }

    fn load_native<Integer, Adopt>(
        self,
        verify_sha256: bool,
        expected_dtype: &'static str,
        expected_bit_width: usize,
        expected_element_bytes: usize,
        adopt: Adopt,
    ) -> Result<LoadedModelBva<Integer>, ModelBvaError>
    where
        Integer: BvaInteger + bytemuck::Pod,
        Adopt: FnOnce(
            &[(usize, usize)],
            Vec<Integer>,
        ) -> Result<DenseMatrixBva<Integer>, MatrixCollectionError>,
    {
        if self.storage_dtype != expected_dtype
            || self.bit_width != expected_bit_width
            || self.element_bytes != expected_element_bytes
        {
            return Err(ModelBvaError::UnsupportedNativeLoad {
                preset: self.preset,
                storage_dtype: self.storage_dtype.clone(),
                bit_width: self.bit_width,
                requested_storage_dtype: expected_dtype,
            });
        }
        if cfg!(target_endian = "big") && expected_element_bytes > 1 {
            return Err(ModelBvaError::invalid(
                "storage_encoding",
                "little-endian i16 BVA loading requires a little-endian target",
            ));
        }

        let total_started = Instant::now();
        let allocation_started = Instant::now();
        let mut values = Vec::<Integer>::new();
        values
            .try_reserve_exact(self.logical_entries)
            .map_err(|source| ModelBvaError::Allocation {
                entries: self.logical_entries,
                message: source.to_string(),
            })?;
        values.resize(self.logical_entries, Integer::from(false));
        let allocation_seconds = allocation_started.elapsed().as_secs_f64();

        let read_started = Instant::now();
        let mut bva_file = File::open(&self.bva_path)
            .map_err(|source| ModelBvaError::io(&self.bva_path, source))?;
        let current_size = bva_file
            .metadata()
            .map_err(|source| ModelBvaError::io(&self.bva_path, source))?
            .len();
        if current_size != self.bva_size_bytes {
            return Err(ModelBvaError::FileSizeMismatch {
                path: self.bva_path.clone(),
                expected: self.bva_size_bytes,
                actual: current_size,
            });
        }
        bva_file
            .read_exact(bytemuck::cast_slice_mut(values.as_mut_slice()))
            .map_err(|source| ModelBvaError::io(&self.bva_path, source))?;
        let read_seconds = read_started.elapsed().as_secs_f64();

        let hash_started = Instant::now();
        if verify_sha256 {
            let actual = sha256_hex(bytemuck::cast_slice(values.as_slice()));
            if actual != self.expected_sha256 {
                return Err(ModelBvaError::Sha256Mismatch {
                    path: self.bva_path.clone(),
                    expected: self.expected_sha256.clone(),
                    actual,
                });
            }
        }
        let sha256_seconds = hash_started.elapsed().as_secs_f64();

        let adopt_started = Instant::now();
        let bva = adopt(&self.matrix_shapes, values)?;
        let adopt_seconds = adopt_started.elapsed().as_secs_f64();
        let profile = ModelBvaLoadProfile {
            allocation_seconds,
            read_seconds,
            sha256_seconds,
            adopt_seconds,
            total_seconds: total_started.elapsed().as_secs_f64(),
            bytes_read: self.bva_size_bytes,
            sha256_verified: verify_sha256,
        };
        Ok(LoadedModelBva {
            manifest: self,
            bva,
            profile,
        })
    }
}

/// Stage-level timings for direct BVA loading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelBvaLoadProfile {
    pub allocation_seconds: f64,
    pub read_seconds: f64,
    pub sha256_seconds: f64,
    pub adopt_seconds: f64,
    pub total_seconds: f64,
    pub bytes_read: u64,
    pub sha256_verified: bool,
}

/// A native short-integer model BVA ready for the matrix-collection pipeline.
pub struct LoadedModelBva<Integer: BvaInteger> {
    pub manifest: ModelBvaManifest,
    pub bva: DenseMatrixBva<Integer>,
    pub profile: ModelBvaLoadProfile,
}

pub type LoadedI8ModelBva = LoadedModelBva<i8>;
pub type LoadedI16ModelBva = LoadedModelBva<i16>;

#[derive(Debug)]
pub enum ModelBvaError {
    UnknownPreset(String),
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidManifest {
        field: &'static str,
        message: String,
    },
    FileSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    Sha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    UnsupportedNativeLoad {
        preset: ModelPreset,
        storage_dtype: String,
        bit_width: usize,
        requested_storage_dtype: &'static str,
    },
    Allocation {
        entries: usize,
        message: String,
    },
    MatrixCollection(MatrixCollectionError),
}

impl ModelBvaError {
    fn io(path: &Path, source: io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            source,
        }
    }

    fn invalid(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidManifest {
            field,
            message: message.into(),
        }
    }
}

impl Display for ModelBvaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPreset(value) => write!(
                formatter,
                "unknown model preset {value:?}; expected llama8b-i8, llama8b-i16, or llama70b-i8"
            ),
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Json { path, source } => {
                write!(formatter, "invalid JSON manifest {}: {source}", path.display())
            }
            Self::InvalidManifest { field, message } => {
                write!(formatter, "invalid manifest field {field}: {message}")
            }
            Self::FileSizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "BVA file {} has {actual} bytes; expected {expected}",
                path.display()
            ),
            Self::Sha256Mismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "BVA SHA-256 mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::UnsupportedNativeLoad {
                preset,
                storage_dtype,
                bit_width,
                requested_storage_dtype,
            } => write!(
                formatter,
                "model preset {preset} stores {storage_dtype}/{bit_width}-bit elements; the {requested_storage_dtype} loader does not convert it"
            ),
            Self::Allocation { entries, message } => {
                write!(formatter, "could not allocate {entries} native BVA entries: {message}")
            }
            Self::MatrixCollection(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for ModelBvaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::MatrixCollection(source) => Some(source),
            _ => None,
        }
    }
}

impl From<MatrixCollectionError> for ModelBvaError {
    fn from(source: MatrixCollectionError) -> Self {
        Self::MatrixCollection(source)
    }
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    format: String,
    version: u64,
    #[serde(default)]
    source: Option<RawSource>,
    quantization: RawQuantization,
    layout: RawLayout,
    tensors: Vec<RawTensor>,
    buckets: Vec<RawBucket>,
}

#[derive(Debug, Deserialize)]
struct RawSource {
    model_family: String,
}

#[derive(Debug, Deserialize)]
struct RawQuantization {
    #[serde(default)]
    scheme: Option<String>,
    #[serde(default)]
    native_i8_tensor_count: Option<u64>,
    #[serde(default)]
    converted_float_tensor_count: Option<u64>,
    #[serde(default)]
    bitcast_bf16_tensor_count: Option<u64>,
    #[serde(default)]
    bit_preserving: Option<bool>,
    #[serde(default)]
    numeric_conversion: Option<bool>,
    #[serde(default)]
    source_to_storage: Option<String>,
    #[serde(default)]
    source_storage_encoding: Option<String>,
    #[serde(default)]
    field_interpretation: Option<String>,
    storage_dtype: String,
    storage_encoding: String,
    #[serde(default)]
    storage_item_bytes: Option<u64>,
    bit_width: u64,
}

#[derive(Debug, Deserialize)]
struct RawLayout {
    tensor_count: u64,
    bucket_count: u64,
    physical_entries: u64,
    active_entries: u64,
    logical_entries: u64,
    boundary_padding_entries: u64,
    final_padding_entries: u64,
    splitting_budget: u64,
    bucket_order: String,
    within_bucket_order: String,
    rank1_matrix_mapping: String,
    #[serde(default)]
    element_dtype: Option<String>,
    #[serde(default)]
    element_bytes: Option<u64>,
    bva_path: PathBuf,
    bva_size_bytes: u64,
    bva_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    index: u64,
    #[serde(default)]
    source_dtype: Option<String>,
    #[serde(default)]
    conversion: Option<String>,
    #[serde(default)]
    max_abs: Option<f64>,
    #[serde(default)]
    scale: Option<f64>,
    matrix_shape: [u64; 2],
    element_count: u64,
    bucket_indices: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct RawBucket {
    index: u64,
    tensor_index: u64,
    row_offset: u64,
    row_len: u64,
    physical_row_len: u64,
    col_offset: u64,
    col_len: u64,
    physical_col_len: u64,
    dense_offset: u64,
    source_offset: u64,
    entry_count: u64,
    physical_entry_count: u64,
}

fn resolve_manifest_path(
    preset: ModelPreset,
    model_data_root: &Path,
    manifest_override: Option<&Path>,
) -> PathBuf {
    match manifest_override {
        Some(path) if path.is_absolute() => path.to_owned(),
        Some(path) => model_data_root.join(path),
        None => model_data_root.join(preset.manifest_relative_path()),
    }
}

fn resolve_bva_path(
    manifest_path: &Path,
    recorded_path: &Path,
    bva_override: Option<&Path>,
) -> Result<PathBuf, ModelBvaError> {
    let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    if let Some(path) = bva_override {
        return Ok(if path.is_absolute() {
            path.to_owned()
        } else {
            parent.join(path)
        });
    }
    let basename = recorded_path.file_name().ok_or_else(|| {
        ModelBvaError::invalid(
            "layout.bva_path",
            format!("{} has no file basename", recorded_path.display()),
        )
    })?;
    Ok(parent.join(basename))
}

fn validate_scalar_metadata(preset: ModelPreset, raw: &RawManifest) -> Result<(), ModelBvaError> {
    if !matches!(raw.version, LEGACY_MANIFEST_VERSION | MANIFEST_VERSION) {
        return Err(ModelBvaError::invalid(
            "version",
            format!(
                "expected {LEGACY_MANIFEST_VERSION} or {MANIFEST_VERSION}, got {}",
                raw.version
            ),
        ));
    }
    expect_eq(
        "format",
        raw.format.as_str(),
        format!("parcomm-dense-{}-bva", preset.storage_dtype()).as_str(),
    )?;
    expect_eq(
        "quantization.storage_dtype",
        raw.quantization.storage_dtype.as_str(),
        preset.storage_dtype(),
    )?;
    expect_eq(
        "quantization.bit_width",
        raw.quantization.bit_width,
        preset.bit_width() as u64,
    )?;
    let expected_bytes = (preset.bit_width() / 8) as u64;
    if let Some(item_bytes) = raw.quantization.storage_item_bytes {
        expect_eq(
            "quantization.storage_item_bytes",
            item_bytes,
            expected_bytes,
        )?;
    }
    if let Some(dtype) = &raw.layout.element_dtype {
        expect_eq(
            "layout.element_dtype",
            dtype.as_str(),
            preset.storage_dtype(),
        )?;
    }
    if let Some(element_bytes) = raw.layout.element_bytes {
        expect_eq("layout.element_bytes", element_bytes, expected_bytes)?;
    }
    let valid_encoding = match preset.storage_dtype() {
        "i8" => matches!(
            raw.quantization.storage_encoding.as_str(),
            "signed_twos_complement_byte" | "signed_twos_complement_little_endian"
        ),
        "i16" => raw.quantization.storage_encoding == "signed_twos_complement_little_endian",
        _ => false,
    };
    if !valid_encoding {
        return Err(ModelBvaError::invalid(
            "quantization.storage_encoding",
            format!(
                "unsupported {:?} encoding for {}",
                raw.quantization.storage_encoding,
                preset.storage_dtype()
            ),
        ));
    }
    if preset == ModelPreset::Llama8bI16 {
        validate_bf16_bitcast_metadata(raw)?;
    }
    validate_sha256(&raw.layout.bva_sha256)
}

fn validate_bf16_bitcast_metadata(raw: &RawManifest) -> Result<(), ModelBvaError> {
    expect_eq("version", raw.version, MANIFEST_VERSION)?;
    let source = raw
        .source
        .as_ref()
        .ok_or_else(|| ModelBvaError::invalid("source", "missing BF16 source metadata"))?;
    expect_eq(
        "source.model_family",
        source.model_family.as_str(),
        "llama-bf16",
    )?;
    expect_eq(
        "quantization.scheme",
        raw.quantization.scheme.as_deref(),
        Some(BF16_BITCAST_SCHEME),
    )?;
    expect_eq(
        "quantization.native_i8_tensor_count",
        raw.quantization.native_i8_tensor_count,
        Some(0),
    )?;
    expect_eq(
        "quantization.converted_float_tensor_count",
        raw.quantization.converted_float_tensor_count,
        Some(0),
    )?;
    expect_eq(
        "quantization.bitcast_bf16_tensor_count",
        raw.quantization.bitcast_bf16_tensor_count,
        Some(raw.tensors.len() as u64),
    )?;
    expect_eq(
        "quantization.bit_preserving",
        raw.quantization.bit_preserving,
        Some(true),
    )?;
    expect_eq(
        "quantization.numeric_conversion",
        raw.quantization.numeric_conversion,
        Some(false),
    )?;
    expect_eq(
        "quantization.source_to_storage",
        raw.quantization.source_to_storage.as_deref(),
        Some(BF16_SOURCE_TO_STORAGE),
    )?;
    expect_eq(
        "quantization.source_storage_encoding",
        raw.quantization.source_storage_encoding.as_deref(),
        Some("bf16_bit_pattern_little_endian"),
    )?;
    expect_eq(
        "quantization.field_interpretation",
        raw.quantization.field_interpretation.as_deref(),
        Some("signed_twos_complement_i16"),
    )?;
    for (index, tensor) in raw.tensors.iter().enumerate() {
        if tensor.source_dtype.as_deref() != Some("BF16")
            || tensor.conversion.as_deref() != Some(BF16_BITCAST_CONVERSION)
            || tensor.max_abs.is_some()
            || tensor.scale.is_some()
        {
            return Err(ModelBvaError::invalid(
                "tensors[]",
                format!("tensor {index} does not use the scale-free BF16 bitcast"),
            ));
        }
    }
    Ok(())
}

fn validate_tensor_metadata(raw: &[RawTensor]) -> Result<Vec<(usize, usize)>, ModelBvaError> {
    if raw.is_empty() {
        return Err(ModelBvaError::invalid(
            "tensors",
            "the matrix collection is empty",
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(index, tensor)| {
            expect_eq("tensors[].index", tensor.index, index as u64)?;
            let rows = usize_field("tensors[].matrix_shape[0]", tensor.matrix_shape[0])?;
            let cols = usize_field("tensors[].matrix_shape[1]", tensor.matrix_shape[1])?;
            if rows == 0 || cols == 0 {
                return Err(ModelBvaError::invalid(
                    "tensors[].matrix_shape",
                    format!("tensor {index} has zero shape ({rows}, {cols})"),
                ));
            }
            let entries = rows.checked_mul(cols).ok_or_else(|| {
                ModelBvaError::invalid(
                    "tensors[].matrix_shape",
                    format!("tensor {index} shape ({rows}, {cols}) overflows usize"),
                )
            })?;
            expect_eq(
                "tensors[].element_count",
                tensor.element_count,
                entries as u64,
            )?;
            Ok((rows, cols))
        })
        .collect()
}

fn validate_layout(
    raw: &RawLayout,
    descriptor: &DenseBvaDescriptor,
    tensor_count: usize,
    preset: ModelPreset,
) -> Result<(), ModelBvaError> {
    expect_eq("layout.tensor_count", raw.tensor_count, tensor_count as u64)?;
    expect_eq(
        "layout.bucket_count",
        raw.bucket_count,
        descriptor.bucket_count() as u64,
    )?;
    expect_eq(
        "layout.physical_entries",
        raw.physical_entries,
        descriptor.physical_len() as u64,
    )?;
    expect_eq(
        "layout.active_entries",
        raw.active_entries,
        descriptor.layout().active_len() as u64,
    )?;
    expect_eq(
        "layout.logical_entries",
        raw.logical_entries,
        descriptor.logical_len() as u64,
    )?;
    expect_eq(
        "layout.boundary_padding_entries",
        raw.boundary_padding_entries,
        descriptor.conceptual_boundary_zeros() as u64,
    )?;
    expect_eq(
        "layout.final_padding_entries",
        raw.final_padding_entries,
        descriptor.conceptual_final_zeros() as u64,
    )?;
    expect_eq(
        "layout.splitting_budget",
        raw.splitting_budget,
        DenseBvaDescriptor::SPLITTING_BUDGET as u64,
    )?;
    expect_eq(
        "layout.bucket_order",
        raw.bucket_order.as_str(),
        BUCKET_ORDER,
    )?;
    expect_eq(
        "layout.within_bucket_order",
        raw.within_bucket_order.as_str(),
        WITHIN_BUCKET_ORDER,
    )?;
    expect_eq(
        "layout.rank1_matrix_mapping",
        raw.rank1_matrix_mapping.as_str(),
        RANK1_MATRIX_MAPPING,
    )?;
    let expected_size = raw
        .logical_entries
        .checked_mul((preset.bit_width() / 8) as u64)
        .ok_or_else(|| ModelBvaError::invalid("layout.bva_size_bytes", "size overflow"))?;
    expect_eq("layout.bva_size_bytes", raw.bva_size_bytes, expected_size)
}

fn validate_bucket_metadata(
    raw: &RawManifest,
    descriptor: &DenseBvaDescriptor,
) -> Result<(), ModelBvaError> {
    expect_eq(
        "buckets.length",
        raw.buckets.len(),
        descriptor.bucket_count(),
    )?;
    let mut expected_tensor_buckets = vec![Vec::<u64>::new(); raw.tensors.len()];
    for (index, (bucket, expected)) in raw.buckets.iter().zip(descriptor.buckets()).enumerate() {
        let prefix = "buckets[]";
        expect_eq(prefix, bucket.index, index as u64)?;
        expect_eq(prefix, bucket.tensor_index, expected.matrix() as u64)?;
        expect_eq(prefix, bucket.row_offset, expected.row_offset() as u64)?;
        expect_eq(prefix, bucket.row_len, expected.row_len() as u64)?;
        expect_eq(
            prefix,
            bucket.physical_row_len,
            expected.physical_row_len() as u64,
        )?;
        expect_eq(prefix, bucket.col_offset, expected.col_offset() as u64)?;
        expect_eq(prefix, bucket.col_len, expected.col_len() as u64)?;
        expect_eq(
            prefix,
            bucket.physical_col_len,
            expected.physical_col_len() as u64,
        )?;
        expect_eq(prefix, bucket.dense_offset, expected.dense_offset() as u64)?;
        expect_eq(
            prefix,
            bucket.source_offset,
            expected.source_offset() as u64,
        )?;
        expect_eq(prefix, bucket.entry_count, expected.entry_count() as u64)?;
        expect_eq(
            prefix,
            bucket.physical_entry_count,
            expected.physical_entry_count() as u64,
        )?;
        expected_tensor_buckets[expected.matrix()].push(index as u64);
    }
    for (index, (tensor, expected)) in raw.tensors.iter().zip(expected_tensor_buckets).enumerate() {
        if tensor.bucket_indices != expected {
            return Err(ModelBvaError::invalid(
                "tensors[].bucket_indices",
                format!(
                    "tensor {index} has {:?}; expected {:?}",
                    tensor.bucket_indices, expected
                ),
            ));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), ModelBvaError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ModelBvaError::invalid(
            "layout.bva_sha256",
            "expected 64 hexadecimal digits",
        ));
    }
    Ok(())
}

fn usize_field(field: &'static str, value: u64) -> Result<usize, ModelBvaError> {
    usize::try_from(value)
        .map_err(|_| ModelBvaError::invalid(field, format!("{value} does not fit usize")))
}

fn expect_eq<T>(field: &'static str, actual: T, expected: T) -> Result<(), ModelBvaError>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual == expected {
        Ok(())
    } else {
        Err(ModelBvaError::invalid(
            field,
            format!("expected {expected:?}, got {actual:?}"),
        ))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing into a String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::{json, Value};
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn preset_parser_accepts_benchmark_names() {
        assert_eq!(
            "llama8b-i8".parse::<ModelPreset>().unwrap(),
            ModelPreset::Llama8bI8
        );
        assert_eq!(
            "llama8b-i16".parse::<ModelPreset>().unwrap(),
            ModelPreset::Llama8bI16
        );
        assert_eq!(
            "llama70b-i8".parse::<ModelPreset>().unwrap(),
            ModelPreset::Llama70bI8
        );
    }

    #[test]
    fn valid_i8_fixture_uses_manifest_sibling_and_loads_without_conversion() {
        let fixture = Fixture::new(ModelPreset::Llama8bI8);
        let manifest = ModelBvaManifest::load(
            ModelPreset::Llama8bI8,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap();
        assert_eq!(manifest.bva_path(), fixture.bva_path);
        assert_eq!(manifest.matrix_shapes(), &[(3, 2)]);

        let loaded = manifest.load_i8(true).unwrap();
        assert_eq!(loaded.bva.values(), fixture.values);
        assert!(loaded.profile.sha256_verified);
        assert_eq!(loaded.profile.bytes_read, fixture.values.len() as u64);
    }

    #[test]
    fn altered_bucket_metadata_is_rejected() {
        let fixture = Fixture::new(ModelPreset::Llama8bI8);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest_path).unwrap()).unwrap();
        manifest["buckets"][0]["dense_offset"] = json!(1);
        fs::write(
            &fixture.manifest_path,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = ModelBvaManifest::load(
            ModelPreset::Llama8bI8,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("buckets[]"));
    }

    #[test]
    fn i16_fixture_is_validated_but_never_converted_by_i8_loader() {
        let fixture = Fixture::new(ModelPreset::Llama8bI16);
        let manifest = ModelBvaManifest::load(
            ModelPreset::Llama8bI16,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap();
        let error = match manifest.clone().load_i8(false) {
            Err(error) => error,
            Ok(_) => panic!("the i8 loader must not convert an i16 BVA"),
        };
        assert!(matches!(error, ModelBvaError::UnsupportedNativeLoad { .. }));
        let loaded = manifest.load_i16(true).unwrap();
        let expected = fixture
            .values
            .iter()
            .copied()
            .map(i16::from)
            .collect::<Vec<_>>();
        assert_eq!(loaded.bva.values(), expected);
    }

    #[test]
    fn legacy_i16_manifest_version_is_rejected() {
        let fixture = Fixture::new(ModelPreset::Llama8bI16);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest_path).unwrap()).unwrap();
        manifest["version"] = json!(1);
        fs::write(
            &fixture.manifest_path,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = ModelBvaManifest::load(
            ModelPreset::Llama8bI16,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn numeric_i16_manifest_scheme_is_rejected() {
        let fixture = Fixture::new(ModelPreset::Llama8bI16);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest_path).unwrap()).unwrap();
        manifest["quantization"]["scheme"] = json!("symmetric_per_tensor");
        fs::write(
            &fixture.manifest_path,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = ModelBvaManifest::load(
            ModelPreset::Llama8bI16,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("quantization.scheme"));
    }

    #[test]
    fn version_two_i8_manifest_remains_supported() {
        let fixture = Fixture::new(ModelPreset::Llama8bI8);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&fixture.manifest_path).unwrap()).unwrap();
        manifest["version"] = json!(2);
        fs::write(
            &fixture.manifest_path,
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        ModelBvaManifest::load(
            ModelPreset::Llama8bI8,
            fixture.directory.path(),
            Some(&fixture.manifest_path),
            None,
        )
        .unwrap();
    }

    struct Fixture {
        directory: TempDir,
        manifest_path: PathBuf,
        bva_path: PathBuf,
        values: Vec<i8>,
    }

    impl Fixture {
        fn new(preset: ModelPreset) -> Self {
            let directory = TempDir::new().unwrap();
            let manifest_path = directory.path().join("fixture.manifest.json");
            let extension = preset.storage_dtype();
            let bva_path = directory.path().join(format!("fixture.bva.{extension}"));
            let shapes = [(3, 2)];
            let i8_bva =
                DenseMatrixBva::from_fn(&shapes, |_, row, col| (row * shapes[0].1 + col + 1) as i8)
                    .unwrap();
            let values = i8_bva.values().to_vec();
            let descriptor = i8_bva.descriptor();
            let element_bytes = preset.bit_width() / 8;
            let mut file_bytes = Vec::with_capacity(values.len() * element_bytes);
            if element_bytes == 1 {
                file_bytes.extend_from_slice(bytemuck::cast_slice(values.as_slice()));
            } else {
                for &value in &values {
                    file_bytes.extend_from_slice(&(value as i16).to_le_bytes());
                }
            }
            fs::write(&bva_path, &file_bytes).unwrap();
            let hash = sha256_hex(&file_bytes);
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
            let tensor_bucket_indices = descriptor
                .buckets()
                .iter()
                .enumerate()
                .filter_map(|(index, bucket)| (bucket.matrix() == 0).then_some(index))
                .collect::<Vec<_>>();
            let encoding = if element_bytes == 1 {
                "signed_twos_complement_byte"
            } else {
                "signed_twos_complement_little_endian"
            };
            let mut manifest = json!({
                "format": format!("parcomm-dense-{}-bva", preset.storage_dtype()),
                "version": if preset == ModelPreset::Llama8bI16 { 2 } else { 1 },
                "quantization": {
                    "storage_dtype": preset.storage_dtype(),
                    "storage_encoding": encoding,
                    "storage_item_bytes": element_bytes,
                    "bit_width": preset.bit_width(),
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
                    "within_bucket_order": WITHIN_BUCKET_ORDER,
                    "rank1_matrix_mapping": RANK1_MATRIX_MAPPING,
                    "element_dtype": preset.storage_dtype(),
                    "element_bytes": element_bytes,
                    "bva_path": format!("/artifact-builder/elsewhere/fixture.bva.{extension}"),
                    "bva_size_bytes": file_bytes.len(),
                    "bva_sha256": hash,
                },
                "tensors": [{
                    "index": 0,
                    "matrix_shape": [3, 2],
                    "element_count": 6,
                    "bucket_indices": tensor_bucket_indices,
                }],
                "buckets": buckets,
            });
            if preset == ModelPreset::Llama8bI16 {
                manifest["source"] = json!({ "model_family": "llama-bf16" });
                manifest["quantization"]["scheme"] = json!(BF16_BITCAST_SCHEME);
                manifest["quantization"]["native_i8_tensor_count"] = json!(0);
                manifest["quantization"]["converted_float_tensor_count"] = json!(0);
                manifest["quantization"]["bitcast_bf16_tensor_count"] = json!(1);
                manifest["quantization"]["bit_preserving"] = json!(true);
                manifest["quantization"]["numeric_conversion"] = json!(false);
                manifest["quantization"]["source_to_storage"] = json!(BF16_SOURCE_TO_STORAGE);
                manifest["quantization"]["source_storage_encoding"] =
                    json!("bf16_bit_pattern_little_endian");
                manifest["quantization"]["field_interpretation"] =
                    json!("signed_twos_complement_i16");
                manifest["tensors"][0]["source_dtype"] = json!("BF16");
                manifest["tensors"][0]["conversion"] = json!(BF16_BITCAST_CONVERSION);
                manifest["tensors"][0]["max_abs"] = Value::Null;
                manifest["tensors"][0]["scale"] = Value::Null;
            }
            fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            Self {
                directory,
                manifest_path,
                bva_path,
                values,
            }
        }
    }
}
