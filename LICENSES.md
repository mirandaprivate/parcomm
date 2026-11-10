# Attribution and licenses

The CPU FlashProve benchmark includes upstream shared blocks under Apache-2.0;
its LICENSE, NOTICE and THIRD_PARTY.md are retained in
`benchmarks/flashprove/upstream/`. The Hyrax snapshot retains its declared
MIT/Apache-2.0 license and original source attribution. These benchmark additions
do not relicense upstream code.

Preserve the license files bundled with each dependency. Jolt's server snapshot
includes LICENSE-MIT and LICENSE-APACHE; the patched ark-ec crate includes both
licenses and PATCHES.md. The QAFold backend includes LICENSE_MIT and its source
provenance. Cargo dependencies retain their original license and copyright files.

The parent ParComm workspace declares MIT/Apache-2.0 in Cargo.toml. This packaging
step does not assign new authorship or relicense any component. Model weights
are excluded and must be obtained separately under the model owner's terms.

This package has not been anonymized: source attribution and original reference
logs are retained. Apply the target venue's submission policy before uploading
to an anonymous review system.
