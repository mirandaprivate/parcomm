# Attribution and licenses

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
