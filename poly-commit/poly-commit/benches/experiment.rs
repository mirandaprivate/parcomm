use std::fs::{self, File};
use std::io;
use std::os::unix::io::AsRawFd;
use libc;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use ark_bls12_381::Bls12_381;
use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
};
use ark_ec::pairing::Pairing;
use ark_pcs_bench_templates::*;
use ark_poly::DenseUVPolynomial;
use ark_poly::univariate::DensePolynomial as DenseUnivariatePoly;
use ark_poly::DenseMultilinearExtension;

use ark_poly_commit::{
    ipa_pc::InnerProductArgPC,
    marlin_pc::MarlinKZG10,
    smart_pc::experiment_smart_template,
    sonic_pc::SonicKZG10
};
use ark_poly_commit::linear_codes::{
    LinearCodePCS,
    MultilinearBrakedown,
    MultilinearLigero
};
use ark_poly_commit::hyrax::HyraxPC;
use blake2::Blake2s256;

use ark_ff::PrimeField;

use rand_chacha::ChaCha20Rng;

const MIN_NUM_VARS: usize = 22;
const MAX_NUM_VARS: usize = 23;
const BIT_WIDTH: usize = 8;

#[allow(dead_code)]
fn main() -> io::Result<()> {
    println!("Running experiments for 2^{} to 2^{} variables",
        MIN_NUM_VARS, MAX_NUM_VARS - 1);

    let max_memory = Arc::new(Mutex::new(0));

    let max_memory_clone = Arc::clone(&max_memory);
    let monitor_thread = thread::spawn(move || {
        loop {
            if let Some(memory_usage) = get_memory_usage() {
                let mut max = max_memory_clone.lock().unwrap();
                if memory_usage > *max {
                    *max = memory_usage;
                }
            } 
            thread::sleep(Duration::from_secs(1)); // 暂停 1 秒
        }
    });

    // let log_file = File::create("experiment.log")?;
    // // experiment_lookups();

    // println!("Redirecting stdout to experiment.log...");


    // unsafe {
    //     let stdout_fd = io::stdout().as_raw_fd();
    //     let _ = libc::dup2(log_file.as_raw_fd(), stdout_fd);
    // }

    println!("*********************************************");
    println!("Running experiments for 2^{} to 2^{} variables",
        MIN_NUM_VARS, MAX_NUM_VARS - 1);
    
    for num_vars in (MIN_NUM_VARS..MAX_NUM_VARS).step_by(2) {
        
        println!("\n\n***********************************************");
        println!("Running experiments for 2e{} variables", num_vars);

        // experiment_smart(num_vars);


        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        // experiment_brakedown(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        // experiment_hyrax(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        // experiment_lingero(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);



        // experiment_kzg(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        // experiment_marlin(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        // experiment_sonic(num_vars);

        // let mut max = max_memory.lock().unwrap();
        // println!("\n ****Max RAM: {} KB\n", *max);
        // println!("\n\n***********************************************");
        // *max = 0;
        // std::mem::drop(max);


        experiment_ipa(num_vars);

        let max = max_memory.lock().unwrap();
        println!("\n ****Max RAM: {} KB\n", *max);
        println!("\n\n***********************************************");
    }

    Ok(())
}

fn get_memory_usage() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse().ok());
        }
    }
    None
}


fn experiment_smart(num_vars: usize) {
    type E = Bls12_381;
    // ark_poly_commit::smart_pc::test_smart::<E>();

    println!("\nSMART-PC on BLS12-381:");
    experiment_smart_template::<E>(num_vars);
}

fn experiment_brakedown(num_vars: usize) {
    struct MerkleTreeParams;
    type LeafH = LeafIdentityHasher;
    type CompressH = Sha256;
    impl Config for MerkleTreeParams {
        type Leaf = Vec<u8>;

        type LeafDigest = <LeafH as CRHScheme>::Output;
        type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
        type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

        type LeafHash = LeafH;
        type TwoToOneHash = CompressH;
    }

    pub type MLE<F> = DenseMultilinearExtension<F>;
    type MTConfig = MerkleTreeParams;
    type ColHasher<F> = FieldToBytesColHasher<F, Blake2s256>;
    type Brakedown<F> = LinearCodePCS<
        MultilinearBrakedown<F, MTConfig, MLE<F>, ColHasher<F>>,
        F,
        MLE<F>,
        MTConfig,
        ColHasher<F>,
    >;

    fn rand_poly_brakedown_ml<F: PrimeField>(
        num_vars: usize,
        rng: &mut ChaCha20Rng,
    ) -> DenseMultilinearExtension<F> {
        // DenseMultilinearExtension::rand(num_vars, rng)

        use rand::Rng;

        // Generate a random polynomial with random short-bit-width coefficients
        let n_coeffs = 1 << num_vars;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..n_coeffs).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseMultilinearExtension::from_evaluations_slice(
            num_vars,&coeffs[0..n_coeffs])
    }
    
    fn rand_point_brakedown_ml<F: PrimeField>(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<F> {
        (0..num_vars).map(|_| F::rand(rng)).collect()
    }

    println!("\nBrakedown on BN254:");
    experiment::<ark_bn254::Fr, MLE<ark_bn254::Fr>, Brakedown<ark_bn254::Fr>>(
        num_vars,
        rand_poly_brakedown_ml,
        rand_point_brakedown_ml
    );
}

fn experiment_hyrax(num_vars: usize) {
    type Hyrax254 = HyraxPC<ark_bn254::G1Affine, DenseMultilinearExtension<ark_bn254::Fr>>;

    fn rand_poly_hyrax<F: PrimeField>(
        num_vars: usize,
        rng: &mut ChaCha20Rng,
    ) -> DenseMultilinearExtension<F> {
        // DenseMultilinearExtension::rand(num_vars, rng)

        use rand::Rng;
        // Generate a random polynomial with random short-bit-width coefficients
        let n_coeffs = 1 << num_vars;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..n_coeffs).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseMultilinearExtension::from_evaluations_slice(
            num_vars,&coeffs[0..n_coeffs])
    }

    fn rand_point_hyrax<F: PrimeField>(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<F> {
        (0..num_vars).map(|_| F::rand(rng)).collect()
    }

    println!("\nHyrax on BN254:");
    experiment::<ark_bn254::Fr, DenseMultilinearExtension<ark_bn254::Fr>, Hyrax254>(
        num_vars,
        rand_poly_hyrax,
        rand_point_hyrax
    );
}

fn experiment_lingero(num_vars: usize) {
    struct MerkleTreeParams;
    type LeafH = LeafIdentityHasher;
    type CompressH = Sha256;
    impl Config for MerkleTreeParams {
        type Leaf = Vec<u8>;

        type LeafDigest = <LeafH as CRHScheme>::Output;
        type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
        type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

        type LeafHash = LeafH;
        type TwoToOneHash = CompressH;
    }

    pub type MLE<F> = DenseMultilinearExtension<F>;
    type MTConfig = MerkleTreeParams;
    type ColHasher<F> = FieldToBytesColHasher<F, Blake2s256>;
    type Ligero<F> = LinearCodePCS<
        MultilinearLigero<F, MTConfig, MLE<F>, ColHasher<F>>,
        F,
        MLE<F>,
        MTConfig,
        ColHasher<F>,
    >;

    fn rand_poly_ligero_ml<F: PrimeField>(
        num_vars: usize,
        rng: &mut ChaCha20Rng,
    ) -> DenseMultilinearExtension<F> {
        // DenseMultilinearExtension::rand(num_vars, rng)

        // Generate a random polynomial with random short-bit-width coefficients
        use rand::Rng;
        let n_coeffs = 1 << num_vars;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..n_coeffs).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseMultilinearExtension::from_evaluations_slice(
            num_vars,&coeffs[0..n_coeffs])
    }

    fn rand_point_ligero_ml<F: PrimeField>(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<F> {
        (0..num_vars).map(|_| F::rand(rng)).collect()
    }

    println!("\nLingero on BN254:");
    experiment::<ark_bn254::Fr, MLE<ark_bn254::Fr>, Ligero<ark_bn254::Fr>>(
        num_vars,
        rand_poly_ligero_ml,
        rand_point_ligero_ml
    );

}


fn experiment_marlin(num_vars: usize) {
    #![allow(non_camel_case_types)]
    use ark_poly::univariate::DensePolynomial as DensePoly;

    type UniPoly_381 = DensePoly<<Bls12_381 as Pairing>::ScalarField>;

    type PC<E, P> = MarlinKZG10<E, P>;
    type PC_Bls12_381 = PC<Bls12_381, UniPoly_381>;

    fn rand_poly_marlin<F: PrimeField>(
        degree: usize,
        rng: &mut ChaCha20Rng,
    ) -> DensePoly<F> {
        // DenseUVPolynomial::rand(degree, rng)

        // Generate a random polynomial with random short-bit-width coefficients
        use rand::Rng;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..degree).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseUVPolynomial::from_coefficients_vec(coeffs)        

    }

    fn rand_point_marlin<F: PrimeField>(_: usize, rng: &mut ChaCha20Rng) -> F {
        F::rand(rng)
    }

    println!("\nMarlin-PC on BLS12-381:");
    experiment::<_,_, PC_Bls12_381>(
        2_usize.pow(num_vars as u32),
        rand_poly_marlin,
        rand_point_marlin // Add the type parameter here
    );
}

fn experiment_sonic(num_vars: usize) {
    #![allow(non_camel_case_types)]
    use ark_poly::univariate::DensePolynomial as DensePoly;

    type UniPoly_381 = DensePoly<<Bls12_381 as Pairing>::ScalarField>;

    type PC<E, P> = SonicKZG10<E, P>;
    type PC_Bls12_381 = PC<Bls12_381, UniPoly_381>;

    fn rand_poly_sonic<F: PrimeField>(
        degree: usize,
        rng: &mut ChaCha20Rng,
    ) -> DensePoly<F> {
        // DenseUVPolynomial::rand(degree, rng)

        // Generate a random polynomial with random short-bit-width coefficients
        use rand::Rng;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..degree).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseUVPolynomial::from_coefficients_vec(coeffs)        

    }

    fn rand_point_sonic<F: PrimeField>(_: usize, rng: &mut ChaCha20Rng) -> F {
        F::rand(rng)
    }

    println!("\nSonic on BLS12-381:");
    experiment::<_,_, PC_Bls12_381>(
        2_usize.pow(num_vars as u32),
        rand_poly_sonic,
        rand_point_sonic // Add the type parameter here
    );

}

fn experiment_ipa(num_vars: usize) {

    type UniPoly = DenseUnivariatePoly<ark_ed_on_bls12_381::Fr>;
    // IPA_PC over the JubJub curve with Blake2s as the hash function
    #[allow(non_camel_case_types)]
    type IPA_JubJub = InnerProductArgPC<ark_ed_on_bls12_381::EdwardsAffine, Blake2s256, UniPoly>;

    fn rand_poly_ipa_pc<F: PrimeField>(degree: usize, rng: &mut ChaCha20Rng) -> DenseUnivariatePoly<F> {
        // DenseUnivariatePoly::rand(degree, rng)

        // Generate a random polynomial with random short-bit-width coefficients
        use rand::Rng;
        // Test short bit-width polynomials
        let coeffs: Vec<F> = (0..degree).into_iter().map(|_|{
            F::from( rng.gen_range(0..(1<<BIT_WIDTH - 2)) as i64 - (1<<BIT_WIDTH-1) )
        }).collect();
        DenseUnivariatePoly::from_coefficients_vec(coeffs)  
    }

    fn rand_point_ipa_pc<F: PrimeField>(_: usize, rng: &mut ChaCha20Rng) -> F {
        F::rand(rng)
    }

    println!("\nIPA on BLS12-381:");
    experiment::<ark_ed_on_bls12_381::Fr, UniPoly, IPA_JubJub>(
        2_usize.pow(num_vars as u32),
        rand_poly_ipa_pc,
        rand_point_ipa_pc
    );
    
}

fn experiment_kzg(num_vars: usize) {
    #![allow(non_camel_case_types)]
    use ark_poly::univariate::DensePolynomial as DensePoly;

    type UniPoly_381 = DensePoly<<Bls12_381 as Pairing>::ScalarField>;

    println!("\nKZG on BLS12-381:");
    ark_poly_commit::kzg10::experiment_kzg_template::<Bls12_381, UniPoly_381>(
    2_usize.pow(num_vars as u32),
    );
    
}
