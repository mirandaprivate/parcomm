/// Define sparse matrix operations
///
use ark_ff::PrimeField;

use rayon::prelude::*;
use std::f64::consts::E as Exponential;

use super::linear;
use super::sparse::{DenseBlockMat, DenseMatCM, MatOps, RotationMatIndexFormat};

/// Mock data for a Mat-Circ with
/// M numbers of nxn matrix multiplication
/// short bit caped at k-2k bit
///
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct MockMultiMuls<F>
where
    F: PrimeField + From<i32>,
{
    pub num_mat: usize,
    pub n: usize,
    pub k: usize,
    pub al_mat: RotationMatIndexFormat<i32, F>,
    pub ar_mat: RotationMatIndexFormat<i32, F>,
    pub ao_mat: RotationMatIndexFormat<i32, F>,
    pub block_w: DenseBlockMat<i32, F>,
    pub block_mat_l: DenseBlockMat<i32, F>,
    pub block_mat_r: DenseBlockMat<i32, F>,
    pub block_mat_o: DenseBlockMat<i32, F>,
    pub block_mat_o_full: DenseBlockMat<i32, F>,
    /// Such that 2^k block_mat_o +
    /// booleas_low + 2^{2k} booleas_up
    /// = sing * block_mat_l * block_mat_r
    pub sign_o: DenseBlockMat<bool, F>,
    pub abs_o: DenseBlockMat<i32, F>,
    pub abs_o_res: DenseBlockMat<i32, F>,
    pub abs_o_trailing: DenseBlockMat<i32, F>,
    pub booleans_o: Vec<DenseBlockMat<bool, F>>,
    pub booleans_o_res: Vec<DenseBlockMat<bool, F>>,
    pub booleans_o_trailing: Vec<DenseBlockMat<bool, F>>,
}

impl<F> MockMultiMuls<F>
where
    F: PrimeField + From<i32>,
{
    pub fn new(num_mat: usize, n: usize, k: usize) -> Self {
        let al_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let ar_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let ao_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let block_w = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_l = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_r = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_o = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_o_full = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let sign_o = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
        let abs_o = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let abs_o_res = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let abs_o_trailing = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let booleans_o = Vec::new();
        let booleans_o_res = Vec::new();
        let booleans_o_trailing = Vec::new();

        if (1 << 2 * k) * n > 1 << 31 {
            panic!("The size of the matrix is too large for i32");
        }

        Self {
            num_mat,
            n,
            k,
            al_mat,
            ar_mat,
            ao_mat,
            block_w,
            block_mat_l,
            block_mat_r,
            block_mat_o,
            block_mat_o_full,
            sign_o,
            abs_o,
            abs_o_res,
            abs_o_trailing,
            booleans_o,
            booleans_o_res,
            booleans_o_trailing,
        }
    }

    pub fn gen_mock(&mut self) {
        let k = self.k;

        let n = self.n;
        let num_mat = self.num_mat;

        let log_n = n.ilog2() as usize;
        let max_bit = k + log_n;

        let mut l_blocks = Vec::new();
        let mut r_blocks = Vec::new();
        let mut o_blocks = Vec::new();
        let mut w_blocks = Vec::new();
        let mut sign_blocks = Vec::new();
        let mut abs_o_blocks = Vec::new();
        let mut abs_o_res_blocks = Vec::new();
        let mut abs_o_trailing = Vec::new();
        let mut o_blocks_full = Vec::new();

        let mut booleans_o_blocks = Vec::new();
        let mut booleans_o_res_blocks = Vec::new();
        let mut booleans_o_trailing = Vec::new();

        for _ in 0..k {
            booleans_o_blocks.push(Vec::new());
        }
        for _ in 0..max_bit {
            booleans_o_res_blocks.push(Vec::new());
        }
        for _ in 0..k {
            booleans_o_trailing.push(Vec::new());
        }

        let mut l_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
        l_mat.gen_rand(self.k);

        let mut data_l = l_mat.data.clone();

        let mut r_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
        r_mat.gen_rand(self.k);

        let mut data_r = r_mat.data.clone();
        let mut data_o;

        let data_o_full = linear::mat_mul_shortint::<i32>(&data_l, &data_r);

        let mut o_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
        o_mat.set_data(data_o_full.clone());

        let sign_o_mat_data = data_o_full
            .iter()
            .map(|col| col.par_iter().map(|&x| x >= 0).collect::<Vec<bool>>())
            .collect::<Vec<Vec<bool>>>();

        let mut sign_o_mat = DenseMatCM::<bool, F>::new(n, n);
        sign_o_mat.set_data(sign_o_mat_data);

        let clamp_mat = o_mat.clamp(k);

        data_o = clamp_mat.data.clone();

        let (_, abs_o_mat_clamp, boolean_o_mats_clamp) = clamp_mat.bit_decomposition(k);

        let factor = 1 << k;

        let res_mat_data = (0..n)
            .into_iter()
            .map(|j| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        if sign_o_mat.data[j][i] {
                            o_mat.data[j][i] / factor - abs_o_mat_clamp.data[j][i]
                        } else {
                            abs_o_mat_clamp.data[j][i] + o_mat.data[j][i] / factor
                        }
                    })
                    .collect::<Vec<i32>>()
            })
            .collect::<Vec<Vec<i32>>>();

        let trailing_mat_data = (0..n)
            .into_iter()
            .map(|j| {
                (0..n)
                    .into_par_iter()
                    .map(|i| o_mat.data[j][i] % factor)
                    .collect::<Vec<i32>>()
            })
            .collect::<Vec<Vec<i32>>>();

        let mut res_mat = DenseMatCM::<i32, F>::new(n, n);
        res_mat.set_data(res_mat_data);

        let mut trailing_mat = DenseMatCM::<i32, F>::new(n, n);
        trailing_mat.set_data(trailing_mat_data);

        let (_, abs_o_mat_res, boolean_o_mats_res) = res_mat.bit_decomposition(max_bit);

        let (_, abs_trailing_mat, boolean_o_mat_trailing) = trailing_mat.bit_decomposition(k);

        l_blocks.push(data_l.clone());
        r_blocks.push(data_r.clone());
        o_blocks.push(data_o.clone());
        o_blocks_full.push(data_o_full.clone());

        w_blocks.push(data_l.clone());
        w_blocks.push(data_r.clone());
        w_blocks.push(data_o.clone());

        sign_blocks.push(sign_o_mat.data.clone());
        abs_o_blocks.push(abs_o_mat_clamp.data.clone());
        abs_o_res_blocks.push(abs_o_mat_res.data.clone());
        abs_o_trailing.push(abs_trailing_mat.data.clone());

        for i in 0..k {
            booleans_o_blocks[i].push(boolean_o_mats_clamp[i].data.clone());
        }

        for i in 0..max_bit {
            booleans_o_res_blocks[i].push(boolean_o_mats_res[i].data.clone());
        }

        for i in 0..k {
            booleans_o_trailing[i].push(boolean_o_mat_trailing[i].data.clone());
        }

        for _ in 1..(self.num_mat - 2) {
            let mut l_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
            l_mat.set_data(data_r.clone());

            data_l = data_r;

            let mut r_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
            r_mat.set_data(data_o.clone());

            data_r = data_o;

            let data_o_full = linear::mat_mul_shortint::<i32>(&data_l, &data_r);

            let mut o_mat = DenseMatCM::<i32, F>::new(self.n, self.n);
            o_mat.set_data(data_o_full.clone());

            let sign_o_mat_data = data_o_full
                .iter()
                .map(|col| col.par_iter().map(|&x| x >= 0).collect::<Vec<bool>>())
                .collect::<Vec<Vec<bool>>>();

            let mut sign_o_mat = DenseMatCM::<bool, F>::new(n, n);
            sign_o_mat.set_data(sign_o_mat_data);

            let clamp_mat = o_mat.clamp(k);

            data_o = clamp_mat.data.clone();

            let (_, abs_o_mat_clamp, boolean_o_mats_clamp) = clamp_mat.bit_decomposition(k);

            let res_mat_data = (0..n)
                .into_iter()
                .map(|j| {
                    (0..n)
                        .into_par_iter()
                        .map(|i| {
                            if sign_o_mat.data[j][i] {
                                o_mat.data[j][i] / factor - abs_o_mat_clamp.data[j][i]
                            } else {
                                abs_o_mat_clamp.data[j][i] + o_mat.data[j][i] / factor
                            }
                        })
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>();

            let trailing_mat_data = (0..n)
                .into_iter()
                .map(|j| {
                    (0..n)
                        .into_par_iter()
                        .map(|i| o_mat.data[j][i] % factor)
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>();

            let mut res_mat = DenseMatCM::<i32, F>::new(n, n);
            res_mat.set_data(res_mat_data);

            let mut trailing_mat = DenseMatCM::<i32, F>::new(n, n);
            trailing_mat.set_data(trailing_mat_data);

            let (_, abs_o_mat_res, boolean_o_mats_res) = res_mat.bit_decomposition(max_bit);

            let (_, abs_trailing_mat, boolean_o_mat_trailing) = trailing_mat.bit_decomposition(k);

            l_blocks.push(data_l.clone());
            r_blocks.push(data_r.clone());
            o_blocks.push(data_o.clone());
            w_blocks.push(data_o.clone());
            o_blocks_full.push(data_o_full.clone());

            sign_blocks.push(sign_o_mat.data.clone());
            abs_o_blocks.push(abs_o_mat_clamp.data.clone());
            abs_o_res_blocks.push(abs_o_mat_res.data.clone());
            abs_o_trailing.push(abs_trailing_mat.data.clone());

            for i in 0..k {
                booleans_o_blocks[i].push(boolean_o_mats_clamp[i].data.clone());
            }

            for i in 0..max_bit {
                booleans_o_res_blocks[i].push(boolean_o_mats_res[i].data.clone());
            }

            for i in 0..k {
                booleans_o_trailing[i].push(boolean_o_mat_trailing[i].data.clone());
            }
        }

        let zero_mat = vec![vec![0; n]; n];

        l_blocks.push(zero_mat.clone());
        r_blocks.push(zero_mat.clone());
        o_blocks.push(zero_mat.clone());
        o_blocks_full.push(zero_mat.clone());

        l_blocks.push(zero_mat.clone());
        r_blocks.push(zero_mat.clone());
        o_blocks.push(zero_mat.clone());
        o_blocks_full.push(zero_mat.clone());

        let zero_mat_bool = vec![vec![false; n]; n];
        sign_blocks.push(zero_mat_bool.clone());
        sign_blocks.push(zero_mat_bool.clone());

        abs_o_blocks.push(zero_mat.clone());
        abs_o_blocks.push(zero_mat.clone());

        abs_o_res_blocks.push(zero_mat.clone());
        abs_o_res_blocks.push(zero_mat.clone());

        abs_o_trailing.push(zero_mat.clone());
        abs_o_trailing.push(zero_mat.clone());

        for i in 0..k {
            booleans_o_blocks[i].push(zero_mat_bool.clone());
            booleans_o_blocks[i].push(zero_mat_bool.clone());
        }

        for i in 0..max_bit {
            booleans_o_res_blocks[i].push(zero_mat_bool.clone());
            booleans_o_res_blocks[i].push(zero_mat_bool.clone());
        }

        for i in 0..k {
            booleans_o_trailing[i].push(zero_mat_bool.clone());
            booleans_o_trailing[i].push(zero_mat_bool.clone());
        }

        // println!("w_blocks: {:?}", w_blocks.len());
        // println!("l_blocks: {:?}", l_blocks.len());
        // println!("r_blocks: {:?}", r_blocks.len());
        // println!("o_blocks: {:?}", o_blocks.len());
        // println!("o_blocks_full: {:?}", o_blocks_full.len());
        // println!("sign_blocks: {:?}", sign_blocks.len());
        // println!("abs_o_blocks: {:?}", abs_o_blocks.len());
        // println!("abs_o_res_blocks: {:?}", abs_o_res_blocks.len());
        // println!("abs_o_trailing: {:?}", abs_o_trailing.len());
        // println!("booleans_o_blocks: {:?}", booleans_o_blocks[0].len());
        // println!("booleans_o_res_blocks: {:?}", booleans_o_res_blocks[0].len());
        // println!("booleans_o_trailing: {:?}", booleans_o_trailing[0].len());

        let mut sign_block_mat = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
        sign_block_mat.set_data(sign_blocks);

        let mut abs_o_block_mat = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        abs_o_block_mat.set_data(abs_o_blocks);

        let mut abs_o_res_block_mat = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        abs_o_res_block_mat.set_data(abs_o_res_blocks);

        let mut abs_o_trailing_block_mat = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        abs_o_trailing_block_mat.set_data(abs_o_trailing);

        let mut booleans_o_block_mat = Vec::new();
        for i in 0..k {
            let mut block_mat = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
            block_mat.set_data(booleans_o_blocks[i].clone());
            booleans_o_block_mat.push(block_mat);
        }

        let mut booleans_o_res_block_mat = Vec::new();
        for i in 0..max_bit {
            let mut block_mat = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
            block_mat.set_data(booleans_o_res_blocks[i].clone());
            booleans_o_res_block_mat.push(block_mat);
        }

        let mut booleans_o_trailing_block_mat = Vec::new();
        for i in 0..k {
            let mut block_mat = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
            block_mat.set_data(booleans_o_trailing[i].clone());
            booleans_o_trailing_block_mat.push(block_mat);
        }

        let mut al_mat_data = vec![-1 as i32; num_mat];

        let mut ar_mat_data = vec![-1 as i32; num_mat];

        let mut ao_mat_data = vec![-1 as i32; num_mat];

        for i in 0..(num_mat - 2) {
            al_mat_data[i] = i as i32;
            ar_mat_data[i] = (i + 1) as i32;
            ao_mat_data[i] = (i + 2) as i32;
        }

        self.al_mat.set_data(al_mat_data);
        self.ar_mat.set_data(ar_mat_data);
        self.ao_mat.set_data(ao_mat_data);
        self.block_w.set_data(w_blocks);
        self.block_mat_l.set_data(l_blocks);
        self.block_mat_r.set_data(r_blocks);
        self.block_mat_o.set_data(o_blocks);
        self.block_mat_o_full.set_data(o_blocks_full);
        self.sign_o = sign_block_mat;
        self.abs_o = abs_o_block_mat;
        self.abs_o_res = abs_o_res_block_mat;
        self.abs_o_trailing = abs_o_trailing_block_mat;
        self.booleans_o = booleans_o_block_mat;
        self.booleans_o_res = booleans_o_res_block_mat;
        self.booleans_o_trailing = booleans_o_trailing_block_mat;
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + Exponential.powf(-x)) - 0.5
}

pub fn construct_table(k: usize) -> (Vec<i32>, Vec<i32>) {
    let size = 1 << k;

    let vec_key = (0..size)
        .into_par_iter()
        .map(|i| i - (1 << (k - 1)) as i32)
        .collect::<Vec<i32>>();

    let vec_key_clone = vec_key.clone();

    let vec_value = vec_key_clone
        .into_par_iter()
        .map(|x| x as f64 / (1 << k) as f64)
        .map(|x| sigmoid(x) * (1 << (k + 1)) as f64)
        .map(|x| x as i32)
        .collect::<Vec<i32>>();

    // println!("{:?}", vec_key);
    // println!("{:?}", vec_value);

    (vec_key, vec_value)
}

pub fn find_index(k: usize, key: i32) -> usize {
    let size = 1 << k;

    let index = key + (1 << (k - 1)) as i32;

    if index < 0 || index >= size as i32 {
        panic!("Key out of range");
    }

    index as usize
}

pub fn find_index_vec(k: usize, vec_input: &Vec<i32>) -> Vec<i32> {
    vec_input
        .par_iter()
        .map(|&x| find_index(k, x) as i32)
        .collect::<Vec<i32>>()
}

pub fn find_values(k: usize, vec_input: &Vec<i32>, table: (&Vec<i32>, &Vec<i32>)) -> Vec<i32> {
    let (_, vec_value) = table;

    let indices = find_index_vec(k, vec_input);
    let output_vec = indices
        .par_iter()
        .map(|&i| vec_value[i.abs() as usize])
        .collect::<Vec<i32>>();

    output_vec
}

/// Mock data for a Mat-Circ with
/// M numbers of nxn matrix lookups
/// for k-bit sigmoid functions
///
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct MockMultiLookups<F>
where
    F: PrimeField + From<i32>,
{
    pub num_mat: usize,
    pub n: usize,
    pub k: usize,
    pub al_mat: RotationMatIndexFormat<i32, F>,
    pub ao_mat: RotationMatIndexFormat<i32, F>,
    pub block_w: DenseBlockMat<i32, F>,
    pub block_mat_l: DenseBlockMat<i32, F>,
    pub block_mat_o: DenseBlockMat<i32, F>,
    pub rotation_mat: RotationMatIndexFormat<i32, F>,
    pub if_invalid_rotation: DenseBlockMat<bool, F>,
    pub booleans_rotation: Vec<DenseBlockMat<bool, F>>,
    pub key_vec: Vec<i32>,
    pub value_vec: Vec<i32>,
}

impl<F> MockMultiLookups<F>
where
    F: PrimeField + From<i32>,
{
    pub fn new(num_mat: usize, n: usize, k: usize) -> Self {
        let capital_n = num_mat * n * n;
        let table_size = 1 << k;

        let al_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let ao_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let block_w = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_l = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let block_mat_o = DenseBlockMat::<i32, F>::new(num_mat, (n, n));
        let rotation_mat = RotationMatIndexFormat::<i32, F>::new(capital_n, table_size);

        let if_invalid_rotation = DenseBlockMat::<bool, F>::new(num_mat, (n, n));
        let booleans_rotation = Vec::new();

        let (key_vec, value_vec) = construct_table(k);

        Self {
            num_mat: num_mat,
            n: n,
            k: k,
            al_mat: al_mat,
            ao_mat: ao_mat,
            block_w: block_w,
            block_mat_l: block_mat_l,
            block_mat_o: block_mat_o,
            rotation_mat: rotation_mat,
            if_invalid_rotation: if_invalid_rotation,
            booleans_rotation: booleans_rotation,
            key_vec: key_vec,
            value_vec: value_vec,
        }
    }

    pub fn gen_mock(&mut self) {
        println!("*** gen mock");

        let num_mat = self.num_mat;
        let n = self.n;
        let k = self.k;

        let mut l_blocks = Vec::new();
        let mut o_blocks = Vec::new();
        let mut w_blocks = Vec::new();

        let mut indices_blocks = Vec::new();

        let mut l_mat = DenseMatCM::<i32, F>::new(n, n);
        l_mat.gen_rand(self.k);

        let mut data_l = l_mat.data.clone();
        let mut l_vec = l_mat.to_vec();

        let mut o_vec = find_values(k, &l_vec, (&self.key_vec, &self.value_vec));

        let mut o_mat = DenseMatCM::<i32, F>::new(n, n);
        o_mat.from_vec(&o_vec);

        let mut data_o = o_mat.data.clone();

        let mut indices = find_index_vec(k, &l_vec);

        let mut index_mat = DenseMatCM::<i32, F>::new(n, n);
        index_mat.from_vec(&indices);
        let mut data_indices = index_mat.data.clone();

        l_blocks.push(data_l.clone());
        o_blocks.push(data_o.clone());

        indices_blocks.push(data_indices.clone());

        w_blocks.push(data_l.clone());
        w_blocks.push(data_o.clone());

        for _ in 1..(self.num_mat - 1) {
            l_mat = DenseMatCM::<i32, F>::new(n, n);
            l_mat.set_data(data_o.clone());

            data_l = data_o;

            l_vec = l_mat.to_vec();

            o_vec = find_values(k, &l_vec, (&self.key_vec, &self.value_vec));

            o_mat = DenseMatCM::<i32, F>::new(n, n);
            o_mat.from_vec(&o_vec);
            data_o = o_mat.data.clone();

            indices = find_index_vec(k, &l_vec);

            index_mat = DenseMatCM::<i32, F>::new(n, n);
            index_mat.from_vec(&indices);
            data_indices = index_mat.data.clone();

            l_blocks.push(data_l.clone());
            o_blocks.push(data_o.clone());
            indices_blocks.push(data_indices.clone());

            w_blocks.push(data_o.clone());
        }

        let zero_mat = vec![vec![0; n]; n];
        let minus_one_mat = vec![vec![-1; n]; n];

        l_blocks.push(zero_mat.clone());
        o_blocks.push(zero_mat.clone());

        indices_blocks.push(minus_one_mat);

        let mut al_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let mut al_mat_data = vec![-1 as i32; num_mat];

        let mut ao_mat = RotationMatIndexFormat::<i32, F>::new(num_mat, num_mat);
        let mut ao_mat_data = vec![-1 as i32; num_mat];

        for i in 0..(num_mat - 1) {
            al_mat_data[i] = i as i32;
            ao_mat_data[i] = (i + 1) as i32;
        }

        al_mat.set_data(al_mat_data);
        ao_mat.set_data(ao_mat_data);

        self.al_mat = al_mat;
        self.ao_mat = ao_mat;
        self.block_w.set_data(w_blocks);
        self.block_mat_l.set_data(l_blocks);
        self.block_mat_o.set_data(o_blocks);

        let mut indices_block_mat = DenseBlockMat::<i32, F>::new(num_mat, (n, n));

        indices_block_mat.set_data(indices_blocks);

        // println!("*** indices_block_mat {:?}", indices_block_mat.data);

        let indices_vec = indices_block_mat.to_vec();

        // println!("indices_vec {:?}", indices_vec);

        self.rotation_mat.set_data(indices_vec);

        let (if_invalid_vec, booleans_vec) = self.rotation_mat.split_into_boolean();

        // println!("*** if_invalid_vec {:?}", if_invalid_vec);

        self.if_invalid_rotation.from_vec(&if_invalid_vec);

        for kappa in 0..k {
            // println!("boolean_vec {:?}", booleans_vec[kappa]);

            let mut boolean_mat = DenseBlockMat::<bool, F>::new(num_mat, (n, n));

            boolean_mat.from_vec(&booleans_vec[kappa]);

            // println!("*** boolean_mat {:?}", boolean_mat.data);

            self.booleans_rotation.push(boolean_mat);
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::utils::sparse::{DiagBlockMat, SparseKronecker};

    use super::*;

    #[test]
    fn test_mock_lookup() {
        use ark_bls12_381::Fr;

        let k = 8;
        let (vec_key, vec_value) = construct_table(k);

        let maximum = 1 << (k) - 1;
        let minimum = -1 << (k) + 1;

        for i in 0..(1 << k) {
            assert!(vec_key[i] <= maximum);
            assert!(vec_key[i] >= minimum);
            assert!(vec_value[i] <= maximum);
            assert!(vec_value[i] >= minimum);
            assert_eq!(i, find_index(k, vec_key[i]));
        }
        let indices = find_index_vec(k, &vec_key);
        let indices_check = (0..(1 << k)).map(|i| i as i32).collect::<Vec<i32>>();
        assert_eq!(indices, indices_check);

        let vec_input = [vec_key.clone(), vec_key.clone()].concat();

        let vec_output = [vec_value.clone(), vec_value.clone()].concat();

        let value_check = find_values(k, &vec_input, (&vec_key, &vec_value));

        assert_eq!(vec_output, value_check);

        let mut mock = MockMultiLookups::<Fr>::new(8, 4, 8);
        mock.gen_mock();

        let l_vec_large = mock.block_mat_l.to_vec();

        let o_vec_large = mock.block_mat_o.to_vec();

        let b_mat = mock.rotation_mat.to_dense();

        let k_vec = vec![mock.key_vec.clone()];
        let v_vec = vec![mock.value_vec.clone()];

        let l_vec_check = linear::mat_mul_shortint::<i32>(&b_mat.data, &k_vec);

        let l_vec_check = l_vec_check[0].clone();

        let o_vec_check = linear::mat_mul_shortint::<i32>(&b_mat.data, &v_vec);

        let o_vec_check = o_vec_check[0].clone();

        assert_eq!(l_vec_large, l_vec_check);
        assert_eq!(o_vec_large, o_vec_check);

        let identity_mat = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ];

        let al_dense = mock.al_mat.to_dense();
        let mut al_kronecker = SparseKronecker::<i32, Fr>::new((8, 8), (4, 4));
        al_kronecker.set_data((al_dense.data.clone(), identity_mat.clone()));

        let al_large = al_kronecker.to_dense();

        let ao_dense = mock.ao_mat.to_dense();
        let mut ao_kronecker = SparseKronecker::<i32, Fr>::new((8, 8), (4, 4));
        ao_kronecker.set_data((ao_dense.data.clone(), identity_mat.clone()));
        let ao_large = ao_kronecker.to_dense();

        let w_large = mock.block_w.to_dense().data;

        let dense_l_check = linear::mat_mul_shortint::<i32>(&al_large.data, &w_large);

        let dense_l_mock = mock.block_mat_l.to_dense().data;

        let dense_o_check = linear::mat_mul_shortint::<i32>(&ao_large.data, &w_large);

        let dense_o_mock = mock.block_mat_o.to_dense().data;

        assert_eq!(dense_l_mock, dense_l_check);
        assert_eq!(dense_o_mock, dense_o_check);

        println!("All check for MockMul passed");
    }

    #[test]
    fn test_mock_mul() {
        use ark_bls12_381::Fr;

        let mut mock = MockMultiMuls::<Fr>::new(8, 4, 8);
        mock.gen_mock();

        println!("*** res mat abs {:?}", mock.abs_o_res.data);
        println!("*** o_mat {:?}", mock.block_mat_o.data);
        println!("*** trailing mat {:?}", mock.abs_o_trailing.data);
        println!("Mock data generated");

        let mut diag_l = DiagBlockMat::<i32, Fr>::new(8, (4, 4));
        diag_l.set_data(mock.block_mat_l.data.clone());

        let dense_l = diag_l.to_dense().data;

        let mut diag_r = DiagBlockMat::<i32, Fr>::new(8, (4, 4));
        diag_r.set_data(mock.block_mat_r.data.clone());

        let dense_r = diag_r.to_dense().data;

        let mut diag_o = DiagBlockMat::<i32, Fr>::new(8, (4, 4));

        let o_mat_block = mock.block_mat_o.clone();

        let k = 8;

        diag_o.set_data(o_mat_block.data.clone());

        let dense_o = diag_o.to_dense();

        let dense_o_full = linear::mat_mul_shortint::<i32>(&dense_l, &dense_r);

        println!("*** o_mat full {:?}", dense_o_full);

        let mut dense_o_full_mat = DenseMatCM::<i32, Fr>::new(8 * 4, 8 * 4);
        dense_o_full_mat.set_data(dense_o_full);

        let dense_o_check = dense_o_full_mat.clamp(k);

        assert_eq!(dense_o, dense_o_check);

        let identity_mat = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ];

        let al_dense = mock.al_mat.to_dense();
        let ar_dense = mock.ar_mat.to_dense();
        let ao_dense = mock.ao_mat.to_dense();

        let mut al_kronecker = SparseKronecker::<i32, Fr>::new((8, 8), (4, 4));
        al_kronecker.set_data((al_dense.data.clone(), identity_mat.clone()));
        let al_large = al_kronecker.to_dense();

        let mut ar_kronecker = SparseKronecker::<i32, Fr>::new((8, 8), (4, 4));
        ar_kronecker.set_data((ar_dense.data.clone(), identity_mat.clone()));
        let ar_large = ar_kronecker.to_dense();

        let mut ao_kronecker = SparseKronecker::<i32, Fr>::new((8, 8), (4, 4));
        ao_kronecker.set_data((ao_dense.data.clone(), identity_mat.clone()));
        let ao_large = ao_kronecker.to_dense();

        let w_large = mock.block_w.to_dense().data;

        let dense_l_check = linear::mat_mul_shortint::<i32>(&al_large.data, &w_large);

        let dense_l_mock = mock.block_mat_l.to_dense().data;

        let dense_r_check = linear::mat_mul_shortint::<i32>(&ar_large.data, &w_large);

        let dense_r_mock = mock.block_mat_r.to_dense().data;

        let dense_o_check = linear::mat_mul_shortint::<i32>(&ao_large.data, &w_large);

        let dense_o_mock = mock.block_mat_o.to_dense().data;

        assert_eq!(dense_l_mock, dense_l_check);
        assert_eq!(dense_r_mock, dense_r_check);
        assert_eq!(dense_o_mock, dense_o_check);

        println!("All check for MockMul passed");
    }
}
