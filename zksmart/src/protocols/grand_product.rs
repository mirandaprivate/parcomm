/// Implement the Hadamard protocol
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::linear;
use crate::utils::sparse::{MatOps, ShortInt};
use crate::utils::xi;
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};

use super::hadamard::ZkHadamardField;
use super::litebullet::ZkLiteBullet;
use super::scalars::ZkSchnorr;

pub struct ZkGrandProduct<IA, F, G, MA>
where
    IA: ShortInt,
    F: PrimeField + From<IA>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<IA, F>,
{
    pub v_c_com: G,
    pub shape: (usize, usize),
    _marker: PhantomData<(IA, MA)>,
}

impl<IA, F, G, MA> ZkGrandProduct<IA, F, G, MA>
where
    IA: ShortInt,
    F: PrimeField + From<IA>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<IA, F>,
{
    pub fn new(v_c_com_value: G, shape_value: (usize, usize)) -> Self {
        Self {
            v_c_com: v_c_com_value,
            shape: shape_value,
            _marker: PhantomData,
        }
    }

    pub fn gen_vecs(&self, mat_a: &MA) -> Vec<Vec<F>> {
        let (m, n) = self.shape;

        if (m, n) != mat_a.get_shape() {
            panic!("Matrix dimension mismatch");
        }

        // 将矩阵转换为field类型的向量
        let field_vec = mat_a.to_field_vec();

        // 将一维向量整理成按需要的向量
        let flattened: Vec<F> = field_vec;

        // 创建一个向量，用于存储每一层的向量
        let mut result: Vec<Vec<F>> = Vec::new();

        // 第一层是原始flatten后的向量
        result.push(flattened);

        // 递归构造，每次将向量长度减半，直到长度为1
        let mut current_level = 0;
        while result[current_level].len() > 1 {
            let current_vec = &result[current_level];
            let current_len = current_vec.len();

            // 确保长度是2的幂
            if current_len & (current_len - 1) != 0 {
                panic!("Vector length must be a power of 2");
            }

            // 计算新向量的长度，是当前长度的一半
            let new_len = current_len / 2;
            let mut new_vec = Vec::with_capacity(new_len);

            // 执行Hadamard product，左右相应元素相乘
            for i in 0..new_len {
                let left = current_vec[i];
                let right = current_vec[new_len + i];
                new_vec.push(left * right);
            }

            // 将新向量添加到结果中
            result.push(new_vec);
            current_level += 1;
        }

        result
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &MA,
        tilde_c: F,
    ) -> (G, F, (Vec<F>, Vec<F>)) {
        let (m, n) = self.shape;

        if (m, n) != mat_a.get_shape() {
            panic!("Matrix dimension mismatch");
        }

        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;
        println!("Prover: log_m = {}, log_n = {}", log_m, log_n);

        // 生成递归向量序列
        let vecs = self.gen_vecs(mat_a);

        // 打印vecs的长度信息，这就是total_levels
        for i in 0..vecs.len() {
            println!("vecs[{}] len: {}", i, vecs[i].len());
        }

        // 计算总迭代次数，与verify中一致
        let total_levels = log_m + log_n;
        println!(
            "Prover总迭代次数total_levels = {}, vecs.len = {}",
            total_levels - 1,
            vecs.len()
        );

        // 最终product值应该是最后一个向量中的唯一元素
        let final_product = vecs.last().unwrap()[0];

        // 初始化
        let mut com = self.v_c_com;
        let mut com_tilde = tilde_c;

        // 初始化空的挑战点
        let mut cur_xl = Vec::<F>::new();
        let xr = Vec::<F>::new();
        println!(
            "Prover初始: cur_xl.len() = {}, xr.len() = {}",
            cur_xl.len(),
            xr.len()
        );

        // 从倒数第二层开始，因为最后一层只有一个元素
        for level in (0..total_levels).rev() {
            let actual_len = vecs[level].len();

            println!(
                "Prover Level {}: 迭代前 cur_xl.len() = {}, xr.len() = {}",
                level,
                cur_xl.len(),
                xr.len()
            );

            // 将当前层分为左右两部分，直接使用切片而不是创建新向量
            // 为Hadamard协议创建矩阵格式
            let mut left_mat = Vec::new();
            left_mat.push(vecs[level][0..actual_len / 2].to_vec());

            let mut right_mat = Vec::new();
            right_mat.push(vecs[level][actual_len / 2..actual_len].to_vec());

            // 创建正确的point元组
            let point = (cur_xl.clone(), xr.clone());

            // 使用ZkHadamardField
            let hadamard = ZkHadamardField::<F, G>::new(com, point);

            // 调用Hadamard协议的reduce_prover
            let (a_hat_com, b_hat_com, a_hat_tilde, b_hat_tilde, (xll, _), (xlr, _)) =
                hadamard.reduce_prover(zksrs, zk_trans_seq, &left_mat, &right_mat, com_tilde);

            println!(
                "Prover Level {}: 从hadamard获取的 xll.len() = {}, xlr.len() = {}",
                level,
                xll.len(),
                xlr.len()
            );

            // 生成随机挑战
            let cur_rand = zk_trans_seq.gen_challenge();
            println!(
                "Prover Level {}: 生成随机挑战 cur_rand = {:?}",
                level, cur_rand
            );

            // 构建l_vec，这是用于下一步证明的向量
            let xi_l = xi::xi_from_challenges(&xll);
            let xi_r = linear::vec_scalar_mul(&xi::xi_from_challenges(&xlr), &cur_rand);

            println!(
                "Prover Level {}: xi_l.len() = {}, xi_r.len() = {}",
                level,
                xi_l.len(),
                xi_r.len()
            );

            // 正确创建长度匹配的l_vec
            // l_vec的长度需要与current_vec匹配，因为它将用作LiteBullet.reduce_prover的输入
            let mut l_vec = Vec::with_capacity(actual_len);
            l_vec.extend(xi_l);
            l_vec.extend(xi_r);

            // 确保l_vec长度正确
            println!(
                "Prover Level {}: l_vec.len() = {}, current_vec.len() = {}",
                level,
                l_vec.len(),
                actual_len
            );
            assert_eq!(l_vec.len(), actual_len, "l_vec和current_vec长度不匹配");

            // 更新承诺
            let c_com_update = a_hat_com + b_hat_com.mul(&cur_rand);
            let c_tilde_update = a_hat_tilde + b_hat_tilde * cur_rand;

            // 创建LiteBullet实例
            // 使用当前向量的长度
            let litebullet_length = actual_len;
            println!(
                "Prover Level {}: 创建LiteBullet实例，length = {}",
                level, litebullet_length
            );
            let litebullet = ZkLiteBullet::<F, G>::new(c_com_update, litebullet_length);

            // 调用LiteBullet协议进行进一步证明
            // 注意：参数顺序是 vec_a, vec_b, tilde_v
            let (com_update, _, com_tilde_update, _, xl_update, _) =
                litebullet.reduce_prover(zksrs, zk_trans_seq, &vecs[level], &l_vec, c_tilde_update);

            println!(
                "Prover Level {}: 从litebullet获取的 xl_update.len() = {}",
                level,
                xl_update.len()
            );

            // 更新状态
            cur_xl = xl_update;
            com = com_update;
            com_tilde = com_tilde_update;

            println!(
                "Prover Level {}: 迭代后 cur_xl.len() = {}, xr.len() = {}",
                level,
                cur_xl.len(),
                xr.len()
            );
        }

        // 提取最终结果的xl和xr，使用切片而不是创建新向量
        println!(
            "Prover结束: cur_xl.len() = {}, log_m = {}, log_n = {}",
            cur_xl.len(),
            log_m,
            log_n
        );

        let xl = cur_xl[log_n..].to_vec();
        let xr = cur_xl[..log_n].to_vec();

        println!(
            "Prover最终返回: xl.len() = {}, xr.len() = {}",
            xl.len(),
            xr.len()
        );

        // 注意: 在这里我们直接使用xl和xr，而不是l_vec和r_vec
        // proj_lr内部会调用xi_from_challenges生成完整的l_vec和r_vec向量
        // 因此我们应该将相同的挑战向量传递给proj_lr和inner_product
        // let proj = mat_a.proj_lr(&xl, &xr);

        // // 添加调试信息
        // println!("内积 ip_a = {:?}", ip_a);
        // println!("投影 proj = {:?}", proj);
        // println!("xi.len() = {}, l_vec.len() = {}, r_vec.len() = {}", xi.len(), l_vec.len(), r_vec.len());

        // // 输出差值
        // let diff = ip_a - proj;
        // println!("内积与投影的差值: {:?}", diff);

        // // 验证内积与投影等价性
        // assert_eq!(ip_a, proj);

        // let com_proj = zksrs.com_base.mul(proj) + zksrs.blind_base.mul(com_tilde);

        // // 验证承诺一致性
        // assert_eq!(com_proj, com);

        // Bind the initial Pedersen commitment to the actual terminal product.
        // The verifier will reconstruct the same difference using its public
        // expected product; Schnorr proves that the difference contains only
        // the original blinding factor.
        let product_blinding_commitment = self.v_c_com - zksrs.com_base.mul(final_product);
        assert_eq!(
            product_blinding_commitment,
            zksrs.blind_base.mul(tilde_c),
            "grand-product commitment does not open to the computed product",
        );
        ZkSchnorr::new(zksrs, product_blinding_commitment).prove(zk_trans_seq, tilde_c);

        (com, com_tilde, (xl, xr))
    }

    /// Verify only the recursive Hadamard reduction.  This remains private so
    /// callers cannot accidentally omit the public terminal-product binding.
    fn verify_recurrence_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {
        let (m, n) = self.shape;
        let log_m = (m as usize).ilog2() as usize;
        let log_n = (n as usize).ilog2() as usize;
        let total_levels = log_m + log_n; // 总迭代次数应该是log_m + log_n

        println!("Verify: log_m = {}, log_n = {}", log_m, log_n);

        // 记录初始状态
        let mut cur_xl = Vec::<F>::new();
        let xr = Vec::<F>::new();

        println!(
            "Verify初始: cur_xl.len() = {}, xr.len() = {}",
            cur_xl.len(),
            xr.len()
        );

        let mut pointer = trans_seq.pointer;
        let mut flag = true;

        let mut com = self.v_c_com;

        // 迭代器从total_levels-1（最高层）开始递减到0（最低层）
        for level in (0..total_levels).rev() {
            println!("Verify level {}/{}: 迭代前 cur_xl.len() = {}, xr.len() = {}, 当前transcript指针: {}", 
                    level+1, total_levels, cur_xl.len(), xr.len(), pointer);

            // 计算当前层的大小，使用公式
            let formula_len = 1 << (total_levels - level);

            // 创建当前层的point元组
            let point = (cur_xl.clone(), xr.clone());

            let hadamard = ZkHadamardField::<F, G>::new(com, point);

            // 调用Hadamard协议的verify_as_subprotocol
            let (hadamard_flag, a_hat_com, b_hat_com, (_xll, _), (_xlr, _)) =
                hadamard.verify_as_subprotocol(zksrs, trans_seq);

            pointer = trans_seq.pointer;

            if !hadamard_flag {
                println!(
                    "Verify level {}/{}: hadamard验证失败",
                    level + 1,
                    total_levels
                );
                flag = false;
                break;
            }

            // 读取随机挑战
            let cur_rand;
            if pointer < trans_seq.data.len() {
                if let TranElem::Coin(r) = trans_seq.data[pointer].clone() {
                    cur_rand = r;
                    trans_seq.pointer += 1;
                    pointer = trans_seq.pointer;
                    println!(
                        "Verify level {}/{}: 读取到随机挑战 cur_rand: {}",
                        level + 1,
                        total_levels,
                        cur_rand
                    );
                } else {
                    println!("!! Invalid transcript when verifying GrandProduct: 预期Coin但获取到其他类型, 当前指针: {}", pointer);
                    flag = false;
                    break;
                }
            } else {
                println!(
                    "!! Invalid transcript when verifying GrandProduct: 指针超出范围 {} >= {}",
                    pointer,
                    trans_seq.data.len()
                );
                flag = false;
                break;
            }

            let c_com_update = a_hat_com + b_hat_com * cur_rand;

            // 计算当前层的向量长度
            // 此时不需要计算current_len，只记录日志
            println!(
                "Verify level {}/{}: 准备验证，size = {}",
                level + 1,
                total_levels,
                formula_len
            );

            // 记录当前指针位置
            let before_litebullet_pointer = pointer;

            // 使用克隆的承诺值和正确的size创建LiteBullet实例
            let litebullet = ZkLiteBullet::<F, G>::new(c_com_update.clone(), formula_len);

            // 调用LiteBullet协议进行验证
            let (litebullet_flag, com_update, _, challenges_inv, _) =
                litebullet.verify_as_subprotocol(zksrs, trans_seq);

            pointer = trans_seq.pointer;
            println!(
                "Verify level {}/{}: LiteBullet验证后的transcript指针: {} (移动了 {} 位置)",
                level + 1,
                total_levels,
                pointer,
                pointer - before_litebullet_pointer
            );

            if !litebullet_flag {
                println!(
                    "Verify level {}/{}: LiteBullet验证失败",
                    level + 1,
                    total_levels
                );
                flag = false;
                break;
            }

            // 更新状态
            cur_xl = challenges_inv;
            com = com_update;
        }

        println!(
            "Verify结束: cur_xl.len() = {}, log_m = {}, log_n = {}",
            cur_xl.len(),
            log_m,
            log_n
        );

        // 确保cur_xl长度足够，如果不足，则填充
        if cur_xl.len() < log_m + log_n {
            println!(
                "警告：cur_xl长度不足 ({} < {}), 进行填充",
                cur_xl.len(),
                log_m + log_n
            );
            for _ in cur_xl.len()..(log_m + log_n) {
                cur_xl.push(F::zero());
            }
        }

        // 从cur_xl中提取结果
        let xl = cur_xl[log_n..].to_vec();
        let xr = cur_xl[..log_n].to_vec();

        println!(
            "Verify最终返回: xl.len() = {}, xr.len() = {}, transcript总指针移动: {}",
            xl.len(),
            xr.len(),
            pointer
        );

        // 如果flag为false，则输出验证失败的消息
        if !flag {
            println!("GrandProduct验证失败：子协议验证未通过");
        }

        // 使用最后一次循环中计算的com变量，而不是actual_com
        (flag, com, (xl, xr))
    }

    pub fn verify(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
        expected_product: F,
    ) -> bool {
        if trans_seq.pointer != 0 || !trans_seq.check_fiat_shamir() {
            println!("GrandProduct验证失败：Fiat-Shamir transcript 无效");
            return false;
        }

        // 调用verify_as_subprotocol方法获取结果
        let (flag, _final_com, _) = self.verify_recurrence_as_subprotocol(zksrs, trans_seq);

        // 如果verify_as_subprotocol过程中有任何验证失败，直接返回false
        if !flag {
            println!("GrandProduct验证失败：子协议验证未通过");
            return false;
        }

        if trans_seq.data.len().saturating_sub(trans_seq.pointer) < 5 {
            println!("GrandProduct验证失败：缺少公开乘积绑定证明");
            return false;
        }

        let product_blinding_commitment = self.v_c_com - zksrs.com_base.mul(expected_product);
        let product_binding = ZkSchnorr::new(zksrs, product_blinding_commitment);
        if !product_binding.verify_as_subprotocol(trans_seq) {
            println!("GrandProduct验证失败：承诺未绑定到公开乘积");
            return false;
        }
        if !trans_seq.is_fully_consumed() {
            println!("GrandProduct验证失败：证明包含未消费的尾随数据");
            return false;
        }

        println!("GrandProduct验证成功：递归关系及公开乘积绑定均通过");
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::DenseMatCM;
    use ark_std::{One, UniformRand};
    use std::ops::Mul;

    // 使用4x4矩阵，尺寸较小便于测试
    #[test]
    fn test_grand_product() {
        use ark_ec::pairing::Pairing;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        // 使用4x4矩阵，尺寸较小便于测试
        let m = 4;
        let n = 4;

        // 生成随机SRS
        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        // 创建矩阵
        let mut mat_a = DenseMatCM::<i32, F>::new(m, n);

        // 创建简单的数据，全部元素为1
        let mut data = Vec::new();
        for _ in 0..m {
            let mut row = Vec::new();
            for _ in 0..n {
                row.push(1); // 所有元素都设为1
            }
            data.push(row);
        }
        data[0][0] = 2;
        mat_a.set_data(data);

        // Use a nontrivial terminal product so the public binding is exercised.
        let expected_product = F::from(2_u64);

        // 初始化交易序列
        let mut zk_trans_seq = ZkTranSeq::<F, G>::new(&zksrs);

        // 生成随机盲因子
        let tilde_c = F::rand(rng);

        // 创建GrandProduct实例
        let v_c_com = zksrs.com_base * expected_product + zksrs.blind_base * tilde_c;
        let grand_product = ZkGrandProduct::<i32, F, G, DenseMatCM<i32, F>>::new(v_c_com, (m, n));

        println!("开始{}x{}矩阵测试，总元素数: {}", m, n, m * n);

        // 生成递归向量序列，验证正确生成
        let vecs = grand_product.gen_vecs(&mat_a);

        // 打印vecs的大小信息
        for i in 0..vecs.len() {
            println!("vecs[{}] 长度: {}", i, vecs[i].len());
        }

        // 验证最终结果是所有元素的乘积
        assert_eq!(vecs.last().unwrap()[0], expected_product);

        // 生成证明
        println!("开始生成证明...");
        let (com, com_tilde, (xl, xr)) =
            grand_product.reduce_prover(&zksrs, &mut zk_trans_seq, &mat_a, tilde_c);

        println!("证明生成完成，xl.len()={}, xr.len()={}", xl.len(), xr.len());

        // 获取交易序列并验证
        let trans_seq = zk_trans_seq.publish_trans();
        let mut verify_trans_seq = trans_seq.clone();

        // 开始验证
        println!("开始验证...");
        let verification_result =
            grand_product.verify(&zksrs, &mut verify_trans_seq, expected_product);
        assert!(verify_trans_seq.is_fully_consumed());

        // The public product is part of the statement.  Replaying the same
        // proof against a different product must fail.
        let mut wrong_product_transcript = trans_seq.clone();
        assert!(!grand_product.verify(
            &zksrs,
            &mut wrong_product_transcript,
            expected_product + F::one(),
        ));

        let mut truncated_transcript = trans_seq.clone();
        truncated_transcript.data.pop();
        assert!(!grand_product.verify(&zksrs, &mut truncated_transcript, expected_product,));

        let mut tampered_binding_transcript = trans_seq.clone();
        match tampered_binding_transcript.data.last_mut().unwrap() {
            TranElem::Field(response) => *response += F::one(),
            _ => panic!("grand-product binding must end with a Schnorr response"),
        }
        assert!(!grand_product.verify(&zksrs, &mut tampered_binding_transcript, expected_product,));

        let mut trailing_transcript = trans_seq.clone();
        trailing_transcript.data.push(TranElem::Field(F::one()));
        assert!(!grand_product.verify(&zksrs, &mut trailing_transcript, expected_product,));

        let mut nonzero_pointer_transcript = trans_seq.clone();
        nonzero_pointer_transcript.pointer = 1;
        assert!(!grand_product.verify(&zksrs, &mut nonzero_pointer_transcript, expected_product,));

        let mut tampered_coin_transcript = trans_seq.clone();
        let coin = tampered_coin_transcript
            .data
            .iter_mut()
            .find_map(|element| match element {
                TranElem::Coin(coin) => Some(coin),
                _ => None,
            })
            .unwrap();
        *coin += F::one();
        assert!(!grand_product.verify(&zksrs, &mut tampered_coin_transcript, expected_product,));

        let proj = mat_a.proj_lr(&xl, &xr);

        let com_proj = zksrs.com_base.mul(proj) + zksrs.blind_base.mul(com_tilde);

        // 验证承诺一致性
        assert_eq!(com_proj, com);

        // 打印验证结果
        println!("{}x{}矩阵验证结果: {}", m, n, verification_result);
        assert!(verification_result, "验证失败");
        println!("{}x{}矩阵GrandProduct测试完成!", m, n);
    }
}
