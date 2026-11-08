use parcomm_qafold::field::*;
use std::time::Instant;
fn main() {
    let count=1usize<<std::env::args().nth(1).unwrap_or("20".into()).parse::<usize>().unwrap();
    let left=(0..count).map(|index| F::from_coefficients(std::array::from_fn(|lane| B::from_u64((index*1789+lane*3917+291) as u64)))).collect::<Vec<_>>();
    let right=(0..count).map(|index| F::from_coefficients(std::array::from_fn(|lane| B::from_u64((index*2717+lane*9811+719) as u64)))).collect::<Vec<_>>();
    let scalar=right[177];
    let mut old=left.clone(); let started=Instant::now();
    for (left,right) in old.iter_mut().zip(&right) { *left *= *right; }
    let mul_scalar=started.elapsed().as_secs_f64();
    let mut new=left.clone(); let started=Instant::now();
    F::mul_assign_slice(&mut new,&right);
    let mul_simd=started.elapsed().as_secs_f64(); assert_eq!(old,new);
    let mut old=left.clone(); let started=Instant::now();
    for (output,input) in old.iter_mut().zip(&right) { *output=input.mul_add(scalar,*output); }
    let add_scalar=started.elapsed().as_secs_f64();
    let mut new=left.clone(); let started=Instant::now();
    F::mul_add_scalar_slice(&mut new,&right,scalar);
    let add_simd=started.elapsed().as_secs_f64(); assert_eq!(old,new);
    let mut old=left.clone(); let started=Instant::now();
    for index in 0..old.len()/2 { old[index]=old[2*index+1].mul_add(scalar,old[2*index]); }
    let fold_scalar=started.elapsed().as_secs_f64();
    let mut new=left.clone(); let started=Instant::now();
    F::fold_adjacent_in_place(&mut new,scalar);
    let fold_simd=started.elapsed().as_secs_f64(); assert_eq!(&old[..count/2],&new[..count/2]);
    println!("{}",serde_json::json!({"count":count,"runs":1,"mul_scalar_s":mul_scalar,"mul_simd_s":mul_simd,"add_scalar_s":add_scalar,"add_simd_s":add_simd,"fold_scalar_s":fold_scalar,"fold_simd_s":fold_simd,"verified":true}));
}
