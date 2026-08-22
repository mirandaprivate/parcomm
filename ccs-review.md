ACM CCS 2026 A Paper #407 Reviews and Comments
===========================================================================
Paper #407 Batched Multilinear Polynomial Commitments for Large-Scale
Vectors of Heterogeneous Dimensions


Review #407A
===========================================================================

Paper summary
-------------
This work presents a protocol for batch-proving the correctness of evaluation claims from different polynomials of varying sizes. At a high level, the authors achieve this by first concatenating the coefficients of the polynomials in increasing order of size. Then, they invoke the sumcheck protocol over an instance that takes as input two polynomials: the MLE of the concatenated coefficients and the MLE of the concatenation of rts($��_i$) for each evaluation point $x_i$. Furthermore, they provide an algorithm for efficiently evaluating the last MLE in time only logarithmic to the number of polynomials and their total size. Finally, they also propose a concrete optimization for the sumcheck protocol, where the prover folds elements with respect to even and odd indexes and not right half and left half ones. This improves prover time as it exploits memory locality.

Strengths / Reasons to accept
-----------------------------
+ Interesting optimizations for reducing verification time.

Weaknesses / Reasons to reject
------------------------------
- Overall, I am not convinced about the level of novelty in this work. Both even-odd folding and batching heterogeneously sized polynomials have been explored in prior works. 

- The scope is extremely "narrow," focusing only on a specific operation which may or may not be the main bottleneck of a given SNARK prover.

Constructive comments for author
--------------------------------
Thank you for your submission. On the bright side, your work introduces some interesting optimizations on batching, which can concretely reduce the prover's time and space utilization when performing batch opening. However, I have two concerns that I explain below, that make me hesitate recommending accepting it for publication. 

* First and foremost, I think the novelty here is limited. In particular, neither the even-odd folding technique nor the batch opening is really new. Regarding even-odd folding, while I could not find a work that explicitly discusses this optimization, it can be considered as an almost folklore technique, and it has actually been used in various codebases (e.g., see line 377 of [a]). Likewise, batching of different-sized polynomials with different evaluation claims has also been considered previously in the literature. To the best of my knowledge, the first work that attempts to tackle a more general problem is [b] (see the second approach of Section 3.3), which batch opens evaluation claims from different-sized subsets of a vector. Furthermore, there are also some protocols that specifically focus on batch opening of multi-linear polynomials [c,d], which in fact use almost identical techniques to yours. 
Unfortunately, these are entirely ignored in your paper currently. If there are substantial differences, you need to include a clear discussion. If they are similar, this undermines your novelty claims. I do believe that your protocol may have better across-the-board performance, due to the optimizations you propose. However, I am not sure this alone would merit publication and, in any case, this is beyond the point as the current version of the paper omits all this relevant literature. 

* Secondly, as I mentioned above, the scope of this work is somewhat limited, as it focuses solely on optimizing a specific task that typically serves as a sub-protocol within a proving system. To my knowledge, batch proving of evaluation claims from different polynomials is primarily utilized in specialized GKR-like/sumcheck proof systems. It remains unclear how much an optimized protocol for this task would enhance the overall prover efficiency, but also reduce the proof size and verification time, given that GKR-like proof systems already have significant verifier overhead. My suggestion is to revisit the introduction and better highlight the need for a protocol for batch proving evaluation claims, within the context of a SNARK for specific applications. For this, it would be helpful to find applications where batch proving evaluation claims becomes the main bottleneck. This would definitely improve the motivation and importance of this work.            

[a] https://github.com/sunblaze-ucb/Virgo/blob/master/src/linear_gkr/prover_fast_track.cpp

[b] https://eprint.iacr.org/2020/1247.pdf

[c] https://github.com/Lagrange-Labs/deep-prove/blob/master/docs/src/commitments.md

[d] https://eprint.iacr.org/2021/673.pdf

Does the paper raise ethical concerns?
--------------------------------------
1. No

Reviewer expertise in this domain
---------------------------------
4. Expert (I've worked on this topic domain or a closely related topic
   domain)

Reviewer confidence
-------------------
4. Confident

Overall merit
-------------
1. Reject

Open Science Compliance
-----------------------
1. Yes



Review #407B
===========================================================================

Paper summary
-------------
This paper presents a protocol for proving a batch of multilinear polynomials (of different sizes) efficiently without needing to pad the polynomials to the same size. The paper provides asymptotics: for $t$ vectors with heterogeneous lengths $n_1, \ldots, n_t$, of total size $N=n_1+\ldots+n_t$, the prover time is $\Theta(N + t logN)$. The authors also provide experimental results for their batching scheme.

Strengths / Reasons to accept
-----------------------------
* Fastest SNARKs are all based on multilinear polynomials
* In practice, the prover in these SNARKs must prove evaluations of multiple polynomials (of different sizes) and batching is essential to keep proving and verification costs tolerable.

Weaknesses / Reasons to reject
------------------------------
* No comparision with techniques widely used in practice
* No baselines in the experimental evaluation

Constructive comments for author
--------------------------------
"To the best of our knowledge, efficient batch commitment and
opening of heterogeneously sized vectors remains unexplored." --> This is not quite true. Batching multiple heterogenous-sized multilinear polynomials is well-studied and already part of most implementations of multilinear SNARKs like Spartan and works derived from it like Jolt and Nova. For example: https://github.com/microsoft/Nova/blob/fb161aa9c72bc810c00c56688e50bd378ab55998/src/spartan/snark.rs#L407. These implementations do not pad the polynomials to the same size but rather deal with differently-sized polynomials by leveraging properties of multilinear polynomials. See also: https://jolt.a16zcrypto.com/how/architecture/opening-proof.html

* "To reduce memory usage, we
introduce the even-odd (EO) folding technique" --> I'm confused by the phrasing. This technique is also widely used in practice: https://github.com/a16z/jolt/blob/63b94268b895e5077a7c4ac4b7e5e63ce2df7b13/jolt-core/src/poly/multilinear_polynomial.rs#L67. The binding order determines whether we are folding "top" half of the vector with "bottom" half of the vector or even-odd en
tries of the vector.

* "We introduce a class of tensor-bucketed vectors (TBVs), in which a large vector is
represented as the concatenation of multiple buckets, and each bucket
is generated by the tensor product of a small number of scalar values." --> How does this relate to tensor decomposition of large "address" vectors (ra) in Twist and Shout (https://eprint.iacr.org/2025/105.pdf)? A similar thing happens in more recent work: https://eprint.iacr.org/2025/1850.pdf.
 
* "Optimal asymptotic efficiency to date." --> Please relate your protocol with protocols widely used in practice (e.g., in Jolt, Nova, etc.).
 
* "To our knowledge, no prior PCS has handled vectors
at this scale. We implement commitments for vectors totaling 16billion entries using 404 GB of RAM, while keeping the succinct
proof size at 103.03 KB. This is the first demonstrated batched PCS at the
scale of large AI models." --> I'm not sure this is the largest batched PCS.
Routingly, zkVMs are run with billions of cycles of RISC-V, which amounts to up
to 100 billion-sized polynomials. Please discuss the scale of you work in relation to all zk systems including zkVMs.
 
* "Bulletproofs [11] introduced folding techniques" --> This is inaccurate. Bulletproofs is a particular instantiation of the sum-check protocol, which dates back to the 1990s.
 
* In the experimental evaluation, there are no baseline approaches. I think it will be important to compare your work's asymptotics and concrete performance with the approaches that are widely used in practice (e.g., please see the links above).

Areas to address in authors’ response
-------------------------------------
* Please provide asymptotic and concrete comparision with approaches used for batching multilinear polynomials of different sizes in practice

Does the paper raise ethical concerns?
--------------------------------------
1. No

Reviewer expertise in this domain
---------------------------------
4. Expert (I've worked on this topic domain or a closely related topic
   domain)

Reviewer confidence
-------------------
4. Confident

Overall merit
-------------
2. Weak reject

Open Science Compliance
-----------------------
3. Not available, but convincingly justified

