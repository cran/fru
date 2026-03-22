use super::attribute::FYSampler;
use crate::ORDER_CACHE_THRESHOLD;
use crate::attribute::{DfAttribute, DfPivot, SplittingIterator};
use crate::tools::ordering_vector;
use std::{cell::OnceCell, iter, sync::OnceLock};
use xrf::{Mask, RfInput, RfRng, VoteAggregator};

mod da;
mod impurity;
mod votes;
pub use votes::Votes;

pub struct DataFrame {
    features: Vec<DfAttribute>,
    order_cache: Vec<OnceLock<Vec<u32>>>,
    decision: Vec<u32>,
    ncat: u32,
    m: usize,
    n: usize,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = u32;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::ClsDaAggregator;
    type FeatureSampler = FYSampler<Self>;
    fn observation_count(&self) -> usize {
        self.n
    }
    fn feature_count(&self) -> usize {
        self.m
    }
    fn feature_sampler(&self) -> Self::FeatureSampler {
        super::attribute::FYSampler::new(self)
    }
    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DecisionSlice::new(mask, &self.decision, self.ncat)
    }
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, f64)> {
        use DfAttribute::*;
        let feature = &self.features[using as usize];

        match *feature {
            Numeric(x) => {
                if on.len() > ORDER_CACHE_THRESHOLD {
                    impurity::scan_f64_cached(
                        x,
                        y,
                        on,
                        self.order_cache[using as usize].get_or_init(|| ordering_vector(x, self.n)),
                    )
                } else {
                    impurity::scan_f64(x, y, on)
                }
            }
            Integer(x) => {
                if on.len() > ORDER_CACHE_THRESHOLD {
                    impurity::scan_i32_cached(
                        x,
                        y,
                        on,
                        self.order_cache[using as usize].get_or_init(|| ordering_vector(x, self.n)),
                    )
                } else {
                    impurity::scan_i32(x, y, on)
                }
            }
            Logical(x) => impurity::scan_bin(x, y, on),
            Factor(xc, x) => impurity::scan_factor(x, xc, y, on, rng),
        }
    }
    fn split_iter(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        by: &Self::Pivot,
    ) -> impl Iterator<Item = bool> {
        let feature = &self.features[using as usize];
        SplittingIterator::new(feature, by, on.iter())
    }
}

pub struct DecisionSlice {
    values: Vec<u32>,
    multiplicities: OnceCell<Vec<u32>>,
    ncat: u32,
    summary: Votes,
}
impl DecisionSlice {
    fn new(mask: &Mask, values: &[u32], ncat: u32) -> Self {
        let mut summary = Votes::new(ncat);
        let values = mask
            .iter()
            .map(|&e| values[e])
            .inspect(|&x| summary.ingest_vote(x))
            .collect();
        DecisionSlice {
            values,
            multiplicities: OnceCell::new(),
            ncat,
            summary,
        }
    }
    fn provide_mult(&self, mask: &Mask, n: u32) -> &[u32] {
        self.multiplicities.get_or_init(|| {
            let mut ans = vec![0; n as usize];
            for e in mask.iter() {
                ans[*e] += 1;
            }
            ans
        })
    }
}

impl xrf::DecisionSlice<u32> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.summary.is_pure()
    }
    fn condense(&self, rng: &mut RfRng) -> u32 {
        //Leaf always need a single class, so select
        // random one if there is no data
        self.summary.collapse_empty_random(rng)
    }
}

//Boring stuff

impl DataFrame {
    //TOOD: Better order of arguments, maybe?
    pub fn new(
        features: Vec<DfAttribute>,
        decision: Vec<u32>,
        ncat: u32,
        m: usize,
        n: usize,
    ) -> Self {
        Self {
            features,
            order_cache: iter::repeat_with(|| OnceLock::new()).take(m).collect(),
            decision,
            ncat,
            m,
            n,
        }
    }
}
