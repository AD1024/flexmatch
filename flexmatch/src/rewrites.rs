use egg::Rewrite;
use glenside::language::rewrites::*;
use glenside::language::{Language, MyAnalysis};

pub fn get_rewrite_from_string(name: String) -> Rewrite<Language, MyAnalysis> {
    match name.as_str() {
        "bubble-reshape-through-cartesian-product" => bubble_reshape_through_cartesian_product(),
        "flatten-unflatten-all-accesses" => flatten_unflatten_any_access(),
        "bubble-reshape-through-linear" => bubble_reshape_through_linear_generalized(),
        "access-reshape-to-relay" => access_reshape_to_relay(),
        "bubble-reshape-through-compute-dot-product" => {
            bubble_reshape_through_compute_dot_product()
        }

        "flex-linear-rewrite" => linear_layer_accelerator_rewrites(),
        "vta-dense-rewrite" => dot_product_with_vta(),
        _ => {
            eprintln!("{} not implemented", name);
            todo!()
        }
    }
}

pub fn im2col_rewrites() -> Vec<Rewrite<Language, MyAnalysis>> {
    vec![
        flatten_unflatten_any_access(),
        bubble_reshape_through_compute_dot_product(),
        bubble_reshape_through_cartesian_product(),
    ]
}

pub fn linear_rewrites() -> Vec<Rewrite<Language, MyAnalysis>> {
    vec![bubble_reshape_through_linear_generalized()]
}
