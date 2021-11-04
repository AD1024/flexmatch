use egg::{Rewrite, rewrite};
use glenside::language::rewrites::*;
use glenside::language::{Language, MyAnalysis};

pub fn get_rewrite_from_string(name: &String, args: &Box<[i32]>) -> Rewrite<Language, MyAnalysis> {
    match name.as_str() {
        "bubble-reshape-through-cartesian-product" => bubble_reshape_through_cartesian_product(),
        "flatten-unflatten-all-accesses" => flatten_unflatten_any_access(),
        "bubble-reshape-through-linear" => bubble_reshape_through_linear_generalized(),
        "access-reshape-to-relay" => access_reshape_to_relay(),
        "bubble-reshape-through-compute-dot-product" => {
            bubble_reshape_through_compute_dot_product()
        }
        "reassociate-max" => match args[..] {
            [h, w] => reassociate_max(h as usize, w as usize),
            _ => panic!("Expecting H and W from reassociate_max, got {:?}", args),
        },
        "bubble-access-reshape-through-compute-reduce-max" => {
            bubble_access_reshape_through_compute_reduce_max()
        }
        "simplify-multiple-accesses" => simplify_multiple_accesses(),
        "simplify-multiple-transposes" => simplify_multiple_transposes(),
        "simplify-multiple-access-reshapes" => simplify_multiple_access_reshapes(),
        "bubble-access-through-access-transpose" => bubble_access_through_access_transpose(),
        "simplify-reduce-max" => simplify_reduce_max(),

        "flex-linear-rewrite" => linear_layer_accelerator_rewrites(),
        "flex-linear-dense" => dot_product_to_linear(),
        "hlscnn-conv2d" => conv2d_on_hlscnn(),
        "vta-dense-rewrite" => dot_product_with_vta(),
        "flexasr-maxpool" => flexasr_maxpool(),
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
