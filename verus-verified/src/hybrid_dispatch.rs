use vstd::prelude::*;

verus! {

proof fn cache_layer_map_count_consistent(
    num_layers: nat,
    num_attn: nat,
    num_deltanet: nat,
)
    requires
        num_layers >= 1,
        num_attn + num_deltanet == num_layers,
        num_attn >= 1,
    ensures
        num_attn <= num_layers,
        num_deltanet < num_layers,
{
}

proof fn cache_layer_map_index_valid(
    num_layers: nat,
    num_attn: nat,
    cache_idx: nat,
)
    requires
        num_layers >= 1,
        num_attn >= 1,
        num_attn <= num_layers,
        cache_idx < num_attn,
    ensures
        cache_idx < num_attn,
{
}

proof fn deltanet_state_index_valid(
    num_layers: nat,
    num_deltanet: nat,
    dn_idx: nat,
)
    requires
        num_layers >= 1,
        num_deltanet >= 1,
        num_deltanet <= num_layers,
        dn_idx < num_deltanet,
    ensures
        dn_idx < num_deltanet,
{
}

fn main() {}

} // verus!
