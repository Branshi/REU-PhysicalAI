"""Construct backward-compatible SETNODE architecture variants."""

from .graph_network import EncodeProcessDecode
from .graph_network_ablation import GenericAblationEncodeProcessDecode


BASELINE_VARIANT = "baseline"
GENERIC_ABLATION_VARIANT = "generic_ablation"
MODEL_VARIANTS = (BASELINE_VARIANT, GENERIC_ABLATION_VARIANT)


def build_setnode_model(
    *,
    model_variant=BASELINE_VARIANT,
    node_input_dim,
    edge_input_dim,
    latent_dim=128,
    hidden_dim=128,
    num_message_passing_steps=10,
    num_hidden_layers=2,
    node_mean=None,
    node_std=None,
    edge_mean=None,
    edge_std=None,
    num_heads=4,
    distance_dim=16,
    ffn_dim=256,
    epsilon=0.15,
    rbf_initial_scale=1.0,
    global_gate_init=1e-2,
):
    common_arguments = {
        "node_input_dim": node_input_dim,
        "edge_input_dim": edge_input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_message_passing_steps": num_message_passing_steps,
        "num_hidden_layers": num_hidden_layers,
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "num_heads": num_heads,
        "distance_dim": distance_dim,
        "ffn_dim": ffn_dim,
        "epsilon": epsilon,
    }

    if model_variant == BASELINE_VARIANT:
        return EncodeProcessDecode(**common_arguments)
    if model_variant == GENERIC_ABLATION_VARIANT:
        return GenericAblationEncodeProcessDecode(
            **common_arguments,
            rbf_initial_scale=rbf_initial_scale,
            global_gate_init=global_gate_init,
        )

    raise ValueError(
        f"Unknown SETNODE model variant {model_variant!r}. "
        f"Expected one of {MODEL_VARIANTS}."
    )
