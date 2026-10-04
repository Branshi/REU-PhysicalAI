import torch


MASS_FEATURE_MODES = ("raw", "raw_log")


def build_mass_features(masses, mass_feature_mode="raw"):
    """Encode particle masses without discarding their absolute scale."""

    if mass_feature_mode == "raw":
        return masses
    if mass_feature_mode == "raw_log":
        # Raw mass preserves the absolute gravitational scale. Log mass keeps
        # small bodies distinguishable when one dominant body sets the raw-mass
        # normalization statistics.
        log_masses = torch.log10(masses.clamp_min(1e-12))
        return torch.cat([masses, log_masses], dim=-1)

    raise ValueError(
        f"Unknown mass feature mode {mass_feature_mode!r}. "
        f"Expected one of {MASS_FEATURE_MODES}."
    )


def fully_connected_edges(num_bodies, device="cpu"):
    """
    Create a fully connected directed graph without self-edges.

    Returns:
        senders : torch.Tensor
            Shape: [num_edges]

        receivers : torch.Tensor
            Shape : [num_edges]

    """

    # dtype = torch.bool converts the values in the identity matrix into true and false and the ~ simply inverts all those values.
    # torch.where returns the indicies i, j where the value is true.
    receivers, senders = torch.where(
        ~torch.eye(num_bodies, dtype=torch.bool, device=device)
    )

    return senders, receivers


def build_node_features(masses, mass_feature_mode="raw"):
    """
    Build node features for each body.

    Parameters:
        masses: torch.Tensor
            Shape: [num_bodies, 1] for ``raw`` and [num_bodies, 2] for
            ``raw_log``.

    Returns:
        node_features : torch.Tensor
            Shape: [num_bodies, num_mass_features]
    """

    return build_mass_features(
        masses=masses,
        mass_feature_mode=mass_feature_mode,
    )


def build_edge_features(
    masses,
    senders,
    receivers,
    mass_feature_mode="raw",
):
    """
    Build coordinate-independent edge features from particle masses.

    Parameters:
        masses : torch.Tensor
            Shape: [num_bodies, 1]

        senders : torch.Tensor
            Shape: [num_edges]

        receivers : torch.Tensor
            Shape: [num_edges]

        Returns:
            edge_features : torch.Tensor
                Shape : [num_edges, 2] for ``raw`` and [num_edges, 4] for
                ``raw_log``.
    """

    mass_features = build_mass_features(
        masses=masses,
        mass_feature_mode=mass_feature_mode,
    )
    sender_mass_features = mass_features[senders]
    receiver_mass_features = mass_features[receivers]

    edge_features = torch.cat(
        [
            sender_mass_features,
            receiver_mass_features,
        ],
        dim=-1,
    )
    return edge_features


def build_graph(positions, masses, mass_feature_mode="raw"):
    """
    Convert one N-Body time step into graph features

    Parameters:
        positions : torch.Tensor
            Shape : [num_bodies, dim]

        masses : torch.Tensor
            Shape : [num_bodies, 1]

    Returns:
        graph : dict
            Dictionary containing node_features, edge_features, senders, receivers
    """

    device = positions.device
    num_bodies = positions.shape[0]

    senders, receivers = fully_connected_edges(
        num_bodies=num_bodies,
        device=device,
    )

    node_features = build_node_features(
        masses=masses,
        mass_feature_mode=mass_feature_mode,
    )

    edge_features = build_edge_features(
        masses=masses,
        senders=senders,
        receivers=receivers,
        mass_feature_mode=mass_feature_mode,
    )

    graph = {
        "node_features": node_features,
        "edge_features": edge_features,
        "coordinates": positions,
        "masses": masses,
        "senders": senders,
        "receivers": receivers,
    }

    return graph
