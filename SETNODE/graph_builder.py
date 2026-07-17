import torch


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


def build_node_features(masses):
    """
    Build node features for each body.

    Parameters:
        masses: torch.Tensor
            Shape: [num_bodies, 1]

    Returns:
        node_features : torch.Tensor
            Shape: [num_bodies, 1]
    """

    return masses


def build_edge_features(masses, senders, receivers):
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
                Shape : [num_edges, 2]
    """

    sender_mass = masses[senders]
    receiver_mass = masses[receivers]

    edge_features = torch.cat(
        [
            sender_mass,
            receiver_mass,
        ],
        dim=-1,
    )
    return edge_features


def build_graph(positions, masses):
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

    node_features = build_node_features(masses=masses)

    edge_features = build_edge_features(
        masses=masses,
        senders=senders,
        receivers=receivers,
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
