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


def build_node_features(velocities, masses):
    """
    Build node features for each body.

    Parameters:
        velocities: torch.Tensor
            Shape: [num_bodies, dim]

        masses: torch.Tensor
            Shape: [num_bodies, 1]

    Returns:
        node_features : torch.Tensor
            Shape: [num_bodies, dim + 1]
    """

    node_features = torch.cat([velocities, masses], dim=-1)

    return node_features


def build_edge_features(positions, masses, senders, receivers, grav=1, epsilon=1e-8):
    """
    Build edge features using relative displacement and distance

    Parameters:
        positions : torch.Tensor
            Shape: [num_bodies, dim]

        senders : torch.Tensor
            Shape: [num_edges]

        receivers : torch.Tensor
            Shape: [num_edges]

        Returns:
            edge_features : torch.Tensor
                Shape : [num_edges, dim + 1]
    """

    sender_positions = positions[senders]
    receiver_positions = positions[receivers]

    relative_displacement = sender_positions - receiver_positions

    dist_sq = (relative_displacement**2).sum(dim=-1, keepdim=True)
    softened_distance = torch.sqrt(dist_sq + epsilon**2)

    inv_distance = 1.0 / softened_distance
    inv_distance_sq = inv_distance**2
    inv_distance_cubed = inv_distance**3
    sender_mass = masses[senders]
    accel_contribution = grav * sender_mass * relative_displacement * inv_distance_cubed

    edge_features = torch.cat(
        [
            relative_displacement,
            softened_distance,
            inv_distance,
            inv_distance_sq,
            sender_mass,
            accel_contribution,
        ],
        dim=-1,
    )
    return edge_features


def build_graph(positions, velocities, masses, grav=1):
    """
    Convert one N-Body time step into graph features

    Parameters:
        positions : torch.Tensor
            Shape : [num_bodies, dim]

        velocities : torch.Tensor
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

    node_features = build_node_features(velocities=velocities, masses=masses)

    edge_features = build_edge_features(
        positions=positions,
        masses=masses,
        grav=grav,
        senders=senders,
        receivers=receivers,
    )

    graph = {
        "node_features": node_features,
        "edge_features": edge_features,
        "senders": senders,
        "receivers": receivers,
    }

    return graph
