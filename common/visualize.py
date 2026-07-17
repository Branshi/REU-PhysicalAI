from pathlib import Path

import torch


def plot_trajectories(
    true_positions=None,
    predicted_positions=None,
    title="N-body trajectories",
    show=True,
    save_path=None,
    dpi=300,
):
    """Create and optionally save a static 2D trajectory comparison."""

    import matplotlib.pyplot as plt

    if true_positions is None and predicted_positions is None:
        raise ValueError("Provide true_positions, predicted_positions, or both.")

    if true_positions is not None:
        true_positions = true_positions.detach().cpu()

    if predicted_positions is not None:
        predicted_positions = predicted_positions.detach().cpu()

    reference_positions = (
        true_positions if true_positions is not None else predicted_positions
    )
    _, num_bodies, dim = reference_positions.shape

    if dim != 2:
        raise ValueError("This plotting function only supports 2D trajectories.")

    figure, axis = plt.subplots(figsize=(8, 8))

    for body_idx in range(num_bodies):
        if true_positions is not None:
            x_true = true_positions[:, body_idx, 0]
            y_true = true_positions[:, body_idx, 1]

            axis.plot(x_true, y_true, label=f"Body {body_idx} true")
            axis.scatter(x_true[0], y_true[0], marker="o")
            axis.scatter(x_true[-1], y_true[-1], marker="x")

        if predicted_positions is not None:
            x_pred = predicted_positions[:, body_idx, 0]
            y_pred = predicted_positions[:, body_idx, 1]

            axis.plot(
                x_pred,
                y_pred,
                linestyle="--",
                label=f"Body {body_idx} predicted",
            )
            axis.scatter(x_pred[0], y_pred[0], marker="o")
            axis.scatter(x_pred[-1], y_pred[-1], marker="x")

    axis.set_title(title)
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(True)

    resolved_save_path = None
    if save_path is not None:
        resolved_save_path = Path(save_path).expanduser().resolve()
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved static trajectory plot: {resolved_save_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return resolved_save_path


def animate_trajectories(
    true_positions=None,
    predicted_positions=None,
    masses=None,
    dt=None,
    interval=50,
    save_path=None,
    fps=None,
    show=True,
    style="dark",
    trail_length=None,
):
    """Animate true and predicted 2D trajectories."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import writers
    from matplotlib.lines import Line2D

    if true_positions is None and predicted_positions is None:
        raise ValueError("Provide true_positions, predicted_positions, or both.")

    if true_positions is not None:
        true_positions = true_positions.detach().cpu()

    if predicted_positions is not None:
        predicted_positions = predicted_positions.detach().cpu()

    if masses is not None:
        masses = masses.detach().cpu().view(-1)

    reference_positions = (
        true_positions if true_positions is not None else predicted_positions
    )
    num_steps, num_bodies, dim = reference_positions.shape

    if (
        true_positions is not None
        and predicted_positions is not None
        and true_positions.shape != predicted_positions.shape
    ):
        raise ValueError(
            "true_positions and predicted_positions must have the same shape "
            f"for animation. Got {true_positions.shape} and {predicted_positions.shape}."
        )

    if dim != 2:
        raise ValueError("This animation only supports 2D trajectories.")

    dark_style = style == "dark"
    background_color = "#0b1020" if dark_style else "white"
    axes_color = "#111827" if dark_style else "white"
    text_color = "#e5e7eb" if dark_style else "#111827"
    grid_color = "#374151" if dark_style else "#d1d5db"
    spine_color = "#9ca3af" if dark_style else "#111827"

    body_colors = [
        "#38bdf8",
        "#f97316",
        "#a78bfa",
        "#22c55e",
        "#facc15",
        "#fb7185",
    ]
    colors = [body_colors[idx % len(body_colors)] for idx in range(num_bodies)]

    if masses is None:
        marker_sizes = torch.full((num_bodies,), 120.0)
    else:
        mass_min = masses.min()
        mass_range = masses.max() - mass_min
        if mass_range < 1e-8:
            marker_sizes = torch.full((num_bodies,), 140.0)
        else:
            marker_sizes = 90.0 + 130.0 * (masses - mass_min) / mass_range

    marker_sizes = marker_sizes.tolist()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=background_color)
    ax.set_facecolor(axes_color)

    all_positions = []
    if true_positions is not None:
        all_positions.append(true_positions)
    if predicted_positions is not None:
        all_positions.append(predicted_positions)

    combined_positions = torch.cat(all_positions, dim=1)

    x_min = combined_positions[:, :, 0].min().item()
    x_max = combined_positions[:, :, 0].max().item()
    y_min = combined_positions[:, :, 1].min().item()
    y_max = combined_positions[:, :, 1].max().item()

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)
    padding = max(0.25, 0.12 * radius)
    radius = radius + padding

    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
    ax.set_aspect("equal")
    ax.set_title("N-body rollout", color=text_color, fontsize=18, pad=14)
    ax.set_xlabel("x position", color=text_color)
    ax.set_ylabel("y position", color=text_color)
    ax.tick_params(colors=text_color)
    ax.grid(True, color=grid_color, linewidth=0.8, alpha=0.45)

    for spine in ax.spines.values():
        spine.set_color(spine_color)

    if true_positions is not None:
        true_points = ax.scatter(
            true_positions[0, :, 0],
            true_positions[0, :, 1],
            s=marker_sizes,
            c=colors,
            edgecolors="#f9fafb" if dark_style else "#111827",
            linewidths=1.2,
            label="True",
            zorder=5,
        )

        true_trails = [
            ax.plot([], [], color=colors[body_idx], linewidth=2.2, alpha=0.85)[0]
            for body_idx in range(num_bodies)
        ]
    else:
        true_points = None
        true_trails = []

    if predicted_positions is not None:
        predicted_points = ax.scatter(
            predicted_positions[0, :, 0],
            predicted_positions[0, :, 1],
            s=[size * 0.9 for size in marker_sizes],
            c=colors,
            marker="x",
            linewidths=2.4,
            label="Predicted",
            zorder=6,
        )

        predicted_trails = [
            ax.plot(
                [],
                [],
                color=colors[body_idx],
                linestyle="--",
                linewidth=1.8,
                alpha=0.7,
            )[0]
            for body_idx in range(num_bodies)
        ]
    else:
        predicted_points = None
        predicted_trails = []

    frame_text = ax.text(
        0.03,
        0.96,
        "",
        transform=ax.transAxes,
        color=text_color,
        fontsize=12,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#020617" if dark_style else "white",
            "edgecolor": spine_color,
            "alpha": 0.78,
        },
    )

    legend_handles = []
    if true_positions is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=text_color,
                markeredgecolor=text_color,
                markersize=8,
                label="True",
            )
        )
    if predicted_positions is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color=text_color,
                linestyle="none",
                markersize=8,
                markeredgewidth=2,
                label="Predicted",
            )
        )
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.82,
        facecolor="#020617" if dark_style else "white",
        edgecolor=spine_color,
    )

    for label in legend.get_texts():
        label.set_color(text_color)

    def frame_label(frame):
        if dt is None:
            return f"step {frame}/{num_steps - 1}"
        return f"step {frame}/{num_steps - 1}   t={frame * dt:.2f}"

    def init():
        artists = [frame_text]

        if true_points is not None:
            true_points.set_offsets(torch.empty((0, 2)))
            artists.append(true_points)

            for line in true_trails:
                line.set_data([], [])

        frame_text.set_text(frame_label(0))
        artists += true_trails

        if predicted_points is not None:
            predicted_points.set_offsets(torch.empty((0, 2)))

            for line in predicted_trails:
                line.set_data([], [])

            artists += [predicted_points] + predicted_trails

        return artists

    def update(frame):
        frame_text.set_text(frame_label(frame))
        trail_start = 0 if trail_length is None else max(0, frame + 1 - trail_length)

        artists = [frame_text]

        if true_positions is not None and true_points is not None:
            true_xy = true_positions[frame]
            true_points.set_offsets(true_xy)

            for body_idx, line in enumerate(true_trails):
                line.set_data(
                    true_positions[trail_start : frame + 1, body_idx, 0],
                    true_positions[trail_start : frame + 1, body_idx, 1],
                )

            artists += [true_points] + true_trails

        if predicted_positions is not None and predicted_points is not None:
            predicted_xy = predicted_positions[frame]
            predicted_points.set_offsets(predicted_xy)

            for body_idx, line in enumerate(predicted_trails):
                line.set_data(
                    predicted_positions[trail_start : frame + 1, body_idx, 0],
                    predicted_positions[trail_start : frame + 1, body_idx, 1],
                )

            artists += [predicted_points] + predicted_trails

        return artists

    save_writer = None
    save_fps = None
    if save_path is not None:
        save_fps = fps if fps is not None else max(1, round(1000 / interval))
        suffix = Path(save_path).suffix.lower()

        if suffix == ".gif":
            save_writer = "pillow"
        elif suffix in {".mp4", ".m4v", ".mov"}:
            if not writers.is_available("ffmpeg"):
                plt.close(fig)
                raise RuntimeError(
                    "MP4/MOV export requires ffmpeg. Install it with "
                    "`brew install ffmpeg`, or save as a .gif instead."
                )
            save_writer = "ffmpeg"
        else:
            plt.close(fig)
            raise ValueError("save_path must end in .gif, .mp4, .m4v, or .mov.")

    animation = FuncAnimation(
        fig,
        update,
        frames=num_steps,
        init_func=init,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        animation.save(save_path, writer=save_writer, fps=save_fps, dpi=140)
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation
