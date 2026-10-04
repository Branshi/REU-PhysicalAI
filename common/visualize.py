from pathlib import Path

import torch


def _apply_bloom_to_frame(
    frame,
    bloom_source=None,
    radius=12.0,
    strength=1.25,
    brightness_threshold=0.42,
    saturation_threshold=0.18,
):
    """Add selective, additive bloom to a rendered RGB frame.

    When ``bloom_source`` is provided, only bright pixels from that separate
    render contribute to the glow while the effect is composited onto
    ``frame``.
    """

    import numpy as np

    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise RuntimeError(
            "Saved-animation bloom requires Pillow. Install the optional "
            "visualization dependencies from requirements-visualization.txt."
        ) from exc

    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[-1] not in {3, 4}:
        raise ValueError(
            "Bloom expects an RGB or RGBA frame. "
            f"Got an array shaped {frame_array.shape}."
        )

    source_array = (
        frame_array if bloom_source is None else np.asarray(bloom_source)
    )
    if source_array.shape != frame_array.shape:
        raise ValueError(
            "Bloom source and rendered frame must have matching shapes. "
            f"Got {source_array.shape} and {frame_array.shape}."
        )

    rgb = frame_array[..., :3].astype(np.float32) / 255.0
    source_rgb = source_array[..., :3].astype(np.float32) / 255.0
    brightest_channel = source_rgb.max(axis=-1)
    darkest_channel = source_rgb.min(axis=-1)
    saturation = np.divide(
        brightest_channel - darkest_channel,
        brightest_channel,
        out=np.zeros_like(brightest_channel),
        where=brightest_channel > 1e-6,
    )

    brightness_weight = np.clip(
        (brightest_channel - brightness_threshold) / (1.0 - brightness_threshold),
        0.0,
        1.0,
    )
    saturation_weight = np.clip(
        (saturation - saturation_threshold) / (1.0 - saturation_threshold),
        0.0,
        1.0,
    )
    bright_pass = source_rgb * (brightness_weight * saturation_weight)[..., None]

    bright_image = Image.fromarray(
        np.clip(bright_pass * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="RGB",
    )
    tight_glow = (
        np.asarray(
            bright_image.filter(
                ImageFilter.GaussianBlur(radius=max(1.0, radius * 0.35))
            )
        ).astype(np.float32)
        / 255.0
    )
    wide_glow = (
        np.asarray(bright_image.filter(ImageFilter.GaussianBlur(radius=radius))).astype(
            np.float32
        )
        / 255.0
    )

    glow = strength * (0.8 * tight_glow + 1.15 * wide_glow)
    bloomed_rgb = np.clip(rgb + glow, 0.0, 1.0)
    result = np.clip(bloomed_rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    if frame_array.shape[-1] == 4:
        result = np.concatenate((result, frame_array[..., 3:4]), axis=-1)
    return result


def plot_trajectories(
    true_positions=None,
    predicted_positions=None,
    title="N-body trajectories",
    show=True,
    save_path=None,
    dpi=300,
):
    """Create and optionally save a static 2D or 3D trajectory comparison."""

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

    if dim not in {2, 3}:
        raise ValueError(
            f"Trajectory coordinates must be 2D or 3D. Got a final dimension of {dim}."
        )

    figure = plt.figure(figsize=(8, 8))
    if dim == 3:
        axis = figure.add_subplot(111, projection="3d")
    else:
        axis = figure.add_subplot(111)

    start_marker_size = 140
    end_marker_size = 180
    marker_line_width = 2.5

    for body_idx in range(num_bodies):
        if true_positions is not None:
            x_true = true_positions[:, body_idx, 0]
            y_true = true_positions[:, body_idx, 1]

            if dim == 3:
                z_true = true_positions[:, body_idx, 2]
                axis.plot(x_true, y_true, z_true, label=f"Body {body_idx} true")
                axis.scatter(
                    x_true[0],
                    y_true[0],
                    z_true[0],
                    marker="o",
                    s=start_marker_size,
                    linewidths=marker_line_width,
                )
                axis.scatter(
                    x_true[-1],
                    y_true[-1],
                    z_true[-1],
                    marker="x",
                    s=end_marker_size,
                    linewidths=marker_line_width,
                )
            else:
                axis.plot(x_true, y_true, label=f"Body {body_idx} true")
                axis.scatter(
                    x_true[0],
                    y_true[0],
                    marker="o",
                    s=start_marker_size,
                    linewidths=marker_line_width,
                )
                axis.scatter(
                    x_true[-1],
                    y_true[-1],
                    marker="x",
                    s=end_marker_size,
                    linewidths=marker_line_width,
                )

        if predicted_positions is not None:
            x_pred = predicted_positions[:, body_idx, 0]
            y_pred = predicted_positions[:, body_idx, 1]

            if dim == 3:
                z_pred = predicted_positions[:, body_idx, 2]
                axis.plot(
                    x_pred,
                    y_pred,
                    z_pred,
                    linestyle="--",
                    label=f"Body {body_idx} predicted",
                )
                axis.scatter(
                    x_pred[0],
                    y_pred[0],
                    z_pred[0],
                    marker="o",
                    s=start_marker_size,
                    linewidths=marker_line_width,
                )
                axis.scatter(
                    x_pred[-1],
                    y_pred[-1],
                    z_pred[-1],
                    marker="x",
                    s=end_marker_size,
                    linewidths=marker_line_width,
                )
            else:
                axis.plot(
                    x_pred,
                    y_pred,
                    linestyle="--",
                    label=f"Body {body_idx} predicted",
                )
                axis.scatter(
                    x_pred[0],
                    y_pred[0],
                    marker="o",
                    s=start_marker_size,
                    linewidths=marker_line_width,
                )
                axis.scatter(
                    x_pred[-1],
                    y_pred[-1],
                    marker="x",
                    s=end_marker_size,
                    linewidths=marker_line_width,
                )

    axis.set_title(title)
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    if dim == 3:
        axis.set_zlabel("z position")
        combined = []
        if true_positions is not None:
            combined.append(true_positions)
        if predicted_positions is not None:
            combined.append(predicted_positions)
        combined_positions = torch.cat(combined, dim=1).flatten(0, 1)
        position_range = (
            combined_positions.amax(dim=0) - combined_positions.amin(dim=0)
        ).clamp_min(1e-8)
        axis.set_box_aspect(position_range.numpy())
    else:
        axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=13, markerscale=1.15, framealpha=0.92)
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
    loop=False,
    bloom=False,
    bloom_largest_only=False,
    body_size_scale=1.5,
):
    """Animate trajectories with Matplotlib in 2D or PyVista in 3D.

    The renderer is selected from the final coordinate dimension. PyVista is
    imported lazily so existing 2D workflows do not require it.
    """

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
            "for animation. Got "
            f"{true_positions.shape} and {predicted_positions.shape}."
        )

    if dim == 3:
        return animate_trajectories_3d(
            true_positions=true_positions,
            predicted_positions=predicted_positions,
            masses=masses,
            dt=dt,
            interval=interval,
            save_path=save_path,
            fps=fps,
            show=show,
            style=style,
            trail_length=trail_length,
            loop=loop,
            bloom=bloom,
            bloom_largest_only=bloom_largest_only,
            body_size_scale=body_size_scale,
        )

    if dim != 2:
        raise ValueError(
            f"Trajectory coordinates must be 2D or 3D. Got a final dimension of {dim}."
        )

    if bloom:
        raise ValueError("Bloom is only available for saved 3D animations.")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import writers
    from matplotlib.lines import Line2D

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


def animate_trajectories_3d(
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
    loop=False,
    bloom=False,
    bloom_largest_only=False,
    window_size=(1000, 900),
    star_count=0,
    body_size_scale=1.5,
):
    """Render an interactive or saved 3D N-body animation with PyVista.

    Solid spheres and trails show reference trajectories. Wireframe spheres and
    translucent trails show predictions when both are present. Prediction-only
    animations use solid neon bodies. When masses are provided, sphere radii
    follow cube-root mass scaling, corresponding to equal-density bodies.
    ``body_size_scale`` enlarges every body without changing those relative
    proportions. The default dark style uses a clean black background with no
    star field.
    """

    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "3D trajectory animation requires PyVista. Install the optional "
            "visualization dependencies with `python -m pip install pyvista "
            "imageio imageio-ffmpeg`."
        ) from exc

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

    if num_steps == 0:
        raise ValueError("Trajectory animation requires at least one time step.")

    if dim != 3:
        raise ValueError(
            "PyVista animation requires positions shaped [steps, bodies, 3]. "
            f"Got {tuple(reference_positions.shape)}."
        )

    if (
        true_positions is not None
        and predicted_positions is not None
        and true_positions.shape != predicted_positions.shape
    ):
        raise ValueError(
            "true_positions and predicted_positions must have the same shape "
            "for animation. Got "
            f"{true_positions.shape} and {predicted_positions.shape}."
        )

    if masses is not None and masses.numel() != num_bodies:
        raise ValueError(
            f"masses must contain one value per body ({num_bodies}). "
            f"Got {masses.numel()}."
        )

    if interval <= 0:
        raise ValueError("interval must be positive.")

    if trail_length is not None and trail_length <= 0:
        raise ValueError("trail_length must be positive when provided.")

    if star_count < 0:
        raise ValueError("star_count cannot be negative.")

    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive when provided.")

    if body_size_scale <= 0:
        raise ValueError("body_size_scale must be positive.")

    if loop and not show:
        raise ValueError("Looping requires an interactive window; omit --no-show.")

    if bloom and save_path is None:
        raise ValueError("Bloom requires a saved 3D animation; provide --save-path.")

    if bloom_largest_only and not bloom:
        raise ValueError("bloom_largest_only requires bloom to be enabled.")

    if bloom_largest_only and masses is None:
        raise ValueError("Largest-body-only bloom requires body masses.")

    dark_style = style == "dark"
    background_color = "#000000" if dark_style else "#f8fafc"
    text_color = "#f8fafc" if dark_style else "#111827"
    accent_color = "#94a3b8" if dark_style else "#475569"
    body_colors = [
        "#00E5FF",
        "#FF5A1F",
        "#C084FC",
        "#39FF88",
        "#FFE600",
        "#FF3CAC",
    ]
    colors = [body_colors[idx % len(body_colors)] for idx in range(num_bodies)]
    largest_body_idx = None if masses is None else int(torch.argmax(masses).item())

    all_positions = []
    if true_positions is not None:
        all_positions.append(true_positions)
    if predicted_positions is not None:
        all_positions.append(predicted_positions)
    combined_positions = torch.cat(all_positions, dim=1)
    bounds_min = combined_positions.amin(dim=(0, 1)).numpy()
    bounds_max = combined_positions.amax(dim=(0, 1)).numpy()
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    scene_radius = max(0.5 * extent, 1.0)

    if masses is None:
        body_radii = np.full(
            num_bodies,
            0.045 * scene_radius * body_size_scale,
        )
    else:
        mass_values = masses.numpy()
        if not np.all(np.isfinite(mass_values)) or np.any(mass_values <= 0.0):
            raise ValueError("3D body sizing requires finite, positive masses.")

        # For equal-density spheres, mass is proportional to volume, so radius
        # scales as the cube root of mass. Normalize by the largest body and
        # preserve the mass ratios while applying one shared visual-size
        # multiplier. A value of 1.0 reproduces the previous rendered sizes.
        radius_scales = np.cbrt(mass_values / float(mass_values.max()))
        body_radii = (
            0.063 * scene_radius * body_size_scale * radius_scales
        )

    resolved_save_path = None
    save_fps = fps if fps is not None else max(1, round(1000 / interval))
    if save_path is not None:
        resolved_save_path = Path(save_path).expanduser().resolve()
        if resolved_save_path.suffix.lower() not in {".gif", ".mp4", ".m4v", ".mov"}:
            raise ValueError("save_path must end in .gif, .mp4, .m4v, or .mov.")
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)

    if loop and resolved_save_path is not None:
        raise ValueError(
            "Looping is only available for interactive, unsaved animations. "
            "Omit --loop-animation when using --save-path."
        )

    plotter = pv.Plotter(
        notebook=False,
        off_screen=not show,
        window_size=list(window_size),
    )
    plotter.set_background(background_color)

    if dark_style and star_count > 0:
        rng = np.random.default_rng(2026)
        directions = rng.normal(size=(star_count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        distances = rng.uniform(1.7, 2.6, size=(star_count, 1)) * scene_radius
        stars = center + directions * distances
        star_cloud = pv.PolyData(stars)
        plotter.add_points(
            star_cloud,
            style="points",
            color="#dbeafe",
            point_size=3.0,
            render_points_as_spheres=True,
            lighting=False,
            opacity=0.95,
        )

    def new_trail_mesh(initial_point):
        points = np.repeat(np.asarray(initial_point, dtype=float)[None, :], 2, axis=0)
        return pv.PolyData(points, lines=np.array([2, 0, 1]))

    def update_trail_mesh(mesh, points):
        points = np.asarray(points, dtype=float)
        if len(points) == 1:
            points = np.repeat(points, 2, axis=0)
        mesh.points = points
        mesh.lines = np.concatenate((
            [len(points)],
            np.arange(len(points), dtype=np.int64),
        ))

    true_actors = []
    true_halos = []
    true_trails = []
    true_trail_halos = []
    if true_positions is not None:
        for body_idx in range(num_bodies):
            sphere = pv.Sphere(
                radius=float(body_radii[body_idx]),
                theta_resolution=32,
                phi_resolution=24,
            )
            actor = plotter.add_mesh(
                sphere,
                color=colors[body_idx],
                smooth_shading=True,
                ambient=0.75,
                diffuse=0.55,
                specular=1.0,
                specular_power=90.0,
            )
            actor.position = true_positions[0, body_idx].numpy()
            true_actors.append(actor)

            halo_sphere = pv.Sphere(
                radius=float(body_radii[body_idx] * 1.38),
                theta_resolution=32,
                phi_resolution=24,
            )
            halo = plotter.add_mesh(
                halo_sphere,
                color=colors[body_idx],
                smooth_shading=True,
                opacity=0.13,
                lighting=False,
            )
            halo.position = true_positions[0, body_idx].numpy()
            true_halos.append(halo)

            trail = new_trail_mesh(true_positions[0, body_idx].numpy())
            plotter.add_mesh(
                trail,
                color=colors[body_idx],
                line_width=4.0,
                render_lines_as_tubes=True,
                opacity=1.0,
                lighting=False,
            )
            true_trails.append(trail)

            trail_halo = new_trail_mesh(true_positions[0, body_idx].numpy())
            plotter.add_mesh(
                trail_halo,
                color=colors[body_idx],
                line_width=10.0,
                render_lines_as_tubes=True,
                opacity=0.13,
                lighting=False,
            )
            true_trail_halos.append(trail_halo)

    predicted_only = true_positions is None
    predicted_actors = []
    predicted_halos = []
    predicted_trails = []
    predicted_trail_halos = []
    if predicted_positions is not None:
        for body_idx in range(num_bodies):
            sphere = pv.Sphere(
                radius=float(body_radii[body_idx]),
                theta_resolution=32,
                phi_resolution=24,
            )
            if predicted_only:
                actor = plotter.add_mesh(
                    sphere,
                    color=colors[body_idx],
                    smooth_shading=True,
                    ambient=0.8,
                    diffuse=0.5,
                    specular=1.0,
                    specular_power=100.0,
                )
            else:
                actor = plotter.add_mesh(
                    sphere,
                    color=colors[body_idx],
                    style="wireframe",
                    line_width=3.0,
                    opacity=1.0,
                )
            actor.position = predicted_positions[0, body_idx].numpy()
            predicted_actors.append(actor)

            halo_sphere = pv.Sphere(
                radius=float(body_radii[body_idx] * 1.1),
                theta_resolution=32,
                phi_resolution=24,
            )
            if predicted_only:
                halo = plotter.add_mesh(
                    halo_sphere,
                    color=colors[body_idx],
                    smooth_shading=True,
                    opacity=0.15,
                    lighting=False,
                )
            else:
                halo = plotter.add_mesh(
                    halo_sphere,
                    color=colors[body_idx],
                    style="wireframe",
                    line_width=6.0,
                    opacity=0.16,
                    lighting=False,
                )
            halo.position = predicted_positions[0, body_idx].numpy()
            predicted_halos.append(halo)

            trail = new_trail_mesh(predicted_positions[0, body_idx].numpy())
            plotter.add_mesh(
                trail,
                color=colors[body_idx],
                line_width=3.5 if predicted_only else 2.5,
                render_lines_as_tubes=True,
                opacity=0.95 if predicted_only else 0.65,
                lighting=False,
            )
            predicted_trails.append(trail)

            trail_halo = new_trail_mesh(predicted_positions[0, body_idx].numpy())
            plotter.add_mesh(
                trail_halo,
                color=colors[body_idx],
                line_width=9.0 if predicted_only else 7.0,
                render_lines_as_tubes=True,
                opacity=0.14,
                lighting=False,
            )
            predicted_trail_halos.append(trail_halo)

    def frame_label(frame):
        if dt is None:
            return f"step {frame}/{num_steps - 1}"
        return f"step {frame}/{num_steps - 1}    t={frame * dt:.2f}"

    comparison_label = []
    if true_positions is not None:
        comparison_label.append("solid = true")
    if predicted_positions is not None:
        comparison_label.append(
            "predicted trajectories" if predicted_only else "wireframe = predicted"
        )

    plotter.add_text(
        "Predicted N-body rollout" if predicted_only else "N-body rollout",
        position="upper_edge",
        color=text_color,
        font_size=16,
        name="title",
    )
    plotter.add_text(
        "   |   ".join(comparison_label),
        position="upper_right",
        color=accent_color,
        font_size=10,
        name="comparison-label",
    )
    frame_label_actor = plotter.add_text(
        frame_label(0),
        position="upper_left",
        color=text_color,
        font_size=11,
        name="frame-label",
    )
    plotter.add_axes(color=accent_color)
    plotter.camera.zoom(0.75)
    plotter.camera_position = [
        center + scene_radius * np.array([2.4, 1.8, 1.6]),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.clipping_range = (0.01 * scene_radius, 12.0 * scene_radius)

    def update(frame):
        trail_start = 0 if trail_length is None else max(0, frame + 1 - trail_length)

        if true_positions is not None:
            for body_idx, actor in enumerate(true_actors):
                actor.position = true_positions[frame, body_idx].numpy()
                true_halos[body_idx].position = true_positions[frame, body_idx].numpy()
                update_trail_mesh(
                    true_trails[body_idx],
                    true_positions[trail_start : frame + 1, body_idx].numpy(),
                )
                update_trail_mesh(
                    true_trail_halos[body_idx],
                    true_positions[trail_start : frame + 1, body_idx].numpy(),
                )

        if predicted_positions is not None:
            for body_idx, actor in enumerate(predicted_actors):
                actor.position = predicted_positions[frame, body_idx].numpy()
                predicted_halos[body_idx].position = predicted_positions[
                    frame, body_idx
                ].numpy()
                update_trail_mesh(
                    predicted_trails[body_idx],
                    predicted_positions[trail_start : frame + 1, body_idx].numpy(),
                )
                update_trail_mesh(
                    predicted_trail_halos[body_idx],
                    predicted_positions[trail_start : frame + 1, body_idx].numpy(),
                )

        # Mutate the existing annotation instead of replacing its renderer
        # actor every frame. Replacing it can race with renderer teardown when
        # an interactive window is closed while a timer callback is running.
        frame_label_actor.set_text("upper_left", frame_label(frame))

    def render_largest_body_bloom_source():
        """Render only the maximum-mass body for an isolated bloom mask."""

        target_actors = []
        if true_positions is not None:
            target_actors.append(true_actors[largest_body_idx])
        if predicted_positions is not None:
            target_actors.append(predicted_actors[largest_body_idx])

        actor_visibilities = []
        for actor in plotter.actors.values():
            if hasattr(actor, "GetVisibility") and hasattr(actor, "SetVisibility"):
                actor_visibilities.append((actor, actor.GetVisibility()))
                actor.SetVisibility(False)

        for actor in target_actors:
            actor.SetVisibility(True)

        try:
            plotter.render()
            return plotter.screenshot(return_img=True)
        finally:
            for actor, visibility in actor_visibilities:
                actor.SetVisibility(visibility)

    bloom_writer = None
    try:
        if resolved_save_path is not None:
            if bloom:
                try:
                    import imageio.v2 as imageio
                except ImportError as exc:
                    raise RuntimeError(
                        "Saved-animation bloom requires imageio. Install the "
                        "optional visualization dependencies from "
                        "requirements-visualization.txt."
                    ) from exc

                writer_options = {"fps": save_fps}
                if resolved_save_path.suffix.lower() == ".gif":
                    writer_options["loop"] = 0
                else:
                    writer_options["quality"] = 8
                    writer_options["macro_block_size"] = 1
                bloom_writer = imageio.get_writer(
                    str(resolved_save_path),
                    **writer_options,
                )
            elif resolved_save_path.suffix.lower() == ".gif":
                plotter.open_gif(str(resolved_save_path), fps=save_fps)
            else:
                plotter.open_movie(
                    str(resolved_save_path),
                    framerate=save_fps,
                    quality=8,
                )

        if loop:
            update(0)

            def loop_callback(step):
                renderer = getattr(plotter, "renderer", None)
                if (
                    plotter._closed
                    or plotter.render_window is None
                    or renderer is None
                    or getattr(renderer, "_closed", False)
                    or not hasattr(renderer, "_actors")
                ):
                    return
                update(step % num_steps)

            # A timer keeps the VTK event loop interactive while advancing the
            # animation. It runs until the user closes the window.
            plotter.add_timer_event(
                max_steps=1_000_000_000,
                duration=interval,
                callback=loop_callback,
            )
            plotter.show()
        elif show:
            plotter.show(interactive_update=True, auto_close=False)
        elif resolved_save_path is not None and (
            bloom or resolved_save_path.suffix.lower() != ".gif"
        ):
            # Screenshots and movies require an initialized off-screen window.
            plotter.show(auto_close=False)

        if not loop:
            for frame in range(num_steps):
                update(frame)
                if resolved_save_path is not None:
                    if bloom:
                        plotter.render()
                        rendered_frame = plotter.screenshot(return_img=True)
                        bloom_source = (
                            render_largest_body_bloom_source()
                            if bloom_largest_only
                            else None
                        )
                        bloom_writer.append_data(
                            _apply_bloom_to_frame(
                                rendered_frame,
                                bloom_source=bloom_source,
                            )
                        )
                    else:
                        plotter.write_frame()
                if show:
                    plotter.update(stime=interval, force_redraw=True)
    except ImportError as exc:
        raise RuntimeError(
            "Saving a PyVista animation requires imageio; MP4/MOV output "
            "also requires imageio-ffmpeg."
        ) from exc
    finally:
        try:
            if bloom_writer is not None:
                bloom_writer.close()
        finally:
            plotter.close()

    if resolved_save_path is not None:
        print(f"Saved 3D animation to {resolved_save_path}")

    return resolved_save_path
