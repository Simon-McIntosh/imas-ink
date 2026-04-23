"""IMAS → 3D mesh extraction for PF coils, TF coils, and the vessel wall.

All heavy imports (pyvista, numpy beyond basic) are at function scope
so that ``import imas_ink`` never pulls in VTK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyvista as pv


@dataclass
class CoilMesh:
    """Container for a 3D coil mesh with associated metadata.

    Attributes
    ----------
    name : str
        Human-readable coil identifier (e.g. ``"PF1"``, ``"CS3U"``).
    mesh : pyvista.PolyData
        The 3D surface mesh.
    metadata : dict[str, Any]
        Arbitrary metadata — typically includes ``r_center``,
        ``z_center``, ``element_count``, ``is_cs_segment``.
    """

    name: str
    mesh: Any  # pv.PolyData — typed as Any to avoid import
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_pf_coils(pf_active) -> list[CoilMesh]:
    """Extract PF/CS coil meshes from a ``pf_active`` IDS.

    For each coil and each element within it, the cross-section geometry
    is revolved 360° about the z-axis to produce a toroidal ring.

    Supported ``geometry_type`` values:

    - **rectangle** (type 2) — ``rectangle.r``, ``.z``, ``.width``,
      ``.height`` → :func:`ring_from_rectangle`.
    - **outline** (type 1) — ``outline.r``, ``.z`` polygon → revolve.
    - **oblique** (type 4) — four-corner parallelogram → revolve.
    - **annulus** (type 5) — ``annulus.r``, ``.z`` → revolve.

    Falls back to outline if the declared type is unsupported.

    Parameters
    ----------
    pf_active
        ``pf_active`` IDS object.

    Returns
    -------
    list[CoilMesh]
    """
    import numpy as np

    from .primitives import revolve_polygon, ring_from_rectangle

    coils: list[CoilMesh] = []

    for i, coil in enumerate(pf_active.coil):
        coil_name = _safe_name(coil, fallback=f"PF{i + 1}")
        is_cs = "cs" in coil_name.lower() or "CS" in coil_name

        for j, elem in enumerate(coil.element):
            geom = elem.geometry
            geom_type = _safe_int(getattr(geom, "geometry_type", 0))
            mesh = None

            if geom_type == 2 or _has_rectangle(geom):
                # Rectangle
                rect = geom.rectangle
                r_c = float(rect.r)
                z_c = float(rect.z)
                dr = float(rect.width)
                dz = float(rect.height)
                mesh = ring_from_rectangle(r_c, z_c, dr, dz)
            elif geom_type == 1 or _has_outline(geom):
                # Outline polygon
                outline = geom.outline
                r = np.asarray(outline.r, dtype=float)
                z = np.asarray(outline.z, dtype=float)
                if r.size >= 3:
                    mesh = revolve_polygon(r, z)
            elif geom_type == 4 and _has_attr(geom, "oblique"):
                # Oblique (parallelogram) — build from four corners
                obl = geom.oblique
                r = np.asarray(obl.r, dtype=float)
                z = np.asarray(obl.z, dtype=float)
                if r.size >= 3:
                    mesh = revolve_polygon(r, z)
            elif geom_type == 5 and _has_attr(geom, "annulus"):
                ann = geom.annulus
                r = np.asarray(ann.r, dtype=float)
                z = np.asarray(ann.z, dtype=float)
                if r.size >= 3:
                    mesh = revolve_polygon(r, z)

            # Fallback: try outline if we still have nothing
            if mesh is None and _has_outline(geom):
                outline = geom.outline
                r = np.asarray(outline.r, dtype=float)
                z = np.asarray(outline.z, dtype=float)
                if r.size >= 3:
                    mesh = revolve_polygon(r, z)

            if mesh is not None:
                label = f"{coil_name}" if len(coil.element) == 1 else f"{coil_name}_{j}"
                coils.append(
                    CoilMesh(
                        name=label,
                        mesh=mesh,
                        metadata={
                            "coil_index": i,
                            "element_index": j,
                            "element_count": len(coil.element),
                            "is_cs_segment": is_cs,
                        },
                    )
                )

    return coils


def extract_tf_coils(tf, n_coils: int | None = None) -> list[CoilMesh]:
    """Extract TF coil meshes from a ``tf`` IDS.

    Reads the conductor centerline from
    ``coil[i].conductor[0].elements.{start_points,intermediate_points,
    end_points}`` in cylindrical (r, φ, z) coordinates, converts to
    Cartesian, and sweeps a rectangular cross-section along the Frenet-
    framed path.

    If only one coil is stored and ``tf.coils_n`` is set, the coil is
    replicated at evenly spaced toroidal angles.

    Parameters
    ----------
    tf
        ``tf`` IDS object.
    n_coils : int | None
        Override for the number of TF coils to render.  ``None`` means
        "use all stored coils, or replicate if only one is stored".

    Returns
    -------
    list[CoilMesh]
    """
    import numpy as np

    from .primitives import sweep_section_along_path

    coils: list[CoilMesh] = []

    stored_coils = list(tf.coil)
    if not stored_coils:
        return coils

    # Determine number of physical coils
    total_n = n_coils
    if total_n is None:
        total_n = _safe_int(getattr(tf, "coils_n", 0))
        if total_n <= 0:
            total_n = len(stored_coils)

    # Extract cross-section geometry from DD paths
    #   tf/coil/conductor/cross_section — introduced in DD 3.42.0 (AoS)
    #   geometry_type identifier: 1=polygon, 2=circle, 3=rectangle,
    #                             4=square, 5=annulus.
    # Fall back to representative ITER TF winding-pack dimensions when the
    # cross_section block is absent (DD < 3.42.0 or unpopulated).
    first_conductor = stored_coils[0].conductor[0] if stored_coils[0].conductor else None
    section = _extract_tf_section(first_conductor)

    replicate = len(stored_coils) == 1 and total_n > 1

    for idx, coil in enumerate(stored_coils):
        if not coil.conductor:
            continue

        conductor = coil.conductor[0]
        centerline_xyz = _extract_conductor_centerline(conductor)
        if centerline_xyz is None or len(centerline_xyz) < 2:
            continue

        if replicate:
            # Replicate at N toroidal angles
            for k in range(total_n):
                phi_offset = 2 * np.pi * k / total_n
                rotated = _rotate_z(centerline_xyz, phi_offset)
                mesh = sweep_section_along_path(section, rotated, frame="frenet")
                coils.append(
                    CoilMesh(
                        name=f"TF{k + 1}",
                        mesh=mesh,
                        metadata={"coil_index": k, "toroidal_angle_deg": np.degrees(phi_offset)},
                    )
                )
        else:
            mesh = sweep_section_along_path(section, centerline_xyz, frame="frenet")
            coils.append(
                CoilMesh(
                    name=f"TF{idx + 1}",
                    mesh=mesh,
                    metadata={"coil_index": idx},
                )
            )

    return coils


def extract_wall(wall) -> pv.PolyData:
    """Extract the vessel / limiter wall as a revolved 3D mesh.

    Attempts (in order):

    1. ``wall.description_2d[0].vessel.unit[*].annular.centreline`` +
       thickness → revolved annular ring per unit.
    2. ``wall.description_2d[0].limiter.unit[0].outline`` → revolved
       2D polygon (no thickness offset).

    Returns an empty :class:`pyvista.PolyData` if no geometry is found.

    Parameters
    ----------
    wall
        ``wall`` IDS object.

    Returns
    -------
    pyvista.PolyData
    """
    import numpy as np
    import pyvista as pv

    from .primitives import revolve_polygon

    try:
        desc = wall.description_2d[0]
    except (AttributeError, IndexError):
        return pv.PolyData()

    # Try vessel annular centreline
    meshes: list[pv.PolyData] = []
    try:
        for unit in desc.vessel.unit:
            ann = unit.annular
            r_cl = np.asarray(ann.centreline.r, dtype=float)
            z_cl = np.asarray(ann.centreline.z, dtype=float)
            if r_cl.size < 3:
                continue
            # Build a thin annular shell using thickness if available
            thickness = _safe_float(getattr(ann, "thickness", 0.0))
            if thickness > 0:
                # Offset the centreline inward/outward
                r_outer, z_outer = _offset_polygon(r_cl, z_cl, thickness / 2)
                r_inner, z_inner = _offset_polygon(r_cl, z_cl, -thickness / 2)
                # Combine into a closed annular cross-section
                r_section = np.concatenate([r_outer, r_inner[::-1]])
                z_section = np.concatenate([z_outer, z_inner[::-1]])
                meshes.append(revolve_polygon(r_section, z_section))
            else:
                meshes.append(revolve_polygon(r_cl, z_cl))
    except (AttributeError, IndexError):
        pass

    if meshes:
        combined = meshes[0]
        for m in meshes[1:]:
            combined = combined.merge(m)
        return combined

    # Fallback: limiter outline
    try:
        limiter = desc.limiter.unit[0].outline
        r = np.asarray(limiter.r, dtype=float)
        z = np.asarray(limiter.z, dtype=float)
        if r.size >= 3:
            return revolve_polygon(r, z)
    except (AttributeError, IndexError):
        pass

    return pv.PolyData()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _extract_conductor_centerline(conductor) -> np.ndarray | None:
    """Build a Cartesian centerline array from conductor element data.

    Reads ``start_points``, ``intermediate_points``, ``end_points``
    (cylindrical r, φ, z) and concatenates them into an ordered path.
    """
    import numpy as np

    from .primitives import cylindrical_to_cartesian

    try:
        elems = conductor.elements
    except AttributeError:
        return None

    segments: list[np.ndarray] = []
    for attr in ("start_points", "intermediate_points", "end_points"):
        pts = getattr(elems, attr, None)
        if pts is None:
            continue
        r = np.asarray(getattr(pts, "r", []), dtype=float)
        phi = np.asarray(getattr(pts, "phi", []), dtype=float)
        z = np.asarray(getattr(pts, "z", []), dtype=float)
        if r.size == 0:
            continue
        x, y, z_out = cylindrical_to_cartesian(r, phi, z)
        segments.append(np.column_stack([x, y, z_out]))

    if not segments:
        return None

    # Concatenate and remove consecutive duplicates
    path = np.vstack(segments)
    if len(path) > 1:
        mask = np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]
        path = path[mask]

    return path


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points about the z-axis by *angle* radians."""
    import numpy as np

    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ rot.T


def _offset_polygon(r: np.ndarray, z: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Naïve inward/outward offset of a 2D polygon.

    Moves each vertex along the local outward normal by *offset*.
    Positive offset = outward, negative = inward.
    """
    import numpy as np

    n = len(r)
    r_off = np.empty(n)
    z_off = np.empty(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        dr = r[i_next] - r[i_prev]
        dz = z[i_next] - z[i_prev]
        length = np.hypot(dr, dz)
        if length < 1e-12:
            r_off[i] = r[i]
            z_off[i] = z[i]
        else:
            # Outward normal (assuming CCW winding)
            r_off[i] = r[i] + offset * dz / length
            z_off[i] = z[i] - offset * dr / length
    return r_off, z_off


def _safe_name(coil, fallback: str = "coil") -> str:
    """Extract a coil name, falling back to *fallback*."""
    try:
        name = coil.name
        if name and isinstance(name, str) and len(name.strip()) > 0:
            return name.strip()
    except (AttributeError, TypeError):
        pass
    return fallback


def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, falling back to *default*."""
    try:
        v = int(value)
        return v if abs(v) < 1e9 else default
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    """Convert *value* to float, falling back to *default*."""
    try:
        v = float(value)
        return v if abs(v) < 1e30 else default
    except (TypeError, ValueError):
        return default


# Representative ITER TF winding-pack half-dimensions used when the
# DD cross_section block is absent (DD < 3.42.0 or unpopulated).
_TF_DEFAULT_WIDTH = 0.4  # metres, normal (radial) direction
_TF_DEFAULT_HEIGHT = 0.6  # metres, binormal (toroidal) direction


def _extract_tf_section(conductor) -> np.ndarray:
    """Build a 2D TF conductor cross-section polygon in (normal, binormal).

    Reads ``conductor.cross_section[0]`` (IMAS DD >= 3.42.0) and dispatches
    on ``cross_section[0].geometry_type.index``:

    ==  ============  ===============================================
    1   polygon       outline.normal / outline.binormal vertices
    2   circle        regular 32-gon of diameter ``width``
    3   rectangle     ``width`` (normal) x ``height`` (binormal)
    4   square        ``width`` x ``width``
    5   annulus       outer 32-gon of diameter ``width``
                      (inner radius is not rendered - surface-only sweep)
    ==  ============  ===============================================

    Falls back to representative ITER TF winding-pack dimensions
    (``_TF_DEFAULT_WIDTH`` x ``_TF_DEFAULT_HEIGHT``) when the DD
    cross_section AoS is absent or geometry_type is unrecognised.
    """
    import numpy as np

    def _rectangle(w: float, h: float) -> np.ndarray:
        return np.array(
            [
                [-w / 2.0, -h / 2.0],
                [w / 2.0, -h / 2.0],
                [w / 2.0, h / 2.0],
                [-w / 2.0, h / 2.0],
            ]
        )

    def _circle(diameter: float, n: int = 32) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = diameter / 2.0
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    fallback = _rectangle(_TF_DEFAULT_WIDTH, _TF_DEFAULT_HEIGHT)

    if conductor is None:
        return fallback

    cs_aos = getattr(conductor, "cross_section", None)
    if cs_aos is None:
        return fallback
    try:
        cs = cs_aos[0]
    except (TypeError, IndexError):
        return fallback

    geom_type = 0
    try:
        geom_type = int(cs.geometry_type.index)
    except (AttributeError, TypeError, ValueError):
        geom_type = 0

    if geom_type == 1:  # polygon
        try:
            normal = np.asarray(cs.outline.normal, dtype=float)
            binormal = np.asarray(cs.outline.binormal, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return fallback
        if normal.size >= 3 and normal.size == binormal.size:
            return np.column_stack((normal, binormal))
        return fallback

    if geom_type == 3:  # rectangle
        w = _safe_float(getattr(cs, "width", 0.0))
        h = _safe_float(getattr(cs, "height", 0.0))
        if w > 0.0 and h > 0.0:
            return _rectangle(w, h)
        return fallback

    if geom_type == 4:  # square
        w = _safe_float(getattr(cs, "width", 0.0))
        if w > 0.0:
            return _rectangle(w, w)
        return fallback

    if geom_type in (2, 5):  # circle or annulus (outer silhouette)
        diameter = _safe_float(getattr(cs, "width", 0.0))
        if diameter > 0.0:
            return _circle(diameter)
        return fallback

    return fallback


def _has_rectangle(geom) -> bool:
    """Check if *geom* has a populated rectangle sub-structure."""
    try:
        rect = geom.rectangle
        return float(rect.r) != 0 or float(rect.width) != 0
    except (AttributeError, TypeError, ValueError):
        return False


def _has_outline(geom) -> bool:
    """Check if *geom* has a populated outline sub-structure."""
    import numpy as np

    try:
        r = np.asarray(geom.outline.r)
        return r.size >= 3
    except (AttributeError, TypeError, ValueError):
        return False


def _has_attr(obj, name: str) -> bool:
    """Check if *obj* has attribute *name* that is not None."""
    try:
        return getattr(obj, name, None) is not None
    except Exception:
        return False
