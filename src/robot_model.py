"""Build the MuJoCo scene's robot bodies from the checked-in Panda URDF.

MuJoCo can load a URDF as a stand-alone model, but an URDF cannot be included
inside an MJCF scene.  The task scene also contains a table, a rod and two
robots, so this small adapter translates the URDF robot description into the
MJCF body/asset nodes that are needed by the scene.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _vector(value: str | None, size: int = 3) -> tuple[float, ...]:
    values = [float(item) for item in (value or "").split()]
    if len(values) != size:
        raise ValueError(f"expected {size} values, got {value!r}")
    return tuple(values)


def _quat_from_rpy(rpy: tuple[float, float, float]) -> str:
    """Return MuJoCo's wxyz quaternion for a URDF fixed-frame RPY rotation."""

    roll, pitch, yaw = (angle * 0.5 for angle in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return " ".join(
        f"{value:.12g}"
        for value in (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _mesh_path(urdf_path: Path, filename: str) -> Path:
    filename = filename.removeprefix("package://")
    candidates = (
        urdf_path.parent / filename,
        urdf_path.parent.parent / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"mesh {filename!r} referenced by {urdf_path} was not found"
    )


def _link_geometries(
    link: ET.Element,
    *,
    prefix: str,
    link_name: str,
    urdf_path: Path,
    assets: ET.Element,
) -> list[ET.Element]:
    """Create visible MuJoCo mesh geoms for a URDF link's collision meshes.

    The supplied URDF uses DAE files for visual meshes. MuJoCo does not ship a
    DAE decoder, so the URDF collision STL is used as the rendered geometry as
    well. It is the robot model's collision geometry and keeps the scene
    portable with the installed MuJoCo runtime.
    """

    geoms: list[ET.Element] = []
    for index, collision in enumerate(link.findall("collision")):
        mesh = collision.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        mesh_file = _mesh_path(urdf_path, mesh.get("filename", ""))
        mesh_name = f"{prefix}_{link_name}_mesh{index}"
        ET.SubElement(assets, "mesh", name=mesh_name, file=str(mesh_file))
        origin = collision.find("origin")
        geom = ET.Element(
            "geom",
            name=f"{prefix}_{link_name}_collision{index}",
            type="mesh",
            mesh=mesh_name,
            rgba=".72 .72 .76 1",
            density="0",
        )
        if origin is not None:
            geom.set("pos", " ".join(f"{v:.12g}" for v in _vector(origin.get("xyz", "0 0 0"))))
            geom.set("quat", _quat_from_rpy(_vector(origin.get("rpy", "0 0 0"))))
        geoms.append(geom)
    return geoms


def _add_finger_hand(parent: ET.Element, prefix: str, assets: ET.Element, urdf_path: Path) -> None:
    """Add the task gripper at Panda link 8 (the arm URDF is arm-only)."""

    for part in ("hand", "finger"):
        ET.SubElement(assets, "mesh", name=f"{prefix}_{part}_mesh",
                      file=str(_mesh_path(urdf_path, f"meshes/collision/{part}.stl")))
    # Keep the control/grasp frame between the fingertips, 100 mm past the
    # Panda flange. Mesh offsets follow hand.xacro's 58.4 mm finger mount.
    hand = ET.SubElement(parent, "body", name=f"{prefix}_hand", pos="0 0 .207")
    # Palm contact belongs above the finger gap, not at the tool centre:
    # a sphere at the TCP would prop the gripper up on top of the rod.
    ET.SubElement(hand, "geom", name=f"{prefix}_hand_geom", type="box",
                  pos="0 0 -.065", size=".03 .075 .03", group="3", mass=".3")
    ET.SubElement(hand, "geom", name=f"{prefix}_hand_visual", type="mesh",
                  mesh=f"{prefix}_hand_mesh", pos="0 0 -.100", group="2",
                  contype="0", conaffinity="0", density="0", rgba=".72 .72 .76 1")
    ET.SubElement(hand, "site", name=f"{prefix}_ee", size=".012", group="3", rgba="1 0 0 1" if prefix == "left" else "0 0 1 1")
    for index, (axis, y) in enumerate((("0 1 0", ".0075"), ("0 -1 0", "-.0075")), 1):
        finger = ET.SubElement(hand, "body", name=f"{prefix}_finger{index}")
        ET.SubElement(
            finger,
            "joint",
            name=f"{prefix}_finger_joint{index}",
            type="slide",
            axis=axis,
            range="0 .045",
            damping="5",
        )
        ET.SubElement(
            finger,
            "geom",
            name=f"{prefix}_finger{index}_geom",
            type="box",
            pos=f"0 {y} .0034",
            size=".0087 .0075 .0087",
            margin=".0005",
            group="3",
            mass=".08",
        )
        ET.SubElement(finger, "geom", name=f"{prefix}_finger{index}_visual",
                      type="mesh", mesh=f"{prefix}_finger_mesh", pos="0 0 -.0416",
                      quat="1 0 0 0" if index == 1 else "0 0 0 1", group="2",
                      contype="0", conaffinity="0", density="0", rgba=".18 .18 .20 1")


def _add_robot(
    worldbody: ET.Element,
    assets: ET.Element,
    urdf_root: ET.Element,
    urdf_path: Path,
    prefix: str,
    base_pos: tuple[float, float, float],
    base_quat: str = "1 0 0 0",
) -> None:
    links = {link.get("name"): link for link in urdf_root.findall("link")}
    children: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            children[child.get("link", "")] = joint

    root_link = "panda_link0"
    base = ET.SubElement(
        worldbody,
        "body",
        name=f"{prefix}_arm_base",
        pos=" ".join(f"{value:.12g}" for value in base_pos),
        quat=base_quat,
    )
    for geom in _link_geometries(
        links[root_link],
        prefix=prefix,
        link_name="link0",
        urdf_path=urdf_path,
        assets=assets,
    ):
        base.append(geom)

    current = base
    for index in range(1, 8):
        link_name = f"panda_link{index}"
        joint = children[link_name]
        origin = joint.find("origin")
        xyz = _vector(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _vector(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        body = ET.SubElement(
            current,
            "body",
            name=f"{prefix}_link{index}",
            pos=" ".join(f"{value:.12g}" for value in xyz),
            quat=_quat_from_rpy(rpy),
        )
        axis = joint.find("axis")
        limit = joint.find("limit")
        joint_attrs = {
            "name": f"{prefix}_joint{index}",
            "type": "hinge",
            "axis": " ".join(f"{value:.12g}" for value in _vector(axis.get("xyz", "0 0 1") if axis is not None else "0 0 1")),
            "damping": joint.findtext("dynamics/damping", default="2"),
        }
        if limit is not None:
            joint_attrs["range"] = f"{limit.get('lower', '-3.14')} {limit.get('upper', '3.14')}"
        ET.SubElement(body, "joint", **joint_attrs)
        inertial = links[link_name].find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            mass = inertial.find("mass")
            iorigin = inertial.find("origin")
            inertial_attrs = {
                "mass": mass.get("value", "1") if mass is not None else "1",
                "fullinertia": " ".join(
                    inertia.get(key, "0")
                    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ) if inertia is not None else ".01 .01 .01 0 0 0",
            }
            if iorigin is not None:
                inertial_attrs["pos"] = " ".join(
                    f"{value:.12g}" for value in _vector(iorigin.get("xyz", "0 0 0"))
                )
            ET.SubElement(body, "inertial", **inertial_attrs)
        for geom in _link_geometries(
            links[link_name],
            prefix=prefix,
            link_name=f"link{index}",
            urdf_path=urdf_path,
            assets=assets,
        ):
            body.append(geom)
        current = body

    _add_finger_hand(current, prefix, assets, urdf_path)


def build_scene_xml(scene_path: Path, urdf_path: Path) -> bytes:
    """Return a scene XML document whose arms are generated from ``urdf_path``."""

    scene_root = ET.parse(scene_path).getroot()
    assets = scene_root.find("asset")
    worldbody = scene_root.find("worldbody")
    if assets is None or worldbody is None:
        raise ValueError(f"{scene_path} must contain asset and worldbody sections")

    for body in list(worldbody.findall("body")):
        if body.get("name") in {"left_arm_base", "right_arm_base"}:
            worldbody.remove(body)

    # Both task layouts share one model so the passive viewer stays alive.
    ET.SubElement(scene_root.find("equality"), "weld", name="box_orientation_grasp",
                  body1="left_hand", body2="rod", active="false", solref=".02 1")
    # The existing free body is reused by both layouts. New geoms have zero
    # mass so task1 keeps its original inertia; TaskScene selects cube inertia.
    rod = worldbody.find("body[@name='rod']")
    ET.SubElement(rod, "geom", name="box_shell", type="box", size=".30 .30 .30",
                  rgba=".85 .48 .12 1", mass="0", contype="4", conaffinity="11")
    for side, sign in (("left", -1), ("right", 1)):
        x = sign * .42
        ET.SubElement(rod, "geom", name=f"box_handle_{side}", type="capsule",
                      fromto=f"{x-.055} 0 .26 {x+.055} 0 .26", size=".028",
                      rgba=".15 .18 .22 1", mass="0", contype="4", conaffinity="11")
        for index, end in enumerate((x-.07, x+.07)):
            ET.SubElement(rod, "geom", name=f"box_mount_{side}_{index}", type="capsule",
                          fromto=f"{sign*.30} 0 .13 {end} 0 .13", size=".012",
                          rgba=".15 .18 .22 1", mass="0", contype="4", conaffinity="11")
            ET.SubElement(rod, "geom", name=f"box_join_{side}_{index}", type="capsule",
                          fromto=f"{end} 0 .13 {end} 0 .26", size=".012",
                          rgba=".15 .18 .22 1", mass="0", contype="4", conaffinity="11")
    for i, (pos, size) in enumerate([
        ((-.34,-.42,.002),(.006,.34,.002)), ((.34,-.42,.002),(.006,.34,.002)),
        ((0,-.76,.002),(.34,.006,.002)), ((0,-.08,.002),(.34,.006,.002))]):
        ET.SubElement(worldbody, "geom", name=f"box_target_{i}", type="box",
                      pos=" ".join(map(str,pos)), size=" ".join(map(str,size)),
                      rgba=".1 .95 .35 1", contype="0", conaffinity="0")
    for side, x in (("left",-.62),("right",.62)):
        ET.SubElement(worldbody, "geom", name=f"box_pedestal_{side}", type="box",
                      pos=f"{x} .30 .20", size=".13 .13 .20", rgba=".3 .33 .38 1",
                      contype="0", conaffinity="0")
    ET.SubElement(rod, "geom", name="handover_object", type="box",
                  size=".18 .028 .24", rgba=".95 .62 .12 1", mass="0",
                  contype="4", conaffinity="11")
    for name, pos, color in (("source", "-.32 0 .436", ".15 .35 .9 .4"),
                             ("target", ".32 -.25 .436", ".1 .9 .3 .4")):
        ET.SubElement(worldbody, "geom", name=f"handover_{name}", type="box",
                      pos=pos, size=".23 .10 .006", rgba=color,
                      contype="0", conaffinity="0")
    for side in ("left", "right"):
        ET.SubElement(scene_root.find("equality"), "weld", name=f"handover_{side}_weld",
                      body1=f"{side}_hand", body2="rod", active="false", solref=".02 1")
    urdf_root = ET.parse(urdf_path).getroot()
    # The real Panda has a longer base-to-wrist reach than the former toy
    # chain.  Place the bases closer to the work area so both the source and
    # destination waypoints remain inside the robot workspace.
    _add_robot(
        worldbody, assets, urdf_root, urdf_path, "left", (-.62, .30, .43), "1 0 0 0"
    )
    _add_robot(
        worldbody, assets, urdf_root, urdf_path, "right", (.62, .30, .43), "0 0 0 1"
    )
    # MuJoCo counts gravity-compensated bodies at model compilation. Setting
    # body_gravcomp only afterwards leaves ngravcomp=0 and produces no force.
    for body in worldbody.iter("body"):
        if body.get("name", "").startswith(("left_", "right_")):
            body.set("gravcomp", "1")
    return ET.tostring(scene_root, encoding="utf-8", xml_declaration=True)
