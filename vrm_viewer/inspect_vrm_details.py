import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise ValueError("Файл не похож на GLB/VRM.")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Ожидался GLB version 2, получен {version}.")
    offset = 12
    json_chunk = None
    while offset + 8 <= min(total_length, len(data)):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if kind == 0x4E4F534A:
            json_chunk = chunk
            break
    if json_chunk is None:
        raise ValueError("JSON chunk не найден.")
    return json.loads(json_chunk.decode("utf-8").rstrip("\\x00 \\t\\r\\n"))


def accessor_summary(gltf, index):
    accessors = gltf.get("accessors", [])
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        return None
    a = accessors[index]
    return {
        "accessor_index": index,
        "componentType": a.get("componentType"),
        "type": a.get("type"),
        "count": a.get("count"),
        "min": a.get("min"),
        "max": a.get("max"),
        "normalized": a.get("normalized", False),
        "bufferView": a.get("bufferView"),
        "byteOffset": a.get("byteOffset", 0),
    }


def main():
    base = Path(__file__).resolve().parent
    default_path = base / "vrm_viewer" / "BaseTest.vrm"
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not source.exists():
        print(f"Файл не найден: {source}")
        print('Запуск с путём: python inspect_vrm_details.py "D:\\путь\\BaseTest.vrm"')
        raise SystemExit(1)

    gltf = read_glb(source)
    extensions = gltf.get("extensions", {})
    meshes = gltf.get("meshes", [])
    nodes = gltf.get("nodes", [])

    morph_catalog = []
    morph_lookup = {}
    for mi, mesh in enumerate(meshes):
        names = (mesh.get("extras") or {}).get("targetNames") or []
        max_targets = len(names)
        primitives = []
        for pi, primitive in enumerate(mesh.get("primitives", [])):
            targets = primitive.get("targets", [])
            max_targets = max(max_targets, len(targets))
            target_details = []
            for ti, target in enumerate(targets):
                target_details.append({
                    "target_index": ti,
                    "attributes": {
                        semantic: accessor_summary(gltf, ai)
                        for semantic, ai in target.items()
                    }
                })
            primitives.append({
                "primitive_index": pi,
                "target_count": len(targets),
                "targets": target_details,
                "attributes": {
                    semantic: accessor_summary(gltf, ai)
                    for semantic, ai in primitive.get("attributes", {}).items()
                }
            })

        morphs = []
        for ti in range(max_targets):
            name = names[ti] if ti < len(names) else f"morph_{ti}"
            entries = []
            for pi, primitive in enumerate(mesh.get("primitives", [])):
                targets = primitive.get("targets", [])
                if ti < len(targets):
                    entries.append({
                        "primitive_index": pi,
                        "attributes": {
                            semantic: accessor_summary(gltf, ai)
                            for semantic, ai in targets[ti].items()
                        }
                    })
            item = {
                "mesh_index": mi,
                "mesh_name": mesh.get("name", f"mesh_{mi}"),
                "morph_index": ti,
                "morph_name": name,
                "primitive_target_data": entries,
            }
            morphs.append(item)
            morph_lookup[(mi, ti)] = item

        if morphs:
            morph_catalog.append({
                "mesh_index": mi,
                "mesh_name": mesh.get("name", f"mesh_{mi}"),
                "weights": mesh.get("weights"),
                "morph_count": len(morphs),
                "morphs": morphs,
                "primitives": [
                    {
                        "primitive_index": x["primitive_index"],
                        "target_count": x["target_count"],
                        "attributes": x["attributes"],
                    } for x in primitives
                ],
            })

    expressions = {}
    vrm0 = extensions.get("VRM", {})
    for group in vrm0.get("blendShapeMaster", {}).get("blendShapeGroups", []):
        preset = group.get("presetName", "")
        name = group.get("name") or preset or "unnamed"
        key = f"{preset}:{name}" if preset else name
        binds = []
        for bind in group.get("binds", []):
            mi, ti = bind.get("mesh"), bind.get("index")
            weight = bind.get("weight", 0)
            morph = morph_lookup.get((mi, ti), {})
            binds.append({
                "mesh_index": mi,
                "mesh_name": meshes[mi].get("name", f"mesh_{mi}") if isinstance(mi, int) and 0 <= mi < len(meshes) else None,
                "morph_index": ti,
                "morph_name": morph.get("morph_name", f"morph_{ti}"),
                "weight_percent": weight,
                "weight_normalized": float(weight) / 100.0,
            })
        expressions[key] = {
            "name": name,
            "presetName": preset,
            "isBinary": group.get("isBinary", False),
            "binds": binds,
            "materialValues": group.get("materialValues", []),
        }

    vrm1 = extensions.get("VRMC_vrm", {})
    for category in ("preset", "custom"):
        for name, group in vrm1.get("expressions", {}).get(category, {}).items():
            binds = []
            for bind in group.get("morphTargetBinds", []):
                ni = bind.get("node")
                node = nodes[ni] if isinstance(ni, int) and 0 <= ni < len(nodes) else {}
                mi, ti = node.get("mesh"), bind.get("index")
                weight = bind.get("weight", 0)
                morph = morph_lookup.get((mi, ti), {})
                binds.append({
                    "node_index": ni,
                    "node_name": node.get("name"),
                    "mesh_index": mi,
                    "mesh_name": meshes[mi].get("name", f"mesh_{mi}") if isinstance(mi, int) and 0 <= mi < len(meshes) else None,
                    "morph_index": ti,
                    "morph_name": morph.get("morph_name", f"morph_{ti}"),
                    "weight": weight,
                    "weight_percent": float(weight) * 100.0,
                })
            expressions[f"{category}:{name}"] = {
                "name": name,
                "category": category,
                "overrideBlink": group.get("overrideBlink"),
                "overrideLookAt": group.get("overrideLookAt"),
                "overrideMouth": group.get("overrideMouth"),
                "binds": binds,
            }

    usage = defaultdict(list)
    for exp_name, exp in expressions.items():
        for bind in exp["binds"]:
            k = f"mesh_{bind.get('mesh_index')}:morph_{bind.get('morph_index')}"
            usage[k].append({
                "expression": exp_name,
                "weight": bind.get("weight", bind.get("weight_percent", 0)),
            })

    categories = defaultdict(list)
    for mesh_entry in morph_catalog:
        for morph in mesh_entry["morphs"]:
            name = morph["morph_name"].upper()
            if "EYE" in name or "IRIS" in name or "BLINK" in name:
                cat = "eyes"
            elif "BRW" in name or "BROW" in name:
                cat = "eyebrows"
            elif "MTH" in name or "MOUTH" in name:
                cat = "mouth"
            elif "TEETH" in name or "TOOTH" in name or "HA_" in name:
                cat = "teeth_or_mouth_interior"
            elif "ALL" in name:
                cat = "whole_face_expressions"
            else:
                cat = "other"
            item = dict(morph)
            item["used_by_expressions"] = usage.get(
                f"mesh_{morph['mesh_index']}:morph_{morph['morph_index']}", []
            )
            categories[cat].append(item)

    report = {
        "source_file": source.name,
        "asset": gltf.get("asset", {}),
        "vrm_format": "VRM 0.x" if vrm0 else ("VRM 1.0" if vrm1 else "Unknown"),
        "mesh_count": len(meshes),
        "node_count": len(nodes),
        "expression_count": len(expressions),
        "morph_count": sum(x["morph_count"] for x in morph_catalog),
        "expressions": expressions,
        "morphs_by_category": dict(categories),
        "morph_catalog": morph_catalog,
        "morph_usage": [
            {
                "mesh_index": int(k.split(":")[0].replace("mesh_", "")),
                "morph_index": int(k.split(":")[1].replace("morph_", "")),
                "used_by": v,
            } for k, v in usage.items()
        ],
        "vrm_metadata": {
            "extension_names": sorted(extensions.keys()),
            "VRM_meta": vrm0.get("meta"),
            "VRM_humanoid": vrm0.get("humanoid"),
            "VRM_firstPerson": vrm0.get("firstPerson"),
            "VRM_lookAt": vrm0.get("lookAt"),
            "VRM_materialProperties": vrm0.get("materialProperties"),
            "VRMC_vrm": vrm1 or None,
        },
        "limitations": [
            "This extracts morph names, expression bindings, and morph accessor metadata.",
            "It does not infer the exact visual meaning of a morph from its name.",
            "VRoid Studio UI slider settings may not be stored as a complete slider-to-morph table in the exported VRM.",
            "Morph geometry is represented by accessor references, counts, and min/max metadata; vertex arrays are not duplicated into this report.",
        ],
    }

    out = base / "vrm_inspection"
    out.mkdir(exist_ok=True)
    (out / "vrm_details_full.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "vrm_expressions.json").write_text(json.dumps(expressions, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "vrm_morphs_by_category.json").write_text(json.dumps(dict(categories), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["category\tmesh_index\tmesh_name\tmorph_index\tmorph_name\tused_by"]
    for cat, items in categories.items():
        for m in items:
            used = "; ".join(x["expression"] for x in m["used_by_expressions"]) or "(not bound to VRM expression)"
            lines.append(f"{cat}\t{m['mesh_index']}\t{m['mesh_name']}\t{m['morph_index']}\t{m['morph_name']}\t{used}")
    (out / "vrm_morph_index.tsv").write_text("\n".join(lines), encoding="utf-8")

    print("Готово.")
    print(f"Файл: {source}")
    print(f"Формат: {report['vrm_format']}")
    print(f"Мешей: {report['mesh_count']}; узлов: {report['node_count']}; морфов: {report['morph_count']}")
    print(f"VRM-выражений: {report['expression_count']}")
    print(f"Результаты: {out}")
    print("  vrm_details_full.json")
    print("  vrm_expressions.json")
    print("  vrm_morphs_by_category.json")
    print("  vrm_morph_index.tsv")


if __name__ == "__main__":
    main()
