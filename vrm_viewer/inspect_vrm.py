
import json
import struct
import sys
from pathlib import Path


def read_vrm(path):
    """Читает JSON-структуру внутри VRM/GLB без дополнительных библиотек."""
    data = Path(path).read_bytes()

    if data[:4] != b"glTF":
        raise ValueError("Файл не является GLB/VRM в формате GLB.")

    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Неподдерживаемая версия GLB: {version}")

    offset = 12
    while offset + 8 <= min(total_length, len(data)):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length

        # JSON chunk
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))

    raise ValueError("JSON-часть внутри VRM не найдена.")


def main():
    default_path = Path(__file__).parent / "vrm_viewer" / "BaseTest.vrm"
    vrm_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not vrm_path.exists():
        print(f"Файл не найден: {vrm_path}")
        print("Передай путь к VRM аргументом командной строки.")
        return

    gltf = read_vrm(vrm_path)
    extensions = gltf.get("extensions", {})

    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])

    def node_name(index):
        if isinstance(index, int) and 0 <= index < len(nodes):
            return nodes[index].get("name", f"node_{index}")
        return f"node_{index}"

    def mesh_name(index):
        if isinstance(index, int) and 0 <= index < len(meshes):
            return meshes[index].get("name", f"mesh_{index}")
        return f"mesh_{index}"

    def morph_name(mesh_index, morph_index):
        if not isinstance(mesh_index, int) or not (
            0 <= mesh_index < len(meshes)
        ):
            return f"morph_{morph_index}"

        mesh = meshes[mesh_index]
        names = mesh.get("extras", {}).get("targetNames", [])

        if 0 <= morph_index < len(names):
            return names[morph_index]

        # Некоторые файлы не содержат targetNames.
        return f"{mesh.get('name', f'mesh_{mesh_index}')}_morph_{morph_index}"

    expressions = {}
    version_found = None

    # VRM 1.0
    vrmc = extensions.get("VRMC_vrm", {})
    vrm1_expressions = vrmc.get("expressions", {})

    if vrm1_expressions:
        version_found = "VRM 1.0"

        for category in ("preset", "custom"):
            for name, expression in vrm1_expressions.get(category, {}).items():
                binds = []

                for bind in expression.get("morphTargetBinds", []):
                    node_index = bind.get("node")
                    target_index = bind.get("index")
                    node = (
                        nodes[node_index]
                        if isinstance(node_index, int)
                        and 0 <= node_index < len(nodes)
                        else {}
                    )
                    mesh_index = node.get("mesh")
                    weight = float(bind.get("weight", 0.0))

                    binds.append({
                        "node_index": node_index,
                        "node_name": node_name(node_index),
                        "mesh_index": mesh_index,
                        "mesh_name": mesh_name(mesh_index),
                        "morph_index": target_index,
                        "morph_name": morph_name(mesh_index, target_index),
                        "weight": weight,
                        "weight_percent": round(weight * 100, 4),
                    })

                key = f"{category}:{name}"
                expressions[key] = {
                    "category": category,
                    "name": name,
                    "overrideBlink": expression.get("overrideBlink"),
                    "overrideLookAt": expression.get("overrideLookAt"),
                    "overrideMouth": expression.get("overrideMouth"),
                    "binds": binds,
                }

    # VRM 0.x
    vrm0 = extensions.get("VRM", {})
    blend_shape_master = vrm0.get("blendShapeMaster", {})

    if blend_shape_master:
        version_found = version_found or "VRM 0.x"

        for group in blend_shape_master.get("blendShapeGroups", []):
            preset = group.get("presetName", "")
            name = group.get("name") or preset or "unnamed"
            key = f"{preset}:{name}" if preset else name
            binds = []

            for bind in group.get("binds", []):
                mesh_index = bind.get("mesh")
                target_index = bind.get("index")
                raw_weight = float(bind.get("weight", 0.0))
                normalized = raw_weight / 100.0

                binds.append({
                    "mesh_index": mesh_index,
                    "mesh_name": mesh_name(mesh_index),
                    "morph_index": target_index,
                    "morph_name": morph_name(mesh_index, target_index),
                    "weight": raw_weight,
                    "weight_percent": raw_weight,
                    "normalized_weight": normalized,
                })

            expressions[key] = {
                "category": "preset" if preset else "custom",
                "name": name,
                "presetName": preset,
                "binds": binds,
            }

    # Каталог всех доступных морфов мешей
    morph_catalog = []
    for mesh_index, mesh in enumerate(meshes):
        names = mesh.get("extras", {}).get("targetNames", [])
        target_count = max(
            [len(primitive.get("targets", []))
             for primitive in mesh.get("primitives", [])] + [len(names)]
        )

        morphs = []
        for index in range(target_count):
            morphs.append({
                "index": index,
                "name": names[index] if index < len(names)
                else f"{mesh.get('name', f'mesh_{mesh_index}')}_morph_{index}",
            })

        if morphs:
            morph_catalog.append({
                "mesh_index": mesh_index,
                "mesh_name": mesh.get("name", f"mesh_{mesh_index}"),
                "morphs": morphs,
            })

    # Обратный индекс: какие выражения затрагивают каждый морф
    morph_usage = {}
    for expression_name, expression in expressions.items():
        for bind in expression["binds"]:
            identity = (
                bind.get("mesh_index"),
                bind.get("morph_index"),
            )
            key = f"mesh_{identity[0]}:morph_{identity[1]}"

            morph_usage.setdefault(key, {
                "mesh_index": identity[0],
                "mesh_name": bind.get("mesh_name"),
                "morph_index": identity[1],
                "morph_name": bind.get("morph_name"),
                "used_by": [],
            })
            morph_usage[key]["used_by"].append({
                "expression": expression_name,
                "weight": bind.get("weight"),
                "weight_percent": bind.get("weight_percent"),
            })

    result = {
        "source_file": str(vrm_path),
        "gltf_version": gltf.get("asset", {}).get("version"),
        "vrm_version": version_found or "Версия VRM не определена",
        "expression_count": len(expressions),
        "expressions": expressions,
        "morph_catalog": morph_catalog,
        "morph_usage": list(morph_usage.values()),
        "notes": [
            "VRM 0.x weights are stored as percentages.",
            "VRM 1.0 weights are stored in the 0..1 range.",
            "The dump lists declared morph bindings, not measured runtime deformation.",
        ],
    }

    output_path = Path(__file__).parent / "vrm_expression_dump.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"VRM: {vrm_path}")
    print(f"Версия: {result['vrm_version']}")
    print(f"Выражений: {len(expressions)}")
    print(f"Мешей с морфами: {len(morph_catalog)}")
    print(f"Результат: {output_path}")

    for name, expression in expressions.items():
        print(f"\n{name}:")
        if not expression["binds"]:
            print("  Нет привязанных морфов")
        for bind in expression["binds"]:
            print(
                f"  {bind['morph_name']} "
                f"(mesh={bind['mesh_name']}, "
                f"index={bind['morph_index']}) "
                f"weight={bind['weight_percent']}%"
            )


if __name__ == "__main__":
    main()
