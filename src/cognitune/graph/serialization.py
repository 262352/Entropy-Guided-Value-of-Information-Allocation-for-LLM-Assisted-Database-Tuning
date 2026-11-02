from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from .typed_graph import TypedRelationGraph


def save_graph(graph: TypedRelationGraph, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cognitive_graph.json"
    graphml_path = output / "cognitive_graph.graphml"
    json_path.write_text(json.dumps(graph.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    root = Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    body = SubElement(root, "graph", edgedefault="directed")
    for node in graph.nodes.values():
        element = SubElement(body, "node", id=node.name)
        element.set("kind", node.kind)
        element.set("category", node.category)
    for index, edge in enumerate(graph.edges):
        element = SubElement(body, "edge", id=f"e{index}", source=edge.source, target=edge.target)
        element.set("type", edge.edge_type)
        element.set("weight", str(edge.weight))
        element.set("status", edge.status)
    ElementTree(root).write(graphml_path, encoding="utf-8", xml_declaration=True)
    return json_path, graphml_path
