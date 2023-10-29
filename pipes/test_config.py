import json
from pathlib import Path

file_path = Path(__file__).parent / "2023-07-13-video-audio-pipeline-final.json"

expected_nodes = {
    "oele-03": ["webcam-oele-03", "audio-oele-03"],
    "oele-13": ["webcam-oele-13", "audio-oele-13"],
    "oele-11": ["webcam-oele-11", "audio-oele-11"],
    "oele-12": ["webcam-oele-12", "audio-oele-12"],
    "local": ["show"]
}

with file_path.open("r") as json_file:
    config = json.load(json_file)
    assert len(config["nodes"]) == 9
    node_names = set([
        node["name"] for node in config["nodes"]
    ])
    assert node_names == set([
        "webcam-oele-12",
        "audio-oele-12",
        "show",
        "webcam-oele-03",
        "audio-oele-03",
        "webcam-oele-13",
        "audio-oele-13",
        "webcam-oele-11",
        "audio-oele-11",
    ])
    # Save Names
    for node in config["nodes"]:
        if node.get("save_name"):
            assert node["save_name"]
            assert node["registry_name"] in {"MMLAPipe_Audio", "MMLAPipe_Video"}
    # Workers
    worker_names = [worker["name"] for worker in config["workers"]["instances"]]
    for name, nodes in config["mappings"].items():
        assert name in worker_names, name
        assert set(nodes) == set(expected_nodes[name])

