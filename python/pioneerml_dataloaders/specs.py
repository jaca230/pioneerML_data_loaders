import json
import importlib.resources as pkg_resources


def _load_spec(name: str) -> dict:
    with pkg_resources.files("pioneerml_dataloaders").joinpath(f"../specs/{name}.json").open("r") as f:
        return json.load(f)


GROUP_CLASSIFIER_SPEC = _load_spec("group_classifier")
