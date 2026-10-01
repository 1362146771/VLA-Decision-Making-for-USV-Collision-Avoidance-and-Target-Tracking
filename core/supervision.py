from .actions import action_id


PROXY_FIELDS = frozenset({"pseudo_action_id", "candidate_action_id", "executed_proxy",
                          "risk_proxy", "executed_actions", "confidence"})
CAPTION_COUNTS = {"public": 7000, "synthetic": 8000}
CAPTION_METHODS = {"public": "human", "synthetic": "programmatic"}


def reject_proxy_fields(record):
    if PROXY_FIELDS.intersection(record):
        raise ValueError("Recorded expert actions and collision events are required")


def validate_caption(record):
    reject_proxy_fields(record)
    source_kind = record.get("source_kind")
    if source_kind not in CAPTION_METHODS:
        raise ValueError("Caption source_kind must be public or synthetic")
    if record.get("annotation_method") != CAPTION_METHODS[source_kind]:
        raise ValueError("Public captions require human annotation; synthetic captions require programmatic annotation")
    for key in ("caption", "caption_mirrored", "source", "license"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing caption field {key}")


def validate_episode_sample(record, cfg):
    reject_proxy_fields(record)
    action_id(record.get("target"))
    action_id(record.get("last_action"))
    if len(record.get("history", [])) != cfg.memory_frames + 1:
        raise ValueError("Expected M history frames and one current frame")
    if "world_images" in record:
        expected = cfg.memory_frames + 1 + cfg.prediction_steps
        if len(record["world_images"]) != expected or len(record.get("world_actions", [])) != expected:
            raise ValueError("World frames and recorded expert actions must align")
        for action in record["world_actions"]:
            action_id(action)
        if type(record.get("risk")) is not int or record["risk"] not in (0, 1):
            raise ValueError("Binary collision-event label is required")
