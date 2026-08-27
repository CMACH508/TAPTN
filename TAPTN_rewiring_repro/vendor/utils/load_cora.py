"""Cora loader is not part of the rewiring package."""


def get_raw_text_cora(*args, **kwargs):
    raise RuntimeError(
        "Cora is not shipped in TAPTN_rewiring_repro. "
        "This bundle only re-runs WebKB rewiring experiments."
    )
