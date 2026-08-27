"""arXiv-2023 loader is not part of the rewiring package."""


def get_raw_text_arxiv_2023(*args, **kwargs):
    raise RuntimeError(
        "arXiv-2023 is not shipped in TAPTN_rewiring_repro. "
        "This bundle only re-runs WebKB rewiring experiments."
    )
