"""The ID <-> filename mapping, including the stems it must refuse to decode."""

import pytest

from arxiv_mcp_server.paper_storage import (
    paper_id_from_stem,
    paper_id_to_stem,
    paper_path,
)
from arxiv_mcp_server.tools.list_papers import is_valid_arxiv_id


@pytest.mark.parametrize(
    "paper_id",
    [
        "2401.12345",
        "2401.12345v3",
        "hep-th/9901001",
        "hep-th/9901001v2",
        "math.GT/0309136",
        "hep-th%2F9901001",  # a literal percent must survive its own escaping
    ],
)
def test_an_id_survives_the_trip_to_a_filename_and_back(paper_id):
    assert paper_id_from_stem(paper_id_to_stem(paper_id)) == paper_id


def test_a_slash_becomes_part_of_the_name_not_part_of_the_path(tmp_path):
    """The whole point: an old-style ID must not open a subdirectory."""
    path = paper_path(tmp_path, "hep-th/9901001")
    assert path.parent == tmp_path
    assert path.name == "hep-th%2F9901001.md"


@pytest.mark.parametrize(
    "stem",
    [
        "2401%2E12345",  # decodes to a valid-looking 2401.12345
        "2401.12345%0A",  # decodes to "2401.12345\n"
        "..%2F..%2Fetc%2Fpasswd",
        "hep-th%2F9901001%2F..",
    ],
)
def test_a_junk_filename_never_becomes_a_paper_the_server_advertises(stem):
    """The invariant that matters: nothing here reaches `list_papers`.

    Two different mechanisms hold it, which is why they are asserted apart
    below — `%2E` is refused at the decode, the rest are refused by the ID
    pattern once decoded.
    """
    assert not is_valid_arxiv_id(paper_id_from_stem(stem))


def test_a_stem_this_module_would_not_write_is_not_decoded_at_all():
    """`2401%2E12345.md` is not a name `paper_id_to_stem` can produce.

    Decoded blindly it becomes `2401.12345`, which `list_papers` would
    advertise and `read_paper` would then fail to open, because it re-encodes
    that ID to `2401.12345.md` — a different file. One directory entry would
    become a paper the server offers and then denies. So a non-canonical stem
    is left exactly as it sits on disk, and judged as that.
    """
    assert paper_id_from_stem("2401%2E12345") == "2401%2E12345"


def test_a_trailing_newline_does_not_pass_for_an_id():
    """`$` matches before a final newline; the anchor has to be `\\Z`."""
    assert not is_valid_arxiv_id("2401.12345\n")
