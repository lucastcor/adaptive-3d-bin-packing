from __future__ import annotations

from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from adaptive_3dbp.io.spreadsheet import (
    instance_from_spreadsheet,
    template_bytes,
    write_instance_template,
)


def test_template_round_trip(tmp_path: Path) -> None:
    path = write_instance_template(tmp_path / "template.xlsx")
    instance = instance_from_spreadsheet(path)
    assert instance.instance_id == "exemplo-planilha-001"
    assert len(instance.items) == 3
    assert instance.items[0].allowed_rotations == "all_orthogonal"
    assert instance.items[1].allowed_rotations == "fixed"
    assert instance.items[2].allowed_rotations == [0, 1, 2]
    assert len(instance.constraints.min_distance) == 1
    assert instance.constraints.min_distance[0].distance == 5.0


def test_template_bytes_parse() -> None:
    payload = template_bytes()
    instance = instance_from_spreadsheet(payload)
    assert instance.bin.length == 100


def test_missing_items_sheet_raises(tmp_path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "bin"
    sheet.append(["length", "width", "height", "max_weight"])
    sheet.append([10, 10, 10, None])
    path = tmp_path / "broken.xlsx"
    workbook.save(path)
    with pytest.raises(ValueError, match="items"):
        instance_from_spreadsheet(path)
