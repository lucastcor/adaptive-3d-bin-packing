"""Spreadsheet (.xlsx) import/export for problem instances.

Layout (one workbook):

- sheet ``meta``: key/value rows (``instance_id``, ``problem_type``,
  ``best_known_bins``);
- sheet ``bin``: header ``length, width, height, max_weight`` and one data row;
- sheet ``items``: header ``id, length, width, height, weight,
  allowed_rotations`` and one row per item;
- sheet ``min_distance`` (optional): header ``item_id_a, item_id_b, distance``;
- sheet ``LEIA-ME``: human instructions (ignored by the parser).

``allowed_rotations`` accepts: empty (= all_orthogonal), ``all_orthogonal``,
``fixed`` or a list of rotation ids like ``0,1,2``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from adaptive_3dbp.domain.models import (
    BinSpec,
    ItemSpec,
    MinDistanceConstraint,
    PackingConstraints,
    ProblemInstance,
    SolverResult,
)

_ITEM_HEADERS = ["id", "length", "width", "height", "weight", "allowed_rotations"]
_BIN_HEADERS = ["length", "width", "height", "max_weight"]
_DISTANCE_HEADERS = ["item_id_a", "item_id_b", "distance"]
_RESULT_PLACEMENT_HEADERS = [
    "item_id",
    "item_index",
    "bin_id",
    "bin_number",
    "x",
    "y",
    "z",
    "rotation_id",
    "placed_length",
    "placed_width",
    "placed_height",
    "item_length",
    "item_width",
    "item_height",
    "item_weight",
    "volume",
]
_RESULT_BIN_HEADERS = ["bin_id", "bin_number", "utilization", "occupied_volume", "unused_volume", "compactness"]
_RESULT_CONVERGENCE_HEADERS = [
    "evaluation",
    "elapsed_seconds",
    "best_bins",
    "best_objective_scalar",
    "current_bins",
    "current_objective_scalar",
    "bins_used",
    "objective_scalar",
    "accepted",
    "temperature",
    "delta",
    "improved",
]
_RESULT_DISTANCE_HEADERS = ["item_id_a", "item_id_b", "distance"]

_README_LINES = [
    "MODELO DE INSTANCIA - ADAPTIVE 3D BIN PACKING",
    "",
    "Preencha somente as celulas amarelas.",
    "Aba 'meta': identificacao da instancia.",
    "Aba 'bin': dimensoes da caixa. Use uma unica linha.",
    "Aba 'items': um item por linha. Cada id deve ser unico.",
    "Aba 'min_distance': opcional; use somente quando dois itens precisam ficar afastados.",
    "",
    "Unidades: use a mesma unidade em todas as medidas (cm, mm, m etc.).",
    "Campos vazios permitidos: max_weight, weight, best_known_bins e min_distance.",
    "allowed_rotations: vazio/all_orthogonal = todas | fixed = sem rotacao | lista tipo 0,1,2.",
    "",
    "Importacao:",
    "  CLI:  PYTHONPATH=src python3 -m adaptive_3dbp.cli.main import-spreadsheet arquivo.xlsx",
    "  CLI:  ... solve --instance arquivo.xlsx (importa direto)",
    "  API:  POST /api/instances/import-spreadsheet (corpo = bytes do .xlsx)",
]

_RESULT_README_LINES = [
    "EXPORTACAO DA SOLUCAO - ADAPTIVE 3D BIN PACKING",
    "",
    "Aba 'resumo': status, solver, seed e principais metricas da execucao.",
    "Aba 'placements': onde cada item ficou: caixa, coordenadas e dimensoes posicionadas.",
    "Aba 'bin_metrics': utilizacao e volumes por caixa.",
    "Aba 'convergence': historico amostrado da busca, util para graficos e auditoria.",
    "Aba 'min_distance': restricoes de distancia minima usadas na instancia.",
    "",
    "Sistema de coordenadas:",
    "  x = comprimento da caixa (length)",
    "  y = largura/profundidade da caixa (width)",
    "  z = altura da caixa (height)",
    "",
    "bin_id e item_index sao baseados em zero para compatibilidade com a API.",
    "bin_number e uma coluna amigavel, baseada em um, para leitura humana.",
]

_HEADER_COMMENTS = {
    "length": "Comprimento no eixo X da caixa/peca. Deve ser maior que zero.",
    "width": "Largura/profundidade no eixo Y do modelo. Deve ser maior que zero.",
    "height": "Altura no eixo Z do modelo. Deve ser maior que zero.",
    "max_weight": "Peso maximo da caixa. Opcional; deixe vazio se nao houver limite.",
    "id": "Identificador unico do item. Exemplo: item-001.",
    "weight": "Peso do item. Opcional.",
    "allowed_rotations": "Use vazio/all_orthogonal, fixed ou uma lista como 0,1,2.",
    "item_id_a": "ID do primeiro item do par.",
    "item_id_b": "ID do segundo item do par.",
    "distance": "Distancia minima entre os AABBs quando os itens estiverem na mesma caixa.",
}


def _decorate_workbook(workbook, min_item_rows: int = 50, min_distance_rows: int = 25) -> None:
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.table import Table, TableStyleInfo

    dark_fill = PatternFill("solid", fgColor="172033")
    input_fill = PatternFill("solid", fgColor="FFF7D6")
    optional_fill = PatternFill("solid", fgColor="EEF2F7")
    note_fill = PatternFill("solid", fgColor="DBEAFE")
    white_font = Font(color="FFFFFF", bold=True)
    title_font = Font(color="172033", bold=True, size=14)
    body_font = Font(color="334155", size=11)
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    wrap = Alignment(vertical="top", wrap_text=True)

    def style_header(sheet, row: int, columns: int) -> None:
        for cell in sheet[row][:columns]:
            cell.fill = dark_fill
            cell.font = white_font
            cell.border = border
            cell.alignment = center
            if cell.value in _HEADER_COMMENTS:
                cell.comment = Comment(_HEADER_COMMENTS[str(cell.value)], "Adaptive 3DBP")

    def style_grid(sheet, start_row: int, end_row: int, columns: int, optional_columns: set[int] | None = None) -> None:
        optional_columns = optional_columns or set()
        for row in sheet.iter_rows(min_row=start_row, max_row=end_row, min_col=1, max_col=columns):
            for cell in row:
                cell.border = border
                cell.alignment = center
                cell.fill = optional_fill if cell.column in optional_columns else input_fill

    def add_table(sheet, name: str, ref: str) -> None:
        table = Table(displayName=name, ref=ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        sheet.add_table(table)

    readme = workbook["LEIA-ME"]
    readme.sheet_view.showGridLines = False
    readme.column_dimensions["A"].width = 96
    for row in readme.iter_rows(min_col=1, max_col=1):
        cell = row[0]
        cell.alignment = wrap
        cell.font = title_font if cell.row == 1 else body_font
        if cell.row == 3:
            cell.fill = note_fill
            cell.border = border
    readme.freeze_panes = "A2"

    meta = workbook["meta"]
    meta.sheet_view.showGridLines = False
    meta.column_dimensions["A"].width = 22
    meta.column_dimensions["B"].width = 34
    meta.column_dimensions["C"].width = 18
    meta.column_dimensions["D"].width = 58
    meta.insert_rows(1)
    meta["A1"] = "campo"
    meta["B1"] = "valor"
    meta["C1"] = "obrigatorio"
    meta["D1"] = "como preencher"
    meta["C2"] = "sim"
    meta["D2"] = "Nome curto da instancia. Ex.: pedido-001."
    meta["C3"] = "sim"
    meta["D3"] = "Mantenha offline_3d_bpp."
    meta["C4"] = "nao"
    meta["D4"] = "Melhor numero conhecido de caixas. Deixe vazio se nao souber."
    style_header(meta, 1, 4)
    style_grid(meta, 2, 4, 4, optional_columns={3, 4})
    meta.freeze_panes = "A2"
    add_table(meta, "meta_table", "A1:D4")

    bin_sheet = workbook["bin"]
    bin_sheet.sheet_view.showGridLines = False
    for column, width in zip(("A", "B", "C", "D"), (18, 18, 18, 18), strict=True):
        bin_sheet.column_dimensions[column].width = width
    style_header(bin_sheet, 1, 4)
    style_grid(bin_sheet, 2, 2, 4, optional_columns={4})
    bin_sheet.freeze_panes = "A2"
    add_table(bin_sheet, "bin_table", "A1:D2")

    items = workbook["items"]
    items.sheet_view.showGridLines = False
    for column, width in zip(("A", "B", "C", "D", "E", "F"), (22, 14, 14, 14, 14, 24), strict=True):
        items.column_dimensions[column].width = width
    target_item_row = max(items.max_row, min_item_rows + 1)
    for row_index in range(items.max_row + 1, target_item_row + 1):
        for column_index in range(1, 7):
            items.cell(row=row_index, column=column_index, value=None)
    style_header(items, 1, 6)
    style_grid(items, 2, target_item_row, 6, optional_columns={5, 6})
    items.freeze_panes = "A2"
    # Nota: nao definir sheet.auto_filter junto com Table no mesmo intervalo —
    # o Excel considera o arquivo corrompido (a Table ja tem filtro proprio).
    add_table(items, "items_table", f"A1:F{target_item_row}")

    distance = workbook["min_distance"]
    distance.sheet_view.showGridLines = False
    for column, width in zip(("A", "B", "C"), (22, 22, 18), strict=True):
        distance.column_dimensions[column].width = width
    target_distance_row = max(distance.max_row, min_distance_rows + 1)
    for row_index in range(distance.max_row + 1, target_distance_row + 1):
        for column_index in range(1, 4):
            distance.cell(row=row_index, column=column_index, value=None)
    style_header(distance, 1, 3)
    style_grid(distance, 2, target_distance_row, 3)
    distance.freeze_panes = "A2"
    add_table(distance, "min_distance_table", f"A1:C{target_distance_row}")

    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    cell.alignment = Alignment(
                        horizontal=cell.alignment.horizontal,
                        vertical=cell.alignment.vertical,
                        wrap_text=True,
                    )


def _require_openpyxl():
    try:
        import openpyxl

        return openpyxl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is required for spreadsheet support: pip install openpyxl") from exc


def _rows(sheet) -> list[list[Any]]:
    return [list(row) for row in sheet.iter_rows(values_only=True)]


def _header_index(rows: list[list[Any]], expected: list[str], sheet_name: str) -> dict[str, int]:
    if not rows:
        raise ValueError(f"sheet '{sheet_name}' is empty")
    header = [str(cell).strip().lower() if cell is not None else "" for cell in rows[0]]
    index: dict[str, int] = {}
    for name in expected:
        if name in header:
            index[name] = header.index(name)
    missing = [name for name in expected if name not in index and name not in ("max_weight", "weight", "allowed_rotations")]
    if missing:
        raise ValueError(f"sheet '{sheet_name}' is missing required columns: {missing}")
    return index


def _cell(row: list[Any], index: dict[str, int], name: str) -> Any:
    position = index.get(name)
    if position is None or position >= len(row):
        return None
    value = row[position]
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _parse_rotations(value: Any) -> str | list[int]:
    if value is None:
        return "all_orthogonal"
    text = str(value).strip().lower()
    if text in ("", "all", "all_orthogonal"):
        return "all_orthogonal"
    if text == "fixed":
        return "fixed"
    separators = text.replace(";", ",")
    return [int(part.strip()) for part in separators.split(",") if part.strip() != ""]


def instance_from_spreadsheet(source: Path | bytes) -> ProblemInstance:
    openpyxl = _require_openpyxl()
    if isinstance(source, bytes):
        workbook = openpyxl.load_workbook(io.BytesIO(source), data_only=True)
    else:
        workbook = openpyxl.load_workbook(source, data_only=True)
    sheet_names = {name.strip().lower(): name for name in workbook.sheetnames}

    meta: dict[str, Any] = {}
    if "meta" in sheet_names:
        for row in _rows(workbook[sheet_names["meta"]]):
            if row and row[0] is not None and len(row) > 1:
                meta[str(row[0]).strip().lower()] = row[1]

    if "bin" not in sheet_names:
        raise ValueError("workbook must contain a 'bin' sheet")
    bin_rows = _rows(workbook[sheet_names["bin"]])
    bin_index = _header_index(bin_rows, _BIN_HEADERS, "bin")
    bin_data = next((row for row in bin_rows[1:] if any(cell is not None for cell in row)), None)
    if bin_data is None:
        raise ValueError("sheet 'bin' must contain one data row")
    max_weight = _cell(bin_data, bin_index, "max_weight")
    bin_spec = BinSpec(
        length=float(_cell(bin_data, bin_index, "length")),
        width=float(_cell(bin_data, bin_index, "width")),
        height=float(_cell(bin_data, bin_index, "height")),
        max_weight=float(max_weight) if max_weight is not None else None,
    )

    if "items" not in sheet_names:
        raise ValueError("workbook must contain an 'items' sheet")
    item_rows = _rows(workbook[sheet_names["items"]])
    item_index = _header_index(item_rows, _ITEM_HEADERS, "items")
    items: list[ItemSpec] = []
    for row in item_rows[1:]:
        if not any(cell is not None for cell in row):
            continue
        weight = _cell(row, item_index, "weight")
        items.append(
            ItemSpec(
                id=str(_cell(row, item_index, "id")),
                length=float(_cell(row, item_index, "length")),
                width=float(_cell(row, item_index, "width")),
                height=float(_cell(row, item_index, "height")),
                weight=float(weight) if weight is not None else None,
                allowed_rotations=_parse_rotations(_cell(row, item_index, "allowed_rotations")),
            )
        )
    if not items:
        raise ValueError("sheet 'items' must contain at least one item")

    min_distance: list[MinDistanceConstraint] = []
    if "min_distance" in sheet_names:
        distance_rows = _rows(workbook[sheet_names["min_distance"]])
        if distance_rows and any(cell is not None for cell in distance_rows[0]):
            distance_index = _header_index(distance_rows, _DISTANCE_HEADERS, "min_distance")
            for row in distance_rows[1:]:
                if not any(cell is not None for cell in row):
                    continue
                min_distance.append(
                    MinDistanceConstraint(
                        item_id_a=str(_cell(row, distance_index, "item_id_a")),
                        item_id_b=str(_cell(row, distance_index, "item_id_b")),
                        distance=float(_cell(row, distance_index, "distance")),
                    )
                )

    best_known = meta.get("best_known_bins")
    return ProblemInstance(
        instance_id=str(meta.get("instance_id") or "spreadsheet-instance"),
        problem_type=str(meta.get("problem_type") or "offline_3d_bpp"),
        bin=bin_spec,
        items=items,
        constraints=PackingConstraints(min_distance=min_distance),
        best_known_bins=int(best_known) if best_known is not None else None,
    )


def build_instance_template_workbook():
    openpyxl = _require_openpyxl()
    from openpyxl.styles import Font

    workbook = openpyxl.Workbook()
    bold = Font(bold=True)

    readme = workbook.active
    readme.title = "LEIA-ME"
    for line_number, line in enumerate(_README_LINES, start=1):
        cell = readme.cell(row=line_number, column=1, value=line)
        if line_number == 1:
            cell.font = bold
    readme.column_dimensions["A"].width = 90

    meta = workbook.create_sheet("meta")
    meta.append(["instance_id", "exemplo-planilha-001"])
    meta.append(["problem_type", "offline_3d_bpp"])
    meta.append(["best_known_bins", None])

    bin_sheet = workbook.create_sheet("bin")
    bin_sheet.append(_BIN_HEADERS)
    for cell in bin_sheet[1]:
        cell.font = bold
    bin_sheet.append([100, 100, 100, None])

    items = workbook.create_sheet("items")
    items.append(_ITEM_HEADERS)
    for cell in items[1]:
        cell.font = bold
    items.append(["item-a", 30, 20, 10, None, "all_orthogonal"])
    items.append(["item-b", 25, 25, 25, None, "fixed"])
    items.append(["item-c", 40, 15, 12, None, "0,1,2"])

    distance = workbook.create_sheet("min_distance")
    distance.append(_DISTANCE_HEADERS)
    for cell in distance[1]:
        cell.font = bold
    distance.append(["item-a", "item-b", 5])

    _decorate_workbook(workbook)
    return workbook


def write_instance_template(path: Path) -> Path:
    workbook = build_instance_template_workbook()
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


def template_bytes() -> bytes:
    workbook = build_instance_template_workbook()
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _rotations_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(rotation) for rotation in value)
    return str(value)


def build_instance_workbook(instance: ProblemInstance):
    """Build a workbook representing an existing instance (export path)."""
    openpyxl = _require_openpyxl()
    from openpyxl.styles import Font

    workbook = openpyxl.Workbook()
    bold = Font(bold=True)

    readme = workbook.active
    readme.title = "LEIA-ME"
    for line_number, line in enumerate(_README_LINES, start=1):
        cell = readme.cell(row=line_number, column=1, value=line)
        if line_number == 1:
            cell.font = bold
    readme.column_dimensions["A"].width = 90

    meta = workbook.create_sheet("meta")
    meta.append(["instance_id", instance.instance_id])
    meta.append(["problem_type", instance.problem_type])
    meta.append(["best_known_bins", instance.best_known_bins])

    bin_sheet = workbook.create_sheet("bin")
    bin_sheet.append(_BIN_HEADERS)
    for cell in bin_sheet[1]:
        cell.font = bold
    bin_sheet.append(
        [instance.bin.length, instance.bin.width, instance.bin.height, instance.bin.max_weight]
    )

    items = workbook.create_sheet("items")
    items.append(_ITEM_HEADERS)
    for cell in items[1]:
        cell.font = bold
    for item in instance.items:
        items.append(
            [item.id, item.length, item.width, item.height, item.weight, _rotations_to_text(item.allowed_rotations)]
        )

    distance = workbook.create_sheet("min_distance")
    distance.append(_DISTANCE_HEADERS)
    for cell in distance[1]:
        cell.font = bold
    for rule in instance.constraints.min_distance:
        distance.append([rule.item_id_a, rule.item_id_b, rule.distance])

    _decorate_workbook(
        workbook,
        min_item_rows=max(50, len(instance.items) + 10),
        min_distance_rows=max(25, len(instance.constraints.min_distance) + 10),
    )
    return workbook


def instance_to_spreadsheet_bytes(instance: ProblemInstance) -> bytes:
    workbook = build_instance_workbook(instance)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _decorate_result_workbook(workbook) -> None:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.table import Table, TableStyleInfo

    dark_fill = PatternFill("solid", fgColor="172033")
    input_fill = PatternFill("solid", fgColor="F8FAFC")
    note_fill = PatternFill("solid", fgColor="DBEAFE")
    white_font = Font(color="FFFFFF", bold=True)
    title_font = Font(color="172033", bold=True, size=14)
    body_font = Font(color="334155", size=11)
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="top", wrap_text=True)

    def style_header(sheet, row: int, columns: int) -> None:
        for cell in sheet[row][:columns]:
            cell.fill = dark_fill
            cell.font = white_font
            cell.border = border
            cell.alignment = center

    def style_grid(sheet, start_row: int, end_row: int, columns: int) -> None:
        if end_row < start_row:
            return
        for row in sheet.iter_rows(min_row=start_row, max_row=end_row, min_col=1, max_col=columns):
            for cell in row:
                cell.border = border
                cell.alignment = center
                cell.fill = input_fill

    def add_table(sheet, name: str, columns: int) -> None:
        if sheet.max_row < 2:
            return
        last_column = sheet.cell(row=1, column=columns).column_letter
        ref = f"A1:{last_column}{sheet.max_row}"
        table = Table(displayName=name, ref=ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        sheet.add_table(table)

    readme = workbook["LEIA-ME"]
    readme.sheet_view.showGridLines = False
    readme.column_dimensions["A"].width = 96
    for row in readme.iter_rows(min_col=1, max_col=1):
        cell = row[0]
        cell.alignment = left
        cell.font = title_font if cell.row == 1 else body_font
        if cell.row == 3:
            cell.fill = note_fill
            cell.border = border
    readme.freeze_panes = "A2"

    widths = {
        "resumo": (24, 38),
        "placements": (22, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 14, 14, 14, 14, 16),
        "bin_metrics": (12, 12, 16, 18, 18, 16),
        "convergence": (14, 18, 12, 22, 14, 24, 12, 18, 12, 14, 14, 12),
        "min_distance": (22, 22, 18),
    }
    columns_by_sheet = {
        "resumo": 2,
        "placements": len(_RESULT_PLACEMENT_HEADERS),
        "bin_metrics": len(_RESULT_BIN_HEADERS),
        "convergence": len(_RESULT_CONVERGENCE_HEADERS),
        "min_distance": len(_RESULT_DISTANCE_HEADERS),
    }

    for sheet_name, columns in columns_by_sheet.items():
        sheet = workbook[sheet_name]
        sheet.sheet_view.showGridLines = False
        for index, width in enumerate(widths[sheet_name], start=1):
            sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = width
        style_header(sheet, 1, columns)
        style_grid(sheet, 2, sheet.max_row, columns)
        sheet.freeze_panes = "A2"
        # Sem sheet.auto_filter aqui: a Table do intervalo ja inclui filtro, e a
        # duplicacao faz o Excel pedir "reparo" do arquivo.
        add_table(sheet, f"{sheet_name}_table", columns)


def build_result_workbook(instance: ProblemInstance, result: SolverResult, run_id: int | None = None):
    openpyxl = _require_openpyxl()

    workbook = openpyxl.Workbook()
    readme = workbook.active
    readme.title = "LEIA-ME"
    for line_number, line in enumerate(_RESULT_README_LINES, start=1):
        readme.cell(row=line_number, column=1, value=line)

    resumo = workbook.create_sheet("resumo")
    resumo.append(["campo", "valor"])
    summary_rows = [
        ("run_id", run_id),
        ("instance_id", instance.instance_id),
        ("result_instance_id", result.instance_id),
        ("status", result.status),
        ("solver", result.solver),
        ("solver_version", result.solver_version),
        ("configuration_id", result.configuration_id),
        ("seed", result.seed),
        ("best_found_bins", result.best_found_bins),
        ("lower_bound", result.lower_bound),
        ("best_known_bins", result.best_known_bins),
        ("optimal_proven", result.optimal_proven),
        ("gap_to_lower_bound", result.gap_to_lower_bound),
        ("gap_to_best_known", result.gap_to_best_known),
        ("runtime_seconds", result.runtime_seconds),
        ("evaluations", result.evaluations),
        ("objective_scalar", result.objective_scalar),
        ("item_count", len(instance.items)),
        ("placement_count", len(result.placements)),
        ("bin_length", instance.bin.length),
        ("bin_width", instance.bin.width),
        ("bin_height", instance.bin.height),
        ("min_distance_rule_count", len(instance.constraints.min_distance)),
    ]
    for key, value in summary_rows:
        resumo.append([key, value])

    item_by_id = {item.id: item for item in instance.items}
    placements = workbook.create_sheet("placements")
    placements.append(_RESULT_PLACEMENT_HEADERS)
    for placement in result.placements:
        item = item_by_id.get(placement.item_id)
        placements.append(
            [
                placement.item_id,
                placement.item_index,
                placement.bin_id,
                placement.bin_id + 1,
                placement.x,
                placement.y,
                placement.z,
                placement.rotation_id,
                placement.placed_length,
                placement.placed_width,
                placement.placed_height,
                item.length if item else None,
                item.width if item else None,
                item.height if item else None,
                item.weight if item else None,
                placement.volume,
            ]
        )

    bin_metrics = workbook.create_sheet("bin_metrics")
    bin_metrics.append(_RESULT_BIN_HEADERS)
    for metric in result.bin_metrics:
        bin_metrics.append(
            [
                metric.bin_id,
                metric.bin_id + 1,
                metric.utilization,
                metric.occupied_volume,
                metric.unused_volume,
                metric.compactness,
            ]
        )

    convergence = workbook.create_sheet("convergence")
    convergence.append(_RESULT_CONVERGENCE_HEADERS)
    for point in result.convergence:
        convergence.append(
            [
                point.evaluation,
                point.elapsed_seconds,
                point.best_bins,
                point.best_objective_scalar,
                point.current_bins,
                point.current_objective_scalar,
                point.bins_used,
                point.objective_scalar,
                point.accepted,
                point.temperature,
                point.delta,
                point.improved,
            ]
        )

    distance = workbook.create_sheet("min_distance")
    distance.append(_RESULT_DISTANCE_HEADERS)
    for rule in instance.constraints.min_distance:
        distance.append([rule.item_id_a, rule.item_id_b, rule.distance])

    _decorate_result_workbook(workbook)
    return workbook


def result_to_spreadsheet_bytes(instance: ProblemInstance, result: SolverResult, run_id: int | None = None) -> bytes:
    workbook = build_result_workbook(instance, result, run_id)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
