from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from wham.expr import Expr, ExprError, parse


@pytest.mark.parametrize("dtype", [pa.bool_(), pa.int8(), pa.int32()])
@pytest.mark.parametrize("source", [
    "flag == 1", "flag == 0", "1 == flag", "flag != 1",
    "flag == 2", "flag < 0.5", "flag >= -1", "flag == True",
    "flag == False", "True == flag", "flag == other",
    "(flag == 1 | other == 0) & pt > 26",
])
def test_boolean_numeric_comparisons(dtype, source):
    from wham.expr import widened_arrow

    table = pa.table({
        "flag": pa.array([0, 1, 0, 1], type=pa.int8()).cast(dtype),
        "other": [True, True, False, False],
        "pt": [20.0, 25.0, 30.0, 40.0],
        "row": [0, 1, 2, 3],
    })
    cols = {name: table[name].to_numpy() for name in table.column_names}
    parsed = parse(source)
    expected = cols["row"][parsed.evaluate(cols)].tolist()
    actual = ds.dataset(table).to_table(filter=parsed.arrow(table.schema))
    assert actual["row"].to_pylist() == expected

    filt = widened_arrow(parsed, {"pt": (0.9, 1.1)}, table.schema)
    rows = set(ds.dataset(table).to_table(filter=filt)["row"].to_pylist())
    for scale in [0.9, 1.0, 1.1]:
        shifted = dict(cols, pt=cols["pt"] * scale)
        assert set(cols["row"][parsed.evaluate(shifted)]) <= rows


def test_bare_boolean_filter_with_schema():
    table = pa.table({"flag": [True, False, True], "other": [False, False, True]})
    parsed = parse("flag & ~other")
    assert ds.dataset(table).to_table(filter=parsed.arrow(table.schema)).num_rows == 1


@pytest.fixture
def table() -> pa.Table:
    rng = np.random.default_rng(7)
    n = 5000
    return pa.table(
        {
            "pt_1": rng.uniform(0, 100, n),
            "eta_1": rng.uniform(-3, 3, n),
            "iso_1": rng.uniform(0, 0.5, n),
            "id_2": rng.integers(0, 8, n).astype(np.int32),
            "os": rng.integers(0, 2, n).astype(np.int32),
            "weight": rng.uniform(0.5, 1.5, n),
        }
    )


CUTS = [
    "pt_1 > 26",
    "abs(eta_1) < 2.4",
    "iso_1 < 0.15 & pt_1 > 30",
    "(id_2 >= 5) | (iso_1 < 0.1)",
    "~(os == 1)",
    "1 < id_2 < 5",
    "abs(eta_1) < 2.4 & (1 < id_2 < 5) & pt_1 > 26",
    "pt_1 / 2 > 20",
    "pt_1 + iso_1 * 10 > 30",
    "-eta_1 > 0",
    "id_2 != 3",
    "pt_1 > 26 and iso_1 < 0.15",
    "id_2 >= 5 or iso_1 < 0.1",
    "not (os == 1)",
    "pt_1 > 26 & iso_1 < 0.15 | id_2 == 7",
]


@pytest.mark.parametrize("source", CUTS)
def test_arrow_matches_numpy(table: pa.Table, source: str) -> None:
    parsed = parse(source)

    cols = {name: table.column(name).to_numpy() for name in parsed.columns}
    np_mask = parsed.evaluate(cols)
    assert np_mask.dtype == bool

    dataset = ds.dataset(table)
    arrow_rows = dataset.to_table(filter=parsed.arrow()).num_rows
    assert arrow_rows == int(np_mask.sum()), source


def test_columns_extracted() -> None:
    parsed = parse("abs(eta_1) < 2.4 & pt_1 > 26 & (1 < id_2 < 5)")
    assert parsed.columns == {"eta_1", "pt_1", "id_2"}


def test_weight_expression(table: pa.Table) -> None:
    parsed = parse("weight * 2 + 1")
    cols = {"weight": table.column("weight").to_numpy()}
    out = parsed.evaluate(cols)
    assert np.allclose(out, cols["weight"] * 2 + 1)


def test_amp_precedence_matches_intent(table: pa.Table) -> None:
    # Without preprocessing, python would parse this as iso_1 < (0.15 & pt_1) > 30.
    a = parse("iso_1 < 0.15 & pt_1 > 30")
    b = parse("(iso_1 < 0.15) & (pt_1 > 30)")
    cols = {n: table.column(n).to_numpy() for n in ("iso_1", "pt_1")}
    assert np.array_equal(a.evaluate(cols), b.evaluate(cols))
    assert a.evaluate(cols).sum() > 0


@pytest.mark.parametrize(
    "source, match",
    [
        ("sin(eta_1) < 1", "only abs"),
        ("pt_1.size > 3", "unsupported syntax"),
        ("__import__('os')", "only abs"),
        ("pt_1 > 'x'", "numeric literals"),
        ("[1,2]", "unsupported syntax"),
        ("", "empty"),
        ("pt_1 >", "syntax error"),
    ],
)
def test_rejections(source: str, match: str) -> None:
    with pytest.raises(ExprError, match=match):
        parse(source)


def test_missing_column_error() -> None:
    parsed = parse("nope > 1")
    with pytest.raises(ExprError, match="missing columns"):
        parsed.evaluate({"pt_1": np.ones(3)})


def test_parse_cached() -> None:
    assert parse("pt_1 > 26") is parse("pt_1 > 26")


def test_expr_is_picklable_via_source() -> None:
    # Workers re-parse from source strings; Expr itself need not pickle.
    parsed = parse("pt_1 > 26")
    assert isinstance(parsed, Expr)
    assert parse(parsed.source).columns == parsed.columns


# ------------------------------------------------- widened read filters


def test_widened_arrow_superset(table: pa.Table) -> None:
    """Every row passing the selection with scaled columns (any factor within
    bounds) must pass the widened read filter."""
    from wham.expr import widened_arrow

    parsed = parse("pt_1 > 26 & abs(eta_1) < 2.4 & iso_1 < 0.15")
    bounds = {"pt_1": (0.9, 1.1), "eta_1": (0.9, 1.1)}
    filt = widened_arrow(parsed, bounds)
    assert filt is not None

    indexed = table.append_column("idx", pa.array(np.arange(table.num_rows)))
    picked = ds.dataset(indexed).to_table(filter=filt).column("idx").to_numpy()
    wide = np.zeros(table.num_rows, dtype=bool)
    wide[picked] = True

    cols = {n: table.column(n).to_numpy() for n in ("pt_1", "eta_1", "iso_1")}
    for s_pt in (0.9, 1.0, 1.1):
        for s_eta in (0.9, 1.0, 1.1):
            scaled = dict(cols, pt_1=cols["pt_1"] * s_pt, eta_1=cols["eta_1"] * s_eta)
            mask = parsed.evaluate(scaled)
            assert not np.any(mask & ~wide), (s_pt, s_eta)
    # and it stays a filter: atoms on unscaled columns still cut
    assert wide.sum() < table.num_rows


def test_widened_arrow_bounds_math(table: pa.Table) -> None:
    from wham.expr import widened_arrow

    dataset = ds.dataset(table)
    pt = table.column("pt_1").to_numpy()
    eta = table.column("eta_1").to_numpy()

    # lower bound relaxes by the max factor: pt_1 > 26 -> pt_1 > 26/1.1
    n = dataset.to_table(filter=widened_arrow(parse("pt_1 > 26"),
                                              {"pt_1": (0.9, 1.1)})).num_rows
    assert n == int((pt > 26 / 1.1).sum())

    # upper bound relaxes by the min factor; constant-on-the-left mirrored
    n = dataset.to_table(filter=widened_arrow(parse("26 > pt_1"),
                                              {"pt_1": (0.9, 1.1)})).num_rows
    assert n == int((pt < 26 / 0.9).sum())

    # abs() is scale-covariant for positive factors
    n = dataset.to_table(filter=widened_arrow(parse("abs(eta_1) < 2.4"),
                                              {"eta_1": (0.9, 1.1)})).num_rows
    assert n == int((np.abs(eta) < 2.4 / 0.9).sum())

    # negative constant on a lower bound relaxes by the MIN factor
    n = dataset.to_table(filter=widened_arrow(parse("eta_1 > -2.4"),
                                              {"eta_1": (0.9, 1.1)})).num_rows
    assert n == int((eta > -2.4 / 0.9).sum())


def test_widened_arrow_drops_unwidenable(table: pa.Table) -> None:
    from wham.expr import widened_arrow

    dataset = ds.dataset(table)
    pt = table.column("pt_1").to_numpy()

    # == on a scaled column can't be relaxed: dropped, the pt atom survives
    filt = widened_arrow(parse("id_2 == 3 & pt_1 > 26"), {"id_2": (0.9, 1.1)})
    assert dataset.to_table(filter=filt).num_rows == int((pt > 26).sum())

    # an OR-group touching a scaled column is one atom: dropped whole
    assert widened_arrow(parse("(pt_1 > 26) | (iso_1 < 0.1)"),
                         {"pt_1": (0.9, 1.1)}) is None

    # everything dropped -> None (caller reads unfiltered)
    assert widened_arrow(parse("id_2 == 3"), {"id_2": (0.9, 1.1)}) is None

    # untouched expressions pass through exactly
    filt = widened_arrow(parse("pt_1 > 26"), {"m_vis": (0.9, 1.1)})
    assert dataset.to_table(filter=filt).num_rows == int((pt > 26).sum())


def test_column_bounds() -> None:
    from wham.expr import column_bounds

    e = parse("id_2 == 1 & iso_1 > 0.15 & pt_1 >= 20 & pt_1 < 30")
    assert column_bounds(e, "pt_1") == (20.0, 30.0)
    assert column_bounds(e, "iso_1") == (0.15, None)
    assert column_bounds(e, "id_2") == (None, None)   # == is not a window
    assert column_bounds(parse("20 <= pt_1 & 30 > pt_1"), "pt_1") == (20.0, 30.0)
