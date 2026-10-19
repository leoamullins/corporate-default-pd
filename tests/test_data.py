from src.data import load_data, split_by_year, year_folds


def test_one_default_per_failed_firm(toy_panel_csv):
    df = load_data(toy_panel_csv)
    per_firm = df.groupby("company_name")["default"].agg("sum")
    assert per_firm.to_dict() == {"A": 0, "B": 1, "C": 1}


def test_default_only_in_final_year(toy_panel_csv):
    df = load_data(toy_panel_csv)
    defaults = df[df["default"] == 1]
    assert set(zip(defaults["company_name"], defaults["year"])) == {
        ("B", 2011),
        ("C", 2016),
    }


def test_labels_are_cleaned(toy_panel_csv):
    df = load_data(toy_panel_csv)

    assert "Year " not in df.columns

    failed = df.groupby("company_name")["firm_failed"].agg(["min", "max"])
    assert failed.loc["A"].tolist() == [0, 0]
    assert failed.loc["B"].tolist() == [1, 1]
    assert failed.loc["C"].tolist() == [1, 1]


def test_splits_have_disjoint_years(toy_panel_csv):
    train, val, test = split_by_year(load_data(toy_panel_csv))

    tr, va, te = (set(s["year"]) for s in (train, val, test))

    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)

    defaults = [int(s["default"].sum()) for s in (train, val, test)]
    assert defaults == [1, 0, 1]


def test_year_folds_never_look_ahead(toy_panel_csv):
    df = load_data(toy_panel_csv)
    folds = list(year_folds(df))
    assert folds

    for fit, check in folds:
        assert fit["year"].max() < check["year"].min()
