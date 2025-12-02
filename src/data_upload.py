import pandas as pd
from pathlib import Path

WX_SHEETS = ["07-10--06-11", "07-11--06-12"]
PV_SHEETS = ["07-10--06-11", "07-11--06-12"]


def _read_excel_sheets(path: Path, sheets):
    """
    Legge più sheet Excel assicurandosi che sia installato l'engine richiesto.
    Alza un messaggio chiaro se manca openpyxl.
    """
    try:
        return [pd.read_excel(path, sheet_name=s) for s in sheets]
    except ImportError as exc:
        raise ImportError(
            "Per leggere i file .xlsx serve il pacchetto 'openpyxl'. "
            "Installa le dipendenze con `pip install -r requirements.txt`."
        ) from exc


def load_wx(wx_path: str, sheets=WX_SHEETS) -> pd.DataFrame:
    """Carica e concatena i fogli meteo (wx_dataset)."""
    path = Path(wx_path)
    dfs = _read_excel_sheets(path, sheets)
    X = pd.concat(dfs, axis=0, ignore_index=True)
    return X


def load_pv(pv_path: str, sheets=PV_SHEETS) -> pd.DataFrame:
    """
    Carica e concatena i fogli PV (pv_dataset).
    Prima colonna = datetime, seconda = kwp (label).
    """
    path = Path(pv_path)
    dfs = _read_excel_sheets(path, sheets)
    y = pd.concat(dfs, axis=0, ignore_index=True)

    # Il file aveva intestazioni un po' strane → sistemiamo noi:
    y.columns = ["datetime", "kwp"]
    return y


def load_datasets(wx_path: str, pv_path: str):
    """Wrapper unico richiamato dal main."""
    X = load_wx(wx_path)
    y = load_pv(pv_path)
    return X, y
