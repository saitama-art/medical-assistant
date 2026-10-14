import pandas as pd
from config import DATA_PATH

_REQUIRED_COLUMNS = {
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
}

def load_retail_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请下载 UCI Online Retail.xlsx 并放到项目根目录的 data 文件夹。"
        )

    df = pd.read_excel(DATA_PATH)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要字段：{sorted(missing)}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")
    return df
