import io
import re
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# PDF
import pdfplumber


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="TR PDF Portfolio Analyzer", page_icon="📄", layout="wide")
st.title("📄 Trade Republic · Portfolio Analyzer (PDF/CSV/XLSX)")
st.caption("Sube un PDF de Trade Republic. La app intentará extraer tablas y analizar tu cartera. Todo local en memoria.")


# =========================
# HELPERS
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_numeric_series(x: pd.Series) -> pd.Series:
    if x is None:
        return x
    s = x.astype(str).str.strip()

    # quitar símbolos raros (moneda, espacios, etc.)
    s = s.str.replace(r"[^\d,\.\-\+]", "", regex=True)

    # heurística: 1.234,56 -> 1234.56
    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)
    both = has_comma & has_dot
    only_comma = has_comma & ~has_dot

    s.loc[both] = s.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def read_any(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".csv"):
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(data), engine="python")

    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")

    raise ValueError("Formato no soportado para read_any")


def extract_tables_from_pdf(pdf_bytes: bytes, max_pages: int = 30) -> List[pd.DataFrame]:
    """
    Intenta extraer tablas del PDF con pdfplumber.
    Devuelve una lista de DataFrames candidatos.
    """
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            try:
                page_tables = page.extract_tables()
            except Exception:
                page_tables = None

            if not page_tables:
                continue

            for t in page_tables:
                if not t or len(t) < 2:
                    continue
                df = pd.DataFrame(t)
                # quitar filas totalmente vacías
                df = df.dropna(how="all")
                # filtrar cosas muy pequeñas
                if df.shape[0] >= 3 and df.shape[1] >= 2:
                    tables.append(df)
    return tables


def guess_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heurística: si la primera fila parece cabecera (mucho texto), úsala como header.
    """
    if df.empty:
        return df

    first = df.iloc[0].astype(str).str.strip()
    # si hay al menos 2 celdas con letras, asumimos cabecera
    alpha_cells = sum(bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", v)) for v in first.values)
    if alpha_cells >= max(2, int(df.shape[1] * 0.4)):
        new_cols = [str(x).strip() if str(x).strip() else f"col_{i}" for i, x in enumerate(first.values)]
        out = df.iloc[1:].copy()
        out.columns = new_cols
        return out.reset_index(drop=True)

    # si no, columnas genéricas
    out = df.copy()
    out.columns = [f"col_{i}" for i in range(out.shape[1])]
    return out.reset_index(drop=True)


def auto_map_positions(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    ncols = [_norm(c) for c in cols]

    def pick(cands):
        for cand in cands:
            for i, nc in enumerate(ncols):
                if cand in nc:
                    return cols[i]
        return None

    return {
        "name": pick(["name", "nombre", "instrument", "producto", "asset", "valor"]),
        "isin": pick(["isin"]),
        "quantity": pick(["quantity", "cantidad", "shares", "unidades", "units", "stücke"]),
        "avg_price": pick(["average price", "precio medio", "avg price", "precio compra", "purchase price", "avg cost"]),
        "cost": pick(["cost", "importe invertido", "invested", "cost basis", "acquisition", "valor compra"]),
        "current_price": pick(["current price", "precio actual", "price", "cotización"]),
        "market_value": pick(["market value", "valor de mercado", "current value", "valor actual", "valuation", "valor"]),
        "currency": pick(["currency", "divisa", "ccy", "währung"]),
    }


def choose_col(label: str, cols: List[str], default: Optional[str]) -> Optional[str]:
    opts = ["(ninguna)"] + cols
    idx = cols.index(default) + 1 if default in cols else 0
    picked = st.selectbox(label, opts, index=idx)
    return None if picked == "(ninguna)" else picked


# =========================
# UPLOAD
# =========================
with st.sidebar:
    st.header("1) Subir archivo")
    uploaded = st.file_uploader("PDF / CSV / XLSX", type=["pdf", "csv", "xlsx"])
    show_raw = st.checkbox("Mostrar tablas crudas", value=False)
    top_n = st.slider("Top N", 5, 30, 12)

if not uploaded:
    st.info("⬅️ Sube tu PDF (o CSV/XLSX) para empezar.")
    st.stop()

name = uploaded.name.lower()
if name.endswith(".pdf"):
    st.subheader("Extracción desde PDF")
    pdf_bytes = uploaded.getvalue()

    with st.spinner("Leyendo PDF y buscando tablas..."):
        tables = extract_tables_from_pdf(pdf_bytes)

    if not tables:
        st.error(
            "No he podido extraer ninguna tabla del PDF con lectura normal.\n\n"
            "Esto suele pasar si el PDF es una imagen escaneada o si la tabla está muy 'maquetada'.\n"
            "Solución más fácil: intenta conseguir CSV/XLSX (o un PDF diferente: 'statement' con tablas)."
        )
        st.stop()

    # preparar candidatos (con cabecera estimada)
    candidates = [guess_header_row(t) for t in tables]

    st.write(f"Tablas encontradas: **{len(candidates)}**. Elige la que tenga tu cartera/posiciones:")
    labels = [f"Tabla #{i+1} · {c.shape[0]} filas × {c.shape[1]} cols" for i, c in enumerate(candidates)]
    idx = st.selectbox("Tabla detectada", list(range(len(candidates))), format_func=lambda i: labels[i])
    df = candidates[idx].copy()

    if show_raw:
        st.caption("Vista previa (tabla seleccionada)")
        st.dataframe(df, use_container_width=True)

else:
    df = read_any(uploaded)
    st.subheader("Archivo tabular cargado")
    if show_raw:
        st.dataframe(df, use_container_width=True)

# =========================
# POSITIONS ANALYSIS (simple)
# =========================
st.divider()
st.header("Análisis de posiciones (holdings)")

cols = list(df.columns)
suggested = auto_map_positions(df)

with st.expander("Mapeo de columnas (ajústalo si hace falta)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        col_name = choose_col("Nombre/Activo", cols, suggested["name"])
        col_isin = choose_col("ISIN", cols, suggested["isin"])
        col_qty = choose_col("Cantidad", cols, suggested["quantity"])
    with c2:
        col_avg = choose_col("Precio medio", cols, suggested["avg_price"])
        col_cost = choose_col("Coste total / invertido", cols, suggested["cost"])
        col_ccy = choose_col("Divisa", cols, suggested["currency"])
    with c3:
        col_px = choose_col("Precio actual", cols, suggested["current_price"])
        col_mv = choose_col("Valor actual / mercado", cols, suggested["market_value"])

if col_name is None:
    st.error("Necesito al menos la columna de Nombre/Activo para analizar.")
    st.stop()

work = pd.DataFrame()
work["name"] = df[col_name].astype(str).str.strip()
work["isin"] = df[col_isin].astype(str).str.strip() if col_isin else ""
work["currency"] = df[col_ccy].astype(str).str.strip().str.upper() if col_ccy else "N/A"
work["quantity"] = _to_numeric_series(df[col_qty]) if col_qty else np.nan
work["avg_price"] = _to_numeric_series(df[col_avg]) if col_avg else np.nan
work["cost"] = _to_numeric_series(df[col_cost]) if col_cost else np.nan
work["current_price"] = _to_numeric_series(df[col_px]) if col_px else np.nan
work["market_value"] = _to_numeric_series(df[col_mv]) if col_mv else np.nan

# completar si faltan
if work["market_value"].isna().all():
    work["market_value"] = work["quantity"] * work["current_price"]
if work["cost"].isna().all():
    work["cost"] = work["quantity"] * work["avg_price"]

work = work.replace([np.inf, -np.inf], np.nan)
work["market_value"] = work["market_value"].fillna(0.0)
work["cost"] = work["cost"].fillna(0.0)

total_mv = float(work["market_value"].sum())
total_cost = float(work["cost"].sum())
pnl = total_mv - total_cost
pnl_pct = (pnl / total_cost) if total_cost else np.nan

work["pnl"] = work["market_value"] - work["cost"]
work["pnl_pct"] = np.where(work["cost"] != 0, work["pnl"] / work["cost"], np.nan)
work["weight"] = np.where(total_mv != 0, work["market_value"] / total_mv, 0.0)

work = work.sort_values("market_value", ascending=False)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Valor total", f"{total_mv:,.2f}")
m2.metric("Coste total", f"{total_cost:,.2f}")
m3.metric("P&L", f"{pnl:,.2f}")
m4.metric("P&L %", f"{pnl_pct*100:,.2f}%" if np.isfinite(pnl_pct) else "N/A")

st.subheader("Tabla limpia")
st.dataframe(
    work[["name", "isin", "currency", "quantity", "avg_price", "current_price", "cost", "market_value", "pnl", "pnl_pct", "weight"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Gráficos rápidos (sin librerías extra)")
top = work.head(top_n).copy()

# barras pesos
chart_w = top.set_index("name")[["weight"]]
st.caption(f"Pesos (Top {top_n})")
st.bar_chart(chart_w)

# barras P&L
chart_p = top.set_index("name")[["pnl"]]
st.caption(f"P&L (Top {top_n})")
st.bar_chart(chart_p)

# descarga
csv_out = work.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar tabla limpia (CSV)", data=csv_out, file_name="tr_positions_clean.csv", mime="text/csv")
