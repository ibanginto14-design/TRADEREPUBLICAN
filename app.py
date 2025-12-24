import io
import re
import math
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Trade Republic · Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Trade Republic · Portfolio Analyzer (local, sin login)")
st.caption("Sube tu CSV/XLSX exportado y obtén resumen, pesos, P&L, concentración y (opcional) XIRR con operaciones.")


# =========================
# HELPERS
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("€", "eur").replace("$", "usd")
    return s


def _to_numeric_series(x: pd.Series) -> pd.Series:
    """
    Convierte a numérico tolerando:
    - separador decimal coma
    - miles con punto o coma
    - símbolos moneda
    """
    if x is None:
        return x
    s = x.astype(str).str.strip()
    s = s.str.replace(r"[^\d,\.\-\+]", "", regex=True)

    # Heurística: si hay coma y punto, asumimos miles con punto y decimal coma (1.234,56)
    # si solo coma, asumimos decimal coma (1234,56)
    has_comma = s.str.contains(",", na=False)
    has_dot = s.str.contains(r"\.", na=False)

    both = has_comma & has_dot
    only_comma = has_comma & ~has_dot

    s.loc[both] = s.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def _read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        # intenta separadores típicos
        raw = file.getvalue()
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
        # último intento
        return pd.read_csv(io.BytesIO(raw), engine="python")
    else:
        # xlsx
        return pd.read_excel(file, engine="openpyxl")


def _guess_dataset_type(df: pd.DataFrame) -> str:
    cols = [_norm(c) for c in df.columns]

    # Señales de "posiciones"
    holdings_signals = {"isin", "valor de mercado", "market value", "quantity", "cantidad", "average price", "precio medio",
                        "current price", "precio actual", "valor", "instrument", "producto", "nombre"}

    # Señales de "transacciones"
    tx_signals = {"date", "fecha", "tipo", "type", "buy", "sell", "importe", "amount", "fees", "comisión", "dividend"}

    h = sum(any(sig in c for c in cols) for sig in holdings_signals)
    t = sum(any(sig in c for c in cols) for sig in tx_signals)

    return "Posiciones (cartera/holdings)" if h >= t else "Operaciones (transactions)"


def _auto_map_positions(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Devuelve un mapeo sugerido para columnas de posiciones.
    keys:
      - name, isin, quantity, avg_price, cost, current_price, market_value, currency
    """
    cols = list(df.columns)
    ncols = [_norm(c) for c in cols]

    def pick(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for i, nc in enumerate(ncols):
                if cand in nc:
                    return cols[i]
        return None

    return {
        "name": pick(["name", "nombre", "instrument", "producto", "asset", "valor"]),
        "isin": pick(["isin"]),
        "quantity": pick(["quantity", "cantidad", "shares", "unidades", "units", "stücke"]),
        "avg_price": pick(["average price", "precio medio", "avg price", "purchase price", "precio compra", "avg cost"]),
        "cost": pick(["cost", "importe invertido", "invested", "valor compra", "acquisition value", "cost basis", "costs"]),
        "current_price": pick(["current price", "precio actual", "price", "cotización"]),
        "market_value": pick(["market value", "valor de mercado", "current value", "valor actual", "valor", "valuation"]),
        "currency": pick(["currency", "divisa", "ccy", "währung"]),
    }


def _auto_map_transactions(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    keys:
      - date, type, amount, fees, name, isin, currency
    """
    cols = list(df.columns)
    ncols = [_norm(c) for c in cols]

    def pick(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for i, nc in enumerate(ncols):
                if cand in nc:
                    return cols[i]
        return None

    return {
        "date": pick(["date", "fecha", "datum"]),
        "type": pick(["type", "tipo", "transaction", "acción", "vorgang"]),
        "amount": pick(["amount", "importe", "total", "betrag", "cash"]),
        "fees": pick(["fee", "fees", "comisión", "commission", "kosten"]),
        "name": pick(["name", "nombre", "instrument", "producto", "asset", "valor"]),
        "isin": pick(["isin"]),
        "currency": pick(["currency", "divisa", "ccy", "währung"]),
    }


def _choose_col(label: str, cols: List[str], default: Optional[str]) -> Optional[str]:
    opts = ["(ninguna)"] + cols
    idx = 0
    if default in cols:
        idx = cols.index(default) + 1
    chosen = st.selectbox(label, opts, index=idx)
    return None if chosen == "(ninguna)" else chosen


def _xirr(dates: np.ndarray, cashflows: np.ndarray, guess: float = 0.10) -> Optional[float]:
    """
    XIRR por Newton-Raphson. Devuelve tasa anual.
    Requiere al menos un flujo negativo y uno positivo.
    """
    if len(cashflows) < 2:
        return None
    if not (np.any(cashflows < 0) and np.any(cashflows > 0)):
        return None

    d0 = dates.min()
    years = (dates - d0) / np.timedelta64(365, "D")

    def npv(r):
        return np.sum(cashflows / np.power(1 + r, years))

    def d_npv(r):
        return np.sum(-years * cashflows / np.power(1 + r, years + 1))

    r = guess
    for _ in range(100):
        f = npv(r)
        df = d_npv(r)
        if df == 0:
            break
        nr = r - f / df
        if not np.isfinite(nr):
            break
        if abs(nr - r) < 1e-10:
            r = nr
            break
        r = nr

    if np.isfinite(r) and r > -0.9999:
        return float(r)
    return None


# =========================
# SIDEBAR: UPLOAD
# =========================
with st.sidebar:
    st.header("1) Subir archivo")
    file = st.file_uploader("Export CSV/XLSX de Trade Republic", type=["csv", "xlsx"])

    st.divider()
    st.header("2) Preferencias")
    top_n = st.slider("Top N posiciones a mostrar", 5, 30, 12)
    show_raw = st.checkbox("Mostrar tabla original (raw)", value=False)
    currency_hint = st.text_input("Divisa principal (opcional, ej. EUR)", value="")

if not file:
    st.info("⬅️ Sube un CSV/XLSX exportado de Trade Republic para empezar.")
    st.stop()

df_raw = _read_any(file)
df = df_raw.copy()

if show_raw:
    st.subheader("Archivo cargado (raw)")
    st.dataframe(df_raw, use_container_width=True)

dataset_type_guess = _guess_dataset_type(df)

st.subheader("Selecciona qué contiene tu archivo")
dataset_type = st.radio(
    "Tipo de datos",
    ["Posiciones (cartera/holdings)", "Operaciones (transactions)"],
    index=0 if "Posiciones" in dataset_type_guess else 1,
    horizontal=True,
)

st.divider()


# =========================
# POSITIONS FLOW
# =========================
if dataset_type == "Posiciones (cartera/holdings)":
    st.header("✅ Análisis de posiciones")

    cols = list(df.columns)
    suggested = _auto_map_positions(df)

    with st.expander("Mapeo de columnas (si no coincide, cámbialo aquí)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            col_name = _choose_col("Nombre/Activo", cols, suggested["name"])
            col_isin = _choose_col("ISIN", cols, suggested["isin"])
            col_qty = _choose_col("Cantidad", cols, suggested["quantity"])
        with c2:
            col_avg = _choose_col("Precio medio", cols, suggested["avg_price"])
            col_cost = _choose_col("Coste total / invertido", cols, suggested["cost"])
            col_ccy = _choose_col("Divisa", cols, suggested["currency"])
        with c3:
            col_px = _choose_col("Precio actual", cols, suggested["current_price"])
            col_mv = _choose_col("Valor de mercado / actual", cols, suggested["market_value"])

    if col_name is None:
        st.error("Necesito al menos la columna de Nombre/Activo para continuar.")
        st.stop()

    work = pd.DataFrame()
    work["name"] = df[col_name].astype(str).str.strip()
    if col_isin:
        work["isin"] = df[col_isin].astype(str).str.strip()
    else:
        work["isin"] = ""

    if col_ccy:
        work["currency"] = df[col_ccy].astype(str).str.strip().str.upper()
    else:
        work["currency"] = (currency_hint.strip().upper() if currency_hint.strip() else "N/A")

    # numéricas
    work["quantity"] = _to_numeric_series(df[col_qty]) if col_qty else np.nan
    work["avg_price"] = _to_numeric_series(df[col_avg]) if col_avg else np.nan
    work["cost"] = _to_numeric_series(df[col_cost]) if col_cost else np.nan
    work["current_price"] = _to_numeric_series(df[col_px]) if col_px else np.nan
    work["market_value"] = _to_numeric_series(df[col_mv]) if col_mv else np.nan

    # Derivaciones
    # market_value: si no está, qty * current_price
    if work["market_value"].isna().all():
        work["market_value"] = work["quantity"] * work["current_price"]

    # cost: si no está, qty * avg_price
    if work["cost"].isna().all():
        work["cost"] = work["quantity"] * work["avg_price"]

    # Limpieza básica
    work = work.replace([np.inf, -np.inf], np.nan)
    work["market_value"] = work["market_value"].fillna(0.0)
    work["cost"] = work["cost"].fillna(0.0)

    total_mv = float(work["market_value"].sum())
    total_cost = float(work["cost"].sum())
    pnl = total_mv - total_cost
    pnl_pct = (pnl / total_cost) if total_cost != 0 else np.nan

    work["pnl"] = work["market_value"] - work["cost"]
    work["pnl_pct"] = np.where(work["cost"] != 0, work["pnl"] / work["cost"], np.nan)
    work["weight"] = np.where(total_mv != 0, work["market_value"] / total_mv, 0.0)

    # Concentración
    weights_sorted = np.sort(work["weight"].values)[::-1]
    top5 = float(np.sum(weights_sorted[:5])) if len(weights_sorted) >= 5 else float(np.sum(weights_sorted))
    top10 = float(np.sum(weights_sorted[:10])) if len(weights_sorted) >= 10 else float(np.sum(weights_sorted))
    hhi = float(np.sum(np.square(work["weight"].values)))  # 0..1

    # ====== METRICS
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Valor cartera", f"{total_mv:,.2f}")
    m2.metric("Coste total", f"{total_cost:,.2f}")
    m3.metric("P&L total", f"{pnl:,.2f}", delta=None)
    m4.metric("P&L %", f"{(pnl_pct*100):,.2f}%" if np.isfinite(pnl_pct) else "N/A")
    m5.metric("Concentración Top 5", f"{top5*100:,.1f}%")

    st.caption(f"Top 10: {top10*100:,.1f}% · HHI (0–1): {hhi:,.3f}  (más alto = más concentrado)")

    # ====== TABLE + DOWNLOAD
    st.subheader("Posiciones (limpias)")
    show_cols = ["name", "isin", "currency", "quantity", "avg_price", "current_price", "cost", "market_value", "pnl", "pnl_pct", "weight"]
    work_show = work.copy()
    work_show = work_show.sort_values("market_value", ascending=False)

    st.dataframe(
        work_show[show_cols],
        use_container_width=True,
        hide_index=True
    )

    csv_out = work_show[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar posiciones limpias (CSV)", data=csv_out, file_name="positions_clean.csv", mime="text/csv")

    st.divider()

    # ====== CHARTS
    st.subheader("Visualizaciones")
    c1, c2 = st.columns(2)

    top = work_show.head(top_n).copy()

    with c1:
        fig_pie = px.pie(top, names="name", values="market_value", title=f"Pesos · Top {top_n} por valor")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        top_pnl = work_show.sort_values("pnl", ascending=True).tail(top_n)
        fig_bar = px.bar(top_pnl, x="pnl", y="name", orientation="h", title=f"P&L absoluto · Top {top_n}")
        st.plotly_chart(fig_bar, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_treemap = px.treemap(
            top,
            path=["currency", "name"],
            values="market_value",
            title=f"Treemap · Top {top_n} (por divisa)"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with c4:
        fig_scatter = px.scatter(
            work_show,
            x="weight",
            y="pnl_pct",
            hover_name="name",
            size="market_value",
            title="Peso vs P&L % (tamaño = valor)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.success("Listo. Si tus columnas no encajan, ajusta el mapeo arriba y se recalcula al instante.")


# =========================
# TRANSACTIONS FLOW
# =========================
else:
    st.header("✅ Análisis de operaciones (transactions) + XIRR opcional")

    cols = list(df.columns)
    suggested = _auto_map_transactions(df)

    with st.expander("Mapeo de columnas (si no coincide, cámbialo aquí)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            col_date = _choose_col("Fecha", cols, suggested["date"])
            col_type = _choose_col("Tipo (buy/sell/dividend...)", cols, suggested["type"])
        with c2:
            col_amount = _choose_col("Importe (cashflow)", cols, suggested["amount"])
            col_fees = _choose_col("Comisiones (opcional)", cols, suggested["fees"])
        with c3:
            col_name = _choose_col("Nombre/Activo (opcional)", cols, suggested["name"])
            col_isin = _choose_col("ISIN (opcional)", cols, suggested["isin"])

    if col_date is None or col_type is None or col_amount is None:
        st.error("Necesito al menos Fecha + Tipo + Importe para continuar.")
        st.stop()

    tx = pd.DataFrame()
    tx["date_raw"] = df[col_date]
    tx["type_raw"] = df[col_type].astype(str)

    # Parse date
    tx["date"] = pd.to_datetime(tx["date_raw"], errors="coerce", dayfirst=True)
    tx["amount"] = _to_numeric_series(df[col_amount])

    tx["fees"] = _to_numeric_series(df[col_fees]) if col_fees else 0.0
    tx["fees"] = tx["fees"].fillna(0.0)

    tx["name"] = df[col_name].astype(str).str.strip() if col_name else ""
    tx["isin"] = df[col_isin].astype(str).str.strip() if col_isin else ""

    tx = tx.dropna(subset=["date", "amount"]).copy()
    if tx.empty:
        st.error("No he podido leer fechas/importes válidos. Revisa el mapeo y el formato del archivo.")
        st.stop()

    # Clasificación simple de flujos: compras negativas, ventas/dividendos positivos (heurística)
    tnorm = tx["type_raw"].str.lower()

    is_buy = tnorm.str.contains(r"\bbuy\b|compra|kauf|purchase", regex=True)
    is_sell = tnorm.str.contains(r"\bsell\b|venta|verkauf", regex=True)
    is_div = tnorm.str.contains(r"dividend|dividendo|zins|interest", regex=True)

    # Si en el CSV las compras ya vienen con signo negativo, no lo cambiamos.
    # Si vienen siempre positivas, aplicamos signo por tipo.
    already_signed = (tx["amount"].min() < 0) and (tx["amount"].max() > 0)

    cashflow = tx["amount"].copy()
    if not already_signed:
        cashflow = np.where(is_buy, -np.abs(tx["amount"]), np.abs(tx["amount"]))
        cashflow = np.where(is_sell | is_div, np.abs(tx["amount"]), cashflow)

    # comisiones: siempre restan
    cashflow = cashflow - np.abs(tx["fees"].values)

    tx["cashflow"] = cashflow

    st.subheader("Operaciones (limpias)")
    st.dataframe(
        tx.sort_values("date", ascending=False)[["date", "type_raw", "name", "isin", "amount", "fees", "cashflow"]],
        use_container_width=True,
        hide_index=True
    )

    # Resumen por tipo
    st.divider()
    st.subheader("Resumen de flujos")
    total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum())
    total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum())
    net = float(tx["cashflow"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Entradas (+)", f"{total_in:,.2f}")
    m2.metric("Salidas (-)", f"{total_out:,.2f}")
    m3.metric("Netto", f"{net:,.2f}")

    # Serie temporal
    tx_ts = tx.sort_values("date").copy()
    tx_ts["cum_net"] = tx_ts["cashflow"].cumsum()

    fig_ts = px.line(tx_ts, x="date", y="cum_net", title="Acumulado de cashflows (neto)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # XIRR
    st.divider()
    st.subheader("XIRR (rentabilidad anualizada por flujos) — opcional")
    st.caption("Para XIRR “real”, necesitas que el fichero incluya también el valor final (o una venta/cierre). "
               "Si solo hay compras, XIRR no se puede calcular.")

    xirr_guess = st.slider("Guess inicial (%)", 0, 50, 10) / 100.0

    dates_np = tx["date"].values.astype("datetime64[ns]")
    cf_np = tx["cashflow"].values.astype(float)

    rate = _xirr(dates_np, cf_np, guess=xirr_guess)
    if rate is None:
        st.warning("No se puede calcular XIRR con estos flujos (falta al menos un flujo positivo y uno negativo, o hay datos insuficientes).")
    else:
        st.success(f"XIRR estimada: {rate*100:,.2f}% anual")

    # Download
    csv_out = tx.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar operaciones limpias (CSV)", data=csv_out, file_name="transactions_clean.csv", mime="text/csv")

    st.info("Si quieres, puedo añadir un modo mixto: posiciones + operaciones, para calcular rendimiento más completo y gráficos más avanzados.")
