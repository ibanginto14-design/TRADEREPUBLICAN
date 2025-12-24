import io
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber

import plotly.express as px
import plotly.graph_objects as go


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Trade Republic · PDF Analyzer", page_icon="📄", layout="wide")
st.title("📄 Trade Republic · PDF Analyzer (Extracto de cuenta)")
st.caption(
    "Sube un PDF de Trade Republic (extracto de cuenta). "
    "Parseo robusto de 'TRANSACCIONES DE CUENTA' + dashboard pro con gráficas."
)

# =========================
# CONSTANTS / HELPERS
# =========================
MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12,
}

DROP_PATTERNS = [
    r"^TRADE REPUBLIC BANK",
    r"^Trade Republic Bank",
    r"^www\.traderepublic",
    r"^Página \d+ de \d+",
    r"^RESUMEN DE ESTADO DE CUENTA",
    r"^TRANSACCIONES DE CUENTA$",
    r"^FECHA\s+TIPO\s+DESCRIPCIÓN",
    r"\bENTRADA\b",
    r"\bSALIDA\b",
    r"\bBALANCE\b",
]

END_MARKERS = ("RESUMEN DEL BALANCE", "NOTAS SOBRE")


def _to_float_eu(s: str) -> Optional[float]:
    """Convierte números tipo 1.001,00 o 100,00 o -4,18 a float."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"[^\d,\.\-\+]", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _extract_text_all_pages(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join([(p.extract_text() or "") for p in pdf.pages])


def _slice_transaction_section(lines: List[str]) -> List[str]:
    """Coge la parte entre 'TRANSACCIONES DE CUENTA' y 'RESUMEN...' y limpia headers/footers."""
    up = [l.upper() for l in lines]

    start = 0
    for i, l in enumerate(up):
        if "TRANSACCIONES DE CUENTA" in l:
            start = i
            break

    end = len(lines)
    for i, l in enumerate(up):
        if l.startswith(END_MARKERS):
            end = i
            break

    sub = lines[start:end]
    cleaned = []
    for l in sub:
        l = l.strip()
        if not l:
            continue
        if any(re.search(p, l) for p in DROP_PATTERNS):
            continue
        cleaned.append(l)
    return cleaned


def _date_prefix(line: str) -> Optional[Tuple[int, str, str]]:
    """
    Detecta líneas que empiezan por '10 may' o '18 dic Transacción'.
    Devuelve (day, mon_str, rest).
    """
    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    day = int(m.group(1))
    mon = m.group(2)
    rest = (m.group(3) or "").strip()
    return day, mon, rest


def _year_prefix(line: str) -> Optional[Tuple[int, str]]:
    """Detecta líneas tipo '2025 con tarjeta'. Devuelve (year, rest)."""
    m = re.match(r"^\s*(\d{4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    year = int(m.group(1))
    rest = (m.group(2) or "").strip()
    return year, rest


def _infer_type(desc: str) -> str:
    candidates = [
        "Transacción con tarjeta",
        "Transferencia",
        "Operar",
        "Comisión",
        "Rentabilidad",
        "Interés",
        "Interest",
        "Transacción",
    ]
    low = (desc or "").lower()
    for c in candidates:
        if low.startswith(c.lower()):
            return c
    return desc.split(" ", 1)[0] if desc else "Unknown"


def _infer_side_and_cashflow(tx_type: str, desc: str, amount: Optional[float]) -> Tuple[str, Optional[float]]:
    """
    Devuelve (side, cashflow):
    - side: BUY/SELL/NA (solo para Operar)
    - cashflow: signo inferido (entrada + / salida -)
    """
    if amount is None or not np.isfinite(amount):
        return "NA", None
    if amount < 0:
        return "NA", float(amount)

    t = (tx_type or "").lower()
    d = (desc or "").lower()

    if "operar" in t:
        is_sell = bool(re.search(r"\bsell\b|venta|ejecución venta", d))
        side = "SELL" if is_sell else "BUY"
        return side, float(+amount if is_sell else -amount)

    if ("rentabilidad" in t) or ("interés" in t) or ("interest" in t):
        return "NA", float(+amount)

    if "comisión" in t or "comision" in t:
        return "NA", float(-amount)

    if "transacción con tarjeta" in t or (("transacción" in t) and ("tarjeta" in d)):
        return "NA", float(-amount)

    if "transferencia" in t:
        if any(k in d for k in ["top up", "incoming", "ingreso", "accepted"]):
            return "NA", float(+amount)
        if any(k in d for k in ["payout", "outgoing", "retirada"]):
            return "NA", float(-amount)
        return "NA", float(+amount)

    return "NA", float(+amount)


def _extract_isin(desc: str) -> str:
    m = re.search(r"\b[A-Z]{2}[A-Z0-9]{10}\b", desc or "")
    return m.group(0) if m else ""


def _extract_quantity(desc: str) -> Optional[float]:
    m = re.search(r"quantity:\s*([0-9\.,]+)", desc or "", flags=re.IGNORECASE)
    if not m:
        return None
    q = m.group(1).replace(",", ".")
    try:
        return float(q)
    except Exception:
        return None


def _extract_asset_name(desc: str, isin: str) -> str:
    if not desc or not isin or isin not in desc:
        return ""
    after = desc.split(isin, 1)[1].strip()
    if re.match(r"^[-+]?\d", after):
        return ""
    name = re.split(r",\s*quantity:|\s+[-+]?\d{1,3}(?:\.\d{3})*(?:,\d{2})\s*€", after)[0].strip()
    return name.strip(", ")


def parse_tr_pdf_transactions(pdf_bytes: bytes) -> pd.DataFrame:
    text = _extract_text_all_pages(pdf_bytes)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = _slice_transaction_section(lines)

    recs = []
    i = 0
    while i < len(lines):
        dp = _date_prefix(lines[i])
        if not dp:
            i += 1
            continue

        day, mon_str, rest = dp
        i += 1

        year = None
        chunks: List[str] = []
        if rest:
            chunks.append(rest)

        while i < len(lines) and not _date_prefix(lines[i]):
            yp = _year_prefix(lines[i])
            if yp:
                y, yrest = yp
                year = y
                if yrest:
                    chunks.append(yrest)
            else:
                chunks.append(lines[i].strip())
            i += 1

        mon_num = MONTHS.get(mon_str.lower())
        date = pd.NaT
        if year and mon_num:
            date = pd.Timestamp(year, mon_num, day)

        desc = " ".join([c for c in chunks if c]).strip()

        amts = re.findall(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d{2})", desc)
        amount = _to_float_eu(amts[-2]) if len(amts) >= 2 else (_to_float_eu(amts[-1]) if len(amts) == 1 else None)
        balance = _to_float_eu(amts[-1]) if len(amts) >= 1 else None

        tx_type = _infer_type(desc)
        side, cashflow = _infer_side_and_cashflow(tx_type, desc, amount)

        isin = _extract_isin(desc)
        qty = _extract_quantity(desc)
        asset = _extract_asset_name(desc, isin)

        recs.append(
            {
                "date": date,
                "type": tx_type,
                "desc": desc,
                "isin": isin,
                "asset": asset,
                "quantity": qty,
                "side": side,
                "amount": amount,
                "cashflow": cashflow,
                "balance": balance,
            }
        )

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.dropna(subset=["date"], how="all").sort_values("date").reset_index(drop=True)
    return df


def _category(row_type: str, desc: str) -> str:
    t = (row_type or "").lower()
    d = (desc or "").lower()

    if "operar" in t:
        return "Trading"
    if "transacción con tarjeta" in t or ("tarjeta" in d and "transacción" in t):
        return "Tarjeta"
    if "comisión" in t or "comision" in t:
        return "Comisiones"
    if "rentabilidad" in t or "interés" in t or "interest" in t:
        return "Intereses/Rentabilidad"
    if "transferencia" in t:
        if any(k in d for k in ["top up", "incoming", "ingreso", "accepted"]):
            return "Aportaciones"
        if any(k in d for k in ["payout", "outgoing", "retirada"]):
            return "Retiradas"
        return "Transferencias"
    return "Otros"


def _xirr(dates: np.ndarray, cashflows: np.ndarray, guess: float = 0.10) -> Optional[float]:
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


def compute_asset_pnl_avg_cost(tx: pd.DataFrame) -> pd.DataFrame:
    op = tx[tx["type"].str.lower().eq("operar")].copy()
    op = op[op["isin"].astype(str).str.len() > 0].copy()
    if op.empty:
        return pd.DataFrame()

    op["quantity"] = pd.to_numeric(op["quantity"], errors="coerce")
    op["amount"] = pd.to_numeric(op["amount"], errors="coerce")
    op = op.dropna(subset=["date", "quantity", "amount"]).sort_values("date").copy()

    rows = []
    for isin, g in op.groupby("isin"):
        pos_qty = 0.0
        avg_cost = 0.0
        realized = 0.0

        buy_qty = buy_amt = 0.0
        sell_qty = sell_amt = 0.0
        n_trades = 0
        asset_name = ""

        for _, r in g.iterrows():
            n_trades += 1
            qty = float(r["quantity"])
            amt = float(r["amount"])
            side = (r.get("side", "NA") or "NA").upper()

            if not asset_name:
                cand = str(r.get("asset", "") or "").strip()
                if cand:
                    asset_name = cand

            if side == "BUY":
                total_cost_before = pos_qty * avg_cost
                total_cost_after = total_cost_before + amt
                pos_qty += qty
                avg_cost = (total_cost_after / pos_qty) if pos_qty > 0 else 0.0
                buy_qty += qty
                buy_amt += amt

            elif side == "SELL":
                proceeds = amt
                cost_basis = qty * avg_cost
                realized += (proceeds - cost_basis)

                pos_qty -= qty
                if pos_qty <= 1e-12:
                    pos_qty = 0.0
                    avg_cost = 0.0

                sell_qty += qty
                sell_amt += amt

            else:
                desc = str(r.get("desc", "")).lower()
                is_sell = bool(re.search(r"\bsell\b|venta|ejecución venta", desc))
                if is_sell:
                    proceeds = amt
                    cost_basis = qty * avg_cost
                    realized += (proceeds - cost_basis)
                    pos_qty -= qty
                    if pos_qty <= 1e-12:
                        pos_qty = 0.0
                        avg_cost = 0.0
                    sell_qty += qty
                    sell_amt += amt
                else:
                    total_cost_before = pos_qty * avg_cost
                    total_cost_after = total_cost_before + amt
                    pos_qty += qty
                    avg_cost = (total_cost_after / pos_qty) if pos_qty > 0 else 0.0
                    buy_qty += qty
                    buy_amt += amt

        net_invested = buy_amt - sell_amt
        # ratio buy (para color en bubble)
        denom = (buy_amt + sell_amt)
        buy_ratio = (buy_amt / denom) if denom > 0 else np.nan

        rows.append({
            "isin": isin,
            "asset": asset_name,
            "trades": n_trades,
            "buy_qty": buy_qty,
            "buy_amount": buy_amt,
            "sell_qty": sell_qty,
            "sell_amount": sell_amt,
            "net_qty": pos_qty,
            "avg_cost": avg_cost,
            "realized_pnl": realized,
            "net_invested": net_invested,
            "buy_ratio": buy_ratio,
            "first_trade": g["date"].min(),
            "last_trade": g["date"].max(),
        })

    out = pd.DataFrame(rows).sort_values("net_invested", ascending=False).reset_index(drop=True)
    return out


# =========================
# VISUAL HELPERS (PLOTLY)
# =========================
def plot_waterfall_net_by_category(tx: pd.DataFrame):
    by_cat = tx.groupby("category")["cashflow"].sum().sort_values()
    cats = by_cat.index.tolist()
    vals = by_cat.values.tolist()
    net = float(np.nansum(vals))

    fig = go.Figure(go.Waterfall(
        name="Neto",
        orientation="v",
        measure=["relative"] * len(cats) + ["total"],
        x=cats + ["NETO"],
        y=vals + [net],
        connector={"line": {"width": 1}},
    ))
    fig.update_layout(title="Waterfall · ¿De dónde sale mi neto?", showlegend=False, height=420)
    return fig


def plot_sankey_flows(tx: pd.DataFrame):
    """
    Sankey simple: ENTRADAS -> categorías positivas ; categorías negativas -> SALIDAS
    (Usa netos por categoría; es súper explicativo aunque no sea doble entrada perfecta)
    """
    by_cat = tx.groupby("category")["cashflow"].sum().sort_values()
    pos = by_cat[by_cat > 0]
    neg = by_cat[by_cat < 0]

    nodes = ["ENTRADAS"] + pos.index.tolist() + neg.index.tolist() + ["SALIDAS"]
    node_index = {n: i for i, n in enumerate(nodes)}

    sources = []
    targets = []
    values = []

    # entradas a categorías positivas
    for c, v in pos.items():
        sources.append(node_index["ENTRADAS"])
        targets.append(node_index[c])
        values.append(float(v))

    # categorías negativas a salidas (valor absoluto)
    for c, v in neg.items():
        sources.append(node_index[c])
        targets.append(node_index["SALIDAS"])
        values.append(float(abs(v)))

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=14),
        link=dict(source=sources, target=targets, value=values),
    )])
    fig.update_layout(title="Sankey · Flujo neto (entradas vs salidas por categoría)", height=440)
    return fig


def plot_calendar_heatmap(tx: pd.DataFrame, value_col: str = "cashflow", mode: str = "sum"):
    """
    Heatmap tipo GitHub.
    mode: 'sum' (€/día) o 'count' (#transacciones)
    """
    df = tx.dropna(subset=["date"]).copy()
    df["day"] = df["date"].dt.date
    if mode == "count":
        daily = df.groupby("day").size().rename("value").reset_index()
    else:
        daily = df.groupby("day")[value_col].sum().rename("value").reset_index()

    daily["day"] = pd.to_datetime(daily["day"])
    iso = daily["day"].dt.isocalendar()
    daily["iso_year"] = iso["year"].astype(int)
    daily["iso_week"] = iso["week"].astype(int)
    daily["dow"] = daily["day"].dt.weekday  # 0 Mon..6 Sun
    dow_map = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    daily["dow_name"] = daily["dow"].map(dow_map)

    daily["week_label"] = daily["iso_year"].astype(str) + "-W" + daily["iso_week"].astype(str).str.zfill(2)
    pivot = daily.pivot_table(index="dow_name", columns="week_label", values="value", aggfunc="sum").fillna(0.0)

    # Orden días
    pivot = pivot.reindex(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=("Heatmap calendario · " + ("# transacciones" if mode == "count" else "€ netos/día")),
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_donuts(tx: pd.DataFrame):
    """
    Donuts de distribución de entradas y salidas por categoría
    """
    df = tx.copy()
    inflow = df[df["cashflow"] > 0].groupby("category")["cashflow"].sum().reset_index()
    outflow = df[df["cashflow"] < 0].groupby("category")["cashflow"].sum().abs().reset_index()

    fig1 = px.pie(inflow, names="category", values="cashflow", hole=0.55, title="Donut · Entradas por categoría")
    fig2 = px.pie(outflow, names="category", values="cashflow", hole=0.55, title="Donut · Salidas por categoría")
    fig1.update_layout(height=380)
    fig2.update_layout(height=380)
    return fig1, fig2


def plot_histograms(tx: pd.DataFrame):
    df = tx.dropna(subset=["cashflow"]).copy()
    df["abs_cashflow"] = df["cashflow"].abs()

    fig = px.histogram(
        df,
        x="abs_cashflow",
        color="category",
        nbins=35,
        title="Distribución de importes (|cashflow|) por categoría",
    )
    fig.update_layout(height=420)
    return fig


def plot_lollipop_top_days(tx: pd.DataFrame, kind: str = "out", top_n: int = 10):
    """
    kind='out' -> top días con mayor salida (más negativo)
    kind='in' -> top días con mayor entrada (más positivo)
    """
    df = tx.dropna(subset=["date", "cashflow"]).copy()
    df["day"] = df["date"].dt.date
    daily = df.groupby("day")["cashflow"].sum().reset_index()
    daily["day"] = pd.to_datetime(daily["day"])

    if kind == "out":
        sel = daily.nsmallest(top_n, "cashflow").copy()
        title = f"Top {top_n} días con mayor SALIDA"
    else:
        sel = daily.nlargest(top_n, "cashflow").copy()
        title = f"Top {top_n} días con mayor ENTRADA"

    sel = sel.sort_values("cashflow")
    x = sel["cashflow"].values
    y = sel["day"].dt.strftime("%Y-%m-%d").values

    fig = go.Figure()
    for xi, yi in zip(x, y):
        fig.add_trace(go.Scatter(x=[0, xi], y=[yi, yi], mode="lines", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="día"))

    fig.update_layout(title=title, height=420, xaxis_title="€ netos del día", yaxis_title="Fecha")
    return fig


def plot_double_line_balance_vs_cum(tx: pd.DataFrame):
    df = tx.dropna(subset=["date"]).sort_values("date").copy()
    df["cum_net"] = df["cashflow"].fillna(0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["cum_net"], mode="lines", name="Cashflow acumulado"))
    if df["balance"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["balance"], mode="lines", name="Balance (PDF)"))
    fig.update_layout(title="Balance (PDF) vs Cashflow acumulado", height=420)
    return fig


def plot_monthly_cohort(tx: pd.DataFrame):
    df = tx.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    g = df.groupby(["month", "category"])["cashflow"].sum().reset_index()

    # % de composición por mes en términos de |cashflow|
    g["abs"] = g["cashflow"].abs()
    tot = g.groupby("month")["abs"].transform("sum").replace(0, np.nan)
    g["share"] = (g["abs"] / tot) * 100.0

    pivot = g.pivot_table(index="category", columns="month", values="share", aggfunc="sum").fillna(0.0)
    fig = px.imshow(pivot, aspect="auto", title="Cohort mensual · % composición por categoría (|cashflow|)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig, pivot


def plot_bubble_assets(assets: pd.DataFrame):
    df = assets.copy()
    df["label"] = np.where(df["asset"].astype(str).str.len() > 0, df["asset"], df["isin"])
    fig = px.scatter(
        df,
        x="net_invested",
        y="realized_pnl",
        size="trades",
        color="buy_ratio",
        hover_name="label",
        hover_data=["isin", "asset", "net_qty", "avg_cost", "buy_amount", "sell_amount"],
        title="Bubble · Activos: neto invertido vs P&L realizado (tamaño = nº trades)",
    )
    fig.update_layout(height=520)
    return fig


def plot_drawdown_cum(tx: pd.DataFrame):
    df = tx.dropna(subset=["date"]).sort_values("date").copy()
    df["cum_net"] = df["cashflow"].fillna(0).cumsum()
    df["peak"] = df["cum_net"].cummax()
    df["drawdown"] = df["cum_net"] - df["peak"]

    fig1 = px.line(df, x="date", y="cum_net", title="Cashflow acumulado")
    fig1.update_layout(height=360)

    fig2 = px.line(df, x="date", y="drawdown", title="Drawdown del cashflow acumulado (desde máximos)")
    fig2.update_layout(height=360)
    return fig1, fig2


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Subir PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])
    st.divider()
    top_n = st.slider("Top N", 5, 30, 12)
    show_full_tx = st.checkbox("Mostrar transacciones completas", value=False)
    show_debug = st.checkbox("Modo diagnóstico", value=False)
    xirr_guess_pct = st.slider("XIRR guess inicial (%)", 0, 50, 10)

if not up:
    st.info("⬅️ Sube tu PDF de Trade Republic para empezar.")
    st.stop()

pdf_bytes = up.getvalue()

# =========================
# PARSE (SAFE)
# =========================
try:
    tx = parse_tr_pdf_transactions(pdf_bytes)
except Exception as e:
    st.error("Error parseando el PDF. No crasheo la app: revisa que sea un extracto de cuenta.")
    st.exception(e)
    st.stop()

if tx.empty:
    st.error(
        "No he encontrado transacciones parseables. "
        "Suele pasar si el PDF no incluye 'TRANSACCIONES DE CUENTA' o si es un formato distinto."
    )
    st.stop()

# Features base
tx = tx.copy()
tx["category"] = [_category(t, d) for t, d in zip(tx["type"].astype(str), tx["desc"].astype(str))]
tx["cashflow"] = pd.to_numeric(tx["cashflow"], errors="coerce")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
tx["balance"] = pd.to_numeric(tx["balance"], errors="coerce")

# =========================
# PRECOMPUTE
# =========================
ts = tx.dropna(subset=["date"]).sort_values("date").copy()
ts["cum_net"] = ts["cashflow"].fillna(0).cumsum()

last_balance = tx["balance"].dropna()
last_balance_val = float(last_balance.iloc[-1]) if not last_balance.empty else float("nan")
last_date = tx["date"].dropna()
last_date_val = last_date.iloc[-1] if not last_date.empty else pd.NaT

assets = compute_asset_pnl_avg_cost(tx)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Resumen", "🧾 Transacciones", "📦 Activos (P&L)", "📅 Mensual", "📊 Gráficas Pro"]
)

with tab1:
    st.subheader("Resumen de cuenta (a partir del PDF)")
    total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum(skipna=True))
    total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum(skipna=True))
    net = float(tx["cashflow"].sum(skipna=True))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entradas (+)", f"{total_in:,.2f} €")
    c2.metric("Salidas (-)", f"{total_out:,.2f} €")
    c3.metric("Neto", f"{net:,.2f} €")
    c4.metric("Balance final (PDF)", f"{last_balance_val:,.2f} €" if np.isfinite(last_balance_val) else "N/A")

    st.subheader("Neto por categoría")
    by_cat = tx.groupby("category")["cashflow"].sum().sort_values()
    fig_cat = px.bar(by_cat.reset_index(), x="cashflow", y="category", orientation="h", title="Neto por categoría")
    fig_cat.update_layout(height=420)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Evolución (cashflow acumulado)")
    fig_ts = px.line(ts, x="date", y="cum_net", title="Cashflow acumulado")
    fig_ts.update_layout(height=380)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("XIRR (cashflows + balance final como valor final)")
    st.caption(
        "Estimación XIRR usando cashflows y añadiendo el balance final como flujo positivo final. "
        "Útil para extractos aunque no tengas precios de mercado por activo."
    )

    if np.isfinite(last_balance_val) and pd.notna(last_date_val):
        dates_np = ts["date"].values.astype("datetime64[ns]")
        cf_np = ts["cashflow"].fillna(0).values.astype(float)

        dates_np2 = np.append(dates_np, np.datetime64(pd.Timestamp(last_date_val)))
        cf_np2 = np.append(cf_np, float(last_balance_val))

        rate = _xirr(dates_np2, cf_np2, guess=xirr_guess_pct / 100.0)
        if rate is None:
            st.warning("No se pudo calcular XIRR con estabilidad (faltan flujos con ambos signos o datos insuficientes).")
        else:
            st.success(f"XIRR estimada: {rate*100:,.2f}% anual")
    else:
        st.warning("No he encontrado balance final fiable en el PDF para calcular XIRR.")

with tab2:
    st.subheader("Transacciones parseadas")
    if show_full_tx:
        st.dataframe(tx, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            tx[["date", "type", "category", "isin", "asset", "quantity", "side", "amount", "cashflow", "balance", "desc"]],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "⬇️ Descargar transacciones (CSV)",
        data=tx.to_csv(index=False).encode("utf-8"),
        file_name="tr_transactions_parsed.csv",
        mime="text/csv",
    )

    if show_debug:
        st.info("Diagnóstico")
        st.write("Filas:", len(tx))
        st.write("Tipos detectados:", sorted(tx["type"].dropna().unique().tolist()))
        st.write("Categorías:", sorted(tx["category"].dropna().unique().tolist()))

with tab3:
    st.subheader("Ganado/Perdido por activo (P&L REALIZADO)")
    if assets.empty:
        st.warning("No hay suficientes operaciones 'Operar' con ISIN para calcular P&L por activo.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Activos con trades", f"{len(assets)}")
        c2.metric("P&L realizado total", f"{assets['realized_pnl'].sum():,.2f} €")
        c3.metric("Dinero neto invertido total", f"{assets['net_invested'].sum():,.2f} €")

        st.caption(
            "• realized_pnl = ganancias/pérdidas ya realizadas por ventas (no incluye lo no realizado). "
            "• net_qty y avg_cost son aproximados a partir de compras/ventas."
        )
        st.dataframe(assets, use_container_width=True, hide_index=True)

        top_pnl = assets.sort_values("realized_pnl", ascending=False).head(top_n)
        fig_top_pnl = px.bar(top_pnl, x="realized_pnl", y="isin", orientation="h", title=f"Top {top_n} · P&L realizado")
        fig_top_pnl.update_layout(height=420)
        st.plotly_chart(fig_top_pnl, use_container_width=True)

        st.download_button(
            "⬇️ Descargar activos (CSV)",
            data=assets.to_csv(index=False).encode("utf-8"),
            file_name="tr_assets_realized_pnl.csv",
            mime="text/csv",
        )

with tab4:
    st.subheader("Resumen mensual (neto por categoría)")
    mdf = tx.dropna(subset=["date"]).copy()
    mdf["month"] = mdf["date"].dt.to_period("M").astype(str)
    monthly = mdf.groupby(["month", "category"])["cashflow"].sum().reset_index()

    pivot = monthly.pivot(index="month", columns="category", values="cashflow").fillna(0.0).sort_index()
    st.dataframe(pivot, use_container_width=True)

    fig_month = px.bar(
        monthly,
        x="month",
        y="cashflow",
        color="category",
        title="Mensual · Neto por categoría",
    )
    fig_month.update_layout(height=460)
    st.plotly_chart(fig_month, use_container_width=True)

    st.download_button(
        "⬇️ Descargar resumen mensual (CSV)",
        data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="tr_monthly_summary.csv",
        mime="text/csv",
    )

with tab5:
    st.subheader("Gráficas Pro (más visual y comprensible)")

    # 1) Waterfall
    st.plotly_chart(plot_waterfall_net_by_category(tx), use_container_width=True)

    # 2) Sankey
    st.plotly_chart(plot_sankey_flows(tx), use_container_width=True)

    # 3) Heatmap calendario (dos modos)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_calendar_heatmap(tx, mode="count"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_calendar_heatmap(tx, mode="sum"), use_container_width=True)

    # 4) Donuts
    d1, d2 = plot_donuts(tx)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(d1, use_container_width=True)
    with c4:
        st.plotly_chart(d2, use_container_width=True)

    # 5) Histograma importes
    st.plotly_chart(plot_histograms(tx), use_container_width=True)

    # 6) Lollipop top días
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_lollipop_top_days(tx, kind="out", top_n=10), use_container_width=True)
    with c6:
        st.plotly_chart(plot_lollipop_top_days(tx, kind="in", top_n=10), use_container_width=True)

    # 7) Línea doble balance vs cum_net
    st.plotly_chart(plot_double_line_balance_vs_cum(tx), use_container_width=True)

    # 8) Cohort mensual (heatmap)
    fig_cohort, cohort_pivot = plot_monthly_cohort(tx)
    st.plotly_chart(fig_cohort, use_container_width=True)

    # 9) Bubble por activo
    if assets.empty:
        st.info("Bubble por activo: no hay suficientes operaciones 'Operar' con ISIN en este PDF.")
    else:
        st.plotly_chart(plot_bubble_assets(assets), use_container_width=True)

    # 10) Drawdown
    fig_cum, fig_dd = plot_drawdown_cum(tx)
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(fig_cum, use_container_width=True)
    with c8:
        st.plotly_chart(fig_dd, use_container_width=True)
