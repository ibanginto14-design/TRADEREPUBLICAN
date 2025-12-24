import io
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber

# Plotly (bonito). Si no está instalado, usamos fallback.
PLOTLY_OK = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False


# =========================
# CONFIG + ESTILO
# =========================
st.set_page_config(page_title="Mi dinero en Trade Republic (PDF)", page_icon="💶", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
h1, h2, h3 { letter-spacing: -0.2px; }

/* Tarjetas KPI */
.kpi-grid { display:grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
@media (max-width: 1200px){ .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.04);
}
.kpi .t { font-size: 12px; opacity: 0.85; margin-bottom: 6px; }
.kpi .v { font-size: 22px; font-weight: 750; }
.kpi .s { font-size: 12px; opacity: 0.72; margin-top: 6px; }

.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 12px;
  margin-right: 8px;
  margin-bottom: 6px;
}

.hr { height: 1px; background: rgba(255,255,255,0.12); margin: 16px 0; border-radius: 999px;}
.small { font-size: 12px; opacity: 0.78; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("💶 Mi dinero en Trade Republic")
st.caption(
    "Sube tu **Extracto de cuenta (PDF)** y te lo traduzco a lenguaje fácil. "
    "Tienes **Modo SIMPLE** y **Modo PRO**: el primero para entender rápido, el segundo para profundizar."
)


# =========================
# PARSER (robusto para PDFs maquetados)
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
    """Detecta líneas que empiezan por '10 may' ... Devuelve (day, mon_str, rest)."""
    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    day = int(m.group(1))
    mon = m.group(2)
    rest = (m.group(3) or "").strip()
    return day, mon, rest


def _year_prefix(line: str) -> Optional[Tuple[int, str]]:
    """Detecta líneas tipo '2025 ...'. Devuelve (year, rest)."""
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


@st.cache_data(show_spinner=False)
def parse_tr_pdf_transactions(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Parser robusto para extractos Trade Republic con líneas partidas.
    Extrae: date, type, desc, amount, balance, isin, asset, quantity, side, cashflow
    """
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


# =========================
# LENGUAJE SIMPLE + UTILIDADES
# =========================
def category_simple(row_type: str, desc: str) -> str:
    t = (row_type or "").lower()
    d = (desc or "").lower()

    if "transacción con tarjeta" in t or ("tarjeta" in d and "transacción" in t):
        return "Gastos con tarjeta"
    if "comisión" in t or "comision" in t:
        return "Comisiones"
    if "rentabilidad" in t or "interés" in t or "interest" in t:
        return "Intereses / rentabilidad"
    if "operar" in t:
        return "Operaciones de inversión"
    if "transferencia" in t:
        if any(k in d for k in ["top up", "incoming", "ingreso", "accepted"]):
            return "Dinero que metiste"
        if any(k in d for k in ["payout", "outgoing", "retirada"]):
            return "Dinero que sacaste"
        return "Transferencias"
    return "Otros"


def fmt_eur(x: float) -> str:
    try:
        return f"{float(x):,.2f} €"
    except Exception:
        return "—"


def short_desc(s: str, n: int = 95) -> str:
    s = str(s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def normalize_desc_for_grouping(s: str) -> str:
    """Para detectar 'repetidos': quita números, ISINs, símbolos…"""
    s = str(s or "").lower()
    s = re.sub(r"\b[A-Z]{2}[A-Z0-9]{10}\b", " ", s)  # isin
    s = re.sub(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d{2})", " ", s)  # importes
    s = re.sub(r"\b\d+\b", " ", s)  # números sueltos
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


# =========================
# ACTIVOS (P&L realizado) + serie temporal
# =========================
def compute_asset_realized_pnl(tx: pd.DataFrame) -> pd.DataFrame:
    op = tx[tx["type"].astype(str).str.lower().eq("operar")].copy()
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
        buy_amt = 0.0
        sell_amt = 0.0
        asset_name = ""

        for _, r in g.iterrows():
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
                buy_amt += amt
            else:
                proceeds = amt
                cost_basis = qty * avg_cost
                realized += (proceeds - cost_basis)
                pos_qty -= qty
                if pos_qty <= 1e-12:
                    pos_qty = 0.0
                    avg_cost = 0.0
                sell_amt += amt

        rows.append(
            {
                "ISIN": isin,
                "Activo": asset_name if asset_name else isin,
                "Dinero metido (compras)": buy_amt,
                "Dinero recuperado (ventas)": sell_amt,
                "Ganado / perdido ya cerrado": realized,
                "Cantidad que te queda (aprox.)": pos_qty,
                "Coste medio (aprox.)": avg_cost,
            }
        )

    out = pd.DataFrame(rows).sort_values("Dinero metido (compras)", ascending=False).reset_index(drop=True)
    return out


def realized_pnl_timeline(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Serie temporal: P&L realizado acumulado de Operar (solo en ventas).
    Aproximación por ISIN usando coste medio (igual que compute_asset_realized_pnl).
    """
    op = tx[tx["type"].astype(str).str.lower().eq("operar")].copy()
    op = op[op["isin"].astype(str).str.len() > 0].copy()
    if op.empty:
        return pd.DataFrame()

    op["quantity"] = pd.to_numeric(op["quantity"], errors="coerce")
    op["amount"] = pd.to_numeric(op["amount"], errors="coerce")
    op = op.dropna(subset=["date", "quantity", "amount"]).sort_values("date").copy()

    # estado por ISIN
    state = {}
    pnl_rows = []

    for _, r in op.iterrows():
        isin = str(r["isin"])
        dt = pd.to_datetime(r["date"])
        qty = float(r["quantity"])
        amt = float(r["amount"])
        side = (r.get("side", "NA") or "NA").upper()

        if isin not in state:
            state[isin] = {"pos": 0.0, "avg": 0.0, "pnl": 0.0}

        pos = state[isin]["pos"]
        avg = state[isin]["avg"]
        pnl = state[isin]["pnl"]

        if side == "BUY":
            total_cost_before = pos * avg
            total_cost_after = total_cost_before + amt
            pos += qty
            avg = (total_cost_after / pos) if pos > 0 else 0.0
        else:
            proceeds = amt
            cost_basis = qty * avg
            pnl += (proceeds - cost_basis)
            pos -= qty
            if pos <= 1e-12:
                pos = 0.0
                avg = 0.0

        state[isin]["pos"] = pos
        state[isin]["avg"] = avg
        state[isin]["pnl"] = pnl

        pnl_rows.append({"date": dt, "isin": isin, "pnl_isin": pnl})

    df = pd.DataFrame(pnl_rows)
    # convertir a pnl total por fecha (tomando último estado de cada ISIN y sumando)
    df = df.sort_values("date")
    # para cada fecha, sumamos el último pnl conocido por ISIN
    last = df.groupby(["date", "isin"])["pnl_isin"].last().reset_index()
    total = last.groupby("date")["pnl_isin"].sum().reset_index()
    total.columns = ["date", "pnl_realizado_total"]
    total["pnl_realizado_total"] = total["pnl_realizado_total"].astype(float)
    return total


# =========================
# GRÁFICOS (SIMPLE + PRO)
# =========================
def fig_in_out_net(total_in: float, total_out: float, net: float):
    if not PLOTLY_OK:
        return None
    df = pd.DataFrame({"Concepto": ["Entradas", "Salidas", "Neto"], "€": [total_in, total_out, net]})
    fig = px.bar(df, x="Concepto", y="€", title="⚖️ Entradas vs Salidas (y el neto)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_out_by_category(by_cat: pd.Series, top_n: int = 12):
    if not PLOTLY_OK:
        return None
    out = by_cat[by_cat < 0].abs().sort_values(ascending=False).head(top_n)
    if out.empty:
        return None
    df = out.reset_index()
    df.columns = ["Concepto", "€"]
    fig = px.bar(df, x="€", y="Concepto", orientation="h", title="🧾 ¿En qué se fue el dinero? (Top categorías)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_balance_or_estimated(txg: pd.DataFrame):
    df = txg.dropna(subset=["date"]).sort_values("date").copy()
    if df.empty:
        return None

    if not PLOTLY_OK:
        return None

    if df["balance"].notna().any():
        d2 = df.dropna(subset=["balance"]).copy()
        fig = px.line(d2, x="date", y="balance", title="📈 Evolución del saldo (balance del PDF)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        return fig

    d2 = df.dropna(subset=["cashflow"]).copy()
    d2["Saldo estimado (desde 0)"] = d2["cashflow"].cumsum()
    fig = px.line(d2, x="date", y="Saldo estimado (desde 0)", title="📈 Evolución estimada (sumando entradas/salidas)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_monthly_net(txg: pd.DataFrame):
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None, None

    df["Mes"] = df["date"].dt.to_period("M").astype(str)
    m = df.groupby("Mes")["cashflow"].sum().reset_index()
    m["Acumulado"] = m["cashflow"].cumsum()

    if not PLOTLY_OK:
        return None, m

    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["Mes"], y=m["cashflow"], name="Neto del mes"))
    fig.add_trace(go.Scatter(x=m["Mes"], y=m["Acumulado"], name="Acumulado", mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        title="📅 Mes a mes: neto y acumulado",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis=dict(title="€ neto del mes"),
        yaxis2=dict(title="€ acumulado", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, m


def fig_timeline_bubbles(txg: pd.DataFrame):
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None
    df["Impacto"] = df["cashflow"].abs()
    p95 = np.nanpercentile(df["Impacto"], 95) if df["Impacto"].notna().any() else 1.0
    df["Impacto_clip"] = np.minimum(df["Impacto"], p95)

    fig = px.scatter(
        df,
        x="date",
        y="cashflow",
        size="Impacto_clip",
        hover_data={"Categoria": True, "desc": True, "cashflow": ":.2f", "date": True, "Impacto_clip": False},
        title="🫧 Línea de tiempo: movimientos (puntos grandes = impactos grandes)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ---- PRO extras ----
def fig_cum_in_out(txg: pd.DataFrame):
    """Cumulativo de entradas y salidas por día."""
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None

    df["day"] = df["date"].dt.date
    daily = df.groupby("day")["cashflow"].sum().reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.sort_values("day")

    daily["in"] = daily["cashflow"].clip(lower=0.0)
    daily["out"] = (-daily["cashflow"].clip(upper=0.0))

    daily["cum_in"] = daily["in"].cumsum()
    daily["cum_out"] = daily["out"].cumsum()
    daily["cum_net"] = daily["cashflow"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["cum_in"], name="Entradas acumuladas", mode="lines"))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["cum_out"], name="Salidas acumuladas", mode="lines"))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["cum_net"], name="Neto acumulado", mode="lines+markers"))
    fig.update_layout(
        title="📈 PRO: Acumulado (entradas vs salidas vs neto)",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis_title="€ acumulado",
    )
    return fig


def fig_stack_monthly_out_by_category(txg: pd.DataFrame, top_n: int = 8):
    """Barras apiladas por mes (solo salidas), para ver qué categoría domina cada mes."""
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date", "cashflow", "Categoria"]).copy()
    if df.empty:
        return None

    out = df[df["cashflow"] < 0].copy()
    if out.empty:
        return None
    out["Mes"] = out["date"].dt.to_period("M").astype(str)
    out["€"] = -out["cashflow"]

    # top categorías por gasto total
    top_cats = out.groupby("Categoria")["€"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    out["Categoria2"] = out["Categoria"].where(out["Categoria"].isin(top_cats), other="Otros (resto)")

    grp = out.groupby(["Mes", "Categoria2"])["€"].sum().reset_index()
    fig = px.bar(grp, x="Mes", y="€", color="Categoria2", title="📊 PRO: Gasto por mes (barras apiladas por categoría)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_distribution(txg: pd.DataFrame):
    """Distribución de tamaños de movimientos (entrada/salida)."""
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["cashflow"]).copy()
    if df.empty:
        return None
    df["Tipo"] = np.where(df["cashflow"] >= 0, "Entrada", "Salida")
    df["€"] = df["cashflow"].abs()
    fig = px.histogram(df, x="€", color="Tipo", nbins=50, title="📐 PRO: Distribución de importes (tamaño de movimientos)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def anomalies_daily(txg: pd.DataFrame, z_thresh: float = 2.5) -> pd.DataFrame:
    """Días raros por Z-score del neto diario."""
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["day"] = df["date"].dt.date
    daily = df.groupby("day")["cashflow"].sum().reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    mu = daily["cashflow"].mean()
    sd = daily["cashflow"].std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.DataFrame()

    daily["z"] = (daily["cashflow"] - mu) / sd
    out = daily[daily["z"].abs() >= z_thresh].copy()
    out = out.sort_values("z", key=lambda s: s.abs(), ascending=False)
    out["Día"] = out["day"].dt.strftime("%Y-%m-%d")
    out["Neto diario"] = out["cashflow"]
    out["Z"] = out["z"]
    return out[["Día", "Neto diario", "Z"]]


def recurring_candidates(txg: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """
    Detecta “parece recurrente” por:
    - descripción normalizada + importe redondeado
    """
    df = txg.dropna(subset=["date", "cashflow", "desc"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["desc_norm"] = df["desc"].apply(normalize_desc_for_grouping)
    df["amt_round"] = df["cashflow"].round(2)
    # opcional: centrarse en salidas de tarjeta y transferencias
    df["key"] = df["desc_norm"] + " | " + df["amt_round"].astype(str)

    grp = df.groupby("key").agg(
        veces=("key", "size"),
        primera=("date", "min"),
        ultima=("date", "max"),
        importe=("cashflow", "mean"),
        ejemplo=("desc", lambda s: short_desc(s.iloc[0], 110)),
    ).reset_index()

    grp = grp[grp["veces"] >= min_count].copy()
    if grp.empty:
        return grp

    grp = grp.sort_values(["veces", "ultima"], ascending=[False, False])
    grp["Primera"] = pd.to_datetime(grp["primera"]).dt.strftime("%Y-%m-%d")
    grp["Última"] = pd.to_datetime(grp["ultima"]).dt.strftime("%Y-%m-%d")
    grp["Importe medio"] = grp["importe"]
    return grp[["veces", "Primera", "Última", "Importe medio", "ejemplo"]].rename(columns={"ejemplo": "Ejemplo"})


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("1) Sube tu PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])

    st.divider()
    st.header("2) Nivel")
    mode = st.radio("Elige vista", ["SIMPLE (entender rápido)", "PRO (análisis completo)"], index=0)

    st.divider()
    st.header("3) Filtros")
    show_assets = st.checkbox("Mostrar sección de activos (si operaste)", value=True)
    show_details = st.checkbox("Ver tabla completa (detalles)", value=False)
    top_cat_n = st.slider("Top categorías", 5, 20, 12)
    top_moves_n = st.slider("Top movimientos", 5, 25, 12)

    st.divider()
    st.caption("Tip: si algo no cuadra, activa detalles y mira el texto exacto del PDF.")


if not up:
    st.info("⬅️ Sube tu **extracto de cuenta PDF** para empezar.")
    st.stop()

pdf_bytes = up.getvalue()

# =========================
# PARSE SAFE
# =========================
with st.spinner("Leyendo tu PDF…"):
    try:
        tx = parse_tr_pdf_transactions(pdf_bytes)
    except Exception as e:
        st.error("No he podido leer el PDF sin errores. Prueba con otro extracto o vuelve a descargarlo.")
        st.exception(e)
        st.stop()

if tx.empty:
    st.error(
        "No he encontrado la sección de transacciones dentro del PDF. "
        "Asegúrate de que es un **Extracto de cuenta** con 'TRANSACCIONES DE CUENTA'."
    )
    st.stop()

# Postprocess
tx = tx.copy()
tx["cashflow"] = pd.to_numeric(tx["cashflow"], errors="coerce")
tx["balance"] = pd.to_numeric(tx["balance"], errors="coerce")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
tx["Categoria"] = [category_simple(t, d) for t, d in zip(tx["type"].astype(str), tx["desc"].astype(str))]
txg = tx.dropna(subset=["date"]).sort_values("date").copy()

# Filtro por rango fechas + categorías
if not txg.empty:
    dmin = pd.to_datetime(txg["date"].min()).date()
    dmax = pd.to_datetime(txg["date"].max()).date()
else:
    dmin = dmax = pd.Timestamp.today().date()

with st.sidebar:
    if dmin <= dmax:
        date_range = st.date_input("Rango de fechas", value=(dmin, dmax))
    else:
        date_range = (dmin, dmax)

cats_all = sorted(txg["Categoria"].dropna().unique().tolist()) if not txg.empty else []
with st.sidebar:
    sel_cats = st.multiselect("Categorías", options=cats_all, default=cats_all)

# aplicar filtros
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = dmin, dmax

txg_f = txg.copy()
txg_f = txg_f[(txg_f["date"].dt.date >= start_d) & (txg_f["date"].dt.date <= end_d)]
if sel_cats:
    txg_f = txg_f[txg_f["Categoria"].isin(sel_cats)]

# recomputar métricas para filtros
tx_f = tx.copy()
# en tx_f también filtra por fechas/categorías si tiene date
tx_f2 = tx_f.dropna(subset=["date"]).copy()
tx_f2 = tx_f2[(tx_f2["date"].dt.date >= start_d) & (tx_f2["date"].dt.date <= end_d)]
if sel_cats:
    tx_f2 = tx_f2[tx_f2["Categoria"].isin(sel_cats)]

total_in = float(tx_f2.loc[tx_f2["cashflow"] > 0, "cashflow"].sum(skipna=True))
total_out = float(-tx_f2.loc[tx_f2["cashflow"] < 0, "cashflow"].sum(skipna=True))
net = float(tx_f2["cashflow"].sum(skipna=True))

by_cat = tx_f2.groupby("Categoria")["cashflow"].sum()

metiste = float(by_cat.get("Dinero que metiste", 0.0))
sacaste = float(abs(by_cat.get("Dinero que sacaste", 0.0)))
tarjeta = float(abs(by_cat.get("Gastos con tarjeta", 0.0)))
comisiones = float(abs(by_cat.get("Comisiones", 0.0)))
intereses = float(by_cat.get("Intereses / rentabilidad", 0.0))

last_balance_val = float(tx_f2["balance"].dropna().iloc[-1]) if tx_f2["balance"].notna().any() else float("nan")

# =========================
# CABECERA
# =========================
st.markdown(
    f"""
<span class="badge">📌 Entradas: <b>{fmt_eur(total_in)}</b></span>
<span class="badge">📤 Salidas: <b>{fmt_eur(total_out)}</b></span>
<span class="badge">🧮 Neto: <b>{fmt_eur(net)}</b></span>
<span class="badge">🗓️ Rango: <b>{start_d} → {end_d}</b></span>
""",
    unsafe_allow_html=True,
)

if np.isfinite(last_balance_val):
    st.success(f"Según el PDF (en este rango), tu **saldo final** es: **{fmt_eur(last_balance_val)}**")
else:
    st.warning("No he encontrado un **balance final** fiable en el PDF. Aun así, analizo entradas/salidas.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# KPIs
# =========================
st.subheader("✅ Resumen (lo esencial)")
st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi"><div class="t">Dinero que metiste</div><div class="v">{fmt_eur(metiste)}</div><div class="s">Ingresos/aportaciones</div></div>
  <div class="kpi"><div class="t">Dinero que sacaste</div><div class="v">{fmt_eur(sacaste)}</div><div class="s">Retiradas fuera</div></div>
  <div class="kpi"><div class="t">Gastos con tarjeta</div><div class="v">{fmt_eur(tarjeta)}</div><div class="s">Pagos / compras</div></div>
  <div class="kpi"><div class="t">Comisiones</div><div class="v">{fmt_eur(comisiones)}</div><div class="s">Costes cobrados</div></div>
  <div class="kpi"><div class="t">Intereses / rentabilidad</div><div class="v">{fmt_eur(intereses)}</div><div class="s">Abonos / rendimientos</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# mini-insights
if not txg_f.empty:
    dfm = txg_f.dropna(subset=["date", "cashflow"]).copy()
    dfm["Mes"] = dfm["date"].dt.to_period("M").astype(str)
    m = dfm.groupby("Mes")["cashflow"].sum()
    if len(m) >= 1:
        best_m = m.idxmax()
        worst_m = m.idxmin()
        st.info(
            f"📌 Lectura rápida: Mes mejor **{best_m}** ({fmt_eur(m.loc[best_m])}) · "
            f"Mes peor **{worst_m}** ({fmt_eur(m.loc[worst_m])})"
        )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# TABS
# =========================
tabA, tabB, tabC, tabD = st.tabs(["📌 Vista", "📈 Evolución", "🫧 Movimientos", "📦 Activos / 🔎 Detalles"])

# ---------- SIMPLE ----------
with tabA:
    st.subheader("📌 Vista principal")
    if mode.startswith("SIMPLE"):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**1) Entradas vs Salidas**")
            fig = fig_in_out_net(total_in, total_out, net)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(pd.Series({"Entradas": total_in, "Salidas": total_out, "Neto": net}))

        with c2:
            st.markdown("**2) ¿En qué se fue el dinero?**")
            fig = fig_out_by_category(by_cat, top_n=top_cat_n)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                outs = by_cat[by_cat < 0].abs().sort_values(ascending=False).head(top_cat_n)
                if outs.empty:
                    st.info("No veo salidas en el rango seleccionado.")
                else:
                    st.bar_chart(outs)

        st.markdown("**3) Saldo / evolución**")
        fig = fig_balance_or_estimated(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se pudo generar la evolución del saldo (o falta Plotly).")

    else:
        st.subheader("PRO: Vista principal + panel extra")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**1) Entradas vs Salidas**")
            fig = fig_in_out_net(total_in, total_out, net)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**2) PRO: Acumulado (entradas/salidas/neto)**")
            fig = fig_cum_in_out(txg_f)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se pudo generar acumulado.")

        with c2:
            st.markdown("**3) ¿En qué se fue el dinero?**")
            fig = fig_out_by_category(by_cat, top_n=top_cat_n)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**4) PRO: Gasto por mes (apilado)**")
            fig = fig_stack_monthly_out_by_category(txg_f, top_n=min(10, top_cat_n))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay suficientes salidas para el apilado mensual.")

with tabB:
    st.subheader("📈 Evolución")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Saldo / evolución**")
        fig = fig_balance_or_estimated(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se pudo generar evolución.")

    with c2:
        st.markdown("**Mes a mes: neto y acumulado**")
        fig, mdf = fig_monthly_net(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            if mdf is None or mdf.empty:
                st.info("No hay suficientes datos para mes a mes.")
            else:
                st.bar_chart(mdf.set_index("Mes")[["cashflow"]])

    if mode.startswith("PRO"):
        st.markdown("**PRO: Distribución de importes**")
        fig = fig_distribution(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se pudo generar distribución (o falta Plotly).")

        pnl_t = realized_pnl_timeline(tx_f2)
        if not pnl_t.empty and PLOTLY_OK:
            figp = px.line(pnl_t, x="date", y="pnl_realizado_total", title="💹 PRO: P&L realizado acumulado (ventas)")
            figp.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(figp, use_container_width=True)

with tabC:
    st.subheader("🫧 Movimientos")
    st.caption("Aquí se entiende el “qué pasó” de verdad: picos y movimientos raros.")

    fig = fig_timeline_bubbles(txg_f)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Plotly no está disponible: muestro tablas.")

    st.markdown("**Movimientos más grandes (por impacto)**")
    df = txg_f.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        st.info("No hay movimientos suficientes.")
    else:
        df["Impacto"] = df["cashflow"].abs()
        df = df.sort_values("Impacto", ascending=False).head(top_moves_n).copy()
        df["Día"] = df["date"].dt.strftime("%Y-%m-%d")
        df["€"] = df["cashflow"]
        df["Descripción"] = df["desc"].apply(lambda x: short_desc(x, 120))
        st.dataframe(df[["Día", "Categoria", "€", "Descripción"]], use_container_width=True, hide_index=True)

    if mode.startswith("PRO"):
        st.markdown("**PRO: Días “anómalos” (raros)**")
        an = anomalies_daily(txg_f, z_thresh=2.5)
        if an.empty:
            st.info("No detecto días anómalos (o no hay varianza suficiente).")
        else:
            st.dataframe(an, use_container_width=True, hide_index=True)

        st.markdown("**PRO: Candidatos a pagos recurrentes**")
        rec = recurring_candidates(txg_f, min_count=3)
        if rec.empty:
            st.info("No veo patrones recurrentes claros en el rango filtrado.")
        else:
            st.dataframe(rec, use_container_width=True, hide_index=True)

with tabD:
    st.subheader("📦 Activos / 🔎 Detalles")

    if show_assets:
        st.markdown("### 📦 Activos (si operaste)")
        assets = compute_asset_realized_pnl(tx_f2)
        if assets.empty:
            st.info("No veo operaciones de inversión suficientes en este rango.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Activos detectados", f"{len(assets)}")
            a2.metric("Ganado/perdido ya cerrado (total)", fmt_eur(assets["Ganado / perdido ya cerrado"].sum()))
            a3.metric("Dinero metido total (compras)", fmt_eur(assets["Dinero metido (compras)"].sum()))

            st.dataframe(
                assets[
                    [
                        "Activo",
                        "ISIN",
                        "Dinero metido (compras)",
                        "Dinero recuperado (ventas)",
                        "Ganado / perdido ya cerrado",
                        "Cantidad que te queda (aprox.)",
                        "Coste medio (aprox.)",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            if PLOTLY_OK:
                top = assets.sort_values("Ganado / perdido ya cerrado", ascending=False).head(12)
                fig = px.bar(top, x="Ganado / perdido ya cerrado", y="Activo", orientation="h",
                             title="🏅 Top · Ganado/perdido ya cerrado")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### 🔎 Detalles (tabla completa)")

    if show_details:
        st.dataframe(
            tx_f2[["date", "type", "Categoria", "cashflow", "balance", "isin", "asset", "quantity", "desc"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Activa “Ver tabla completa (detalles)” en la barra lateral para verlo todo.")

# =========================
# DESCARGA
# =========================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.download_button(
    "⬇️ Descargar los datos parseados (CSV)",
    data=tx_f2.to_csv(index=False).encode("utf-8"),
    file_name="trade_republic_extract_parsed.csv",
    mime="text/csv",
)

st.markdown(
    """
### Nota importante
- Esto explica **lo que pasó en tu cuenta** a partir del PDF (entradas/salidas y, si existe, balance).
- Para saber el **valor actual** de tus inversiones haría falta añadir precios de mercado (no vienen en el PDF).
"""
)
