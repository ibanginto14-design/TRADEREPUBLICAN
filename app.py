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
# CONFIG + NUEVO DISEÑO (sin cambiar “qué datos doy”)
# =========================
st.set_page_config(page_title="Trade Republic · Mi dinero (PDF)", page_icon="💶", layout="wide")

st.markdown(
    """
<style>
/* --- App frame --- */
.block-container { padding-top: 0.9rem; padding-bottom: 2.2rem; max-width: 1260px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.10); }

/* --- Typography --- */
h1,h2,h3 { letter-spacing: -0.35px; }
p, li, label { line-height: 1.35; }
.small { font-size: 12px; opacity: .78; }
.muted { opacity: .82; }

/* --- Top bar --- */
.topbar {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  margin-bottom: 12px;
}
.brand {
  display:flex; align-items:center; gap:10px;
}
.brand .logo {
  width: 36px; height: 36px; border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  font-size: 18px;
}
.brand .txt .t { font-weight: 850; font-size: 16px; letter-spacing:-0.3px; }
.brand .txt .s { font-size: 12px; opacity: .78; margin-top: 2px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.13);
  background: rgba(255,255,255,0.04);
  font-size: 12px;
}

/* --- Cards --- */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 12px 14px;
}
.card .ct { font-weight: 800; margin-bottom: 2px; letter-spacing:-0.25px; }
.card .cs { font-size: 12px; opacity: .78; margin-bottom: 10px; }

/* --- KPI row --- */
.kpis { display:grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
@media (max-width: 1200px){ .kpis { grid-template-columns: repeat(2, 1fr); } }
.kpi {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.035);
  border-radius: 18px;
  padding: 12px 14px;
}
.kpi .t { font-size: 12px; opacity: .80; margin-bottom: 6px; }
.kpi .v { font-size: 22px; font-weight: 900; letter-spacing:-0.5px; }
.kpi .s { font-size: 12px; opacity: .70; margin-top: 6px; }

/* --- Section divider --- */
.hr { height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0; border-radius: 999px; }

/* --- Dataframe --- */
div[data-testid="stDataFrame"] {
  border-radius: 16px; overflow:hidden; border: 1px solid rgba(255,255,255,0.10);
}

/* --- Navigation (radio) look like segmented control --- */
div[role="radiogroup"] > label { padding: 0 !important; }
.navwrap {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 6px 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_eur(x: float) -> str:
    try:
        return f"{float(x):,.2f} €"
    except Exception:
        return "—"


def short_desc(s: str, n: int = 110) -> str:
    s = str(s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


# =========================
# PARSER (sin cambios de datos)
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
    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2), (m.group(3) or "").strip()


def _year_prefix(line: str) -> Optional[Tuple[int, str]]:
    m = re.match(r"^\s*(\d{4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "").strip()


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
# CATEGORÍAS “CLARAS” (sin cambios)
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


# =========================
# ACTIVOS (sin cambios)
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


# =========================
# GRÁFICOS (mismos tipos/datos)
# =========================
def fig_in_out_net(total_in: float, total_out: float, net: float):
    if not PLOTLY_OK:
        return None
    df = pd.DataFrame({"Concepto": ["Entradas", "Salidas", "Neto"], "€": [total_in, total_out, net]})
    fig = px.bar(df, x="Concepto", y="€", title="⚖️ Entradas vs Salidas (y neto)")
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=50, b=8))
    return fig


def donut_outflows(by_cat: pd.Series, top_n: int = 8):
    if not PLOTLY_OK:
        return None
    out = by_cat[by_cat < 0].abs().sort_values(ascending=False)
    if out.empty:
        return None

    top = out.head(top_n)
    rest = out.iloc[top_n:].sum() if len(out) > top_n else 0.0
    df = top.reset_index()
    df.columns = ["Concepto", "€"]
    if rest > 1e-9:
        df = pd.concat([df, pd.DataFrame([{"Concepto": "Otros (resto)", "€": rest}])], ignore_index=True)

    fig = px.pie(df, names="Concepto", values="€", hole=0.62, title="🍩 Donut · Salidas")
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=50, b=8))
    return fig


def fig_balance_or_estimated(txg: pd.DataFrame):
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date"]).sort_values("date").copy()
    if df.empty:
        return None

    if df["balance"].notna().any():
        d2 = df.dropna(subset=["balance"]).copy()
        fig = px.line(d2, x="date", y="balance", title="📈 Evolución del saldo (balance del PDF)")
    else:
        d2 = df.dropna(subset=["cashflow"]).copy()
        d2["Saldo estimado (desde 0)"] = d2["cashflow"].cumsum()
        fig = px.line(d2, x="date", y="Saldo estimado (desde 0)", title="📈 Evolución estimada (entradas/salidas)")
    fig.update_layout(height=360, margin=dict(l=8, r=8, t=50, b=8))
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
        height=360,
        margin=dict(l=8, r=8, t=50, b=8),
        yaxis=dict(title="€ neto"),
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
        title="🫧 Movimientos (puntos grandes = impacto grande)",
    )
    fig.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))
    return fig


def fig_stack_monthly_out_by_category(txg: pd.DataFrame, top_n: int = 8):
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

    top_cats = out.groupby("Categoria")["€"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    out["Categoria2"] = out["Categoria"].where(out["Categoria"].isin(top_cats), other="Otros (resto)")

    grp = out.groupby(["Mes", "Categoria2"])["€"].sum().reset_index()
    fig = px.bar(grp, x="Mes", y="€", color="Categoria2", title="📊 PRO · Gasto por mes (apilado por categoría)")
    fig.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))
    return fig


def biggest_moves_table(txg: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return df
    df["Impacto"] = df["cashflow"].abs()
    df = df.sort_values("Impacto", ascending=False).head(n).copy()
    df["Día"] = df["date"].dt.strftime("%Y-%m-%d")
    df["€ (entrada/salida)"] = df["cashflow"]
    df["Descripción corta"] = df["desc"].apply(lambda x: short_desc(x, 120))
    return df[["Día", "Categoria", "€ (entrada/salida)", "Descripción corta"]]


# =========================
# SIDEBAR (limpia)
# =========================
with st.sidebar:
    st.subheader("📄 Extracto")
    up = st.file_uploader("Sube tu PDF", type=["pdf"])

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("⚙️ Controles")

    view = st.radio("Modo", ["Cómodo", "PRO"], index=0)

    with st.expander("Opciones", expanded=True):
        show_assets = st.checkbox("Mostrar activos (si operaste)", value=True)
        show_details = st.checkbox("Ver tabla completa", value=False)
        donut_top = st.slider("Donut: Top categorías", 4, 12, 8)
        top_moves_n = st.slider("Top movimientos", 5, 25, 12)

if not up:
    st.info("⬅️ Sube tu PDF para empezar.")
    st.stop()

pdf_bytes = up.getvalue()

with st.spinner("Leyendo tu PDF…"):
    tx = parse_tr_pdf_transactions(pdf_bytes)

if tx.empty:
    st.error(
        "No he encontrado la sección de transacciones dentro del PDF. "
        "Asegúrate de que es un **Extracto de cuenta** con 'TRANSACCIONES DE CUENTA'."
    )
    st.stop()

# Postprocess (sin cambiar datos)
tx = tx.copy()
tx["cashflow"] = pd.to_numeric(tx["cashflow"], errors="coerce")
tx["balance"] = pd.to_numeric(tx["balance"], errors="coerce")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
tx["Categoria"] = [category_simple(t, d) for t, d in zip(tx["type"].astype(str), tx["desc"].astype(str))]
txg = tx.dropna(subset=["date"]).sort_values("date").copy()

# Rango de fechas
if not txg.empty:
    dmin = pd.to_datetime(txg["date"].min()).date()
    dmax = pd.to_datetime(txg["date"].max()).date()
else:
    dmin = dmax = pd.Timestamp.today().date()

with st.sidebar:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("🗓️ Rango")
    date_range = st.date_input("Fechas", value=(dmin, dmax))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = dmin, dmax

txg_f = txg[(txg["date"].dt.date >= start_d) & (txg["date"].dt.date <= end_d)].copy()
tx_f2 = tx.dropna(subset=["date"]).copy()
tx_f2 = tx_f2[(tx_f2["date"].dt.date >= start_d) & (tx_f2["date"].dt.date <= end_d)].copy()

# Métricas (sin cambios)
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
# TOPBAR (nuevo look)
# =========================
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">💶</div>
    <div class="txt">
      <div class="t">Trade Republic · Mi dinero</div>
      <div class="s">Entradas, salidas, en qué se fue, evolución y movimientos clave</div>
    </div>
  </div>
  <div class="chips">
    <div class="chip">📌 Entradas <b>{fmt_eur(total_in)}</b></div>
    <div class="chip">📤 Salidas <b>{fmt_eur(total_out)}</b></div>
    <div class="chip">🧮 Neto <b>{fmt_eur(net)}</b></div>
    <div class="chip">🗓️ {start_d} → {end_d}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if np.isfinite(last_balance_val):
    st.success(f"Según el PDF (en este rango), tu **saldo final** es: **{fmt_eur(last_balance_val)}**")
else:
    st.warning("No he encontrado un **balance final** fiable en el PDF. Aun así, analizo entradas/salidas.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# KPIs (mismo contenido)
# =========================
st.markdown(
    f"""
<div class="kpis">
  <div class="kpi"><div class="t">Dinero que metiste</div><div class="v">{fmt_eur(metiste)}</div><div class="s">Ingresos/aportaciones</div></div>
  <div class="kpi"><div class="t">Dinero que sacaste</div><div class="v">{fmt_eur(sacaste)}</div><div class="s">Retiradas fuera</div></div>
  <div class="kpi"><div class="t">Gastos con tarjeta</div><div class="v">{fmt_eur(tarjeta)}</div><div class="s">Pagos / compras</div></div>
  <div class="kpi"><div class="t">Comisiones</div><div class="v">{fmt_eur(comisiones)}</div><div class="s">Costes cobrados</div></div>
  <div class="kpi"><div class="t">Intereses / rentabilidad</div><div class="v">{fmt_eur(intereses)}</div><div class="s">Abonos / rendimientos</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# NAVEGACIÓN (diseño nuevo, datos iguales)
# =========================
cnav1, cnav2 = st.columns([1.5, 4.5], gap="large")

with cnav1:
    st.markdown('<div class="navwrap">', unsafe_allow_html=True)
    page = st.radio(
        "Navegación",
        ["Dashboard", "Movimientos", "PRO", "Activos & Detalles"],
        label_visibility="collapsed",
        index=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.caption("Consejo: si algo no cuadra, ve a **Detalles** y revisamos el texto del PDF.")

with cnav2:
    if page == "Dashboard":
        st.markdown('<div class="card"><div class="ct">📌 Dashboard</div><div class="cs">Vista cómoda: lo esencial sin ruido.</div></div>', unsafe_allow_html=True)
        st.write("")

        r1a, r1b = st.columns([1.25, 0.95], gap="large")
        with r1a:
            st.markdown('<div class="card"><div class="ct">⚖️ Entradas vs Salidas</div><div class="cs">Lo más directo para entender el periodo.</div>', unsafe_allow_html=True)
            fig = fig_in_out_net(total_in, total_out, net)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(pd.Series({"Entradas": total_in, "Salidas": total_out, "Neto": net}))
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="card"><div class="ct">📈 Evolución del saldo</div><div class="cs">Balance del PDF o estimación acumulando entradas/salidas.</div>', unsafe_allow_html=True)
            fig = fig_balance_or_estimated(txg_f)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se pudo generar la evolución (o falta Plotly).")
            st.markdown("</div>", unsafe_allow_html=True)

        with r1b:
            st.markdown('<div class="card"><div class="ct">🍩 Donut de salidas</div><div class="cs">En qué se fue el dinero (agrupado).</div>', unsafe_allow_html=True)
            fig = donut_outflows(by_cat, top_n=donut_top)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                outs = by_cat[by_cat < 0].abs().sort_values(ascending=False).head(donut_top)
                if outs.empty:
                    st.info("No veo salidas en el rango.")
                else:
                    st.bar_chart(outs)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card"><div class="ct">📅 Mes a mes</div><div class="cs">Neto mensual y acumulado (menos ruido que el día a día).</div>', unsafe_allow_html=True)
        fig, mdf = fig_monthly_net(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            if mdf is not None and not mdf.empty:
                st.bar_chart(mdf.set_index("Mes")[["cashflow"]])
            else:
                st.info("No hay suficientes datos para mes a mes.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Movimientos":
        st.markdown('<div class="card"><div class="ct">🫧 Movimientos</div><div class="cs">Picos y días que movieron la aguja.</div></div>', unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="card"><div class="ct">Timeline</div><div class="cs">Puntos grandes = movimientos grandes.</div>', unsafe_allow_html=True)
        fig = fig_timeline_bubbles(txg_f)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly no está disponible.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card"><div class="ct">Top movimientos por impacto</div><div class="cs">Los más importantes para entender “qué pasó”.</div>', unsafe_allow_html=True)
        big = biggest_moves_table(txg_f, n=top_moves_n)
        if big.empty:
            st.info("No hay suficientes movimientos con fecha/importe para listar.")
        else:
            st.dataframe(big, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "PRO":
        st.markdown('<div class="card"><div class="ct">📊 PRO</div><div class="cs">Más detalle, manteniendo limpieza visual.</div></div>', unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="card"><div class="ct">Gasto por mes (apilado por categoría)</div><div class="cs">Qué categoría dominó cada mes.</div>', unsafe_allow_html=True)
        fig = fig_stack_monthly_out_by_category(txg_f, top_n=8)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes salidas para construir este gráfico (o falta Plotly).")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        with st.expander("Ver tabla de categorías netas"):
            tbl = by_cat.sort_values()
            if tbl.empty:
                st.info("No hay datos por categoría.")
            else:
                st.dataframe(
                    tbl.rename("€ neto")
                    .reset_index()
                    .rename(columns={"index": "Categoría"}),
                    use_container_width=True,
                    hide_index=True,
                )

    else:  # Activos & Detalles
        st.markdown('<div class="card"><div class="ct">📦 Activos & 🔎 Detalles</div><div class="cs">Activos (si operaste) y la tabla completa para comprobar.</div></div>', unsafe_allow_html=True)
        st.write("")

        a, b = st.columns([1.05, 0.95], gap="large")

        with a:
            st.markdown('<div class="card"><div class="ct">📦 Activos (si operaste)</div><div class="cs">Resultado ya cerrado y estimación de lo que queda.</div>', unsafe_allow_html=True)
            if show_assets:
                assets = compute_asset_realized_pnl(tx_f2)
                if assets.empty:
                    st.info("No veo operaciones de inversión suficientes en este rango.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Activos", f"{len(assets)}")
                    c2.metric("Ganado/perdido ya cerrado", fmt_eur(assets["Ganado / perdido ya cerrado"].sum()))
                    c3.metric("Dinero metido (compras)", fmt_eur(assets["Dinero metido (compras)"].sum()))

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
                        fig = px.bar(
                            top,
                            x="Ganado / perdido ya cerrado",
                            y="Activo",
                            orientation="h",
                            title="🏅 Top · Ganado/perdido ya cerrado",
                        )
                        fig.update_layout(height=360, margin=dict(l=8, r=8, t=50, b=8))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Activa 'Mostrar activos' en la barra lateral.")
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown('<div class="card"><div class="ct">🔎 Detalles</div><div class="cs">Para verificar líneas exactas del PDF.</div>', unsafe_allow_html=True)
            if show_details:
                st.dataframe(
                    tx_f2[["date", "type", "Categoria", "cashflow", "balance", "isin", "asset", "quantity", "desc"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Activa 'Ver tabla completa' en la barra lateral.")
            st.markdown("</div>", unsafe_allow_html=True)


# =========================
# DESCARGA + NOTA
# =========================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.download_button(
    "⬇️ Descargar datos parseados (CSV)",
    data=tx_f2.to_csv(index=False).encode("utf-8"),
    file_name="trade_republic_extract_parsed.csv",
    mime="text/csv",
)

st.markdown(
    """
<div class="small muted">
<b>Nota:</b> Esto explica tu cuenta a partir del PDF (entradas/salidas y balance si existe).
Para el <b>valor actual</b> de tus inversiones harían falta precios de mercado (no vienen en el PDF).
</div>
""",
    unsafe_allow_html=True,
)
