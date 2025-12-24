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
# CONFIG + ESTILO (más claro)
# =========================
st.set_page_config(page_title="Mi dinero en Trade Republic (PDF)", page_icon="💶", layout="wide")

st.markdown(
    """
<style>
/* Layout más limpio */
.block-container { padding-top: 1.4rem; padding-bottom: 2.0rem; }
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
.kpi .v { font-size: 22px; font-weight: 700; }
.kpi .s { font-size: 12px; opacity: 0.75; margin-top: 6px; }

.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 12px;
  margin-right: 8px;
}

.hr { height: 1px; background: rgba(255,255,255,0.12); margin: 16px 0; border-radius: 999px;}
.small { font-size: 12px; opacity: 0.78; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("💶 Mi dinero en Trade Republic")
st.caption(
    "Sube tu **Extracto de cuenta (PDF)** y te lo traduzco a lenguaje fácil: "
    "**qué metiste**, **qué sacaste**, **en qué se fue**, y **qué días/meses movieron la aguja**."
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

        # Importes: cogemos los 2 últimos números del bloque (importe y balance)
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
                "amount": amount,      # sin signo 100% garantizado
                "cashflow": cashflow,  # con signo inferido
                "balance": balance,
            }
        )

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.dropna(subset=["date"], how="all").sort_values("date").reset_index(drop=True)
    return df


# =========================
# “TRADUCCIÓN” A LENGUAJE SIMPLE
# =========================
def category_simple(row_type: str, desc: str) -> str:
    """Categorías pensadas para entender el dinero, no para finanzas."""
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


def short_desc(s: str, n: int = 70) -> str:
    s = str(s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def compute_asset_realized_pnl(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Por activo (ISIN): muy entendible:
    - dinero metido en ese activo (compras)
    - dinero recuperado (ventas)
    - ganado/perdido REALIZADO (solo lo ya cerrado con ventas)
    - cantidad que te queda (aprox.) y coste medio (aprox.)
    """
    if tx.empty or "type" not in tx.columns:
        return pd.DataFrame()

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
        buy_qty = 0.0
        sell_qty = 0.0
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
                buy_qty += qty
            else:  # SELL (o inferido)
                proceeds = amt
                cost_basis = qty * avg_cost
                realized += (proceeds - cost_basis)

                pos_qty -= qty
                if pos_qty <= 1e-12:
                    pos_qty = 0.0
                    avg_cost = 0.0

                sell_amt += amt
                sell_qty += qty

        rows.append(
            {
                "ISIN": isin,
                "Activo": asset_name if asset_name else isin,
                "Dinero metido (compras)": buy_amt,
                "Dinero recuperado (ventas)": sell_amt,
                "Ganado / perdido ya cerrado": realized,
                "Cantidad que te queda (aprox.)": pos_qty,
                "Coste medio (aprox.)": avg_cost,
                "Primera operación": g["date"].min(),
                "Última operación": g["date"].max(),
            }
        )

    out = pd.DataFrame(rows).sort_values("Dinero metido (compras)", ascending=False).reset_index(drop=True)
    return out


# =========================
# GRÁFICAS (más chulas, pero fáciles)
# =========================
def sankey_money_flow(by_cat: pd.Series):
    """
    Sankey muy visual:
    Categorías de entrada -> TU CUENTA -> categorías de salida (+ ajuste si falta/sobra).
    """
    if not PLOTLY_OK:
        return None

    by_cat = by_cat.copy()
    by_cat = by_cat[by_cat.abs() > 1e-9]

    ins = by_cat[by_cat > 0].sort_values(ascending=False)
    outs = (-by_cat[by_cat < 0]).sort_values(ascending=False)

    total_in = float(ins.sum()) if not ins.empty else 0.0
    total_out = float(outs.sum()) if not outs.empty else 0.0

    center = "Tu cuenta"

    labels = []
    idx = {}

    def add_label(name: str) -> int:
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
        return idx[name]

    # Nodos
    for c in ins.index.tolist():
        add_label(f"⬆️ {c}")
    add_label(center)
    for c in outs.index.tolist():
        add_label(f"⬇️ {c}")

    # Ajuste para cuadrar Sankey
    extra_in = 0.0
    extra_out = 0.0
    if total_out > total_in + 1e-9:
        extra_in = total_out - total_in
        add_label("🧩 Saldo previo (no se ve en este PDF)")
    elif total_in > total_out + 1e-9:
        extra_out = total_in - total_out
        add_label("💰 Saldo que te quedó")

    sources = []
    targets = []
    values = []

    # Entradas -> Cuenta
    for c, v in ins.items():
        sources.append(add_label(f"⬆️ {c}"))
        targets.append(add_label(center))
        values.append(float(v))

    if extra_in > 0:
        sources.append(add_label("🧩 Saldo previo (no se ve en este PDF)"))
        targets.append(add_label(center))
        values.append(float(extra_in))

    # Cuenta -> Salidas
    for c, v in outs.items():
        sources.append(add_label(center))
        targets.append(add_label(f"⬇️ {c}"))
        values.append(float(v))

    if extra_out > 0:
        sources.append(add_label(center))
        targets.append(add_label("💰 Saldo que te quedó"))
        values.append(float(extra_out))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=12, thickness=16),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title="🧭 Mapa del dinero (muy visual)", height=520, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def donut_outflows(by_cat: pd.Series):
    """Donut solo de salidas (en qué se fue el dinero)."""
    if not PLOTLY_OK:
        return None
    out = by_cat[by_cat < 0].abs()
    out = out[out > 0]
    if out.empty:
        return None
    df = out.reset_index()
    df.columns = ["Concepto", "€"]
    fig = px.pie(df, names="Concepto", values="€", hole=0.6, title="🧾 ¿En qué se fue tu dinero? (solo salidas)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def monthly_net(txg: pd.DataFrame):
    """
    Barras por mes (neto) + línea acumulada (para entender tendencia).
    Si Plotly no está, devolvemos df para fallback.
    """
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


def balance_or_estimated(txg: pd.DataFrame):
    """
    Si hay balance fiable: línea de balance.
    Si no: línea de 'saldo estimado' partiendo de 0 (acumulado de cashflow).
    """
    df = txg.dropna(subset=["date"]).sort_values("date").copy()
    if df.empty:
        return None

    has_balance = df["balance"].notna().any()
    if PLOTLY_OK:
        if has_balance:
            d2 = df.dropna(subset=["balance"]).copy()
            fig = px.line(d2, x="date", y="balance", title="📈 Tu saldo en el PDF (balance)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            return fig
        else:
            df = df.dropna(subset=["cashflow"]).copy()
            df["Saldo estimado (desde 0)"] = df["cashflow"].cumsum()
            fig = px.line(df, x="date", y="Saldo estimado (desde 0)", title="📈 Evolución estimada (sumando entradas/salidas)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            return fig

    return None


def biggest_moves(txg: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Movimientos que más impactaron (por valor absoluto del cashflow)."""
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return df
    df["Impacto"] = df["cashflow"].abs()
    df = df.sort_values("Impacto", ascending=False).head(n).copy()
    df["Día"] = df["date"].dt.strftime("%Y-%m-%d")
    df["€ (entrada/salida)"] = df["cashflow"]
    df["Descripción corta"] = df["desc"].apply(lambda x: short_desc(x, 90))
    return df[["Día", "Categoria", "€ (entrada/salida)", "Descripción corta"]]


# =========================
# SIDEBAR (modo simple)
# =========================
with st.sidebar:
    st.header("1) Sube tu PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])

    st.divider()
    st.header("2) Modo")
    simple_mode = st.toggle("Modo súper simple (recomendado)", value=True)
    show_assets = st.checkbox("Mostrar inversiones por activo (si operaste)", value=True)
    show_details = st.checkbox("Ver tabla completa (detalles)", value=False)

    st.divider()
    top_n_moves = st.slider("Movimientos más grandes a mostrar", 5, 20, 10)
    st.caption("Tip: si algo no cuadra, activa **detalles** y miramos el texto exacto del PDF.")


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

# Resumen por categorías
by_cat = tx.groupby("Categoria")["cashflow"].sum()

total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum(skipna=True))
total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum(skipna=True))
net = float(tx["cashflow"].sum(skipna=True))

metiste = float(by_cat.get("Dinero que metiste", 0.0))
sacaste = float(abs(by_cat.get("Dinero que sacaste", 0.0)))
tarjeta = float(abs(by_cat.get("Gastos con tarjeta", 0.0)))
comisiones = float(abs(by_cat.get("Comisiones", 0.0)))
intereses = float(by_cat.get("Intereses / rentabilidad", 0.0))

last_balance_val = float(tx["balance"].dropna().iloc[-1]) if tx["balance"].notna().any() else float("nan")

# =========================
# CABECERA: QUÉ SIGNIFICA
# =========================
st.markdown(
    f"""
<span class="badge">📌 Entradas: <b>{fmt_eur(total_in)}</b></span>
<span class="badge">📤 Salidas: <b>{fmt_eur(total_out)}</b></span>
<span class="badge">🧮 Neto (entradas - salidas): <b>{fmt_eur(net)}</b></span>
""",
    unsafe_allow_html=True,
)

if np.isfinite(last_balance_val):
    st.success(f"Según el PDF, tu **saldo final** es: **{fmt_eur(last_balance_val)}**")
else:
    st.warning("No he encontrado un **balance final** fiable en el PDF. Aun así, puedo explicar entradas/salidas.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# KPIs (tarjetas)
# =========================
st.subheader("✅ Resumen fácil (lo esencial)")

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

if simple_mode:
    st.info(
        "Cómo leerlo (en 15 segundos):\n"
        "- **Entradas** = dinero que entra a tu cuenta.\n"
        "- **Salidas** = dinero que sale (tarjeta, comisiones, retiradas…).\n"
        "- **Neto** = entradas − salidas (si es negativo, salió más de lo que entró en este periodo)."
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# TABS (menos abrumador)
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🧭 Vista rápida", "📅 Mes a mes", "📦 Activos", "🔎 Detalles"])

with tab1:
    st.subheader("🧭 Vista rápida (los 3 gráficos que mejor se entienden)")

    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        st.markdown("**1) Mapa del dinero (Sankey)**")
        st.caption("Visualmente: de dónde vino el dinero, pasa por **tu cuenta**, y a dónde se fue.")
        fig_s = sankey_money_flow(by_cat)
        if fig_s is not None:
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.warning("Plotly no está disponible: muestro un resumen en tabla.")
            st.dataframe(by_cat.sort_values(ascending=False).rename("€ neto"), use_container_width=True)

        st.markdown("**2) Los movimientos que más cambiaron tu dinero**")
        st.caption("Top por impacto (entrada o salida grande).")
        big = biggest_moves(txg, n=top_n_moves)
        if big.empty:
            st.info("No hay suficientes movimientos con fecha/importe para listar.")
        else:
            st.dataframe(big, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**3) ¿En qué se fue tu dinero? (solo salidas)**")
        st.caption("Solo cuenta salidas: tarjeta, comisiones, retiradas…")
        fig_d = donut_outflows(by_cat)
        if fig_d is not None:
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            outs = by_cat[by_cat < 0].abs().sort_values(ascending=False)
            if outs.empty:
                st.info("No veo salidas en el PDF.")
            else:
                st.bar_chart(outs)

        st.markdown("**Extra: evolución del saldo**")
        if np.isfinite(last_balance_val):
            st.caption("Esto usa el **balance** que aparece en tu PDF.")
        else:
            st.caption("Esto es un **saldo estimado** sumando entradas/salidas (empieza en 0).")
        fig_b = balance_or_estimated(txg)
        if fig_b is not None:
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("No se pudo generar la evolución del saldo.")

with tab2:
    st.subheader("📅 Mes a mes (para entender tendencias)")
    st.caption("Barra = neto del mes. Línea = acumulado. Es de lo más fácil para ver si vas “a favor” o “en contra”.")

    fig_m, mdf = monthly_net(txg)
    if fig_m is not None:
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        if mdf is None or mdf.empty:
            st.info("No hay suficientes datos para el mes a mes.")
        else:
            st.bar_chart(mdf.set_index("Mes")[["cashflow"]])

    if mdf is not None and not mdf.empty and simple_mode:
        best = mdf.sort_values("cashflow", ascending=False).head(1)
        worst = mdf.sort_values("cashflow", ascending=True).head(1)
        try:
            st.info(
                f"Mes mejor: **{best['Mes'].iloc[0]}** ({fmt_eur(best['cashflow'].iloc[0])}) · "
                f"Mes peor: **{worst['Mes'].iloc[0]}** ({fmt_eur(worst['cashflow'].iloc[0])})"
            )
        except Exception:
            pass

with tab3:
    st.subheader("📦 Si hiciste inversiones: por activo (simple)")
    st.caption(
        "Aquí no calculo valor actual. Solo lo que se ve en el extracto: compras/ventas y **resultado ya cerrado** (vendido)."
    )

    if not show_assets:
        st.info("Activa “Mostrar inversiones por activo” en la barra lateral.")
    else:
        assets = compute_asset_realized_pnl(tx)
        if assets.empty:
            st.info("No veo operaciones de inversión suficientes en este PDF para calcular por activo.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Activos detectados", f"{len(assets)}")
            a2.metric("Ganado/perdido ya cerrado (total)", fmt_eur(assets["Ganado / perdido ya cerrado"].sum()))
            a3.metric("Dinero metido total (compras)", fmt_eur(assets["Dinero metido (compras)"].sum()))

            st.markdown("**Tabla (lo importante)**")
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

            st.markdown("**Top: resultado ya cerrado por activo**")
            top = assets.sort_values("Ganado / perdido ya cerrado", ascending=False).head(10).copy()
            if PLOTLY_OK:
                fig = px.bar(
                    top,
                    x="Ganado / perdido ya cerrado",
                    y="Activo",
                    orientation="h",
                    title="🏅 Top 10 · Ganado/perdido ya cerrado",
                )
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top.set_index("Activo")[["Ganado / perdido ya cerrado"]])

with tab4:
    st.subheader("🔎 Detalles (por si quieres comprobarlo)")
    if show_details:
        st.caption("Tabla parseada del PDF. Si algo no cuadra, aquí vemos qué línea lo provocó.")
        st.dataframe(
            tx[["date", "type", "Categoria", "cashflow", "balance", "isin", "asset", "quantity", "desc"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Activa “Ver tabla completa (detalles)” en la barra lateral si quieres verlo todo.")

# =========================
# DESCARGA
# =========================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.download_button(
    "⬇️ Descargar los datos parseados (CSV)",
    data=tx.to_csv(index=False).encode("utf-8"),
    file_name="trade_republic_extract_parsed.csv",
    mime="text/csv",
)

st.markdown(
    """
### Nota importante
- Esta app explica **lo que pasó en tu cuenta** según el extracto (entradas/salidas y, si existe, balance).
- Para saber **cuánto valen hoy** tus inversiones, haría falta añadir **precios actuales de mercado** (no vienen en este PDF).
"""
)
