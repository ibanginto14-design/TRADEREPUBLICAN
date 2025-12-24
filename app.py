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
.block-container { padding-top: 1.3rem; padding-bottom: 2.0rem; }
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
.kpi .s { font-size: 12px; opacity: 0.72; margin-top: 6px; }

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


def short_desc(s: str, n: int = 80) -> str:
    s = str(s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


# =========================
# ACTIVOS (P&L realizado)
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
            else:  # SELL
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
# GRÁFICOS ÚTILES (sin Sankey)
# =========================
def fig_in_out_big(total_in: float, total_out: float, net: float):
    """Entradas vs Salidas + Neto (simple)."""
    if not PLOTLY_OK:
        return None

    df = pd.DataFrame(
        {"Concepto": ["Entradas", "Salidas", "Neto"], "€": [total_in, total_out, net]}
    )
    fig = px.bar(df, x="Concepto", y="€", title="⚖️ Entradas vs Salidas (y el neto)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_out_by_category(by_cat: pd.Series, top_n: int = 10):
    """Barras horizontales: en qué se fue (solo salidas)."""
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


def fig_in_by_category(by_cat: pd.Series, top_n: int = 10):
    """Barras horizontales: de dónde vino (solo entradas)."""
    if not PLOTLY_OK:
        return None
    ins = by_cat[by_cat > 0].sort_values(ascending=False).head(top_n)
    if ins.empty:
        return None
    df = ins.reset_index()
    df.columns = ["Concepto", "€"]
    fig = px.bar(df, x="€", y="Concepto", orientation="h", title="💰 ¿De dónde vino el dinero? (Top categorías)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_balance_or_estimated(txg: pd.DataFrame):
    """Balance del PDF si existe; si no, saldo estimado acumulando cashflow desde 0."""
    df = txg.dropna(subset=["date"]).sort_values("date").copy()
    if df.empty:
        return None

    if PLOTLY_OK:
        if df["balance"].notna().any():
            d2 = df.dropna(subset=["balance"]).copy()
            fig = px.line(d2, x="date", y="balance", title="📈 Evolución del saldo (balance del PDF)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            return fig
        else:
            d2 = df.dropna(subset=["cashflow"]).copy()
            d2["Saldo estimado (desde 0)"] = d2["cashflow"].cumsum()
            fig = px.line(d2, x="date", y="Saldo estimado (desde 0)", title="📈 Evolución estimada (sumando entradas/salidas)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            return fig

    return None


def fig_monthly_net(txg: pd.DataFrame):
    """Mes a mes: neto del mes + acumulado."""
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


def biggest_moves_table(txg: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Tabla: los movimientos con más impacto."""
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return df
    df["Impacto"] = df["cashflow"].abs()
    df = df.sort_values("Impacto", ascending=False).head(n).copy()
    df["Día"] = df["date"].dt.strftime("%Y-%m-%d")
    df["€ (entrada/salida)"] = df["cashflow"]
    df["Descripción corta"] = df["desc"].apply(lambda x: short_desc(x, 95))
    return df[["Día", "Categoria", "€ (entrada/salida)", "Descripción corta"]]


def fig_timeline_bubbles(txg: pd.DataFrame):
    """Línea de tiempo con burbujas: movimientos grandes resaltan."""
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None
    df["Impacto"] = df["cashflow"].abs()
    # limitar tamaños extremos para que se vea bien
    p95 = np.nanpercentile(df["Impacto"], 95) if df["Impacto"].notna().any() else 1.0
    df["Impacto_clip"] = np.minimum(df["Impacto"], p95)

    fig = px.scatter(
        df,
        x="date",
        y="cashflow",
        size="Impacto_clip",
        hover_data={"Categoria": True, "desc": True, "cashflow": ":.2f", "date": True, "Impacto_clip": False},
        title="🫧 Línea de tiempo: cada punto es un movimiento (puntos grandes = movimientos grandes)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_weekday_pattern(txg: pd.DataFrame):
    """Patrón por día de la semana (salidas)."""
    if not PLOTLY_OK:
        return None
    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None
    df["weekday"] = df["date"].dt.day_name()
    # solo salidas
    out = df[df["cashflow"] < 0].copy()
    if out.empty:
        return None
    out["€"] = -out["cashflow"]
    w = out.groupby("weekday")["€"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    w = w.dropna()
    if w.empty:
        return None

    # etiquetas en español (por claridad)
    map_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    dfw = w.reset_index()
    dfw.columns = ["Día", "€"]
    dfw["Día"] = dfw["Día"].map(map_es)

    fig = px.bar(dfw, x="Día", y="€", title="📆 ¿Qué día de la semana gastas más? (solo salidas)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_calendar_heatmap(txg: pd.DataFrame):
    """
    Heatmap tipo “calendario” (estilo contribuciones): día vs semana.
    Muestra el neto por día (entrada-salida). Muy visual para detectar rachas.
    """
    if not PLOTLY_OK:
        return None

    df = txg.dropna(subset=["date", "cashflow"]).copy()
    if df.empty:
        return None

    daily = df.groupby(df["date"].dt.date)["cashflow"].sum().reset_index()
    daily.columns = ["day", "net"]
    daily["day"] = pd.to_datetime(daily["day"])

    # rango completo
    start = daily["day"].min()
    end = daily["day"].max()
    all_days = pd.date_range(start, end, freq="D")
    full = pd.DataFrame({"day": all_days}).merge(daily, on="day", how="left").fillna({"net": 0.0})

    # semana (col) y día de la semana (fila)
    # week_index: semanas desde el lunes de la primera semana
    first_monday = (full["day"].min() - pd.to_timedelta(full["day"].min().weekday(), unit="D")).normalize()
    full["week"] = ((full["day"] - first_monday).dt.days // 7).astype(int)
    full["dow"] = full["day"].dt.weekday  # 0=lunes
    dow_names = ["L", "M", "X", "J", "V", "S", "D"]

    pivot = full.pivot(index="dow", columns="week", values="net").reindex(range(7))
    pivot.index = [dow_names[i] for i in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            hovertemplate="Semana %{x}<br>Día %{y}<br>Neto %{z:.2f}€<extra></extra>",
        )
    )
    fig.update_layout(
        title="🗺️ Mapa diario (heatmap): neto por día (entradas - salidas)",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Semanas (desde el inicio del extracto)",
        yaxis_title="Día de la semana",
    )
    return fig


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("1) Sube tu PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])

    st.divider()
    st.header("2) Opciones")
    simple_mode = st.toggle("Modo súper simple (recomendado)", value=True)
    show_assets = st.checkbox("Mostrar inversiones por activo (si operaste)", value=True)
    show_details = st.checkbox("Ver tabla completa (detalles)", value=False)

    st.divider()
    top_moves_n = st.slider("Movimientos más grandes a mostrar", 5, 20, 10)
    top_cat_n = st.slider("Top categorías a mostrar", 5, 20, 10)

    st.caption("Si algo no cuadra, activa **detalles** y revisamos el texto exacto del PDF.")


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

# Métricas simples
total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum(skipna=True))
total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum(skipna=True))
net = float(tx["cashflow"].sum(skipna=True))

by_cat = tx.groupby("Categoria")["cashflow"].sum()
metiste = float(by_cat.get("Dinero que metiste", 0.0))
sacaste = float(abs(by_cat.get("Dinero que sacaste", 0.0)))
tarjeta = float(abs(by_cat.get("Gastos con tarjeta", 0.0)))
comisiones = float(abs(by_cat.get("Comisiones", 0.0)))
intereses = float(by_cat.get("Intereses / rentabilidad", 0.0))

last_balance_val = float(tx["balance"].dropna().iloc[-1]) if tx["balance"].notna().any() else float("nan")

# =========================
# CABECERA
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
# KPIs
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
        "Cómo leerlo (rápido):\n"
        "- **Entradas**: dinero que entra.\n"
        "- **Salidas**: dinero que sale (tarjeta, comisiones, retiradas…).\n"
        "- **Neto**: entradas − salidas (si es negativo, salió más de lo que entró en este periodo)."
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# TABS (más claro: solo cosas útiles)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Resumen visual", "📈 Evolución", "🫧 Movimientos", "📆 Patrones", "📦 Activos / 🔎 Detalles"]
)

with tab1:
    st.subheader("📌 Resumen visual (lo que mejor se entiende)")

    cA, cB = st.columns(2, gap="large")

    with cA:
        st.markdown("**1) Entradas vs Salidas**")
        st.caption("La forma más directa de entender si en este periodo tu cuenta fue “a favor” o “en contra”.")
        fig = fig_in_out_big(total_in, total_out, net)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.Series({"Entradas": total_in, "Salidas": total_out, "Neto": net}))

    with cB:
        st.markdown("**2) ¿En qué se fue el dinero?**")
        st.caption("Top categorías de **salida** (barras = más legible que el Sankey).")
        fig = fig_out_by_category(by_cat, top_n=top_cat_n)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            outs = by_cat[by_cat < 0].abs().sort_values(ascending=False).head(top_cat_n)
            if outs.empty:
                st.info("No veo salidas en el PDF.")
            else:
                st.bar_chart(outs)

    st.markdown("**3) ¿De dónde vino el dinero?**")
    st.caption("Top categorías de **entrada** (si casi todo es 'Dinero que metiste', es normal).")
    fig = fig_in_by_category(by_cat, top_n=top_cat_n)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        ins = by_cat[by_cat > 0].sort_values(ascending=False).head(top_cat_n)
        if ins.empty:
            st.info("No veo entradas en el PDF.")
        else:
            st.bar_chart(ins)

with tab2:
    st.subheader("📈 Evolución (cómo fue cambiando con el tiempo)")
    cC, cD = st.columns(2, gap="large")

    with cC:
        st.markdown("**1) Evolución del saldo**")
        st.caption("Si tu PDF trae balance, lo uso. Si no, muestro un saldo estimado acumulando entradas/salidas.")
        fig = fig_balance_or_estimated(txg)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se pudo generar la evolución del saldo.")

    with cD:
        st.markdown("**2) Mes a mes: neto y acumulado**")
        st.caption("Ideal para ver tendencia sin ruido diario.")
        fig, mdf = fig_monthly_net(txg)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            if mdf is None or mdf.empty:
                st.info("No hay suficientes datos para el mes a mes.")
            else:
                st.bar_chart(mdf.set_index("Mes")[["cashflow"]])

        if simple_mode and mdf is not None and not mdf.empty:
            best = mdf.sort_values("cashflow", ascending=False).head(1)
            worst = mdf.sort_values("cashflow", ascending=True).head(1)
            st.info(
                f"Mes mejor: **{best['Mes'].iloc[0]}** ({fmt_eur(best['cashflow'].iloc[0])}) · "
                f"Mes peor: **{worst['Mes'].iloc[0]}** ({fmt_eur(worst['cashflow'].iloc[0])})"
            )

with tab3:
    st.subheader("🫧 Movimientos (para ver picos claramente)")
    st.caption("Puntos grandes = movimientos grandes. Si quieres entender “qué pasó”, esto es clave.")

    fig = fig_timeline_bubbles(txg)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Plotly no está disponible: muestro solo la tabla de movimientos grandes.")

    st.markdown("**Movimientos más grandes**")
    big = biggest_moves_table(txg, n=top_moves_n)
    if big.empty:
        st.info("No hay suficientes movimientos con fecha/importe para listar.")
    else:
        st.dataframe(big, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("📆 Patrones (hábitos que se repiten)")
    cE, cF = st.columns(2, gap="large")

    with cE:
        st.markdown("**1) ¿Qué día gastas más?**")
        st.caption("Solo salidas (tarjeta, comisiones, etc.).")
        fig = fig_weekday_pattern(txg)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes salidas para calcular patrón por día.")

    with cF:
        st.markdown("**2) Mapa diario (heatmap)**")
        st.caption("Neto por día (entradas - salidas). Muy útil para detectar rachas.")
        fig = fig_calendar_heatmap(txg)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes datos diarios para construir el mapa.")

with tab5:
    st.subheader("📦 Activos (si invertiste)")

    if show_assets:
        assets = compute_asset_realized_pnl(tx)
        if assets.empty:
            st.info("No veo operaciones de inversión suficientes en este PDF para calcular por activo.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Activos detectados", f"{len(assets)}")
            a2.metric("Ganado/perdido ya cerrado (total)", fmt_eur(assets["Ganado / perdido ya cerrado"].sum()))
            a3.metric("Dinero metido total (compras)", fmt_eur(assets["Dinero metido (compras)"].sum()))

            st.caption(
                "Aquí no uso precios actuales: solo compras/ventas del extracto. "
                "**Ganado/perdido ya cerrado** = lo que ya vendiste."
            )
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
                fig.update_layout(height=440, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top.set_index("Activo")[["Ganado / perdido ya cerrado"]])
    else:
        st.info("Activa “Mostrar inversiones por activo” en la barra lateral si quieres ver esta parte.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
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
