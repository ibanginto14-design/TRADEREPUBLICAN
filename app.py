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
# CONFIG
# =========================
st.set_page_config(page_title="Mi dinero en Trade Republic (PDF)", page_icon="💶", layout="wide")

st.title("💶 Mi dinero en Trade Republic")
st.caption(
    "Sube tu **Extracto de cuenta (PDF)**. Te explico, con gráficos sencillos, "
    "qué ha pasado con tu dinero: **entradas**, **salidas**, **en qué se fue** y **cómo cambió el saldo**."
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


def compute_asset_realized_pnl(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Por activo (ISIN): muestra lo más entendible:
    - dinero metido en ese activo (compras)
    - dinero recuperado (ventas)
    - ganado/perdido REALIZADO (solo lo ya cerrado con ventas)
    - cantidad que te queda (aprox.) y coste medio (aprox.)
    """
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

        rows.append({
            "ISIN": isin,
            "Activo": asset_name if asset_name else isin,
            "Dinero metido (compras)": buy_amt,
            "Dinero recuperado (ventas)": sell_amt,
            "Ganado / perdido ya cerrado": realized,
            "Cantidad que te queda (aprox.)": pos_qty,
            "Coste medio (aprox.)": avg_cost,
            "Primera operación": g["date"].min(),
            "Última operación": g["date"].max(),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("Dinero metido (compras)", ascending=False).reset_index(drop=True)
    return out


# =========================
# GRÁFICAS (SENCILLAS + EXPLICADAS)
# =========================
def money_story_waterfall(by_cat: pd.Series):
    """
    Waterfall con etiquetas muy simples.
    by_cat: cashflow neto por categoría (positivo entra, negativo sale)
    """
    if not PLOTLY_OK:
        return None

    # Orden lógico para lectura
    order = [
        "Dinero que metiste",
        "Dinero que sacaste",
        "Gastos con tarjeta",
        "Comisiones",
        "Operaciones de inversión",
        "Intereses / rentabilidad",
        "Transferencias",
        "Otros",
    ]
    # asegurar que existan
    vals = {k: float(by_cat.get(k, 0.0)) for k in order}
    x = []
    y = []
    for k in order:
        if abs(vals[k]) > 1e-9:
            x.append(k)
            y.append(vals[k])

    net = float(sum(y))
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(x) + ["total"],
        x=x + ["CAMBIO NETO"],
        y=y + [net],
        connector={"line": {"width": 1}},
    ))
    fig.update_layout(
        title="📌 Historia del dinero (qué suma y qué resta)",
        height=420,
        showlegend=False,
    )
    return fig


def pie_where_money_went(by_cat: pd.Series):
    """Pie solo de SALIDAS (en qué se fue el dinero)."""
    if not PLOTLY_OK:
        return None
    out = by_cat[by_cat < 0].abs()
    out = out[out > 0]
    if out.empty:
        return None
    df = out.reset_index()
    df.columns = ["Concepto", "€"]
    fig = px.pie(df, names="Concepto", values="€", hole=0.55, title="🧾 ¿En qué se fue tu dinero? (solo salidas)")
    fig.update_layout(height=380)
    return fig


def line_balance(tx: pd.DataFrame):
    """Saldo (balance) a lo largo del tiempo (si existe)."""
    if not PLOTLY_OK:
        return None
    df = tx.dropna(subset=["date"]).sort_values("date").copy()
    if not df["balance"].notna().any():
        return None
    fig = px.line(df, x="date", y="balance", title="📈 Cómo cambió tu saldo (balance del PDF)")
    fig.update_layout(height=380)
    return fig


def monthly_net_bar(tx: pd.DataFrame):
    if not PLOTLY_OK:
        return None, None
    df = tx.dropna(subset=["date"]).copy()
    df["Mes"] = df["date"].dt.to_period("M").astype(str)
    m = df.groupby("Mes")["cashflow"].sum().reset_index()
    fig = px.bar(m, x="Mes", y="cashflow", title="📅 Cambio neto por mes (entradas - salidas)")
    fig.update_layout(height=380)
    return fig, m


def top_days(tx: pd.DataFrame, top_n: int = 8):
    """Top días con más salida y más entrada."""
    df = tx.dropna(subset=["date", "cashflow"]).copy()
    df["Día"] = df["date"].dt.date
    daily = df.groupby("Día")["cashflow"].sum().reset_index()
    daily["Día"] = pd.to_datetime(daily["Día"])

    outs = daily.nsmallest(top_n, "cashflow").copy()
    ins = daily.nlargest(top_n, "cashflow").copy()

    if PLOTLY_OK:
        fig_out = px.bar(
            outs.sort_values("cashflow"),
            x="cashflow", y=outs["Día"].dt.strftime("%Y-%m-%d"),
            orientation="h",
            title=f"⬇️ Días con más salida (Top {top_n})",
        )
        fig_in = px.bar(
            ins.sort_values("cashflow"),
            x="cashflow", y=ins["Día"].dt.strftime("%Y-%m-%d"),
            orientation="h",
            title=f"⬆️ Días con más entrada (Top {top_n})",
        )
        fig_out.update_layout(height=360)
        fig_in.update_layout(height=360)
        return fig_out, fig_in, outs, ins

    return None, None, outs, ins


# =========================
# SIDEBAR (simple)
# =========================
with st.sidebar:
    st.header("1) Sube tu PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])

    st.divider()
    st.header("2) Opciones")
    top_days_n = st.slider("Top días (entradas/salidas)", 5, 15, 8)
    show_details = st.checkbox("Ver detalles (tabla completa)", value=False)
    show_assets = st.checkbox("Mostrar sección de activos (si operaste)", value=True)

    st.divider()
    st.caption("Si algo no cuadra, activa 'ver detalles' y revisamos el texto de cada línea.")

if not up:
    st.info("⬅️ Sube tu **extracto de cuenta PDF** para empezar.")
    st.stop()

pdf_bytes = up.getvalue()

# =========================
# PARSE SAFE
# =========================
try:
    tx = parse_tr_pdf_transactions(pdf_bytes)
except Exception as e:
    st.error("No he podido leer el PDF (pero no voy a crashear). Prueba con otro extracto o vuelve a descargarlo.")
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

# Filtrar solo filas con fecha válida para gráficos
txg = tx.dropna(subset=["date"]).sort_values("date").copy()

# =========================
# MÉTRICAS SENCILLAS
# =========================
total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum(skipna=True))
total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum(skipna=True))
net = float(tx["cashflow"].sum(skipna=True))

by_cat = tx.groupby("Categoria")["cashflow"].sum()
metiste = float(by_cat.get("Dinero que metiste", 0.0))
sacaste = float(abs(by_cat.get("Dinero que sacaste", 0.0)))  # como salida suele ser negativa, pero a veces no
tarjeta = float(abs(by_cat.get("Gastos con tarjeta", 0.0)))
comisiones = float(abs(by_cat.get("Comisiones", 0.0)))
intereses = float(by_cat.get("Intereses / rentabilidad", 0.0))

last_balance_val = float(tx["balance"].dropna().iloc[-1]) if tx["balance"].notna().any() else float("nan")

# =========================
# UI PRINCIPAL
# =========================
st.subheader("✅ Resumen fácil (lo esencial)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dinero que metiste", f"{metiste:,.2f} €")
c2.metric("Dinero que sacaste", f"{sacaste:,.2f} €")
c3.metric("Gastos con tarjeta", f"{tarjeta:,.2f} €")
c4.metric("Comisiones", f"{comisiones:,.2f} €")
c5.metric("Intereses/rentabilidad", f"{intereses:,.2f} €")

st.info(
    "Cómo leer esto:\n"
    "- **Dinero que metiste**: ingresos/aportaciones a la cuenta.\n"
    "- **Dinero que sacaste**: retiradas hacia fuera.\n"
    "- **Gastos con tarjeta**: pagos hechos con la tarjeta.\n"
    "- **Comisiones**: costes cobrados.\n"
    "- **Intereses/rentabilidad**: dinero que te han abonado por intereses u otros rendimientos."
)

if np.isfinite(last_balance_val):
    st.success(f"📌 Según el PDF, tu **saldo (balance) final** es: **{last_balance_val:,.2f} €**")
else:
    st.warning("No he encontrado un balance final fiable en el PDF. Aun así, puedo analizar entradas/salidas.")

st.divider()

# =========================
# GRÁFICAS PRINCIPALES (muy entendibles)
# =========================
st.subheader("📊 Gráficos para entender qué pasó con tu dinero")

colA, colB = st.columns(2)

with colA:
    st.markdown("**1) Historia del dinero (qué suma y qué resta)**")
    st.caption("Si una barra sube, **entró dinero**. Si baja, **salió dinero**.")
    fig_w = money_story_waterfall(by_cat)
    if fig_w is not None:
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        # fallback simple
        tmp = by_cat.sort_values()
        st.bar_chart(tmp)

with colB:
    st.markdown("**2) ¿En qué se fue tu dinero? (solo salidas)**")
    st.caption("Este gráfico solo usa las **salidas** (tarjeta, comisiones, etc.).")
    fig_p = pie_where_money_went(by_cat)
    if fig_p is not None:
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        outs = by_cat[by_cat < 0].abs().sort_values(ascending=False)
        st.bar_chart(outs)

st.divider()

colC, colD = st.columns(2)

with colC:
    st.markdown("**3) Cómo cambió tu saldo con el tiempo**")
    st.caption("Esto es el **balance** que aparece en el PDF tras cada movimiento (si está disponible).")
    fig_bal = line_balance(txg)
    if fig_bal is not None:
        st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.warning("Tu PDF no trae una columna de balance consistente para dibujar esta gráfica.")

with colD:
    st.markdown("**4) Cambio neto por mes**")
    st.caption("Barra positiva = ese mes **acabaste con más** dinero neto. Barra negativa = **salió más** de lo que entró.")
    fig_m, monthly_df = monthly_net_bar(txg)
    if fig_m is not None:
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        # fallback
        if monthly_df is not None:
            st.bar_chart(monthly_df.set_index("Mes")[["cashflow"]])

st.divider()

# Top días
st.subheader("🗓️ Días más importantes (para entender “picos”)")
st.caption("Te señalo los días en los que más dinero salió o entró, para que entiendas los cambios grandes.")
fig_out, fig_in, outs_df, ins_df = top_days(txg, top_n=top_days_n)
cE, cF = st.columns(2)
with cE:
    if fig_out is not None:
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.dataframe(outs_df, use_container_width=True, hide_index=True)
with cF:
    if fig_in is not None:
        st.plotly_chart(fig_in, use_container_width=True)
    else:
        st.dataframe(ins_df, use_container_width=True, hide_index=True)

st.divider()

# =========================
# SECCIÓN ACTIVOS (solo si operaste) — en lenguaje llano
# =========================
if show_assets:
    st.subheader("📦 Si hiciste inversiones: qué pasó por activo (muy simple)")
    st.caption(
        "Aquí **no** usamos precios actuales de mercado. "
        "Esto enseña lo que ya está **cerrado** (vendido) y una estimación de lo que te queda, "
        "según compras/ventas del extracto."
    )

    assets = compute_asset_realized_pnl(tx)
    if assets.empty:
        st.info("No veo operaciones de inversión suficientes en este PDF para calcular por activo.")
    else:
        # tarjetas resumen
        a1, a2, a3 = st.columns(3)
        a1.metric("Activos detectados", f"{len(assets)}")
        a2.metric("Ganado/perdido ya cerrado (total)", f"{assets['Ganado / perdido ya cerrado'].sum():,.2f} €")
        a3.metric("Dinero metido total (compras)", f"{assets['Dinero metido (compras)'].sum():,.2f} €")

        st.info(
            "Cómo leerlo:\n"
            "- **Dinero metido (compras)**: lo que has gastado comprando ese activo.\n"
            "- **Dinero recuperado (ventas)**: lo que has cobrado vendiendo.\n"
            "- **Ganado/perdido ya cerrado**: resultado de lo que ya has vendido (esto sí es real, porque ya ocurrió).\n"
            "- **Cantidad que te queda** y **coste medio** son aproximaciones con la info del extracto."
        )

        # Tabla simple
        st.dataframe(
            assets[[
                "Activo", "ISIN",
                "Dinero metido (compras)", "Dinero recuperado (ventas)",
                "Ganado / perdido ya cerrado",
                "Cantidad que te queda (aprox.)", "Coste medio (aprox.)"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # gráfico muy fácil: top P&L realizado
        top = assets.sort_values("Ganado / perdido ya cerrado", ascending=False).head(10).copy()
        if PLOTLY_OK:
            fig = px.bar(
                top,
                x="Ganado / perdido ya cerrado",
                y="Activo",
                orientation="h",
                title="🏅 Top 10 · Ganado/perdido ya cerrado por activo",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top.set_index("Activo")[["Ganado / perdido ya cerrado"]])

st.divider()

# =========================
# DETALLES (para revisar)
# =========================
st.subheader("🔎 Detalles (por si quieres comprobarlo)")
if show_details:
    st.caption("Aquí está la tabla que la app ha parseado del PDF. Si algo no te cuadra, lo miramos desde aquí.")
    st.dataframe(
        tx[["date", "type", "Categoria", "cashflow", "balance", "isin", "asset", "quantity", "desc"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("Activa “Ver detalles” en la barra lateral si quieres ver la tabla completa.")

# Descarga CSV
st.download_button(
    "⬇️ Descargar los datos parseados (CSV)",
    data=tx.to_csv(index=False).encode("utf-8"),
    file_name="trade_republic_extract_parsed.csv",
    mime="text/csv",
)

# Nota final clara
st.markdown("---")
st.markdown(
    "### Nota importante\n"
    "- Esta app explica el **dinero en tu cuenta** a partir del extracto (entradas/salidas y balance del PDF).\n"
    "- Para saber el **valor actual de tus inversiones** (lo que valen hoy), hace falta también precio actual de mercado.\n"
)
