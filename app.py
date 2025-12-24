import io
import re
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Trade Republic · PDF Analyzer", page_icon="📄", layout="wide")
st.title("📄 Trade Republic · PDF Analyzer (Extracto de cuenta)")
st.caption(
    "Sube un PDF de Trade Republic (extracto de cuenta). "
    "La app parsea 'TRANSACCIONES DE CUENTA' aunque el PDF esté maquetado y NO tenga tablas reales."
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
    """Coge la parte entre 'TRANSACCIONES DE CUENTA' y 'RESUMEN DEL BALANCE/NOTAS...' y limpia headers/footers."""
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
    # Orden importante
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

    # Operar
    if "operar" in t:
        is_sell = bool(re.search(r"\bsell\b|venta|ejecución venta", d))
        side = "SELL" if is_sell else "BUY"
        # SELL entra, BUY sale
        return side, float(+amount if is_sell else -amount)

    # Rentabilidad/Interés -> entrada
    if ("rentabilidad" in t) or ("interés" in t) or ("interest" in t):
        return "NA", float(+amount)

    # Comisiones -> salida
    if "comisión" in t or "comision" in t:
        return "NA", float(-amount)

    # Tarjeta -> salida
    if "transacción con tarjeta" in t or (("transacción" in t) and ("tarjeta" in d)):
        return "NA", float(-amount)

    # Transferencias: heurística
    if "transferencia" in t:
        if any(k in d for k in ["top up", "incoming", "ingreso", "accepted"]):
            return "NA", float(+amount)
        if any(k in d for k in ["payout", "outgoing", "retirada"]):
            return "NA", float(-amount)
        # fallback: entrada
        return "NA", float(+amount)

    # Otros: sin signo fiable => lo dejamos positivo
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
    """
    En operaciones suele ser: "... for ISIN US... <NAME>, quantity: ..."
    """
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

        # Acumular hasta la siguiente fecha
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
                "amount": amount,      # sin signo fiable
                "cashflow": cashflow,  # con signo inferido
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
    """
    XIRR por Newton-Raphson (tasa anual).
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


def compute_asset_pnl_avg_cost(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Ganado/perdido por activo (ISIN) a partir de operaciones "Operar":
    - P&L realizado (ventas vs coste medio)
    - net_qty, avg_cost, buy/sell totals
    Importante: sin precio actual, esto es P&L REALIZADO.
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
                # si el side no está bien, intentamos inferir por desc
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

        net_invested = buy_amt - sell_amt  # >0: dinero neto puesto
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
            "first_trade": g["date"].min(),
            "last_trade": g["date"].max(),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("net_invested", ascending=False).reset_index(drop=True)
    return out


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Subir PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])
    st.divider()
    top_n = st.slider("Top N activos", 5, 30, 12)
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

# Categorización
tx = tx.copy()
tx["category"] = [
    _category(t, d) for t, d in zip(tx["type"].astype(str), tx["desc"].astype(str))
]

# Asegurar numéricos
tx["cashflow"] = pd.to_numeric(tx["cashflow"], errors="coerce")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
tx["balance"] = pd.to_numeric(tx["balance"], errors="coerce")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📌 Resumen", "🧾 Transacciones", "📦 Activos (P&L)", "📅 Mensual"])

with tab1:
    st.subheader("Resumen de cuenta (a partir del PDF)")

    total_in = float(tx.loc[tx["cashflow"] > 0, "cashflow"].sum(skipna=True))
    total_out = float(-tx.loc[tx["cashflow"] < 0, "cashflow"].sum(skipna=True))
    net = float(tx["cashflow"].sum(skipna=True))

    last_balance = tx["balance"].dropna()
    last_balance_val = float(last_balance.iloc[-1]) if not last_balance.empty else float("nan")
    last_date = tx["date"].dropna()
    last_date_val = last_date.iloc[-1] if not last_date.empty else pd.NaT

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entradas (+)", f"{total_in:,.2f} €")
    c2.metric("Salidas (-)", f"{total_out:,.2f} €")
    c3.metric("Neto", f"{net:,.2f} €")
    c4.metric("Balance final (PDF)", f"{last_balance_val:,.2f} €" if np.isfinite(last_balance_val) else "N/A")

    # Neto por categoría
    st.subheader("Neto por categoría")
    by_cat = tx.groupby("category")["cashflow"].sum().sort_values()
    st.bar_chart(by_cat)

    # Serie acumulada
    st.subheader("Evolución (cashflow acumulado)")
    ts = tx.dropna(subset=["date"]).sort_values("date").copy()
    ts["cum_net"] = ts["cashflow"].fillna(0).cumsum()
    st.line_chart(ts.set_index("date")[["cum_net"]])

    # XIRR usando balance final como "valor final"
    st.subheader("XIRR (cashflows + balance final como valor final)")
    st.caption(
        "Como este PDF es un extracto de cuenta, podemos estimar XIRR usando los cashflows "
        "y añadiendo el balance final como flujo positivo final. "
        "Si hay pocos movimientos, puede no ser estable."
    )
    if np.isfinite(last_balance_val) and pd.notna(last_date_val):
        dates_np = ts["date"].values.astype("datetime64[ns]")
        cf_np = ts["cashflow"].fillna(0).values.astype(float)

        # Añadir valor final
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
            tx[["date", "type", "category", "isin", "asset", "quantity", "amount", "cashflow", "balance", "desc"]],
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

    assets = compute_asset_pnl_avg_cost(tx)
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

        # Gráficos top
        top_pnl = assets.sort_values("realized_pnl", ascending=False).head(top_n).set_index("isin")[["realized_pnl"]]
        st.subheader(f"Top {top_n} por P&L realizado")
        st.bar_chart(top_pnl)

        top_invested = assets.sort_values("net_invested", ascending=False).head(top_n).set_index("isin")[["net_invested"]]
        st.subheader(f"Top {top_n} por dinero neto invertido")
        st.bar_chart(top_invested)

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

    # Pivot para chart
    pivot = monthly.pivot(index="month", columns="category", values="cashflow").fillna(0.0).sort_index()
    st.dataframe(pivot, use_container_width=True)

    st.subheader("Barras apiladas (aprox.)")
    st.bar_chart(pivot)

    st.download_button(
        "⬇️ Descargar resumen mensual (CSV)",
        data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="tr_monthly_summary.csv",
        mime="text/csv",
    )
