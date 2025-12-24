import io
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Trade Republic · PDF Analyzer", page_icon="📄", layout="wide")
st.title("📄 Trade Republic · PDF Analyzer (sin errores con extractos maquetados)")
st.caption("Sube un PDF de Trade Republic (extracto de cuenta). La app parsea la sección 'TRANSACCIONES DE CUENTA' aunque no sea una tabla real.")


# =========================
# PARSER
# =========================
MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12,
}

DROP_PATTERNS = [
    r"^TRADE REPUBLIC BANK",
    r"^Trade Republic Bank",
    r"^C/ Velazquez",
    r"^28001,",
    r"^NIF",
    r"^www\.traderepublic",
    r"^NIF-IVA",
    r"^Domicilio social",
    r"^Registrada en",
    r"^Directores generales",
    r"^Creado en",
    r"^Página \d+ de \d+",
    r"^RESUMEN DE ESTADO DE CUENTA",
    r"^TRANSACCIONES DE CUENTA$",
    r"^FECHA\s+TIPO\s+DESCRIPCIÓN",
    r"ENTRADA",
    r"SALIDA",
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

    # miles con punto + decimal coma: 1.001,00 -> 1001.00
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
    """Coge solo la parte entre 'TRANSACCIONES DE CUENTA' y 'RESUMEN DEL BALANCE/NOTAS...'."""
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
    """
    Detecta líneas tipo '2024' o '2025 con tarjeta'.
    Devuelve (year, rest).
    """
    m = re.match(r"^\s*(\d{4})\b(?:\s+(.*))?$", line.strip())
    if not m:
        return None
    year = int(m.group(1))
    rest = (m.group(2) or "").strip()
    return year, rest


def _infer_type(desc: str) -> str:
    # orden importante: primero las más largas
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
    low = desc.lower()
    for c in candidates:
        if low.startswith(c.lower()):
            return c
    # fallback: primera palabra
    return desc.split(" ", 1)[0] if desc else "Unknown"


def _infer_cashflow(tx_type: str, desc: str, amount: Optional[float]) -> Optional[float]:
    """
    Devuelve cashflow con signo:
      + entrada (sell/dividend/interés/top up incoming)
      - salida (buy/card/fees/payout outgoing)
    """
    if amount is None or not np.isfinite(amount):
        return None
    if amount < 0:
        return float(amount)

    t = (tx_type or "").lower()
    d = (desc or "").lower()

    if "operar" in t:
        # sell / venta -> entrada ; buy / compra -> salida
        if (" sell" in d) or (" venta" in d) or ("ejecución venta" in d):
            return float(+amount)
        return float(-amount)

    if ("rentabilidad" in t) or ("interés" in t) or ("interest" in t):
        return float(+amount)

    if "comisión" in t or "comision" in t:
        return float(-amount)

    if "transacción" in t:
        return float(-amount)

    if "transferencia" in t:
        # top up / incoming / ingreso -> entrada
        if ("top up" in d) or ("incoming" in d) or ("ingreso" in d) or ("accepted" in d):
            return float(+amount)
        # payout / outgoing -> salida
        if ("payout" in d) or ("outgoing" in d):
            return float(-amount)
        # si no sabemos, asumimos entrada (lo puedes cambiar luego si quieres)
        return float(+amount)

    # fallback neutro
    return float(amount)


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

    # Si justo después hay un número, no hay nombre (ej: "for ISIN US... 21,45 €")
    if re.match(r"^[-+]?\d", after):
        return ""

    # Corta por ", quantity:" o por el primer importe en euros
    name = re.split(r",\s*quantity:|\s+[-+]?\d{1,3}(?:\.\d{3})*(?:,\d{2})\s*€", after)[0].strip()
    return name.strip(", ")


def parse_tr_pdf_transactions(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Parser robusto para extractos de Trade Republic como el tuyo:
    - filas partidas: '10 may' / 'Transferencia ...' / '2024'
    - filas combinadas: '18 dic Transacción' / '...'/ '2025 con tarjeta'
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

        # importes: cogemos los 2 últimos números del registro (importe y balance)
        amts = re.findall(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d{2})", desc)
        amount = _to_float_eu(amts[-2]) if len(amts) >= 2 else (_to_float_eu(amts[-1]) if len(amts) == 1 else None)
        balance = _to_float_eu(amts[-1]) if len(amts) >= 1 else None

        tx_type = _infer_type(desc)
        isin = _extract_isin(desc)
        qty = _extract_quantity(desc)
        asset = _extract_asset_name(desc, isin)

        cashflow = _infer_cashflow(tx_type, desc, amount)

        recs.append(
            {
                "date": date,
                "type": tx_type,
                "desc": desc,
                "isin": isin,
                "asset": asset,
                "quantity": qty,
                "amount": amount,     # importe del movimiento (sin signo fiable)
                "cashflow": cashflow, # importe con signo inferido
                "balance": balance,
            }
        )

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.dropna(subset=["date"], how="all").sort_values("date").reset_index(drop=True)
    return df


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Subir PDF")
    up = st.file_uploader("Extracto Trade Republic (PDF)", type=["pdf"])
    show_raw = st.checkbox("Mostrar tabla completa", value=False)
    show_debug = st.checkbox("Mostrar diagnóstico (por si algo no cuadra)", value=False)
    top_n = st.slider("Top N activos (por cash invertido)", 5, 30, 12)

if not up:
    st.info("⬅️ Sube tu PDF de Trade Republic para empezar.")
    st.stop()

pdf_bytes = up.getvalue()

# =========================
# PARSE SAFE
# =========================
try:
    tx = parse_tr_pdf_transactions(pdf_bytes)
except Exception as e:
    st.error("He fallado parseando el PDF. No voy a crashear la app: revisa que el PDF sea un extracto de cuenta y vuelve a probar.")
    st.exception(e)
    st.stop()

if tx.empty:
    st.error(
        "No he encontrado transacciones parseables dentro del PDF. "
        "Si este PDF NO es un 'Extracto de cuenta' con sección 'TRANSACCIONES DE CUENTA', necesitaré otro documento."
    )
    st.stop()

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📌 Resumen", "🧾 Transacciones", "📦 Activos (por ISIN)"])

with tab1:
    st.subheader("Resumen")
    tx_cf = tx.copy()
    tx_cf["cashflow"] = pd.to_numeric(tx_cf["cashflow"], errors="coerce")
    total_in = float(tx_cf.loc[tx_cf["cashflow"] > 0, "cashflow"].sum())
    total_out = float(-tx_cf.loc[tx_cf["cashflow"] < 0, "cashflow"].sum())
    net = float(tx_cf["cashflow"].sum())

    last_balance = tx_cf["balance"].dropna()
    last_balance_val = float(last_balance.iloc[-1]) if not last_balance.empty else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entradas (+)", f"{total_in:,.2f} €")
    c2.metric("Salidas (-)", f"{total_out:,.2f} €")
    c3.metric("Neto", f"{net:,.2f} €")
    c4.metric("Balance final (según PDF)", f"{last_balance_val:,.2f} €" if np.isfinite(last_balance_val) else "N/A")

    st.caption("Nota: este PDF es un extracto de cuenta. Aquí analizamos flujos y operaciones; la valoración de cartera actual requiere también posiciones/valor de mercado.")

    # Series simples
    tx_cf2 = tx_cf.dropna(subset=["date"]).sort_values("date").copy()
    tx_cf2["cum_net"] = tx_cf2["cashflow"].fillna(0).cumsum()
    st.line_chart(tx_cf2.set_index("date")[["cum_net"]])

    # Por tipo
    by_type = tx_cf2.groupby("type", dropna=False)["cashflow"].sum().sort_values()
    st.subheader("Neto por tipo")
    st.bar_chart(by_type)

with tab2:
    st.subheader("Transacciones (parseadas)")
    if show_raw:
        st.dataframe(tx, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            tx[["date", "type", "isin", "asset", "quantity", "amount", "cashflow", "balance", "desc"]],
            use_container_width=True,
            hide_index=True,
        )

    csv_out = tx.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar transacciones parseadas (CSV)", data=csv_out, file_name="tr_transactions_parsed.csv", mime="text/csv")

    if show_debug:
        st.info("Diagnóstico rápido")
        st.write("Filas:", len(tx))
        st.write("Tipos detectados:", sorted(tx["type"].dropna().unique().tolist()))

with tab3:
    st.subheader("Activos (solo operaciones 'Operar')")
    op = tx[tx["type"].str.lower().eq("operar")].copy()
    if op.empty:
        st.warning("No he encontrado filas de tipo 'Operar' en este PDF.")
    else:
        op["side"] = np.where(
            op["desc"].str.lower().str.contains(r"\bsell\b|venta|ejecución venta", regex=True, na=False),
            "SELL",
            "BUY"
        )

        # Solo filas con ISIN
        op = op[op["isin"].astype(str).str.len() > 0].copy()

        # Agregados
        def agg_asset(g: pd.DataFrame) -> pd.Series:
            buy = g[g["side"] == "BUY"]
            sell = g[g["side"] == "SELL"]

            buy_qty = float(buy["quantity"].fillna(0).sum())
            sell_qty = float(sell["quantity"].fillna(0).sum())
            buy_amt = float(buy["amount"].fillna(0).sum())
            sell_amt = float(sell["amount"].fillna(0).sum())

            net_qty = buy_qty - sell_qty
            net_cash_invested = buy_amt - sell_amt  # >0 = has puesto dinero neto; <0 = has sacado más de lo que pusiste

            avg_buy = (buy_amt / buy_qty) if buy_qty > 0 else np.nan
            avg_sell = (sell_amt / sell_qty) if sell_qty > 0 else np.nan

            # nombre: el más frecuente no vacío
            asset_name = g["asset"].replace("", np.nan).dropna()
            asset_name = asset_name.mode().iloc[0] if not asset_name.empty else ""

            return pd.Series({
                "asset": asset_name,
                "buy_qty": buy_qty,
                "buy_amount": buy_amt,
                "avg_buy": avg_buy,
                "sell_qty": sell_qty,
                "sell_amount": sell_amt,
                "avg_sell": avg_sell,
                "net_qty": net_qty,
                "net_cash_invested": net_cash_invested,
            })

        assets = op.groupby("isin", dropna=False).apply(agg_asset).reset_index()
        assets = assets.sort_values("net_cash_invested", ascending=False).reset_index(drop=True)

        st.dataframe(assets, use_container_width=True, hide_index=True)

        st.subheader(f"Top {top_n} por cash neto invertido")
        top = assets.head(top_n).set_index("isin")[["net_cash_invested"]]
        st.bar_chart(top)

        csv_assets = assets.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar resumen por ISIN (CSV)", data=csv_assets, file_name="tr_assets_by_isin.csv", mime="text/csv")
