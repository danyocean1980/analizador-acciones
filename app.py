import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Optional

# ------------------ CONFIG GENERAL ------------------ #

st.set_page_config(
    page_title="Analizador de Acciones",
    page_icon="📈",
    layout="wide",
)

# ------------------ FUNCIONES AUXILIARES CÁLCULO ------------------ #

def calcular_rsi(series, window: int = 14) -> pd.Series:
    """
    RSI clásico de 14 periodos.
    Aseguramos que la entrada sea una Serie 1D aunque venga como DataFrame/ndarray 2D.
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    else:
        series = pd.Series(series).squeeze()

    series = series.astype(float)

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_macd(series: pd.Series):
    """MACD estándar (12, 26, 9)."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.Series(series).astype(float)

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calcular_max_drawdown(series: pd.Series) -> float:
    """Máxima caída desde máximos (drawdown) en % negativa."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.Series(series)

    if series.empty:
        return np.nan
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    return float(drawdown.min() * 100.0)


def calcular_volatilidades_multiperiodo(returns: pd.Series):
    """
    Volatilidad anualizada en diferentes ventanas usando rentabilidades diarias.
    1M ≈ 21 días, 3M ≈ 63 días, 1A ≈ 252 días.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = pd.Series(returns)

    vols = {}
    for label, window in [("1M", 21), ("3M", 63), ("1A", 252)]:
        if len(returns) > window:
            vol = returns.tail(window).std() * np.sqrt(252) * 100
        else:
            vol = np.nan
        vols[label] = float(vol) if not np.isnan(vol) else np.nan
    return vols


def calcular_beta_vs_indice(price_series, ticker_indice: str = "^GSPC", interval: str = "1d") -> float:
    """
    Calcula la beta de la acción vs un índice (por defecto S&P 500).
    Usa datos del mismo periodo y frecuencia.
    """
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    else:
        price_series = pd.Series(price_series).squeeze()

    try:
        idx_data = yf.download(
            ticker_indice,
            start=price_series.index.min(),
            end=price_series.index.max(),
            interval=interval,
            progress=False,
        )
        if idx_data.empty:
            return np.nan

        stock_ret = price_series.pct_change().dropna()
        idx_ret = idx_data["Close"].pct_change().dropna()

        df = pd.concat([stock_ret, idx_ret], axis=1, join="inner")
        df.columns = ["stock", "index"]
        df = df.dropna()
        if df.empty or df["index"].var() == 0:
            return np.nan

        cov = np.cov(df["stock"], df["index"])[0][1]
        beta = cov / df["index"].var()
        return float(beta)
    except Exception:
        return np.nan


# ------------------ FUNCIONES DATOS FUNDAMENTALES ------------------ #

def obtener_info_basica(ticker_obj: yf.Ticker) -> dict:
    """Datos fundamentales básicos vía yfinance."""
    info = {}
    try:
        fast = ticker_obj.fast_info
        info["currency"] = getattr(fast, "currency", None)
        info["last_price"] = getattr(fast, "last_price", None)
        info["year_high"] = getattr(fast, "year_high", None)
        info["year_low"] = getattr(fast, "year_low", None)
    except Exception:
        pass

    try:
        full_info = ticker_obj.info  # puede ser lento
        info["longName"] = full_info.get("longName")
        info["sector"] = full_info.get("sector")
        info["industry"] = full_info.get("industry")
        info["marketCap"] = full_info.get("marketCap")
        info["trailingPE"] = full_info.get("trailingPE")
        info["forwardPE"] = full_info.get("forwardPE")
        info["dividendYield"] = full_info.get("dividendYield")
        info["grossMargins"] = full_info.get("grossMargins")
        info["operatingMargins"] = full_info.get("operatingMargins")
        info["profitMargins"] = full_info.get("profitMargins")
        info["returnOnEquity"] = full_info.get("returnOnEquity")
        info["returnOnAssets"] = full_info.get("returnOnAssets")
        info["totalDebt"] = full_info.get("totalDebt")
        info["totalCash"] = full_info.get("totalCash")
        info["freeCashflow"] = full_info.get("freeCashflow")
        info["revenueGrowth"] = full_info.get("revenueGrowth")
        info["earningsGrowth"] = full_info.get("earningsGrowth")
        info["sharesOutstanding"] = full_info.get("sharesOutstanding")
    except Exception:
        pass

    return info


def obtener_datos_analistas(ticker_obj: yf.Ticker) -> dict:
    """Intenta sacar precio objetivo y consenso de analistas (si yfinance lo da)."""
    datos = {
        "price_target_mean": None,
        "price_target_low": None,
        "price_target_high": None,
        "num_analysts": None,
        "recommendation_mean": None,
    }

    try:
        pt = getattr(ticker_obj, "analyst_price_targets", None)
        if pt and isinstance(pt, dict):
            datos["price_target_mean"] = pt.get("mean")
            datos["price_target_low"] = pt.get("low")
            datos["price_target_high"] = pt.get("high")
            datos["num_analysts"] = pt.get("numberOfAnalysts") or pt.get("analystCount")
    except Exception:
        pass

    try:
        rec_sum = getattr(ticker_obj, "recommendations_summary", None)
        if rec_sum and isinstance(rec_sum, dict):
            datos["recommendation_mean"] = rec_sum.get("mean")
    except Exception:
        pass

    return datos


# ---------- NOTICIAS + SENTIMIENTO ---------- #

POSITIVE_KEYWORDS = [
    "beat", "beats", "record", "upgrade", "raises guidance",
    "strong", "surge", "soars", "profit jumps"
]
NEGATIVE_KEYWORDS = [
    "miss", "misses", "downgrade", "cuts guidance",
    "weak", "plunge", "falls", "lawsuit", "probe", "regulator"
]


def obtener_noticias(ticker_obj: yf.Ticker, max_n: int = 10) -> list:
    """
    Noticias recientes del ticker (si yfinance las da).
    Devuelve una lista de diccionarios normalizados:
    {title, publisher, link}
    """

    def normalizar_item(item):
        # admite dict u objeto con atributos
        if isinstance(item, dict):
            get = item.get
        else:
            def get(k, default=None):
                return getattr(item, k, default)

        title = (
            get("title")
            or get("headline")
            or get("summary")
            or get("content")
            or ""
        )
        publisher = (
            get("publisher")
            or get("provider")
            or get("source")
            or get("publisher_name")
            or ""
        )
        link = get("link") or get("url") or get("news_url") or ""

        return {
            "title": str(title).strip(),
            "publisher": str(publisher).strip(),
            "link": str(link).strip(),
        }

    try:
        news_raw = ticker_obj.news
        if not isinstance(news_raw, list) or len(news_raw) == 0:
            return []
        normalizadas = [normalizar_item(it) for it in news_raw[:max_n]]
        return normalizadas
    except Exception:
        return []


def clasificar_noticias(news_list: list) -> dict:
    """
    Clasifica noticias en positivas / negativas / neutras según el titular (muy simplificado).
    Espera que cada noticia venga normalizada con keys: title, publisher, link.
    """
    clasificadas = {"positivas": [], "negativas": [], "neutrales": []}
    for item in news_list:
        title = (item.get("title") or "").lower()
        categoria = "neutrales"
        if any(word in title for word in POSITIVE_KEYWORDS):
            categoria = "positivas"
        elif any(word in title for word in NEGATIVE_KEYWORDS):
            categoria = "negativas"
        clasificadas[categoria].append(item)
    return clasificadas


def obtener_eps_hist(ticker_obj: yf.Ticker, info: dict) -> Optional[pd.DataFrame]:
    """
    Intenta aproximar EPS histórico a partir de 'earnings' (beneficio total) y acciones en circulación.
    Normalmente yfinance solo trae 4 años, no 10.
    """
    try:
        earn_df = ticker_obj.earnings  # index: año, columnas: Revenue, Earnings
        if earn_df is None or earn_df.empty:
            return None
        shares = info.get("sharesOutstanding")
        if not shares:
            return None
        eps_series = earn_df["Earnings"] / shares
        eps_df = pd.DataFrame(
            {"Año": eps_series.index.astype(int), "EPS_aprox": eps_series.values}
        )
        return eps_df.tail(10)
    except Exception:
        return None


# ------------------ FUNCIONES FORMATO ------------------ #

def formatear_numero(n, dec: int = 2) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/D"
    return f"{n:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatear_porcentaje(x, dec: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/D"
    return f"{x*100:,.{dec}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def formatear_capitalizacion(market_cap) -> str:
    if market_cap is None or (isinstance(market_cap, float) and np.isnan(market_cap)):
        return "N/D"
    if market_cap >= 1e12:
        return f"{market_cap / 1e12:.2f} Bn"
    elif market_cap >= 1e9:
        return f"{market_cap / 1e9:.2f} B"
    elif market_cap >= 1e6:
        return f"{market_cap / 1e6:.2f} M"
    else:
        return f"{market_cap:,.0f}".replace(",", ".")


def to_float_or_nan(x):
    if x is None:
        return np.nan
    if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)):
        arr = np.asarray(x).flatten()
        if arr.size == 0:
            return np.nan
        x = arr[0]
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def es_numero_valido(x) -> bool:
    x = to_float_or_nan(x)
    return not (isinstance(x, float) and np.isnan(x))


# ------------------ RATING SIMPLE ------------------ #

def generar_rating_simple(price, sma50, sma200, rsi, ret_3m, volatility):
    """Mini-rating automático muy simplificado para tener una señal rápida."""
    price = to_float_or_nan(price)
    sma50 = to_float_or_nan(sma50)
    sma200 = to_float_or_nan(sma200)
    rsi = to_float_or_nan(rsi)
    ret_3m = to_float_or_nan(ret_3m)
    volatility = to_float_or_nan(volatility)

    score = 0
    motivos = []

    if es_numero_valido(sma50) and es_numero_valido(sma200):
        if price > sma50 > sma200:
            score += 2
            motivos.append("Tendencia alcista (precio > SMA50 > SMA200).")
        elif price > sma50 and sma50 <= sma200:
            score += 1
            motivos.append("Precio por encima de SMA50 (ligera tendencia positiva).")
        elif price < sma50 < sma200:
            score -= 2
            motivos.append("Tendencia bajista (precio < SMA50 < SMA200).")
        else:
            motivos.append("Tendencia poco clara según SMA50 y SMA200.")

    if es_numero_valido(rsi):
        if 40 <= rsi <= 60:
            score += 1
            motivos.append(f"RSI neutro ({rsi:.1f}), sin sobrecompra/sobreventa extrema.")
        elif rsi < 30:
            score += 1
            motivos.append(f"RSI en sobreventa ({rsi:.1f}), posible zona de rebote.")
        elif rsi > 70:
            score -= 1
            motivos.append(f"RSI en sobrecompra ({rsi:.1f}), posible corrección.")

    if es_numero_valido(ret_3m):
        if ret_3m > 10:
            score += 1
            motivos.append(f"Buen comportamiento en 3 meses: {ret_3m:.1f}%.")
        elif ret_3m < -10:
            score -= 1
            motivos.append(f"Mal comportamiento en 3 meses: {ret_3m:.1f}%.")

    if es_numero_valido(volatility):
        if volatility > 45:
            score -= 1
            motivos.append(f"Volatilidad muy alta: {volatility:.1f}%.")
        elif volatility < 20:
            score += 1
            motivos.append(f"Volatilidad moderada/baja: {volatility:.1f}%.")

    if score >= 3:
        label = "✅ Compra potencial (para seguir analizando)."
        color = "#22c55e"
    elif 1 <= score < 3:
        label = "🟡 Interesante pero con cautela."
        color = "#eab308"
    elif -1 <= score < 1:
        label = "⚪ Neutro (ni claro comprar ni vender)."
        color = "#9ca3af"
    else:
        label = "❌ Riesgo elevado / poco atractiva."
        color = "#f97373"

    return label, color, motivos


def map_quick_action(label_rating: str) -> str:
    if "Compra potencial" in label_rating:
        return "➡️ Señal rápida: **COMPRAR (para estudiar más / entrada parcial)**."
    if "Interesante pero con cautela" in label_rating:
        return "➡️ Señal rápida: **MANTENER / COMPRAR UN POCO** si encaja en tu plan."
    if "Neutro" in label_rating:
        return "➡️ Señal rápida: **MANTENER / OBSERVAR**, sin prisa por entrar."
    if "Riesgo elevado" in label_rating or "poco atractiva" in label_rating:
        return "➡️ Señal rápida: **EVITAR / REDUCIR POSICIÓN** según tu situación."
    return "➡️ Señal rápida: **NEUTRO**."


# ------------------ INFORME 1–8 ------------------ #

def construir_informe_estructurado(
    ticker: str,
    info: dict,
    datos_analistas: dict,
    noticias_clas: dict,
    eps_df: Optional[pd.DataFrame],
    price,
    sma50,
    sma200,
    rsi_last,
    ret_3m,
    volatility,
    max_dd,
    vols_multiperiodo: dict,
    beta,
) -> str:
    price = to_float_or_nan(price)
    sma50 = to_float_or_nan(sma50)
    sma200 = to_float_or_nan(sma200)
    rsi_last = to_float_or_nan(rsi_last)
    ret_3m = to_float_or_nan(ret_3m)
    volatility = to_float_or_nan(volatility)
    max_dd = to_float_or_nan(max_dd)
    beta = to_float_or_nan(beta)

    nombre = info.get("longName") or ticker
    sector = info.get("sector") or "N/D"
    industry = info.get("industry") or "N/D"

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    dy = info.get("dividendYield")
    roe = info.get("returnOnEquity")
    roic = info.get("returnOnAssets")
    gross_m = info.get("grossMargins")
    op_m = info.get("operatingMargins")
    net_m = info.get("profitMargins")
    debt = info.get("totalDebt")
    cash = info.get("totalCash")
    fcf = info.get("freeCashflow")
    rev_g = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")

    pt_mean = datos_analistas.get("price_target_mean")
    pt_low = datos_analistas.get("price_target_low")
    pt_high = datos_analistas.get("price_target_high")
    num_analysts = datos_analistas.get("num_analysts")
    rec_mean = datos_analistas.get("recommendation_mean")

    ratio_price_pt = None
    if es_numero_valido(pt_mean) and es_numero_valido(price):
        ratio_price_pt = price / to_float_or_nan(pt_mean)

    informe = []

    informe.append("# Informe de análisis\n")
    informe.append("**Actúo como un analista financiero independiente y gestor de carteras profesional.**")
    informe.append(
        f"Análisis solicitado para **{nombre} ({ticker})**, sector **{sector}**, industria **{industry}**.\n"
    )

    # 1. Dirección probable
    informe.append("## 1. Dirección probable de la cotización")
    escenario_principal = "base"
    if es_numero_valido(sma50) and es_numero_valido(sma200) and es_numero_valido(price):
        if price > sma50 > sma200 and es_numero_valido(ret_3m) and ret_3m > 0:
            escenario_principal = "alcista"
        elif price < sma50 < sma200 and es_numero_valido(ret_3m) and ret_3m < 0:
            escenario_principal = "bajista"

    informe.append(
        f"- En el **corto/medio plazo**, el escenario principal parece **{escenario_principal}** "
        "basado en tendencia, momentum y volatilidad."
    )
    informe.append("- **Escenario alcista:** continuidad de tendencia alcista con resultados y noticias favorables.")
    informe.append("- **Escenario base:** lateralidad con volatilidad moderada y sin catalizadores claros.")
    informe.append("- **Escenario bajista:** ruptura de soportes, revisiones negativas y deterioro macro/sector.")
    informe.append("- **Riesgos:** cambios en tipos, resultados peores de lo esperado, regulación o problemas de ejecución.")
    informe.append(
        f"- **Riesgo cuantitativo (ejemplo):** volatilidad anualizada ≈ {volatility:.1f}% "
        f"y drawdown máximo reciente ≈ {max_dd:.1f}% (negativo)."
    )

    # 2. Fundamentales
    informe.append("## 2. Análisis de fundamentales")
    informe.append(f"- **PER actual (trailing):** {formatear_numero(trailing_pe, 1)}")
    informe.append(f"- **PER futuro (forward):** {formatear_numero(forward_pe, 1)}")
    informe.append(f"- **Crec. ingresos (último dato):** {formatear_porcentaje(rev_g) if es_numero_valido(rev_g) else 'N/D'}")
    informe.append(f"- **Crec. beneficios (último dato):** {formatear_porcentaje(earn_g) if es_numero_valido(earn_g) else 'N/D'}")
    informe.append(f"- **Margen bruto:** {formatear_porcentaje(gross_m) if es_numero_valido(gross_m) else 'N/D'}")
    informe.append(f"- **Margen operativo:** {formatear_porcentaje(op_m) if es_numero_valido(op_m) else 'N/D'}")
    informe.append(f"- **Margen neto:** {formatear_porcentaje(net_m) if es_numero_valido(net_m) else 'N/D'}")
    informe.append(f"- **ROE:** {formatear_porcentaje(roe) if es_numero_valido(roe) else 'N/D'}")
    informe.append(f"- **ROA/ROIC aprox.:** {formatear_porcentaje(roic) if es_numero_valido(roic) else 'N/D'}")
    informe.append(f"- **Deuda total:** {formatear_capitalizacion(debt)}")
    informe.append(f"- **Caja total:** {formatear_capitalizacion(cash)}")
    informe.append(f"- **Free cash flow:** {formatear_capitalizacion(fcf)}")
    informe.append(f"- **Rentabilidad por dividendo:** {formatear_porcentaje(dy) if es_numero_valido(dy) else 'N/D'}")

    valoracion_texto = "difícil de evaluar solo con un dato"
    if es_numero_valido(trailing_pe):
        pe_val = to_float_or_nan(trailing_pe)
        if pe_val < 15:
            valoracion_texto = "aparentemente **barata**."
        elif 15 <= pe_val <= 25:
            valoracion_texto = "en zona de valoración **razonable**."
        else:
            valoracion_texto = "más bien **cara**, probablemente por expectativas de crecimiento o calidad percibida."
    informe.append(
        f"\nEn conjunto, sin comparar en detalle con el sector ni con su propio histórico, la acción parece {valoracion_texto}"
    )

    # 3. Noticias
    informe.append("## 3. Noticias recientes")
    tot_news = sum(len(v) for v in noticias_clas.values())
    if tot_news == 0:
        informe.append("- No se han podido obtener noticias recientes en esta fuente.")
    else:
        informe.append(f"- Se han encontrado **{tot_news} noticias**. Clasificación aproximada:")
        for categoria, lst in noticias_clas.items():
            if not lst:
                continue
            etiqueta = {
                "positivas": "✅ Positivas",
                "negativas": "❌ Negativas",
                "neutrales": "⚪ Neutrales",
            }[categoria]
            informe.append(f"  - {etiqueta}:")
            for n in lst[:5]:
                titulo = (n.get("title") or "Sin título").strip()
                fuente = (n.get("publisher") or "").strip()
                link = (n.get("link") or "").strip()

                texto_base = titulo
                if fuente:
                    texto_base += f" ({fuente})"

                if link:
                    informe.append(f"    - {texto_base} – {link}")
                else:
                    informe.append(f"    - {texto_base}")
        informe.append(
            "\nEstas noticias afectan al **sentimiento** de corto plazo, especialmente resultados, regulación, fusiones o cambios estratégicos."
        )

    # 4. Consenso + técnico
    informe.append("## 4. Consenso de analistas y análisis técnico")
    informe.append(
        f"- **Recomendación media (numérica):** {formatear_numero(rec_mean, 2)} "
        "(≈1 fuerte compra, 5 fuerte venta)."
    )
    informe.append(f"- **Precio objetivo medio:** {formatear_numero(pt_mean) if es_numero_valido(pt_mean) else 'N/D'}")
    informe.append(
        f"- **Rango objetivos (bajo-alto):** "
        f"{formatear_numero(pt_low) if es_numero_valido(pt_low) else 'N/D'} – "
        f"{formatear_numero(pt_high) if es_numero_valido(pt_high) else 'N/D'}"
    )
    informe.append(f"- **Nº analistas aprox.:** {int(num_analysts) if num_analysts else 'N/D'}")
    informe.append(f"- **Precio actual:** {formatear_numero(price)} {info.get('currency') or ''}")
    if ratio_price_pt is not None:
        if ratio_price_pt < 0.9:
            informe.append("- Cotiza **por debajo** del precio objetivo medio → consenso ve potencial alcista.")
        elif 0.9 <= ratio_price_pt <= 1.1:
            informe.append("- Cotiza **cerca** del precio objetivo medio → consenso ve recorrido limitado.")
        else:
            informe.append("- Cotiza **por encima** del precio objetivo medio → mercado podría estar demasiado optimista.")

    informe.append(
        "- A nivel técnico se valoran tendencia (SMA50/SMA200), RSI y MACD para identificar soporte/resistencias dinámicos "
        "y posibles zonas de agotamiento."
    )
    informe.append(
        f"- **Riesgo de mercado (beta):** β ≈ {formatear_numero(beta,2)} frente al índice de referencia (aprox. S&P 500)."
    )

    # 5. Opinión general
    informe.append("## 5. Opinión general del mercado")
    informe.append(
        "- El sentimiento viene de la combinación de resultados, noticias, precio relativo vs índices y cambios de recomendación."
    )
    informe.append(
        "- Sin datos específicos de posiciones cortas/institucionales en esta app, se toma como referencia el consenso y el comportamiento relativo."
    )
    informe.append(
        "- La lectura final debe adaptarse al perfil de riesgo y horizonte del inversor."
    )

    # 6. EPS
    informe.append("## 6. Evolución del BPA (EPS) últimos 10 años")
    if eps_df is None or eps_df.empty:
        informe.append("- No hay serie larga de EPS disponible en esta fuente; conviene consultar informes anuales.")
    else:
        informe.append(
            "- Se muestra una aproximación de la evolución del EPS. Una tendencia creciente y estable suele respaldar una tesis de largo plazo."
        )

    # 7. Competidores
    informe.append("## 7. Competidores y potencial relativo")
    informe.append(f"- {nombre} compite en el sector **{sector}**, industria **{industry}**.")
    informe.append(
        "- Para un análisis profesional completo, se compararía con 3–5 competidores en crecimiento, márgenes, ROIC, endeudamiento y valoración."
    )
    informe.append(
        "- La pregunta clave: “¿Por qué prefiero esta empresa frente a su mejor competidor directo?”."
    )

    # 8. Conclusión
    informe.append("## 8. Conclusión")
    informe.append(
        "- **Tesis resumida:** la combinación de crecimiento, calidad de márgenes, solidez del balance y precio pagado determinará el retorno."
    )
    informe.append(
        "- **Puntos a favor (genéricos, a concretar):**\n"
        "  1. Potencial de crecimiento y/o buena rentabilidad del capital.\n"
        "  2. Márgenes razonables y generación de caja positiva.\n"
        "  3. Posible infravaloración si el precio no refleja plenamente la capacidad de beneficios futura.\n"
    )
    informe.append(
        "- **Riesgos principales:**\n"
        "  1. Deterioro del crecimiento o márgenes.\n"
        "  2. Riesgos regulatorios/tecnológicos/competitivos.\n"
        "  3. Sobrevaloración que limite el retorno incluso si el negocio va bien.\n"
    )
    informe.append(
        "- **Tipo de inversor:** más apta para perfiles que aceptan volatilidad y horizonte medio/largo plazo.\n"
        "- **Aviso:** análisis informativo y educativo, no es recomendación de inversión personalizada."
    )

    return "\n\n".join(informe)


# ------------------ CHECKLIST PRO CON EXPLICACIONES ------------------ #

def construir_checklist_pro(nombre: str, ticker: str, sector: str, industry: str) -> str:
    return f"""
## 📋 Checklist PRO para un análisis muy fiable de **{nombre} ({ticker})**

1. **Marco del análisis**
   - Qué definir: horizonte temporal (corto/medio/largo) y tipo de tesis (crecimiento, valor, defensiva, turnaround).
   - En qué fijarte: que la empresa encaje con tu objetivo (no tratar un valor cíclico como si fuera defensivo).

2. **Entendimiento del negocio**
   - ¿Cómo gana dinero {nombre}? Principales líneas de ingresos y peso de cada una.
   - Segmentos de negocio y presencia geográfica.
   - Ventaja competitiva (moat): marca, costes bajos, regulación, patentes, efecto red…
   - En qué fijarte: que tengas una explicación sencilla y clara del negocio en 3–4 frases. Si no lo entiendes, no inviertas.

3. **Calidad del management y gobierno corporativo**
   - Historial del equipo directivo: ¿cumple objetivos? ¿sorpresas negativas recurrentes?
   - Uso del capital: recompras, dividendos, inversiones, adquisiciones.
   - Estructura de incentivos y posibles conflictos de interés.
   - En qué fijarte: que el equipo haya creado valor de forma consistente y esté alineado con los accionistas.

4. **Cuenta de resultados (crecimiento y márgenes)**
   - Evolución de ingresos y beneficios (idealmente 5–10 años).
   - Márgenes bruto, operativo y neto: nivel y estabilidad.
   - En qué fijarte: crecimiento sano con márgenes estables o en mejora, no crecimiento a base de recortar márgenes.

5. **Balance (solvencia y liquidez)**
   - Deuda neta/EBITDA, calendario de vencimientos y tipo de interés.
   - Caja disponible, líneas de crédito, ratio corriente.
   - Peso de goodwill e intangibles.
   - En qué fijarte: que la empresa pueda atravesar una recesión sin necesidad urgente de refinanciar deuda a cualquier precio.

6. **Flujo de caja (calidad de beneficios)**
   - Free Cash Flow (FCF) y tendencia.
   - Conversión de beneficio neto en caja a lo largo de varios años.
   - CAPEX de mantenimiento vs CAPEX de crecimiento.
   - En qué fijarte: que el beneficio contable esté respaldado por caja real, y no solo por ajustes contables.

7. **Rentabilidad del capital (ROE, ROIC, ROA)**
   - ROE y ROIC comparados con el coste de capital.
   - Estabilidad de estas métricas en el tiempo.
   - En qué fijarte: empresas que de forma consistente ganan más de lo que les cuesta el capital son creadoras de valor.

8. **Valoración (múltiplos y valor intrínseco)**
   - Múltiplos: PER, EV/EBITDA, P/FCF, PEG, etc.
   - Comparar con: histórico de {nombre} y competidores de {sector}/{industry}.
   - Estimación de valor intrínseco con escenarios (bajista/base/alcista).
   - En qué fijarte: que no dependas de supuestos hiper optimistas para justificar el precio actual.

9. **Riesgos de negocio, financieros y de valoración**
   - Negocio: concentración de clientes, disrupción tecnológica, materias primas, regulación.
   - Financieros: deuda alta, riesgo de divisa, sensibilidad a tipos.
   - Valoración: múltiplos muy exigentes o expectativas irreales.
   - En qué fijarte: que identifiques 3–5 riesgos que podrían romper la tesis y cómo los vigilarás.

10. **Contexto y ciclo económico/sectorial**
    - Fase del ciclo del sector (expansión, madurez, contracción).
    - Sensibilidad a recesiones, tipos de interés, inflación.
    - Regulación actual y posibles cambios relevantes.
    - En qué fijarte: que el momento del ciclo no esté en el pico de euforia justo cuando compras.

11. **Comparación con competidores**
    - Identificar 3–5 comparables directos.
    - Comparar crecimiento, márgenes, ROE/ROIC, deuda y múltiplos.
    - En qué fijarte: que tengas una razón clara para preferir {nombre} y no otro competidor mejor y/o más barato.

12. **Riesgo y comportamiento histórico del precio**
    - Volatilidad histórica a 1M, 3M, 1A.
    - Máx. drawdown (caída desde máximos) y velocidad de recuperación.
    - Beta vs índice de referencia (riesgo de mercado).
    - En qué fijarte: si encaja con tu tolerancia a la volatilidad y con el resto de tu cartera.

13. **Análisis técnico y timing**
    - Tendencia principal (alcista, lateral, bajista).
    - Soportes y resistencias relevantes.
    - RSI, MACD, medias móviles (50, 200) como apoyo.
    - En qué fijarte: usar el técnico para mejorar entrada/salida, no como única razón para invertir.

14. **Tesis final y plan de acción**
    - Tesis de inversión en 3–5 frases.
    - 3–5 puntos a favor muy claros.
    - 3–5 riesgos clave que romperían la tesis.
    - Precio/rango de entrada razonable, horizonte temporal y plan de salida.
    - En qué fijarte: que el plan sea coherente con tu perfil y que tengas condiciones claras para vender si la tesis cambia.

15. **Encaje en tu cartera**
    - Peso objetivo de {nombre} como % de la cartera.
    - Correlación aproximada con tus principales posiciones.
    - Rol: defensa, crecimiento, dividendo, apuesta temática, etc.
    - En qué fijarte: que no concentres demasiado riesgo en el mismo tipo de activo o sector.

> Recorre esta checklist al analizar {nombre}: te obliga a mirar tanto números como contexto, gestión, riesgo y encaje en tu propia cartera.
"""


# ------------------ EXPORTACIÓN (MD + PDF) ------------------ #

def construir_markdown_completo(
    informe_md: str,
    checklist_md: str,
    eps_df: Optional[pd.DataFrame],
) -> str:
    partes = [informe_md.strip(), "\n\n---\n", checklist_md.strip()]
    if eps_df is not None and not eps_df.empty:
        partes.append("\n\n### Tabla EPS aproximado\n")
        partes.append(eps_df.to_markdown(index=False))
    return "\n".join(partes)


def generar_pdf_desde_texto(texto: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 40
    y_margin = 40
    max_width_chars = 100

    y = height - y_margin
    text_lines = texto.split("\n")

    for line in text_lines:
        wrapped = textwrap.wrap(line, max_width_chars) or [""]
        for subline in wrapped:
            if y <= y_margin:
                c.showPage()
                y = height - y_margin
            c.drawString(x_margin, y, subline)
            y -= 14

    c.save()
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


# ------------------ INTERFAZ STREAMLIT (ANÁLISIS POR TICKER) ------------------ #

st.markdown("## 📈 Analizador PRO de Acciones (educativo)")
st.caption(
    "Actúo como un **analista financiero independiente y gestor de carteras profesional virtual**. "
    "Analizaré la compañía que me indiques usando datos accesibles (por ejemplo, vía Yahoo Finance). "
    "La información es orientativa y no es una recomendación personalizada."
)

with st.sidebar:
    st.markdown("### ⚙️ Parámetros de análisis por ticker")
    tickers_input = st.text_input(
        "Tickers (separados por comas):",
        value="AAPL, MSFT",
        help="Ejemplo: AAPL, MSFT, TSLA, NVDA",
    )

    period = st.selectbox(
        "Periodo histórico:",
        ["6mo", "1y", "2y", "5y", "10y"],
        index=1,
    )

    interval = st.radio(
        label="Frecuencia de datos:",
        options=["1d", "1wk"],
        index=0,
        help="1d = diario, 1wk = semanal.",
    )

    show_sma20 = st.checkbox("Mostrar SMA 20", value=True)
    show_sma50 = st.checkbox("Mostrar SMA 50", value=True)
    show_sma200 = st.checkbox("Mostrar SMA 200", value=True)

    st.markdown("---")
    show_rsi = st.checkbox("Mostrar RSI 14", value=True)
    show_macd = st.checkbox("Mostrar MACD", value=True)

    st.markdown("---")
    analizar = st.button("🔍 Analizar acciones")


if not analizar:
    st.info("Introduce los tickers en la barra lateral y pulsa **“Analizar acciones”**.")
else:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not tickers:
        st.error("Por favor, introduce al menos un ticker válido.")
    else:
        for ticker in tickers:
            st.markdown("---")
            st.markdown(f"### {ticker}")

            try:
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                )

                if raw.empty:
                    st.warning(f"No se han encontrado datos para {ticker}.")
                    continue

                # ---- Normalizar a DataFrame sencillo con una sola columna Close ----
                if isinstance(raw.columns, pd.MultiIndex):
                    if ('Close', ticker) in raw.columns:
                        close = raw[('Close', ticker)]
                    else:
                        close_cols = [c for c in raw.columns if c[0] == 'Close']
                        if not close_cols:
                            st.error(f"No se ha encontrado columna 'Close' para {ticker}.")
                            continue
                        close = raw[close_cols[0]]
                else:
                    close = raw["Close"]

                data = pd.DataFrame(index=raw.index)
                data["Close"] = pd.Series(close).astype(float)

                # Indicadores de precio
                data["SMA20"] = data["Close"].rolling(window=20).mean()
                data["SMA50"] = data["Close"].rolling(window=50).mean()
                data["SMA200"] = data["Close"].rolling(window=200).mean()
                data["RSI14"] = calcular_rsi(data["Close"])
                macd, macd_signal, macd_hist = calcular_macd(data["Close"])
                data["MACD"] = macd
                data["MACD_signal"] = macd_signal
                data["MACD_hist"] = macd_hist

                clean = data.dropna()
                last_row = clean.iloc[-1]

                current_price = float(last_row["Close"])
                sma50 = float(last_row.get("SMA50", np.nan))
                sma200 = float(last_row.get("SMA200", np.nan))
                rsi_last = float(last_row.get("RSI14", np.nan))

                # Rentabilidad 3 meses (aprox)
                lookback_bars = 63 if interval == "1d" else 12
                if len(clean) > lookback_bars:
                    price_past = clean["Close"].iloc[-lookback_bars]
                    ret_3m = (current_price / price_past - 1) * 100
                else:
                    ret_3m = np.nan

                # Volatilidad y riesgo
                returns = data["Close"].pct_change().dropna()
                if not returns.empty:
                    volatility = returns.std() * np.sqrt(252) * 100
                    vols_multiperiodo = calcular_volatilidades_multiperiodo(returns)
                else:
                    volatility = np.nan
                    vols_multiperiodo = {"1M": np.nan, "3M": np.nan, "1A": np.nan}

                max_dd = calcular_max_drawdown(clean["Close"])
                beta = calcular_beta_vs_indice(clean["Close"], interval=interval)

                # Datos avanzados
                yt = yf.Ticker(ticker)
                info = obtener_info_basica(yt)
                datos_analistas = obtener_datos_analistas(yt)
                news_list = obtener_noticias(yt)
                noticias_clas = clasificar_noticias(news_list)
                eps_df = obtener_eps_hist(yt, info)

                label_rating, color_rating, motivos_rating = generar_rating_simple(
                    current_price, sma50, sma200, rsi_last, ret_3m, volatility
                )
                quick_action = map_quick_action(label_rating)

                nombre = info.get("longName") or ticker
                sector = info.get("sector") or "N/D"
                industry = info.get("industry") or "N/D"

                informe_md = construir_informe_estructurado(
                    ticker=ticker,
                    info=info,
                    datos_analistas=datos_analistas,
                    noticias_clas=noticias_clas,
                    eps_df=eps_df,
                    price=current_price,
                    sma50=sma50,
                    sma200=sma200,
                    rsi_last=rsi_last,
                    ret_3m=ret_3m,
                    volatility=volatility,
                    max_dd=max_dd,
                    vols_multiperiodo=vols_multiperiodo,
                    beta=beta,
                )
                checklist_md = construir_checklist_pro(nombre, ticker, sector, industry)
                markdown_completo = construir_markdown_completo(
                    informe_md, checklist_md, eps_df
                )
                pdf_bytes = generar_pdf_desde_texto(markdown_completo)

                tab_resumen, tab_grafico, tab_indicadores, tab_informe, tab_checklist = st.tabs(
                    [
                        "📌 Resumen",
                        "📉 Gráfico precio",
                        "📊 Indicadores / Riesgo",
                        "🧠 Informe detallado (1–8)",
                        "📋 Checklist PRO + exportar",
                    ]
                )

                # -------- RESUMEN -------- #
                with tab_resumen:
                    col1, col2, col3 = st.columns([2, 2, 2])

                    with col1:
                        st.markdown("**Datos del precio**")
                        st.metric(
                            "Precio actual",
                            f"{formatear_numero(current_price)} {info.get('currency', '')}",
                        )
                        if es_numero_valido(sma50):
                            st.write(f"SMA 50: {formatear_numero(sma50)}")
                        if es_numero_valido(sma200):
                            st.write(f"SMA 200: {formatear_numero(sma200)}")
                        if es_numero_valido(ret_3m):
                            st.write(f"Rentabilidad aprox. 3 meses: {ret_3m:.1f} %")

                    with col2:
                        st.markdown("**Perfil de la empresa**")
                        st.write(f"Nombre: {nombre}")
                        st.write(f"Sector: {sector}")
                        st.write(f"Industria: {industry}")
                        st.write(
                            f"Capitalización: {formatear_capitalizacion(info.get('marketCap'))}"
                        )
                        pe = info.get("trailingPE")
                        if es_numero_valido(pe):
                            st.write(f"PER (trailing): {to_float_or_nan(pe):.1f}")
                        else:
                            st.write("PER (trailing): N/D")
                        dy = info.get("dividendYield")
                        if es_numero_valido(dy):
                            st.write(f"Rentabilidad por dividendo: {to_float_or_nan(dy)*100:.2f} %")
                        else:
                            st.write("Rentabilidad por dividendo: N/D")

                    with col3:
                        st.markdown("**Rating automático (no profesional)**")
                        st.markdown(
                            f"<div style='padding:0.8rem;border-radius:0.5rem;"
                            f"border:1px solid {color_rating};color:{color_rating};'>"
                            f"{label_rating}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Motivos principales:**")
                        for m in motivos_rating:
                            st.write(f"- {m}")
                        st.markdown("---")
                        st.markdown(quick_action)
                        st.caption(
                            "⚠️ Modelo simplificado. No sustituye un análisis profesional "
                            "ni tiene en cuenta tu perfil de riesgo."
                        )

                # -------- GRÁFICO -------- #
                with tab_grafico:
                    st.markdown("#### Evolución del precio y medias móviles")
                    cols_plot = ["Close"]
                    if show_sma20:
                        cols_plot.append("SMA20")
                    if show_sma50:
                        cols_plot.append("SMA50")
                    if show_sma200:
                        cols_plot.append("SMA200")
                    st.line_chart(clean[cols_plot])
                    st.caption("En qué fijarte: si el precio respeta medias largas (50/200) o las pierde con fuerza.")

                # -------- INDICADORES / RIESGO -------- #
                with tab_indicadores:
                    col_i1, col_i2 = st.columns(2)

                    with col_i1:
                        if show_rsi:
                            st.markdown("**RSI 14**")
                            st.line_chart(clean["RSI14"])
                            if es_numero_valido(rsi_last):
                                st.write(f"RSI actual: {rsi_last:.1f}")
                            st.caption("En qué fijarte: RSI < 30 = sobreventa (posibles rebotes); RSI > 70 = sobrecompra.")

                        st.markdown("---")
                        st.markdown("**Volatilidad y riesgo**")
                        st.write(f"Volatilidad anualizada (histórica): {formatear_numero(volatility,1)} %")
                        st.write(
                            f"Vol 1M / 3M / 1A: "
                            f"{formatear_numero(vols_multiperiodo['1M'],1)} % / "
                            f"{formatear_numero(vols_multiperiodo['3M'],1)} % / "
                            f"{formatear_numero(vols_multiperiodo['1A'],1)} %"
                        )
                        st.write(f"Máx. drawdown reciente: {formatear_numero(max_dd,1)} %")
                        st.write(f"Beta vs índice (aprox.): {formatear_numero(beta,2)}")
                        st.caption(
                            "En qué fijarte: si la volatilidad, el drawdown típico y la beta son compatibles "
                            "con tu estómago y el resto de tu cartera."
                        )

                    with col_i2:
                        if show_macd:
                            st.markdown("**MACD (12, 26, 9)**")
                            st.line_chart(clean[["MACD", "MACD_signal"]])
                            st.caption("En qué fijarte: cruces al alza del MACD sobre la señal pueden indicar cambio hacia tendencia alcista.")

                        st.markdown("---")
                        st.markdown("**Comentario técnico básico**")
                        st.write(
                            "- Observa si el precio está por encima o por debajo de SMA50/200.\n"
                            "- Fíjate en zonas donde históricamente el precio rebota (soportes) o se frena (resistencias).\n"
                            "- Usa el técnico para afinar la entrada/salida, no como única razón para invertir."
                        )

                # -------- INFORME 1–8 -------- #
                with tab_informe:
                    st.markdown("### Informe estructurado en 8 puntos")
                    st.markdown(informe_md)

                    if eps_df is not None and not eps_df.empty:
                        st.markdown("**Tabla de EPS aproximado (años recientes):**")
                        st.dataframe(eps_df, hide_index=True)
                        st.markdown("**Gráfico EPS aproximado:**")
                        st.line_chart(eps_df.set_index("Año")["EPS_aprox"])

                # -------- CHECKLIST + EXPORT -------- #
                with tab_checklist:
                    st.markdown("### Checklist PRO para tu análisis manual")
                    st.markdown(
                        "Cada punto incluye una mini guía de **en qué fijarte** para hacerlo de forma sistemática."
                    )
                    st.markdown(checklist_md)

                    st.markdown("---")
                    st.markdown("### 📤 Exportar informe + checklist")

                    st.download_button(
                        label="⬇️ Descargar en Markdown (.md)",
                        data=markdown_completo.encode("utf-8"),
                        file_name=f"{ticker}_analisis_completo.md",
                        mime="text/markdown",
                    )

                    st.download_button(
                        label="⬇️ Descargar en PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"{ticker}_analisis_completo.pdf",
                        mime="application/pdf",
                    )

            except Exception as e:
                st.error(f"Error al procesar {ticker}: {e}")

# ------------------ MÓDULO DE CARTERA ------------------ #

st.markdown("---")
st.markdown("## 📂 Módulo de cartera: exposición por sectores y beta aproximada")

st.caption(
    "Introduce tus posiciones con su peso aproximado en la cartera. "
    "Ejemplo: 5 líneas con `TICKER, porcentaje`. Puedes incluir `CASH` o `EFECTIVO` "
    "para la parte en liquidez."
)

default_cartera = "AAPL, 25\nMSFT, 25\nTSLA, 10\nNVDA, 10\nCASH, 30"
cartera_text = st.text_area(
    "Posiciones (una por línea):",
    value=default_cartera,
    help="Formato: TICKER, peso_porcentaje. Ejemplo: AAPL, 20",
)

if st.button("📊 Calcular análisis de cartera"):
    lineas = [l.strip() for l in cartera_text.splitlines() if l.strip()]
    posiciones = []
    cash_weight = 0.0

    for linea in lineas:
        if "," in linea:
            partes = [p.strip() for p in linea.split(",")]
        else:
            partes = [p.strip() for p in linea.split()]

        if len(partes) < 2:
            continue

        tkr = partes[0].upper()
        try:
            peso = float(partes[1].replace("%", "").replace(",", "."))
        except ValueError:
            continue

        if tkr in ("CASH", "EFECTIVO"):
            cash_weight += peso
        else:
            posiciones.append({"ticker": tkr, "peso": peso})

    if not posiciones and cash_weight == 0:
        st.error("No se han podido interpretar posiciones válidas. Revisa el formato.")
    else:
        df_rows = []
        for pos in posiciones:
            tkr = pos["ticker"]
            peso = pos["peso"]
            try:
                yt = yf.Ticker(tkr)
                info = obtener_info_basica(yt)
                nombre = info.get("longName") or tkr
                sector = info.get("sector") or "N/D"

                raw = yf.download(
                    tkr,
                    period="1y",
                    interval="1d",
                    progress=False,
                )
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ('Close', tkr) in raw.columns:
                            close_hist = raw[('Close', tkr)]
                        else:
                            close_cols = [c for c in raw.columns if c[0] == 'Close']
                            if not close_cols:
                                close_hist = raw.iloc[:, 0]
                            else:
                                close_hist = raw[close_cols[0]]
                    else:
                        close_hist = raw["Close"]

                    hist = pd.DataFrame(index=raw.index)
                    hist["Close"] = pd.Series(close_hist).astype(float)

                    rets = hist["Close"].pct_change().dropna()
                    vol1y = rets.std() * np.sqrt(252) * 100
                    beta_t = calcular_beta_vs_indice(hist["Close"], interval="1d")
                else:
                    vol1y = np.nan
                    beta_t = np.nan

                df_rows.append(
                    {
                        "Ticker": tkr,
                        "Nombre": nombre,
                        "Peso_%": peso,
                        "Sector": sector,
                        "Vol_1A_%": vol1y,
                        "Beta": beta_t,
                    }
                )
            except Exception:
                df_rows.append(
                    {
                        "Ticker": tkr,
                        "Nombre": tkr,
                        "Peso_%": peso,
                        "Sector": "N/D",
                        "Vol_1A_%": np.nan,
                        "Beta": np.nan,
                    }
                )

        if df_rows:
            df_cartera = pd.DataFrame(df_rows)
            st.markdown("### 📋 Tabla de posiciones (sin contar la liquidez)")
            st.dataframe(df_cartera, hide_index=True)

            st.markdown("### 🧩 Exposición por sectores")
            expos_sector = df_cartera.groupby("Sector")["Peso_%"].sum().sort_values(ascending=False)
            st.dataframe(
                expos_sector.reset_index().rename(columns={"Peso_%": "Peso_sector_%"}),
                hide_index=True,
            )
            st.bar_chart(expos_sector)

            valid_beta = df_cartera.dropna(subset=["Beta", "Peso_%"]).copy()
            if not valid_beta.empty:
                valid_beta["Peso_%"] = pd.to_numeric(valid_beta["Peso_%"], errors="coerce")
                valid_beta["Beta"] = pd.to_numeric(valid_beta["Beta"], errors="coerce")
                valid_beta = valid_beta.dropna(subset=["Beta", "Peso_%"])
                if not valid_beta.empty:
                    portfolio_beta = float(
                        np.average(valid_beta["Beta"], weights=valid_beta["Peso_%"])
                    )
                else:
                    portfolio_beta = np.nan
            else:
                portfolio_beta = np.nan

            valid_vol = df_cartera.dropna(subset=["Vol_1A_%", "Peso_%"]).copy()
            if not valid_vol.empty:
                valid_vol["Peso_%"] = pd.to_numeric(valid_vol["Peso_%"], errors="coerce")
                valid_vol["Vol_1A_%"] = pd.to_numeric(valid_vol["Vol_1A_%"], errors="coerce")
                valid_vol = valid_vol.dropna(subset=["Vol_1A_%", "Peso_%"])
                if not valid_vol.empty:
                    portfolio_vol = float(
                        np.average(valid_vol["Vol_1A_%"], weights=valid_vol["Peso_%"])
                    )
                else:
                    portfolio_vol = np.nan
            else:
                portfolio_vol = np.nan

            st.markdown("### ⚖️ Riesgo aproximado de la cartera")
            st.write(f"- Peso en liquidez (CASH/EFECTIVO): **{cash_weight:.1f} %**")
            st.write(f"- Beta media ponderada (solo parte invertida): **{formatear_numero(portfolio_beta,2)}**")
            st.write(f"- Volatilidad media ponderada 1A (solo parte invertida): **{formatear_numero(portfolio_vol,1)} %**")
            st.caption(
                "En qué fijarte: si tu cartera está muy concentrada en un sector o tiene una beta muy alta, "
                "tu riesgo global puede ser más alto de lo que esperas. Ajusta peso y liquidez según tu perfil."
            )
        else:
            st.warning("Solo se ha detectado liquidez (CASH) y ninguna posición invertida.")
