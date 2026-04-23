# ===========================================================================================================================================================================
# IMPORTACIONES
# ===========================================================================================================================================================================
#region
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from colorama import Fore, Back, Style, init
import os
import warnings
#endregion

# ===========================================================================================================================================================================
# CONFIGURACIÓN DEL LUGAR DE TRABAJO
# ===========================================================================================================================================================================
#region
# Ignorar warnings para evitar ruido en consola
warnings.filterwarnings("ignore")

# Limpia la consola (solo visual)
os.system('cls')
#endregion

# ===========================================================================================================================================================================
# CONFIGURACIÓN FINANCIERA ESTÁNDAR
# ===========================================================================================================================================================================
#region

# Número de días de trading en un año (no calendario)
TRADING_DAYS = 252

# Aproximación de días de trading en un mes
DAYS_PER_MONTH = TRADING_DAYS / 12  # ≈ 21
#endregion

# ===========================================================================================================================================================================
# FUNCIONES CREADAS
# ===========================================================================================================================================================================
#region

# FUNCIÓN AUXILIAR

def imprimir_paso(titulo):
    # Solo separador visual para organizar el output del script
    print(Fore.RED + "\n" + "="*60 + Style.RESET_ALL)
    print(Fore.RED + f"{titulo}" + Style.RESET_ALL)
    print(Fore.RED + "="*60 + Style.RESET_ALL)

# INTERPRETACIÓN DEL SHARPE RATIO

def interpretar_sharpe(sharpe, escala):

    # Convertimos a escalar por si viene como array o Series
    sharpe = float(np.array(sharpe).squeeze())

    # Sharpe diario ajustado por escala de mercado
    if escala == "diario":
        if sharpe < (0.5 / np.sqrt(TRADING_DAYS)):
            return "Bajo"
        elif sharpe < (1 / np.sqrt(TRADING_DAYS)):
            return "Conservador"
        elif sharpe < (1.5 / np.sqrt(TRADING_DAYS)):
            return "Moderado"
        else:
            return "Agresivo"

    # Sharpe mensual ajustado por raíz del tiempo
    elif escala == "mensual":
        if sharpe < (0.5 / np.sqrt(DAYS_PER_MONTH)):
            return "Bajo"
        elif sharpe < (1 / np.sqrt(DAYS_PER_MONTH)):
            return "Conservador"
        elif sharpe < (1.5 / np.sqrt(DAYS_PER_MONTH)):
            return "Moderado"
        else:
            return "Agresivo"

    # Sharpe anual (más estándar en finanzas)
    else:
        if sharpe < 0.5:
            return "Bajo"
        elif sharpe < 1:
            return "Conservador"
        elif sharpe < 1.5:
            return "Moderado"
        else:
            return "Agresivo"
def imprimir_portafolio(w, idx, nombre):

    r = resultados[idx,1]
    v = resultados[idx,0]
    s = resultados[idx,2]

    print(f"\n")
    print(nombre)

    for t,p in zip(tickers,w):
        print(f"{t}: {round(p*100,2)}%")

    print(f"""
        RETORNO:
            Diario: {round(r,6)}
            Mensual: {round(r*DAYS_PER_MONTH*100,4)}%
            Anual: {round(r*TRADING_DAYS*100,4)}%

        VOLATILIDAD:
            Diaria: {round(v*100,6)}%
            Mensual: {round(v*np.sqrt(DAYS_PER_MONTH)*100,4)}%
            Anual: {round(v*np.sqrt(TRADING_DAYS)*100,4)}%

        SHARPE ANUAL:
            {round(s*np.sqrt(TRADING_DAYS),2)} → {interpretar_sharpe(s*np.sqrt(TRADING_DAYS),"anual")}
        """
    )
#endregion

# ===========================================================================================================================================================================
# PASO 1: DESCARGA DE DATOS
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 1: Descarga de datos")

# Activos que vamos a analizar (comentar/descomentar según perfil)
#tickers = ['AAPL','MSFT','GOOGL','AMZN']                    # Empresas tecnológicas (perfil variante)
tickers = ['VIG','VTI','SCHD','JNJ','PG','KO','WMT']        # Perfil Conservador
#tickers = ['VOO','QQQ','AAPL','MSFT','AMZN','GOOGL','JPM']  # Perfil Moderado
#tickers = ['TSLA','NVDA','AMD','META','NFLX','PLTR','COIN'] # Perfil Agresivo

# Descarga de precios ajustados (incluye splits y dividendos)
datos = yf.download(tickers, start='2020-01-01', auto_adjust=True)['Close']

# =========================================================
# TASA LIBRE DE RIESGO (^IRX)
# =========================================================

# ^IRX = rendimiento de bonos T-Bill a 13 semanas (en % anual)
irx = yf.download("^IRX", start="2020-01-01", auto_adjust=True)["Close"]

# Rellenamos datos faltantes por días sin cotización
irx = irx.ffill()

# Convertimos tasa anual (%) a tasa diaria decimal
rf_diario = (irx / 100) / TRADING_DAYS

# Alineamos la RF con las fechas de los activos
rf_diario = rf_diario.reindex(datos.index).ffill()

# Debug: ver datos crudos
print(datos)
print("\n")
print(rf_diario)
print("\n")
print("RF calculado desde ^IRX")
#endregion

# ===========================================================================================================================================================================
# PASO 2: RENDIMIENTOS
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 2: Rendimientos")

# Retornos simples diarios (% cambio día a día)
rendimientos = datos.pct_change().dropna()

# Aseguramos que RF tenga mismas fechas que retornos
rf_diario = rf_diario.loc[rendimientos.index]

# Debug visual
print(rendimientos)
print("\n")
print(rf_diario)
#endregion

# ===========================================================================================================================================================================
# PASO 3: MODELO DE MACHINE LEARNING (REGRESIÓN LINEAL): No está prediciendo el mercado, solo está extendiendo una línea.
# SUPOSICIONES:
# 
# 1. Tendencia lineal en los rendimientos (simplificación)
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 3: Predicción de Rendimientos con ML (Regresión Lineal)")

# Variable temporal (índice de tiempo)
X = np.arange(len(rendimientos)).reshape(-1,1)

# Diccionario para guardar predicciones
rend_pred = {}

for col in rendimientos.columns:

    # Modelo lineal simple (tendencia en el tiempo)
    modelo = LinearRegression()

    # Entrenamos con historial de retornos
    modelo.fit(X, rendimientos[col])

    # Predicción del siguiente punto temporal
    pred = float(modelo.predict([[len(rendimientos)]])[0])

    rend_pred[col] = pred

    # Mostrar predicciones escaladas
    print(f"""
        {col}

        Diario: {round(pred*100,6)}%
        Mensual: {round(pred*DAYS_PER_MONTH*100,4)}%
        Anual: {round(pred*TRADING_DAYS*100,4)}%
        """
    )

# Convertimos a Serie para usar en portafolio
rend_pred = pd.Series(rend_pred)
#endregion

# ===========================================================================================================================================================================
# PASO 4: MÉTRICAS ESTADÍSTICAS
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 4: Métricas")

# Media de retornos diarios (esperanza matemática)
media = rendimientos.mean()

# Volatilidad diaria (desviación estándar)
vol = rendimientos.std()

for col in media.index:

    # Retorno esperado diario del activo
    r_d = float(media[col])

    # Escalamiento lineal de retorno (media)
    r_m = r_d * DAYS_PER_MONTH
    r_a = r_d * TRADING_DAYS

    # Volatilidad escala con raíz del tiempo
    v_d = float(vol[col])
    v_m = v_d * np.sqrt(DAYS_PER_MONTH)
    v_a = v_d * np.sqrt(TRADING_DAYS)

    # Tasa libre de riesgo promedio diaria
    rf_d = float(rf_diario.mean())

    # Sharpe ratio = exceso de retorno / riesgo
    sharpe_d = float((r_d - rf_d) / v_d)
    sharpe_m = float((r_m - rf_d * DAYS_PER_MONTH) / v_m)
    sharpe_a = float((r_a - rf_d * TRADING_DAYS) / v_a)

    print(f"""
        {col}

            RETORNOS:
                Diario: {round(r_d*100,6)}%
                Mensual: {round(r_m*100,4)}%
                Anual: {round(r_a*100,4)}%

            VOLATILIDAD:
                Diaria: {round(v_d*100,6)}%
                Mensual: {round(v_m*100,4)}%
                Anual: {round(v_a*100,4)}%

            SHARPE:
                Diario: {round(sharpe_d,2)} → {interpretar_sharpe(sharpe_d,"diario")}
                Mensual: {round(sharpe_m,2)} → {interpretar_sharpe(sharpe_m,"mensual")}
                Anual: {round(sharpe_a,2)} → {interpretar_sharpe(sharpe_a,"anual")}
        """
    )
#endregion

# ===========================================================================================================================================================================
# PASO 5: MARKOWITZ (PORTAFOLIOS ALEATORIOS): Depende fuerte de la covarianza (inestable)
# SUPOSICIONES:
#
# 1. Los inversionistas son racionales y adversos al riesgo.
# 2. Las decisiones se basan únicamente en el rendimiento esperado y la varianza (media-varianza).
# 3. Los rendimientos de los activos siguen una distribución normal (o al menos se describen completamente por media y varianza).
# 4. Los mercados son eficientes (la información está reflejada en los precios).
# 5. No existen costos de transacción ni impuestos.
# 6. Los activos son perfectamente divisibles.
# 7. Se puede prestar y pedir prestado a una tasa libre de riesgo constante.
# 8. Las correlaciones entre activos son estables en el tiempo.
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 5: Modelo de Markowitz")

# Matriz de covarianza entre activos (riesgo conjunto)
cov = rendimientos.cov()

# Guardamos resultados de simulación
resultados = np.zeros((5000, 3))  # [vol, retorno, sharpe]
pesos_lista = []

# RF promedio (benchmark)
rf = float(rf_diario.mean())

for i in range(5000):

    # pesos aleatorios normalizados (suman 1)
    w = np.random.random(len(tickers))
    w /= np.sum(w)

    # retorno esperado del portafolio
    r = float(np.dot(w, rend_pred.values))

    # volatilidad del portafolio (matriz cuadrática)
    v = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

    # Sharpe del portafolio
    s = float((r - rf) / v)

    resultados[i, 0] = v
    resultados[i, 1] = r
    resultados[i, 2] = s

    pesos_lista.append(w)

# Selección de portafolios extremos (Sharpe máximo y mínima varianza)
idx_s = np.argmax(resultados[:,2])  # máximo Sharpe
idx_v = np.argmin(resultados[:,0])  # mínima volatilidad

# Mostrar portafolios extremos
imprimir_portafolio(pesos_lista[idx_s], idx_s, "Portafolio Óptimo (Sharpe máximo)")
imprimir_portafolio(pesos_lista[idx_v], idx_v, "Portafolio Mínima Volatilidad")

# Crear las tres gráficas: Diario, Mensual, Anual
import matplotlib.pyplot as plt

# Para la gráfica diaria
plt.figure(figsize=(10, 6))
plt.scatter(resultados[:,0], resultados[:,1], c=resultados[:,2], cmap='viridis', marker='o', s=30, alpha=0.5)
plt.scatter(resultados[idx_s, 0], resultados[idx_s, 1], color='red', marker='*', s=200, label="Portafolio Óptimo (Sharpe máximo)")
plt.scatter(resultados[idx_v, 0], resultados[idx_v, 1], color='blue', marker='*', s=200, label="Portafolio Mínima Volatilidad")

# Anotaciones para la gráfica diaria
plt.title("Frente de Posibilidades de Markowitz - Diario", fontsize=16)
plt.xlabel("Volatilidad (Riesgo) - Diario", fontsize=12)
plt.ylabel("Retorno Esperado - Diario", fontsize=12)

# Coordenadas para el gráfico diario
plt.annotate(f"Óptimo (Sharpe máximo)\n({round(resultados[idx_s, 0], 4)}, {round(resultados[idx_s, 1], 4)})",
             (resultados[idx_s, 0], resultados[idx_s, 1]),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='red')

plt.annotate(f"Mínima Volatilidad\n({round(resultados[idx_v, 0], 4)}, {round(resultados[idx_v, 1], 4)})",
             (resultados[idx_v, 0], resultados[idx_v, 1]),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='blue')

plt.legend(loc='best')
plt.show()
plt.close('all')

# Para la gráfica mensual
plt.figure(figsize=(10, 6))
plt.scatter(resultados[:,0] * np.sqrt(DAYS_PER_MONTH), resultados[:,1] * DAYS_PER_MONTH, c=resultados[:,2], cmap='viridis', marker='o', s=30, alpha=0.5)
plt.scatter(resultados[idx_s, 0] * np.sqrt(DAYS_PER_MONTH), resultados[idx_s, 1] * DAYS_PER_MONTH, color='red', marker='*', s=200, label="Portafolio Óptimo (Sharpe máximo)")
plt.scatter(resultados[idx_v, 0] * np.sqrt(DAYS_PER_MONTH), resultados[idx_v, 1] * DAYS_PER_MONTH, color='blue', marker='*', s=200, label="Portafolio Mínima Volatilidad")

# Anotaciones para la gráfica mensual
plt.title("Frente de Posibilidades de Markowitz - Mensual", fontsize=16)
plt.xlabel("Volatilidad (Riesgo) - Mensual", fontsize=12)
plt.ylabel("Retorno Esperado - Mensual", fontsize=12)

# Coordenadas para el gráfico mensual
plt.annotate(f"Óptimo (Sharpe máximo)\n({round(resultados[idx_s, 0] * np.sqrt(DAYS_PER_MONTH), 4)}, {round(resultados[idx_s, 1] * DAYS_PER_MONTH, 4)})",
             (resultados[idx_s, 0] * np.sqrt(DAYS_PER_MONTH), resultados[idx_s, 1] * DAYS_PER_MONTH),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='red')

plt.annotate(f"Mínima Volatilidad\n({round(resultados[idx_v, 0] * np.sqrt(DAYS_PER_MONTH), 4)}, {round(resultados[idx_v, 1] * DAYS_PER_MONTH, 4)})",
             (resultados[idx_v, 0] * np.sqrt(DAYS_PER_MONTH), resultados[idx_v, 1] * DAYS_PER_MONTH),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='blue')

plt.legend(loc='best')
plt.show()
plt.close('all')

# Para la gráfica anual
plt.figure(figsize=(10, 6))
plt.scatter(resultados[:,0] * np.sqrt(TRADING_DAYS), resultados[:,1] * TRADING_DAYS, c=resultados[:,2], cmap='viridis', marker='o', s=30, alpha=0.5)
plt.scatter(resultados[idx_s, 0] * np.sqrt(TRADING_DAYS), resultados[idx_s, 1] * TRADING_DAYS, color='red', marker='*', s=200, label="Portafolio Óptimo (Sharpe máximo)")
plt.scatter(resultados[idx_v, 0] * np.sqrt(TRADING_DAYS), resultados[idx_v, 1] * TRADING_DAYS, color='blue', marker='*', s=200, label="Portafolio Mínima Volatilidad")

# Anotaciones para la gráfica anual
plt.title("Frente de Posibilidades de Markowitz - Anual", fontsize=16)
plt.xlabel("Volatilidad (Riesgo) - Anual", fontsize=12)
plt.ylabel("Retorno Esperado - Anual", fontsize=12)

# Coordenadas para el gráfico anual
plt.annotate(f"Óptimo (Sharpe máximo)\n({round(resultados[idx_s, 0] * np.sqrt(TRADING_DAYS), 4)}, {round(resultados[idx_s, 1] * TRADING_DAYS, 4)})",
             (resultados[idx_s, 0] * np.sqrt(TRADING_DAYS), resultados[idx_s, 1] * TRADING_DAYS),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='red')

plt.annotate(f"Mínima Volatilidad\n({round(resultados[idx_v, 0] * np.sqrt(TRADING_DAYS), 4)}, {round(resultados[idx_v, 1] * TRADING_DAYS, 4)})",
             (resultados[idx_v, 0] * np.sqrt(TRADING_DAYS), resultados[idx_v, 1] * TRADING_DAYS),
             textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10, color='blue')

plt.legend(loc='best')
plt.show()
plt.close('all')

#endregion

# ===========================================================================================================================================================================
# PASO 6: CAPM: Supuestos muy restrictivos (expectativas homogéneas, mercado perfecto)
# SUPOSICIONES:
#
# 1. Los inversionistas son racionales y maximizan utilidad (media-varianza).
# 2. Todos los inversionistas tienen expectativas homogéneas (misma información y análisis).
# 3. Existe un activo libre de riesgo al cual se puede prestar y pedir prestado sin límite.
# 4. Los mercados son eficientes (no hay arbitraje).
# 5. No existen impuestos ni costos de transacción.
# 6. Todos los activos son divisibles y negociables.
# 7. El riesgo relevante es únicamente el riesgo sistemático (medido por beta).
# 8. El portafolio de mercado incluye todos los activos disponibles.
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 6: CAPM")

# 1. DATOS DEL MERCADO

# Usamos SPY como proxy del mercado
spy = yf.download("SPY", start='2020-01-01', auto_adjust=True)['Close']

# Retornos del mercado
rm = spy.pct_change().dropna()

# Alineamos la tasa libre de riesgo con el mercado
rf_m = rf_diario.reindex(rm.index).ffill()

# 2. CAPM POR ACTIVO

print("\n")
print("CAPM POR ACTIVO")

for col in rendimientos.columns:

    # Dataset conjunto
    df = pd.concat([rendimientos[col], rm, rf_m], axis=1).dropna()
    df.columns = ['Ri','Rm','Rf']

    # Excess returns
    df['Ri_excess'] = df['Ri'] - df['Rf']
    df['Rm_excess'] = df['Rm'] - df['Rf']

    # Regresión CAPM
    X = df['Rm_excess'].values.reshape(-1,1)
    y = df['Ri_excess'].values

    model = LinearRegression()
    model.fit(X, y)

    beta = float(model.coef_[0])
    alpha = float(model.intercept_)

    # Interpretación beta
    if beta > 1:
        interp_beta = "Activo más volátil que el mercado"
    elif beta < 1:
        interp_beta = "Activo menos volátil que el mercado"
    else:
        interp_beta = "Misma volatilidad que el mercado"

    # Interpretación alpha
    if alpha > 0:
        interp_alpha = "Genera rendimiento adicional (alpha positivo)"
    elif alpha < 0:
        interp_alpha = "Rendimiento inferior al esperado por CAPM"
    else:
        interp_alpha = "Rendimiento igual al esperado por CAPM"

    print(f"""
{col}

Beta: {round(beta,4)} → {interp_beta}

Alpha diario: {round(alpha,6)}
Alpha mensual: {round(alpha * DAYS_PER_MONTH * 100,4)}%
Alpha anual: {round(alpha * TRADING_DAYS * 100,4)}%
→ {interp_alpha}
""")

# 3. CAPM PARA PORTAFOLIOS

print("\n CAPM PARA PORTAFOLIOS")

# Construcción de retornos de portafolio
rend_port_opt = rendimientos.dot(pesos_lista[idx_s])
rend_port_min = rendimientos.dot(pesos_lista[idx_v])

# Función CAPM para portafolio
def capm_portafolio(nombre, rend_port, rm, rf_m):

    df = pd.concat([rend_port, rm, rf_m], axis=1).dropna()
    df.columns = ['Rp','Rm','Rf']

    # Excess returns
    df['Rp_excess'] = df['Rp'] - df['Rf']
    df['Rm_excess'] = df['Rm'] - df['Rf']

    # Regresión
    X = df['Rm_excess'].values.reshape(-1,1)
    y = df['Rp_excess'].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    beta_p = float(modelo.coef_[0])
    alpha_p = float(modelo.intercept_)

    # Interpretación beta
    if beta_p > 1:
        interp_beta = "Portafolio agresivo (más volátil que el mercado)"
    elif beta_p < 1:
        interp_beta = "Portafolio defensivo (menos volátil que el mercado)"
    else:
        interp_beta = "Portafolio neutral al mercado"

    # Interpretación alpha
    if alpha_p > 0:
        interp_alpha = "Genera valor (alpha positivo)"
    elif alpha_p < 0:
        interp_alpha = "Destruye valor (alpha negativo)"
    else:
        interp_alpha = "Neutral"

    print(f"""
        {nombre}

        Beta del portafolio: {round(beta_p,4)} → {interp_beta}

        Alpha diario: {round(alpha_p,6)}
        Alpha mensual: {round(alpha_p * DAYS_PER_MONTH * 100,4)}%
        Alpha anual: {round(alpha_p * TRADING_DAYS * 100,4)}%
        → {interp_alpha}
""")

# Ejecución
capm_portafolio("Portafolio Óptimo (Sharpe máximo)", rend_port_opt, rm, rf_m)
capm_portafolio("Portafolio Mínima Varianza", rend_port_min, rm, rf_m)

# 1. CONSTRUIR RETORNOS DEL PORTAFOLIO

# Portafolio óptimo (Sharpe máximo)
rend_port_opt = rendimientos.dot(pesos_lista[idx_s])

# Portafolio mínima varianza
rend_port_min = rendimientos.dot(pesos_lista[idx_v])

# 2. FUNCIÓN CAPM COMPLETA

def capm_portafolio(nombre, rend_port):

    # Unimos: portafolio + mercado + RF
    df = pd.concat([rend_port, rm, rf_m], axis=1).dropna()
    df.columns = ['Rp','Rm','Rf']

    # 3. EXCESS RETURNS
    df['Rp_excess'] = df['Rp'] - df['Rf']
    df['Rm_excess'] = df['Rm'] - df['Rf']

    # 4. REGRESIÓN LINEAL
    X = df['Rm_excess'].values.reshape(-1,1)
    y = df['Rp_excess'].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    beta_p = float(modelo.coef_[0])
    alpha_p = float(modelo.intercept_)

    # 5. INTERPRETACIÓN

    # Beta
    if beta_p > 1:
        interp_beta = "Portafolio más volátil que el mercado (agresivo)"
    elif beta_p < 1:
        interp_beta = "Portafolio menos volátil que el mercado (defensivo)"
    else:
        interp_beta = "Portafolio con misma volatilidad que el mercado"

    # Alpha
    if alpha_p > 0:
        interp_alpha = "Genera rendimiento superior al esperado por CAPM"
    elif alpha_p < 0:
        interp_alpha = "Rendimiento inferior al esperado por CAPM"
    else:
        interp_alpha = "Rendimiento igual al esperado por CAPM"

    # 6. OUTPUT
    print(f"""
📊 {nombre}

Beta del portafolio: {round(beta_p,4)}
→ {interp_beta}

Alpha diario: {round(alpha_p,6)}
Alpha mensual: {round(alpha_p * DAYS_PER_MONTH * 100,4)}%
Alpha anual: {round(alpha_p * TRADING_DAYS * 100,4)}%
→ {interp_alpha}
""")

# 3. EJECUCIÓN

capm_portafolio("Portafolio Óptimo (Sharpe máximo)", rend_port_opt)
capm_portafolio("Portafolio Mínima Varianza", rend_port_min)
#endregion

# ===========================================================================================================================================================================
# PASO 7: AHP: Subjetividad en los pesos
# SUPOSICIONES:
#
# 1. El problema puede estructurarse en forma jerárquica (objetivo, criterios, alternativas).
# 2. Los criterios son independientes entre sí.
# 3. Las preferencias pueden representarse mediante comparaciones (escala tipo Likert o Saaty).
# 4. Los juicios del decisor son consistentes o suficientemente consistentes.
# 5. Es posible asignar pesos relativos a cada criterio.
# 6. Las alternativas pueden evaluarse cuantitativa o cualitativamente.
# 7. La decisión final se basa en la agregación ponderada de los criterios.
# ===========================================================================================================================================================================
#region

imprimir_paso("Paso 8: AHP (versión tesis con perfiles)")

# =========================
# 1. OBJETIVO
# =========================
# Seleccionar el mejor activo financiero

# =========================
# 2. JERARQUÍA
# =========================
# Criterios: Retorno (beneficio), Volatilidad (costo), Sharpe (beneficio)
# Alternativas: Activos del portafolio

# =========================
# 3. PONDERACIONES (TIPO SAATY POR PERFIL)
# =========================
perfiles_brutos = {
    "Conservador": {'Retorno': 3, 'Volatilidad': 6, 'Sharpe': 5},
    "Moderado": {'Retorno': 4, 'Volatilidad': 3, 'Sharpe': 6},
    "Agresivo": {'Retorno': 6, 'Volatilidad': 2, 'Sharpe': 4}
}

# =========================
# 4. MATRIZ DE DECISIÓN
# =========================
ret = rendimientos.mean()
vol = rendimientos.std()
rf = float(rf_diario.mean())

sharpe = (ret - rf) / vol

matriz_decision = pd.DataFrame({
    'Retorno': ret,
    'Volatilidad': vol,
    'Sharpe': sharpe
})

#print("\n")
#print("MATRIZ DE DECISIÓN:")
#print(matriz_decision)

# =========================
# 5. NORMALIZACIÓN (MIN-MAX)
# =========================
matriz_norm = pd.DataFrame(index=matriz_decision.index)

# Beneficio
matriz_norm['Retorno'] = (
    (matriz_decision['Retorno'] - matriz_decision['Retorno'].min()) /
    (matriz_decision['Retorno'].max() - matriz_decision['Retorno'].min())
)

matriz_norm['Sharpe'] = (
    (matriz_decision['Sharpe'] - matriz_decision['Sharpe'].min()) /
    (matriz_decision['Sharpe'].max() - matriz_decision['Sharpe'].min())
)

# Costo
matriz_norm['Volatilidad'] = (
    (matriz_decision['Volatilidad'].max() - matriz_decision['Volatilidad']) /
    (matriz_decision['Volatilidad'].max() - matriz_decision['Volatilidad'].min())
)

#print("\n")
#print("MATRIZ NORMALIZADA:")
#print(matriz_norm)

# =========================
# 6–8. PROCESO POR PERFIL
# =========================
for nombre, pesos_brutos in perfiles_brutos.items():

    print(f"\n{'='*60}")
    print(f"PERFIL {nombre.upper()}")
    print(f"{'='*60}")

    # 🔹 Normalizar pesos (Paso 3 tesis)
    suma = sum(pesos_brutos.values())
    pesos = {k: v/suma for k,v in pesos_brutos.items()}

    #print("\n")
    #print("PONDERACIONES NORMALIZADAS:")
    #print(pesos)

    # =========================
    # 6. MATRIZ PONDERADA
    # =========================
    matriz_ponderada = matriz_norm.copy()

    for col in matriz_ponderada.columns:
        matriz_ponderada[col] *= pesos[col]

    #print("\n")
    #print("MATRIZ PONDERADA:")
    #print(matriz_ponderada)

    # =========================
    # 7. PUNTAJE TOTAL
    # =========================
    score = matriz_ponderada.sum(axis=1)

    #print("\n")
    #print("PUNTAJE TOTAL:")
    #print(score)

    # =========================
    # 8. RANKING
    # =========================
    ranking = score.sort_values(ascending=False)

    print("\n")
    print("RANKING FINAL:")
    print(ranking)

    mejor = ranking.idxmax()

    print(f"""
MEJOR ACTIVO ({nombre}): {mejor}

Interpretación:
El perfil {nombre} prioriza {max(pesos, key=pesos.get)}.
El activo seleccionado maximiza el puntaje AHP bajo dichas preferencias.
""")
    
#endregion

print("\n")
print("PROCESO COMPLETADO")