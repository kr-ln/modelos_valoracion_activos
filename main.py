# ===========================================================================================================================================================================
# IMPORTACIONES
# ===========================================================================================================================================================================
#region
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
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

from scipy.optimize import minimize
from numpy.random import dirichlet

# =========================================================
# DATOS BASE
# =========================================================

media = rendimientos.mean().values          # retorno diario
cov = rendimientos.cov().values             # cov diaria
n = len(tickers)
rf = float(rf_diario.mean())

# =========================================================
# FUNCIONES OBJETIVO (SOLVER)
# =========================================================

def port_ret(w, mu):
    return np.dot(w, mu)

def port_vol(w, cov):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def sharpe(w, mu, cov, rf):
    return -(port_ret(w, mu) - rf) / port_vol(w, cov)

def vol_only(w, cov):
    return port_vol(w, cov)

# restricciones
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(n)]
w0 = np.ones(n) / n

# =========================================================
# FUNCIÓN MEJORADA: MÚLTIPLES INICIOS PARA MÍNIMA VARIANZA
# =========================================================

def encontrar_min_var_robusto(cov, n, bounds, cons, n_intentos=20):
    """Encuentra mínima varianza con múltiples puntos de inicio aleatorios"""
    mejor_resultado = None
    mejor_vol = np.inf
    
    for intento in range(n_intentos):
        # Punto de inicio aleatorio usando Dirichlet (distribución uniforme en simplex)
        w0_rand = dirichlet(np.ones(n))
        
        sol = minimize(vol_only, w0_rand, args=(cov,),
                      method='SLSQP', bounds=bounds, constraints=cons,
                      options={'ftol': 1e-12, 'eps': 1e-8, 'disp': False})
        
        if sol.success:
            vol_actual = vol_only(sol.x, cov)
            if vol_actual < mejor_vol - 1e-10:  # tolerancia numérica
                mejor_vol = vol_actual
                mejor_resultado = sol
    
    # Si no se encontró ninguna solución válida, usar el inicio uniforme
    if mejor_resultado is None:
        print("  Advertencia: No se encontraron soluciones con múltiples inicios. Usando inicio uniforme.")
        return minimize(vol_only, w0, args=(cov,), method='SLSQP', 
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'eps': 1e-8})
    
    return mejor_resultado

# =========================================================
# SOLVER: PORTAFOLIOS ÓPTIMOS (DIARIO) - VERSIÓN MEJORADA
# =========================================================

print("\nOptimizando portafolios (esto puede tomar unos segundos)...")

# Para Sharpe máximo (suele ser más estable, un solo inicio es suficiente)
sol_sharpe_d = minimize(sharpe, w0, args=(media, cov, rf),
                        method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol': 1e-12, 'eps': 1e-8})

# Para mínima varianza - usando múltiples inicios
sol_minvar_d = encontrar_min_var_robusto(cov, n, bounds, cons)

# Verificar que las soluciones son válidas
if not sol_sharpe_d.success:
    print(f"  Advertencia: Solución Sharpe máximo no convergió: {sol_sharpe_d.message}")
if not sol_minvar_d.success:
    print(f"  Advertencia: Solución mínima varianza no convergió: {sol_minvar_d.message}")

w_max_sharpe_d = sol_sharpe_d.x
w_min_var_d = sol_minvar_d.x

# =========================================================
# ESCALADO TEMPORAL
# =========================================================

mu_m = media * DAYS_PER_MONTH
cov_m = cov * DAYS_PER_MONTH

mu_a = media * TRADING_DAYS
cov_a = cov * TRADING_DAYS

# =========================================================
# SOLVER: MENSUAL
# =========================================================

sol_sharpe_m = minimize(sharpe, w0, args=(mu_m, cov_m, rf * DAYS_PER_MONTH),
                        method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol': 1e-12, 'eps': 1e-8})

sol_minvar_m = minimize(vol_only, w0, args=(cov_m,),
                        method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol': 1e-12, 'eps': 1e-8})

w_max_sharpe_m = sol_sharpe_m.x
w_min_var_m = sol_minvar_m.x

# =========================================================
# SOLVER: ANUAL
# =========================================================

sol_sharpe_a = minimize(sharpe, w0, args=(mu_a, cov_a, rf * TRADING_DAYS),
                        method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol': 1e-12, 'eps': 1e-8})

sol_minvar_a = minimize(vol_only, w0, args=(cov_a,),
                        method='SLSQP', bounds=bounds, constraints=cons,
                        options={'ftol': 1e-12, 'eps': 1e-8})

w_max_sharpe_a = sol_sharpe_a.x
w_min_var_a = sol_minvar_a.x

# =========================================================
# MONTECARLO MEJORADO (USANDO DIRICHLET)
# =========================================================

N = 10000  # Aumentado de 5000 a 10000
resultados = np.zeros((N, 3))
pesos_lista = []

for i in range(N):
    # Usar Dirichlet para muestreo uniforme en el simplex
    w = dirichlet(np.ones(n))
    
    r = np.dot(w, media)
    v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    s = (r - rf) / v
    
    resultados[i] = [v, r, s]
    pesos_lista.append(w)

idx_s = np.argmax(resultados[:, 2])
idx_v = np.argmin(resultados[:, 0])

# Mostrar verificación numérica
print(f"\nVerificación numérica (escala diaria):")
print(f"  Mínima volatilidad Monte Carlo: {resultados[idx_v, 0]*100:.6f}%")
print(f"  Mínima volatilidad Solver: {port_vol(w_min_var_d, cov)*100:.6f}%")
print(f"  Diferencia: {(port_vol(w_min_var_d, cov) - resultados[idx_v, 0])*100:.8f}%")

# =========================================================
# FUNCIÓN IMPRESIÓN MEJORADA (SOLVER)
# =========================================================

def imprimir_solver(nombre, w, mu, cov, rf, escala="diario"):
    
    r = np.dot(w, mu)  # retorno en escala base
    v = np.sqrt(np.dot(w.T, np.dot(cov, w)))  # volatilidad en escala base
    s = (r - rf) / v  # Sharpe ratio
    
    # Escalamos para output (con 4 decimales)
    if escala == "diario":
        r_diario = r * 100
        r_mensual = r * DAYS_PER_MONTH * 100
        r_anual = r * TRADING_DAYS * 100
        v_diario = v * 100
        v_mensual = v * np.sqrt(DAYS_PER_MONTH) * 100
        v_anual = v * np.sqrt(TRADING_DAYS) * 100
        s_anual = s * np.sqrt(TRADING_DAYS)
        
        print(f"\n{nombre}")
        print("-" * 40)
        
        for t, p in zip(tickers, w):
            print(f"  {t}: {round(p*100,2)}%")
        
        print(f"""
  RETORNOS:
    Diario: {round(r_diario,4)}%
    Mensual: {round(r_mensual,4)}%
    Anual: {round(r_anual,4)}%

  VOLATILIDAD:
    Diaria: {round(v_diario,4)}%
    Mensual: {round(v_mensual,4)}%
    Anual: {round(v_anual,4)}%

  SHARPE RATIO:
    Diario: {round(s,4)}
    Anualizado: {round(s_anual,4)} → {interpretar_sharpe(s_anual, 'anual')}
""")
    
    elif escala == "mensual":
        r_mensual = r * 100
        r_anual = r * 12 * 100
        v_mensual = v * 100
        v_anual = v * np.sqrt(12) * 100
        s_anual = s * np.sqrt(12)
        
        print(f"\n{nombre}")
        print("-" * 40)
        
        for t, p in zip(tickers, w):
            print(f"  {t}: {round(p*100,2)}%")
        
        print(f"""
  RETORNOS:
    Mensual: {round(r_mensual,4)}%
    Anual: {round(r_anual,4)}%

  VOLATILIDAD:
    Mensual: {round(v_mensual,4)}%
    Anual: {round(v_anual,4)}%

  SHARPE RATIO:
    Mensual: {round(s,4)}
    Anualizado: {round(s_anual,4)} → {interpretar_sharpe(s_anual, 'anual')}
""")
    
    else:  # anual
        r_anual = r * 100
        v_anual = v * 100
        s_anual = s
        
        print(f"\n{nombre}")
        print("-" * 40)
        
        for t, p in zip(tickers, w):
            print(f"  {t}: {round(p*100,2)}%")
        
        print(f"""
  RETORNOS:
    Anual: {round(r_anual,4)}%

  VOLATILIDAD:
    Anual: {round(v_anual,4)}%

  SHARPE RATIO:
    Anual: {round(s_anual,4)} → {interpretar_sharpe(s_anual, 'anual')}
""")

# =========================================================
# OUTPUT SOLVER
# =========================================================

imprimir_solver("SOLVER - Sharpe Máximo (Diario)", w_max_sharpe_d, media, cov, rf, "diario")
imprimir_solver("SOLVER - Mínima Varianza (Diario)", w_min_var_d, media, cov, rf, "diario")
imprimir_solver("SOLVER - Sharpe Máximo (Mensual)", w_max_sharpe_m, mu_m, cov_m, rf * DAYS_PER_MONTH, "mensual")
imprimir_solver("SOLVER - Mínima Varianza (Mensual)", w_min_var_m, mu_m, cov_m, rf * DAYS_PER_MONTH, "mensual")
imprimir_solver("SOLVER - Sharpe Máximo (Anual)", w_max_sharpe_a, mu_a, cov_a, rf * TRADING_DAYS, "anual")
imprimir_solver("SOLVER - Mínima Varianza (Anual)", w_min_var_a, mu_a, cov_a, rf * TRADING_DAYS, "anual")

# =========================================================
# GRÁFICA DIARIA (Escala Diaria en %)
# =========================================================

plt.figure(figsize=(14, 8))

# Convertir puntos de Monte Carlo a porcentaje diario
vol_mc_diario = resultados[:,0] * 100
ret_mc_diario = resultados[:,1] * 100

# Graficar frontera
scatter = plt.scatter(vol_mc_diario, ret_mc_diario,
                      c=resultados[:,2], cmap='RdYlGn', alpha=0.6, s=15,
                      vmin=np.percentile(resultados[:,2], 5), 
                      vmax=np.percentile(resultados[:,2], 95))
plt.colorbar(scatter, label='Sharpe Ratio (diario)')

# SOLVER Sharpe máximo - escala diaria
v_s = port_vol(w_max_sharpe_d, cov) * 100
r_s = port_ret(w_max_sharpe_d, media) * 100

plt.scatter(v_s, r_s, color='red', marker='*', s=500, edgecolors='black', linewidths=2,
            label='ÓPTIMO SHARPE MÁXIMO', zorder=5)

# SOLVER mínima varianza - escala diaria
v_v = port_vol(w_min_var_d, cov) * 100
r_v = port_ret(w_min_var_d, media) * 100

plt.scatter(v_v, r_v, color='blue', marker='s', s=400, edgecolors='black', linewidths=2,
            label='ÓPTIMO MÍNIMA VARIANZA', zorder=5)

# Agregar anotaciones (Volatilidad, Retorno) con 4 decimales
plt.annotate(f'Sharpe Max\n({v_s:.4f}%, {r_s:.4f}%)',
             xy=(v_s, r_s),
             xytext=(15, 15), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.annotate(f'Mínima Varianza\n({v_v:.4f}%, {r_v:.4f}%)',
             xy=(v_v, r_v),
             xytext=(15, -25), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Activos individuales
for i, ticker in enumerate(tickers):
    vol_i = np.sqrt(cov[i,i]) * 100
    ret_i = media[i] * 100
    plt.scatter(vol_i, ret_i, color='gray', marker='o', s=100, alpha=0.7)
    plt.annotate(ticker, (vol_i, ret_i), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title("Frontera Eficiente de Markowitz (Datos Diarios)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Volatilidad Diaria (%)", fontsize=12)
plt.ylabel("Retorno Esperado Diario (%)", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
plt.close('all')

# =========================================================
# GRÁFICA MENSUAL (Escala Mensual en %)
# =========================================================

plt.figure(figsize=(14, 8))

# Convertir puntos de Monte Carlo a porcentaje mensual
vol_mc_mensual = resultados[:,0] * np.sqrt(DAYS_PER_MONTH) * 100
ret_mc_mensual = resultados[:,1] * DAYS_PER_MONTH * 100

scatter = plt.scatter(vol_mc_mensual, ret_mc_mensual,
                      c=resultados[:,2], cmap='RdYlGn', alpha=0.6, s=15,
                      vmin=np.percentile(resultados[:,2], 5), 
                      vmax=np.percentile(resultados[:,2], 95))
plt.colorbar(scatter, label='Sharpe Ratio (diario)')

# SOLVER mensual
r_s = port_ret(w_max_sharpe_m, mu_m) * 100
v_s = port_vol(w_max_sharpe_m, cov_m) * 100

plt.scatter(v_s, r_s, color='red', marker='*', s=500, edgecolors='black', linewidths=2,
            label='ÓPTIMO SHARPE MÁXIMO', zorder=5)

r_v = port_ret(w_min_var_m, mu_m) * 100
v_v = port_vol(w_min_var_m, cov_m) * 100

plt.scatter(v_v, r_v, color='blue', marker='s', s=400, edgecolors='black', linewidths=2,
            label='ÓPTIMO MÍNIMA VARIANZA', zorder=5)

# Anotaciones con 4 decimales
plt.annotate(f'Sharpe Max\n({v_s:.4f}%, {r_s:.4f}%)',
             xy=(v_s, r_s),
             xytext=(15, 15), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.annotate(f'Mínima Varianza\n({v_v:.4f}%, {r_v:.4f}%)',
             xy=(v_v, r_v),
             xytext=(15, -25), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Activos individuales
for i, ticker in enumerate(tickers):
    vol_i = np.sqrt(cov[i,i]) * np.sqrt(DAYS_PER_MONTH) * 100
    ret_i = media[i] * DAYS_PER_MONTH * 100
    plt.scatter(vol_i, ret_i, color='gray', marker='o', s=100, alpha=0.7)
    plt.annotate(ticker, (vol_i, ret_i), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title("Frontera Eficiente de Markowitz (Datos Mensuales)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Volatilidad Mensual (%)", fontsize=12)
plt.ylabel("Retorno Esperado Mensual (%)", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
plt.close('all')

# =========================================================
# GRÁFICA ANUAL (Escala Anual en %)
# =========================================================

plt.figure(figsize=(14, 8))

# Convertir puntos de Monte Carlo a porcentaje anual
vol_mc_anual = resultados[:,0] * np.sqrt(TRADING_DAYS) * 100
ret_mc_anual = resultados[:,1] * TRADING_DAYS * 100

scatter = plt.scatter(vol_mc_anual, ret_mc_anual,
                      c=resultados[:,2], cmap='RdYlGn', alpha=0.6, s=15,
                      vmin=np.percentile(resultados[:,2], 5), 
                      vmax=np.percentile(resultados[:,2], 95))
plt.colorbar(scatter, label='Sharpe Ratio (diario)')

# SOLVER anual
r_s = port_ret(w_max_sharpe_a, mu_a) * 100
v_s = port_vol(w_max_sharpe_a, cov_a) * 100

plt.scatter(v_s, r_s, color='red', marker='*', s=500, edgecolors='black', linewidths=2,
            label='ÓPTIMO SHARPE MÁXIMO', zorder=5)

r_v = port_ret(w_min_var_a, mu_a) * 100
v_v = port_vol(w_min_var_a, cov_a) * 100

plt.scatter(v_v, r_v, color='blue', marker='s', s=400, edgecolors='black', linewidths=2,
            label='ÓPTIMO MÍNIMA VARIANZA', zorder=5)

# Anotaciones con 4 decimales
plt.annotate(f'Sharpe Max\n({v_s:.4f}%, {r_s:.4f}%)',
             xy=(v_s, r_s),
             xytext=(15, 15), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.annotate(f'Mínima Varianza\n({v_v:.4f}%, {r_v:.4f}%)',
             xy=(v_v, r_v),
             xytext=(15, -25), textcoords='offset points',
             fontsize=10, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Activos individuales
for i, ticker in enumerate(tickers):
    vol_i = np.sqrt(cov[i,i]) * np.sqrt(TRADING_DAYS) * 100
    ret_i = media[i] * TRADING_DAYS * 100
    plt.scatter(vol_i, ret_i, color='gray', marker='o', s=100, alpha=0.7)
    plt.annotate(ticker, (vol_i, ret_i), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title("Frontera Eficiente de Markowitz (Datos Anuales)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Volatilidad Anual (%)", fontsize=12)
plt.ylabel("Retorno Esperado Anual (%)", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
plt.close('all')

# =========================================================
# TABLA COMPARATIVA DE PORTAFOLIOS
# =========================================================

print("\n" + "="*80)
print("RESUMEN COMPARATIVO DE PORTAFOLIOS")
print("="*80)

portafolios = {
    "Sharpe Máximo (Diario)": (w_max_sharpe_d, media, cov, rf, "diario"),
    "Mínima Varianza (Diario)": (w_min_var_d, media, cov, rf, "diario"),
    "Sharpe Máximo (Mensual)": (w_max_sharpe_m, mu_m, cov_m, rf * DAYS_PER_MONTH, "mensual"),
    "Mínima Varianza (Mensual)": (w_min_var_m, mu_m, cov_m, rf * DAYS_PER_MONTH, "mensual"),
    "Sharpe Máximo (Anual)": (w_max_sharpe_a, mu_a, cov_a, rf * TRADING_DAYS, "anual"),
    "Mínima Varianza (Anual)": (w_min_var_a, mu_a, cov_a, rf * TRADING_DAYS, "anual")
}

for nombre, (w, mu, cov_local, rf_local, escala) in portafolios.items():
    r = port_ret(w, mu)
    v = port_vol(w, cov_local)
    s = (r - rf_local) / v
    
    if escala == "diario":
        r_pct = r * 100
        v_pct = v * 100
        r_anual_pct = r * TRADING_DAYS * 100
        v_anual_pct = v * np.sqrt(TRADING_DAYS) * 100
        s_anual = s * np.sqrt(TRADING_DAYS)
        print(f"\n{nombre}:")
        print(f"  Retorno Diario: {r_pct:.4f}%")
        print(f"  Volatilidad Diaria: {v_pct:.4f}%")
        print(f"  Retorno Anual: {r_anual_pct:.4f}%")
        print(f"  Volatilidad Anual: {v_anual_pct:.4f}%")
        print(f"  Sharpe Ratio (anual): {s_anual:.4f} → {interpretar_sharpe(s_anual, 'anual')}")
    
    elif escala == "mensual":
        r_pct = r * 100
        v_pct = v * 100
        r_anual_pct = r * 12 * 100
        v_anual_pct = v * np.sqrt(12) * 100
        s_anual = s * np.sqrt(12)
        print(f"\n{nombre}:")
        print(f"  Retorno Mensual: {r_pct:.4f}%")
        print(f"  Volatilidad Mensual: {v_pct:.4f}%")
        print(f"  Retorno Anual: {r_anual_pct:.4f}%")
        print(f"  Volatilidad Anual: {v_anual_pct:.4f}%")
        print(f"  Sharpe Ratio (anual): {s_anual:.4f} → {interpretar_sharpe(s_anual, 'anual')}")
    
    else:  # anual
        r_pct = r * 100
        v_pct = v * 100
        print(f"\n{nombre}:")
        print(f"  Retorno Anual: {r_pct:.4f}%")
        print(f"  Volatilidad Anual: {v_pct:.4f}%")
        print(f"  Sharpe Ratio: {s:.4f} → {interpretar_sharpe(s, 'anual')}")

# Mostrar el portafolio de Monte Carlo con mejor Sharpe
print(f"\n{'='*80}")
print("COMPARACIÓN CON MONTE CARLO")
print(f"{'='*80}")

mejor_sharpe_mc = resultados[idx_s, 2] * np.sqrt(TRADING_DAYS)
print(f"Mejor Sharpe encontrado por Monte Carlo: {mejor_sharpe_mc:.4f}")
print(f"Mejor Sharpe encontrado por Solver: {(port_ret(w_max_sharpe_d, media) - rf) / port_vol(w_max_sharpe_d, cov) * np.sqrt(TRADING_DAYS):.4f}")

if mejor_sharpe_mc > (port_ret(w_max_sharpe_d, media) - rf) / port_vol(w_max_sharpe_d, cov) * np.sqrt(TRADING_DAYS) + 0.01:
    print("\n⚠️ ADVERTENCIA: Monte Carlo encontró un Sharpe superior al solver.")
    print("   Esto puede indicar que el solver se quedó en un óptimo local.")
else:
    print("\n✓ Solver y Monte Carlo son consistentes.")

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
rend_port_opt = rendimientos.dot(w_max_sharpe_d)
rend_port_min = rendimientos.dot(w_min_var_d)

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
rend_port_opt = rendimientos.dot(w_max_sharpe_d)

# Portafolio mínima varianza
rend_port_min = rendimientos.dot(w_min_var_d)

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

imprimir_paso("Paso 7: AHP")

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
# Perfil Conservador
perfiles_brutos = {
    "Conservador": {'Retorno': 3, 'Volatilidad': 9, 'Sharpe': 6},
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

# =========================
# 5. NORMALIZACIÓN MIN-MAX
# =========================
matriz_norm = pd.DataFrame(index=matriz_decision.index)

# Criterios de Beneficio (Mayor es mejor)
matriz_norm['Retorno'] = (
    (matriz_decision['Retorno'] - matriz_decision['Retorno'].min()) /
    (matriz_decision['Retorno'].max() - matriz_decision['Retorno'].min())
)

matriz_norm['Sharpe'] = (
    (matriz_decision['Sharpe'] - matriz_decision['Sharpe'].min()) /
    (matriz_decision['Sharpe'].max() - matriz_decision['Sharpe'].min())
)

# Criterio de Costo (Menor es mejor)
matriz_norm['Volatilidad'] = (
    (matriz_decision['Volatilidad'].max() - matriz_decision['Volatilidad']) /
    (matriz_decision['Volatilidad'].max() - matriz_decision['Volatilidad'].min())
)

# ==========================================================
# 6. VECTOR DE PRIORIDADES POR CRITERIO
# Normalización por columna: cada valor se divide entre la suma de su columna
# ==========================================================
df_prioridades = matriz_norm.div(matriz_norm.sum(axis=0), axis=1)

# =========================
# 7-8. PROCESO POR PERFIL
# =========================
for nombre, pesos_brutos in perfiles_brutos.items():

    print(f"\n{'='*60}")
    print(f"PERFIL {nombre.upper()}")
    print(f"{'='*60}")

    # Normalizar pesos de importancia del criterio
    suma = sum(pesos_brutos.values())
    pesos_normalizados = {k: v/suma for k, v in pesos_brutos.items()}
    pesos_ser = pd.Series(pesos_normalizados)

    # Puntaje final mediante producto punto con el vector de prioridades
    score = df_prioridades.mul(pesos_ser, axis=1).sum(axis=1)

    # Ranking final
    ranking = score.sort_values(ascending=False)

    print("\n")
    print("RANKING FINAL:")
    print(ranking)

    mejor = ranking.idxmax()

    print(f"""
MEJOR ACTIVO ({nombre}): {mejor}

Interpretación:
El perfil {nombre} prioriza {max(pesos_brutos, key=pesos_brutos.get)}.
El activo seleccionado maximiza el puntaje AHP bajo dichas preferencias.
""")
    
#endregion

print("\n")
print("PROCESO COMPLETADO")