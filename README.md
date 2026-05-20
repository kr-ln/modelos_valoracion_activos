# Modelo de Valoración de Activos y Optimización de Cartera

Este repositorio contiene un framework integral en Python diseñado para la descarga, análisis, modelado predictivo y optimización de portafolios de inversión utilizando datos financieros reales de **Yahoo Finance**. 

El motor combina enfoques clásicos de la teoría financiera con técnicas de optimización numérica y toma de decisiones bajo múltiples criterios (MCDM).

## 🚀 Funcionalidades Clave

* **Ingesta Temática de Datos:** Descarga automatizada de precios de cierre ajustados e integración dinámica de la Tasa Libre de Riesgo en tiempo real mediante el rendimiento de los *T-Bills* a 13 semanas (`^IRX`).
* **Modelado Lineal de Tendencias (ML):** Extrapolación de rendimientos esperados individuales mediante Regresión Lineal (`scikit-learn`) para mapear inercias en series de tiempo.
* **Optimización Robusta de Markowitz (Solver & Monte Carlo):**
    * **Algoritmo SLSQP:** Maximización del Sharpe Ratio y minimización de varianza a escalas diaria, mensual y anual.
    * **Muestreo Dirichlet:** Simulación de Monte Carlo de 10,000 portafolios con asignación uniforme estricta en el simplex de pesos, mitigando sesgos de concentración.
    * **Optimización Multihito:** Inicialización estocástica múltiple en el solver de mínima varianza para garantizar la convergencia global y evitar mínimos locales.
* **Análisis de Riesgo Sistemático (CAPM):** Regresión lineal de retornos en exceso contra el mercado (`SPY`) para extraer dinámicamente las métricas de **Beta** (sensibilidad) y **Alpha** (generación de valor) por activo y por portafolio óptimo.
* **Asignación Jerárquica de Activos (AHP):** Implementación del Proceso de Análisis Jerárquico mediante normalización Min-Max para rankear y seleccionar activos de acuerdo con tres perfiles de inversión (*Conservador*, *Moderado* y *Agresivo*).

## 🛠️ Arquitectura del Código e Implementación

El script está estructurado bajo una lógica modular secuencial:


```

├── IMPORTACIONES ───────────────► Librerías numéricas, financieras y de optimización
├── CONFIGURACIÓN ENTORNO ───────► Inicialización visual y supresión de alertas
├── CONFIGURACIÓN FINANCIERA ────► Parametrización de días de trading (252 anuales / ~21 mensuales)
├── FUNCIONES CREADAS ───────────► Formateadores, clasificadores de Sharpe y reportes numéricos
├── PASO 1: DESCARGA DE DATOS ───► Conexión a yfinance (Precios e ^IRX decimalizado)
├── PASO 2: RENDIMIENTOS ────────► Cálculo de retornos simples y alineación de vectores
├── PASO 3: MODELADO ML ─────────► Ajuste y proyección temporal con Regresión Lineal
├── PASO 4: MÉTRICAS ESTADÍSTICAS► Escalamiento por raíz del tiempo ($\sqrt{t}$) y Sharpe empírico
├── PASO 5: MODELO DE MARKOWITZ ─► Optimización matemática, Monte Carlo y ploteo de Fronteras Eficientes
├── PASO 6: MODELO CAPM ─────────► Cálculo de Alpha/Beta individuales y agregados frente a SPY
└── PASO 7: PROCESO AHP ─────────► Ponderación multicriterio basada en la escala de Saaty

```

---

## 📐 Supuestos Metodológicos Básicos

El motor opera bajo los siguientes pilares de la teoría financiera clásica y moderna:

### 1. Modelo de Media-Varianza de Markowitz
* Los inversionistas son racionales y adversos al riesgo.
* Las decisiones de inversión se basan exclusivamente en el rendimiento esperado y la varianza.
* Los rendimientos de los activos siguen una distribución normal o aproximada.
* Las correlaciones y covarianzas entre los activos son estables en la ventana temporal analizada.
* Los mercados son eficientes, los activos son perfectamente divisibles y no se consideran costos de transacción ni fricciones fiscales.

### 2. Capital Asset Pricing Model (CAPM)
* Existe una tasa libre de riesgo constante y accesible para tomar o prestar capital sin límite.
* Las expectativas del mercado son homogéneas entre los participantes.
* El riesgo relevante de un activo es únicamente su riesgo sistemático o no diversificable, representado por el coeficiente **Beta** ($\beta$).

### 3. Proceso de Análisis Jerárquico (AHP)
* El problema de selección de activos puede estructurarse de manera jerárquica (Objetivo $\rightarrow$ Criterios $\rightarrow$ Alternativas).
* La asignación de pesos relativos a los criterios (Retorno, Volatilidad, Sharpe) refleja la función de utilidad y aversión al riesgo del inversionista según su perfil psicográfico mediante prioridades de Saaty.

---

## 📊 Requisitos e Instalación

Para ejecutar este motor de optimización, asegúrate de contar con Python 3.8+ y las siguientes dependencias instaladas:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn scipy colorama

```

### Ejecución

Modifica la lista de `tickers` en el **Paso 1** según el perfil que desees evaluar (Conservador, Moderado o Agresivo) o según los activos que deseés comparar, y ejecuta el archivo principal:

```bash
python main.py

```

El script imprimirá en la consola los reportes detallados con código de colores para cada dimensión del análisis financiero y desplegará tres gráficas consecutivas con las **Fronteras Eficientes** a escala diaria, mensual y anualizadamente. Cierra cada uno en cuanto acabes de visualizarlo para que muestre las gráficas restantes. Has lo mismo con cada una para continuar con la ejecución del resto del código.
