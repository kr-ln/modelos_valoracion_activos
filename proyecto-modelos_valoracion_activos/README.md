# Modelo de Valoración de Activos Financieros

Este proyecto implementa un modelo integral de análisis financiero combinando:

- **Markowitz** (optimización de portafolios)
- **CAPM** (riesgo sistemático y alpha)
- **AHP** (toma de decisiones multicriterio)

---

## Objetivo

Seleccionar el mejor activo financiero y construir portafolios óptimos en función de:

- Rentabilidad
- Riesgo (volatilidad)
- Ratio de Sharpe

---

## Tecnologías utilizadas

- Python
- pandas
- numpy
- yfinance
- scikit-learn
- matplotlib

---

## Funcionalidades

- Descarga automática de datos financieros
- Cálculo de rendimientos y volatilidad
- Simulación de portafolios (Markowitz)
- Evaluación CAPM (beta y alpha)
- Ranking de activos con AHP según perfil:
  - Conservador
  - Moderado
  - Agresivo

---

## Modelos implementados

### 🔹 Markowitz
Optimización de portafolios mediante simulación de combinaciones de activos.

### 🔹 CAPM
Estimación de:
- Beta (riesgo sistemático)
- Alpha (rendimiento extra)

### 🔹 AHP
Selección del mejor activo con base en:
- Normalización min-max
- Ponderaciones subjetivas
- Ranking multicriterio

---

##  Cómo ejecutar el proyecto

1. Clonar el repositorio:
```bash
git clone https://github.com/kr-ln/proyecto-modelos_valoracion_activos.git

---

## Personalización

El usuario puede adaptar el modelo fácilmente modificando:

### Activos (tickers)
Ubicados en el código principal:

```python
tickers = ['VIG','VTI','SCHD','JNJ','PG','KO','WMT']

### Ponderaciones del modelo AHP
También puedes ajustar las prioridades del modelo según tu perfil de riesgo modificando las ponderaciones:
```python
perfiles_brutos = {
    "Conservador": {'Retorno': 3, 'Volatilidad': 6, 'Sharpe': 5},
    "Moderado": {'Retorno': 4, 'Volatilidad': 3, 'Sharpe': 6},
    "Agresivo": {'Retorno': 6, 'Volatilidad': 2, 'Sharpe': 4}
}