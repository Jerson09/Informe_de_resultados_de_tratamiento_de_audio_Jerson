# Informe de resultados de tratamiento de audio

Experiencia SONIDO · CADI Percepción Computacional (semanas 4, 5 y 6)
Autor: Jerson Baquero Cháves

Pipeline de tratamiento de una señal de voz (`archivo_prueba.m4a`): eliminación de DC,
filtros Butterworth pasa-altas (80 Hz) y pasa-bajas (8 kHz), detección de actividad (VAD),
reducción de ruido por sustracción espectral, normalización y análisis (espectro de Welch,
espectrogramas, F0 por autocorrelación y MFCC).

## Contenido

| Archivo / carpeta | Descripción |
|---|---|
| `tratamiento_audio.py` | Script completo del pipeline |
| `archivo_prueba.m4a` | Audio de entrada |
| `requirements.txt` | Dependencias de Python |
| `resultados/` | Gráficas `fig1`–`fig7`, `audio_procesado.wav`, `audio_mono_original.wav`, `metricas.json` |
| `Informe_Tratamiento_Audio.docx` | Informe final |

## Ejecución

1. Instalar [FFmpeg](https://ffmpeg.org/) (decodifica el `.m4a`).
2. `pip install -r requirements.txt`
3. `python tratamiento_audio.py`

Las salidas se generan en `resultados/`.
