"""
Tratamiento de audio - CADI Percepcion Computacional, experiencia SONIDO
Pipeline: carga -> mono -> quitar DC -> pasa-altas -> reduccion de ruido
(sustraccion espectral) -> normalizacion -> VAD -> MFCC.
Dependencias: numpy, scipy, matplotlib (y ffmpeg instalado para decodificar .m4a).
Uso: python tratamiento_audio.py   (genera todo en la carpeta resultados/)
Repositorio: https://github.com/Jerson09/Informe_de_resultados_de_tratamiento_de_audio_Jerson
"""
import json
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import dct
from scipy.io import wavfile

# Rutas relativas a la carpeta del script (funciona al clonar el repositorio).
BASE = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(BASE, "archivo_prueba.m4a")
SALIDA = os.path.join(BASE, "resultados") + os.sep
os.makedirs(SALIDA, exist_ok=True)
metricas = {}


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(SALIDA + nombre, dpi=150)
    plt.close()


# 1. CARGA ---------------------------------------------------------------
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ENTRADA,
                "-ar", "48000", SALIDA + "original_stereo.wav"], check=True)
sr, est = wavfile.read(SALIDA + "original_stereo.wav")
est = est.astype(np.float64) / 32768.0
x = est.mean(axis=1)                       # mono
t = np.arange(len(x)) / sr
metricas["fs_hz"] = sr
metricas["duracion_s"] = round(len(x) / sr, 2)
metricas["canales"] = est.shape[1]
metricas["correlacion_LR"] = round(float(np.corrcoef(est[:, 0], est[:, 1])[0, 1]), 4)
metricas["pico"] = round(float(np.abs(x).max()), 3)
metricas["rms"] = round(float(np.sqrt(np.mean(x ** 2))), 4)
metricas["offset_DC"] = round(float(x.mean()), 4)
metricas["muestras_saturadas"] = int((np.abs(x) >= 0.999).sum())


def rms_db(v):
    return 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)


# 2. PREPROCESAMIENTO ----------------------------------------------------
# 2.1 quitar DC
y = x - x.mean()
# 2.2 pasa-altas Butterworth 80 Hz (elimina golpe/rumble inicial y DC residual)
sos = signal.butter(4, 80, btype="highpass", fs=sr, output="sos")
y_hp = signal.sosfiltfilt(sos, y)
# 2.3 pasa-bajas 8 kHz (la voz casi no tiene energia util arriba y el ruido si)
sos_lp = signal.butter(6, 8000, btype="lowpass", fs=sr, output="sos")
y_bp = signal.sosfiltfilt(sos_lp, y_hp)

# 2.4 deteccion de actividad (VAD) por energia en tramas de 25 ms
N = int(0.025 * sr)
H = int(0.010 * sr)
nt = 1 + (len(y_bp) - N) // H
E = np.array([rms_db(y_bp[i * H:i * H + N]) for i in range(nt)])
tE = (np.arange(nt) * H + N / 2) / sr
piso = np.percentile(E, 10)
umbral = piso + 12                         # 12 dB sobre el piso de ruido
vad = E > umbral
# suavizado: cierra huecos < 150 ms y descarta rafagas < 60 ms
def suavizar(m, cierre, minimo):
    m = m.copy()
    i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            if 0 < i and j < len(m) and (j - i) < cierre:
                m[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if (j - i) < minimo:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m
vad = suavizar(vad, 15, 6)
metricas["piso_ruido_dB"] = round(float(piso), 1)
metricas["umbral_VAD_dB"] = round(float(umbral), 1)
metricas["pct_actividad"] = round(float(vad.mean() * 100), 1)

# 2.5 reduccion de ruido: sustraccion espectral con perfil de ruido tomado
#     de las tramas inactivas (VAD) de menor energia
nfft, hop = 2048, 512
f, tt, Z = signal.stft(y_bp, sr, nperseg=nfft, noverlap=nfft - hop)
mag, fase = np.abs(Z), np.angle(Z)
# marcar columnas STFT inactivas
act_stft = np.interp(tt, tE, vad.astype(float)) > 0.5
cols_ruido = np.where(~act_stft)[0]
if len(cols_ruido) < 5:                    # respaldo: 10 % de tramas mas bajas
    en = mag.sum(axis=0)
    cols_ruido = np.argsort(en)[:max(5, len(en) // 10)]
perfil = np.median(mag[:, cols_ruido], axis=1, keepdims=True)
alpha, beta = 2.0, 0.05                    # sobre-sustraccion y piso espectral
mag_lim = np.maximum(mag - alpha * perfil, beta * mag)
_, y_ns = signal.istft(mag_lim * np.exp(1j * fase), sr,
                       nperseg=nfft, noverlap=nfft - hop)
y_ns = y_ns[:len(y_bp)]

# 2.6 normalizacion a -3 dBFS de pico (sin saturar)
y_fin = y_ns / (np.abs(y_ns).max() + 1e-12) * 10 ** (-3 / 20)

# SNR estimada (segmentos activos vs inactivos) antes y despues
def snr(sig):
    En = np.array([rms_db(sig[i * H:i * H + N]) for i in range(nt)])
    return float(np.mean(En[vad]) - np.mean(En[~vad])) if (~vad).any() else float("nan")
metricas["SNR_original_dB"] = round(snr(x - x.mean()), 1)
metricas["SNR_filtrado_dB"] = round(snr(y_bp), 1)
metricas["SNR_final_dB"] = round(snr(y_ns), 1)
metricas["pico_final_dBFS"] = round(float(20 * np.log10(np.abs(y_fin).max())), 1)

wavfile.write(SALIDA + "audio_procesado.wav", sr, (y_fin * 32767).astype(np.int16))
wavfile.write(SALIDA + "audio_mono_original.wav", sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))

# 3. ANALISIS -------------------------------------------------------------
# 3.1 forma de onda original vs procesada
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
ax[0].plot(t, x, lw=0.4, color="#c0392b")
ax[0].set_title("Señal original (mono)")
ax[0].axhline(0.999, color="k", ls=":", lw=0.6); ax[0].axhline(-0.999, color="k", ls=":", lw=0.6)
ax[1].plot(t, y_fin, lw=0.4, color="#1f6f4a")
ax[1].set_title("Señal procesada (HP 80 Hz + LP 8 kHz + sustracción espectral + normalización)")
for a in ax:
    a.set_ylabel("Amplitud"); a.grid(alpha=0.3)
ax[1].set_xlabel("Tiempo (s)")
guardar("fig1_forma_onda.png")

# 3.2 espectro (Welch) antes/despues
f1, P1 = signal.welch(x - x.mean(), sr, nperseg=8192)
f2, P2 = signal.welch(y_fin, sr, nperseg=8192)
plt.figure(figsize=(10, 4))
plt.semilogx(f1[1:], 10 * np.log10(P1[1:] + 1e-14), label="Original", color="#c0392b")
plt.semilogx(f2[1:], 10 * np.log10(P2[1:] + 1e-14), label="Procesada", color="#1f6f4a")
plt.axvline(80, color="gray", ls="--", lw=0.8); plt.axvline(8000, color="gray", ls="--", lw=0.8)
plt.xlim(10, 24000); plt.xlabel("Frecuencia (Hz)"); plt.ylabel("DEP (dB/Hz)")
plt.title("Densidad espectral de potencia (Welch)"); plt.legend(); plt.grid(alpha=0.3, which="both")
guardar("fig2_espectro.png")

# 3.3 espectrogramas
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for a, s, ttl in [(ax[0], x - x.mean(), "Espectrograma original"),
                  (ax[1], y_fin, "Espectrograma procesado")]:
    ff, tx, S = signal.spectrogram(s, sr, nperseg=2048, noverlap=1536)
    im = a.pcolormesh(tx, ff, 10 * np.log10(S + 1e-14), shading="auto",
                      vmin=-110, vmax=-30, cmap="magma")
    a.set_ylim(0, 6000); a.set_ylabel("Hz"); a.set_title(ttl)
ax[1].set_xlabel("Tiempo (s)")
fig.colorbar(im, ax=ax, label="dB")
plt.savefig(SALIDA + "fig3_espectrograma.png", dpi=150, bbox_inches="tight"); plt.close()

# 3.4 energia + VAD
plt.figure(figsize=(10, 3.6))
plt.plot(tE, E, color="#2c3e50", label="Energía (dB, trama 25 ms)")
plt.axhline(umbral, color="#e67e22", ls="--", label=f"Umbral VAD ({umbral:.1f} dB)")
plt.fill_between(tE, E.min(), E.max(), where=vad, color="#27ae60", alpha=0.2, label="Actividad detectada")
plt.xlabel("Tiempo (s)"); plt.ylabel("dB"); plt.legend(loc="lower right", fontsize=8)
plt.title("Detección de actividad por energía"); plt.grid(alpha=0.3)
guardar("fig4_vad.png")

# 3.5 respuesta del filtro
w, h = signal.sosfreqz(sos, worN=8192, fs=sr)
w2, h2 = signal.sosfreqz(sos_lp, worN=8192, fs=sr)
plt.figure(figsize=(10, 3.6))
plt.semilogx(w[1:], 20 * np.log10(np.abs(h[1:]) + 1e-9), label="Pasa-altas 80 Hz (orden 4)")
plt.semilogx(w2[1:], 20 * np.log10(np.abs(h2[1:]) + 1e-9), label="Pasa-bajas 8 kHz (orden 6)")
plt.ylim(-60, 5); plt.xlim(10, 24000); plt.grid(alpha=0.3, which="both")
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Ganancia (dB)"); plt.legend()
plt.title("Respuesta en frecuencia de los filtros Butterworth")
guardar("fig5_filtros.png")

# 3.6 frecuencia fundamental (autocorrelacion) en tramas activas
def f0_trama(seg, sr, fmin=70, fmax=400):
    seg = seg - seg.mean()
    if np.sqrt(np.mean(seg ** 2)) < 1e-3:
        return np.nan
    ac = signal.correlate(seg, seg, mode="full")[len(seg) - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = int(sr / fmax), int(sr / fmin)
    k = lo + np.argmax(ac[lo:hi])
    return sr / k if ac[k] > 0.35 else np.nan
Nf0, Hf0 = int(0.04 * sr), int(0.01 * sr)
tf, f0 = [], []
for i in range(0, len(y_fin) - Nf0, Hf0):
    tf.append((i + Nf0 / 2) / sr); f0.append(f0_trama(y_fin[i:i + Nf0], sr))
tf, f0 = np.array(tf), np.array(f0)
val = ~np.isnan(f0)
metricas["f0_mediana_Hz"] = round(float(np.nanmedian(f0)), 1) if val.any() else None
metricas["f0_p10_Hz"] = round(float(np.nanpercentile(f0, 10)), 1) if val.any() else None
metricas["f0_p90_Hz"] = round(float(np.nanpercentile(f0, 90)), 1) if val.any() else None
metricas["pct_sonoro"] = round(float(val.mean() * 100), 1)
plt.figure(figsize=(10, 3.4))
plt.plot(tf, f0, ".", ms=3, color="#8e44ad")
plt.xlabel("Tiempo (s)"); plt.ylabel("F0 (Hz)"); plt.grid(alpha=0.3)
plt.title("Frecuencia fundamental estimada (autocorrelación)")
guardar("fig6_f0.png")

# 3.7 MFCC (banco de 40 filtros mel, 13 coeficientes) implementados con numpy
def hz2mel(h): return 2595 * np.log10(1 + h / 700)
def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
def mfcc(sig, sr, n_mel=40, n_mfcc=13, nfft=2048, hop=480, win=1200):
    pre = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
    _, _, Zm = signal.stft(pre, sr, nperseg=win, noverlap=win - hop, nfft=nfft,
                           window="hamming", boundary=None, padded=False)
    pot = np.abs(Zm) ** 2
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mel + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mel, nfft // 2 + 1))
    for m in range(1, n_mel + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    mel_e = np.log(fb @ pot + 1e-10)
    return dct(mel_e, type=2, axis=0, norm="ortho")[:n_mfcc]
C_o = mfcc(x - x.mean(), sr)
C_p = mfcc(y_fin, sr)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, C, ttl in [(ax[0], C_o, "MFCC señal original"), (ax[1], C_p, "MFCC señal procesada")]:
    im = a.imshow(C, aspect="auto", origin="lower", cmap="coolwarm",
                  extent=[0, len(x) / sr, 0, 13], vmin=-40, vmax=40)
    a.set_title(ttl); a.set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Coeficiente")
fig.colorbar(im, ax=ax)
plt.savefig(SALIDA + "fig7_mfcc.png", dpi=150, bbox_inches="tight"); plt.close()
metricas["mfcc_forma"] = list(C_p.shape)

# 3.8 centroide y planitud espectral
ff, tx, S = signal.spectrogram(y_fin, sr, nperseg=2048, noverlap=1536)
cent = (ff[:, None] * S).sum(0) / (S.sum(0) + 1e-20)
metricas["centroide_mediano_Hz"] = round(float(np.median(cent[np.interp(tx, tE, vad.astype(float)) > 0.5])), 0)

with open(SALIDA + "metricas.json", "w") as fh:
    json.dump(metricas, fh, indent=2, ensure_ascii=False)
print(json.dumps(metricas, indent=2, ensure_ascii=False))
