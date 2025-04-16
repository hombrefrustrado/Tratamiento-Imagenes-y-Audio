import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# La función enfatizará los graves con el suavizado de las frecuencias más altas
# Para ello calcula la media móvil de cada valor con una ventana de tamaño configurable
# mediante parámetro (por defecto 101)
# Esta función realiza el cálculo de modo manual, determinando la posición inicial
# y la posición final de la ventana, calculando luego la media para dicha ventana
def aumentar_graves(data, window_size=101):
   
    # Creamos un array similar al original pero relleno de ceros
    filtered_data = np.zeros_like(data, dtype=np.float32)
    
    # Calculamos el centro de la ventana
    half_window = window_size // 2   #
    
    # Se calcula la media móvil para cada punto del array
    for i in range(len(data)):
        # Se calculan los límites de la venatna
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        ##### Aquí debes incluir la línea de código que calcula la media móvil de un punto
        ##### y lo almacena en el nuevo array filtered_data
        media_movil=np.mean(data[start:end])
        filtered_data[i]=media_movil
   
    return filtered_data

# La función realiza el mismo proceso que la anterior pero aplicando un filtro (kernel) de convolución
def aumentar_graves_conv(data, window_size=101):

    # Crear un kernel de media móvil
    kernel = np.ones(window_size) / window_size
    # Aplicar la convolución para calcular la media móvil con np.convolve
    filtered_data=np.convolve(data,kernel,mode='same')
    # Modes
    #    full: convolución completa, se rellenan con ceros los bordes extendidos
    #    valid: devuelve valores cuando hay solape completo, evitando bordes artificiales
    #    same: devuelve un array del mismo tamaño al original. Se centra el kernel en cada valor
    #          se rellenan con ceros los bordes extendidos y se calcula.
    ##### Aquí debes incluir la línea de código que aplica el filtro de convolución
    return filtered_data

# Esta función enfatiza los graves y luego los elimina de la señal original, obteniendo
# como resultado una señal con énfasis de los agudos
def aumentar_agudos(data, window_size=101):

    graves = aumentar_graves(data, window_size)
    ##### Aquí debes incluir la línea de código que elimina los graves
    
    filtered_data = data-graves
    return filtered_data

def invertir_senal(data):
    #### Aquí debes incluir la línea (con return) que devuelve la señal invertida
    inverted_data = data[::-1]
    return inverted_data

def cambiar_volumen(data, ganancia=1.5):
    #### Aquí debes incluir la línea (con return) que devuelve la señal con el volumen cambiado
    return data*ganancia
# Para calcular el eco, desplazamos la señal original "hacia delante" un determinado número de muestras (delay).
# Luego la reducimos de volumen (para no saturar la señal original) y la sumamos a la señal original
def aplicar_eco(data, delay=5000, factor_eco=0.5):

    # Opción 1 (cada #### es una línea de código)
    #### Desplazamos la señal original con roll (numpy)
    #### Rellenamos las "delay" primeras muestras, ya que no tienen audio de eco de referencia
    #### Reducimos de volumen la señal eco
    #### La sumamos a la señao original
    
    # Opción 2 (cada #### es una línea de código)
    #### Creamos una señal de eco con valores 0
    #### Recalculamos la señal de eco multiplicando la original por el factor de volumen y la desplazamos
    #### Sumamos la señal original al eco.
    eco = np.zeros(len(data)+delay,dtype=np.float32)
    volumen_cambiado=cambiar_volumen(data,factor_eco)
    eco[delay:]=volumen_cambiado[:]
    eco_signal=eco[:len(data)]+data
    return eco_signal

# Cargar el archivo WAV

rutaBase= os.path.join(os.getcwd(),"audios")
ficheroEntrada = os.path.join(rutaBase, 'sampleIn.wav')
ficheroSalidaGraves = os.path.join(rutaBase, 'sampleOutGraves.wav')
ficheroSalidaAgudos = os.path.join(rutaBase, 'sampleOutAgudos.wav')
ficheroSalidaInvertida = os.path.join(rutaBase, 'sampleOutInv.wav')
ficheroSalidaSubidaVolumen = os.path.join(rutaBase, 'sampleOutSubidaVol.wav')
ficheroSalidaBajadaVolumen = os.path.join(rutaBase, 'sampleOutBajadaVol.wav')
ficheroSalidaEco = os.path.join(rutaBase, 'sampleOutEco.wav')

fs, data = wavfile.read(ficheroEntrada)

# Si el audio es estéreo, toma solo un canal
if len(data.shape) > 1:
    data = data[:, 0]

# Mejora: si el audio es estéreo y ambos canales contienen el mismo contenido,
# se procesa solo uno y luego se puede replicar.
# canalesIguales = np.all(data[:, 0] == data[:, 1])

# ¿Y si los canales son prácticamente idénticos, con alguna mínima diferencia
# relativa a errores en el muestreo?
# factorTolerancia=0.1
# canalesCasiIguales = np.allclose(data[:, 0], data[:, 1], atol=factorTolerancia)

# Normalizar los datos al rango [-1, 1] si es necesario
if data.dtype == np.int16:
    data = data / 32768.0
elif data.dtype == np.int32:
    data = data / 2147483648.0

# Aplicar transformaciones (elige las que quieras)
data_graves = aumentar_graves_conv(data, window_size=101)  # Aumentar graves
data_agudos = aumentar_agudos(data, window_size=101)  # Aumentar agudos
data_invertida = invertir_senal(data)  # Invertir la señal
data_subida_volumen = cambiar_volumen(data, ganancia=3.7)  # Subir el volumen
data_bajada_volumen = cambiar_volumen(data, ganancia=0.25)  # Bajar el volumen
data_eco = aplicar_eco(data, delay=5000, factor_eco=0.5)  # Aplicar eco

# Graficar la señal original
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Señal de audio original')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal con subida de graves
plt.figure(figsize=(12, 6))
plt.plot(data_graves)
plt.title('Señal de audio (graves)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal con subida de agudos
plt.figure(figsize=(12, 6))
plt.plot(data_agudos)
plt.title('Señal de audio (agudos)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal invertida
plt.figure(figsize=(12, 6))
plt.plot(data_invertida)
plt.title('Señal de audio invertida')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal con subida de volumen
plt.figure(figsize=(12, 6))
plt.plot(data_subida_volumen)
plt.title('Señal de audio con subida de volumen')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal con bajada de volumen
plt.figure(figsize=(12, 6))
plt.plot(data_bajada_volumen)
plt.title('Señal de audio con bajada de volumen')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Graficar la señal con eco
plt.figure(figsize=(12, 6))
plt.plot(data_eco)
plt.title('Señal de audio con eco')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Desnormalizar los datos si es necesario
if data.dtype == np.int16:
    data_graves = np.int16(data_graves * 32768)
    data_agudos = np.int16(data_agudos * 32768)
    data_invertida = np.int16(data_invertida * 32768)
    data_subida_volumen = np.int16(np.clip(data_subida_volumen * 32768, -32768, 32767))
    data_bajada_volumen = np.int16(np.clip(data_bajada_volumen * 32768, -32768, 32767))
    data_eco = np.int16(data_eco * 32768)
elif data.dtype == np.int32:
    data_graves = np.int32(data_graves * 2147483648)
    data_agudos = np.int32(data_agudos * 2147483648)
    data_invertida = np.int32(data_invertida * 2147483648)
    data_subida_volumen = np.int16(np.clip(data_subida_volumen * 32768, -32768, 32767))
    data_bajada_volumen = np.int16(np.clip(data_bajada_volumen * 32768, -32768, 32767))
    data_eco = np.int32(data_eco * 2147483648)

# Guardar la señal transformada en un nuevo archivo WAV
wavfile.write(ficheroSalidaGraves, fs, data_graves)
wavfile.write(ficheroSalidaAgudos, fs, data_agudos)
wavfile.write(ficheroSalidaInvertida, fs, data_invertida)
wavfile.write(ficheroSalidaSubidaVolumen, fs, data_subida_volumen)
wavfile.write(ficheroSalidaBajadaVolumen, fs, data_bajada_volumen)
wavfile.write(ficheroSalidaEco, fs, data_eco)


print(f"Se han guardado las señales transformadas.")