import os
from PIL import Image
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d


# Cada #### se corresponde con una acción que se puede realizar con una única línea de código

# Ajustar brillo. Positivo => más brillo, negativo => menos brillo
def ajustar_brillo(imagen_array, valor_brillo):
    # Ojo, si la suma supera 255, se realiza un wrap y vuelve a comenzar desde cero
    # Esto puede provocar que las zonas blancas tengan tonos oscuros
    #### Se suma el valor de brillo al array, asegurándonos con la función clip que no devolvemos valores fuera del rango 0..255
    return np.clip(imagen_array+np.array([valor_brillo,valor_brillo,valor_brillo]),0,255).astype(np.uint8)
# Ajustar contraste
def ajustar_contraste(imagen_array, factor_contraste):
    #### Multiplicamos el valor de cada pixel por el factor de contraste, asegurándonos con la función clip que no devolvemos valores fuera del rango 0..255
    return np.clip(imagen_array*factor_contraste,0,255).astype(np.uint8)
# Aplicar transparencia sobre un color (croma)

def aplicar_croma(imagen_array, color_croma):
    #### Creamos un array mascara con valor True para todos los puntos cuyo color sea color_croma
    # Pasamos la imagen de RGB a RGBA, es decir, de forma x,y,3 a x,y,4
    # El cuarto canal es el canal Alfa, de transparencia.
    # shape[:2] = resX,resY
    # Se añade el canal alfa rellenando con valor 255 (opaco)
    #### Ponemos a cero los valores del canal Alfa donde la máscara vale True
    
    mascara = np.all(imagen_array == color_croma, axis=-1)
    imagen_rgba = np.dstack((imagen_array, 255 * np.ones(imagen_array.shape[:2], dtype=np.uint8)))
    
    imagen_rgba[mascara,3] = 0
    
    return imagen_rgba.astype(np.uint8)

# Desplazar imagen
def desplazar_imagen(imagen_array, desplazamiento_x, desplazamiento_y):
    #### Creamos un array de tamaño igual al de la imagen con todos los valores a cero
    #### En dicho array asignamos los valores de la imagen original, pero desplazados según los parámetros
    alto, ancho, canales = imagen_array.shape
    imagen_desplazada = np.zeros((alto+desplazamiento_y,ancho+desplazamiento_x,canales),dtype=np.uint8)
    imagen_desplazada[desplazamiento_y:,desplazamiento_x:]=imagen_array[:,:]
    return imagen_desplazada.astype(np.uint8)


# Invertir colores
def invertir_colores(imagen_array):
    #### Restamos 255-color original
    return np.clip(np.array([255,255,255])-imagen_array,0,255).astype(np.uint8)

# Convertir a escala de grises
def convertir_a_gris(imagen_array):
    # Calcula el tono de gris como una combinación lineal de RGB [0.2989, 0.5870, 0.1140]
    #### Creamos una imagen con un único valor por píxel que resulta de la combinación lineal anterior
    imagen_gris=imagen_array[:,:,0]*0.2989+imagen_array[:,:,1]*0.5870+imagen_array[:,:,2]*0.1140
    return imagen_gris.astype(np.uint8)
#otra forma de convertir a gris, solo que es propia
def otro_gris(imagen_array):
    imagen_gris = np.mean(imagen_array, axis=2)
    return imagen_gris.astype(np.uint8)
# Aplicar desenfoque usando un filtro de promedio (convolución)
def aplicar_desenfoque(imagen_array, tamano_kernel=5):
    # Matriz cuya suma de valores es igual a 1
    kernel = np.ones((tamano_kernel, tamano_kernel)) / (tamano_kernel ** 2)
    #### Creamos un array de tamaño igual al de la imagen con todos los valores a cero
    imagen_desenfocada = np.zeros_like(imagen_array)
    for i in range(3):  # Aplicar a cada canal (R, G, B)
        #### Aplicamos a cada canal el filtro de convolución para difuminar la imagen
        imagen_desenfocada[:,:,i] = convolve(imagen_array[:,:,i],kernel)
        

    return imagen_desenfocada.astype(np.uint8)

# Rotar imagen
def rotar_imagen(imagen_array, angulo=90):
    imagen_aux=np.zeros_like(imagen_array)
    if angulo == 90:
        #### Rotamos la imagen 90 grados
        imagen_aux=np.rot90(imagen_array,1)
    elif angulo == 180:
        #### Rotamos la imagen 180 grados
        imagen_aux=np.rot90(imagen_array,2)
    elif angulo == 270:
        #### Rotamos la imagen 270 grados
        imagen_aux=np.rot90(imagen_array,3)
    else:
        raise ValueError("Ángulo de rotación no válido. Use 90, 180 o 270.")
    return imagen_aux.astype(np.uint8)
# Anular canal. 0=R, 1=G, 2=B
def anular_canal(imagen_array, numCanal):
    #### Creamos una copia de la imagen
    #### Establecemos a cero los valores del canal indicado
    imagen_sincanal = np.copy(imagen_array)
    imagen_sincanal[:,:,numCanal]=0
    return imagen_sincanal.astype(np.uint8)

# Detección de bordes con convolución aplicando filtros de Sobel
def detectar_bordes(imagen_array):
    
    # Convertir a gris
    if len(imagen_array.shape) == 3:
        imagen_gris = convertir_a_gris(imagen_array)
    else:
        imagen_gris = imagen_array
    # Definir los kernels de Sobel para detección de bordes
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])  # Detecta bordes verticales
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])  # Detecta bordes horizontales
    # Aplicar las convoluciones para detectar bordes en las direcciones X e Y
    # Same mantiene el tamaño del resultado
    # symm refleja valores cercanos en bordes para convolucionar en límites de la imagen
    #### Calculamos bordes_x con convolve2d
    #### Calculamos bordes_y con convolve2d
    #### Calculamos magnitud del gradiente
    #### Normalizamos los bordes al rango [0, 255]
    # Convertir a tipo uint8
    bordes_x = convolve2d(imagen_gris,sobel_x,mode='same',boundary='symm')
    bordes_y = convolve2d(imagen_gris,sobel_y,mode='same',boundary='symm')
    magnitud = np.sqrt(bordes_x**2 + bordes_y**2)
    
    magnitud = (magnitud / np.max(magnitud)) * 255
    bordes = magnitud.astype(np.uint8)


    return bordes

# Cargar la imagen
# Si la imagen es en color, shape=resX, resY, 3 (RGB)
# Si la imagen es en escala de grises, shape=resX, resY
print("Para usar la aplicación debes de meter un archivo en la carpeta imagenes y decir el nombre a continuación \n")
nombre = input("Indica con el siguiente formato nombre_del_archivo.extensión: ")
directorio = os.path.join(os.getcwd(),"imagenes",nombre)
imagen = Image.open(directorio)
imagen_array = np.array(imagen)
imagen_array=imagen_array[:,:,:3]

# Aplicar transformaciones
imagen_brillo = ajustar_brillo(imagen_array, 50)
imagen_contraste = ajustar_contraste(imagen_array, 1.5)
imagen_croma = aplicar_croma(np.array(Image.open(os.path.join(os.getcwd(),"imagenes","croma.jpg"))), [0, 255, 1])  # Croma verde
imagen_desplazada = desplazar_imagen(imagen_array, 50, 30)
imagen_invertida = invertir_colores(imagen_array)
imagen_gris = convertir_a_gris(imagen_array)
imagen_desenfocada = aplicar_desenfoque(imagen_array, tamano_kernel=5)
imagen_rotada = rotar_imagen(imagen_array, angulo=90)
imagen_sinCanalRojo = anular_canal(imagen_array,0)
imagen_sinCanalVerde = anular_canal(imagen_array,1)
imagen_sinCanalAzul = anular_canal(imagen_array,2)
imagen_bordes = detectar_bordes(imagen_array)

# Guardar o mostrar las imágenes resultantes
Image.fromarray(imagen_array).show(title="Imagen original")
Image.fromarray(imagen_brillo).show(title="Brillo Ajustado")
Image.fromarray(imagen_contraste).show(title="Contraste Ajustado")
Image.fromarray(imagen_croma, 'RGBA').show(title="Croma Aplicado")
Image.fromarray(imagen_desplazada).show(title="Imagen Desplazada")
Image.fromarray(imagen_invertida).show(title="Colores Invertidos")
Image.fromarray(imagen_gris, 'L').show(title="Escala de Grises")
Image.fromarray(imagen_desenfocada).show(title="Imagen Desenfocada")
Image.fromarray(imagen_rotada).show(title="Imagen Rotada")
Image.fromarray(imagen_sinCanalRojo).show(title="Imagen sin canal R")
Image.fromarray(imagen_sinCanalVerde).show(title="Imagen sin canal G")
Image.fromarray(imagen_sinCanalAzul).show(title="Imagen sin canal B")
Image.fromarray(imagen_bordes).show(title="Imagen (bordes)")
