# Clasificador de Maduración de Kiwi
**Fecha:** 3/2/2026

**Autor:** **Grace Johamy Contreras Montaño**

**Nombre del Proyecto:** **Clasificador de Kiwi**

## **Descripción**
Este sistema de **visión artificial** utiliza la arquitectura **YOLOv11** para la detección y clasificación automática del estado de maduración del kiwi. El modelo procesa **video en tiempo real** capturado por una cámara web para identificar tres categorías clave:

* **Verde:** Kiwis con semillas claras y centro blanco.
* **Maduro:** Kiwis con semillas negras y pulpa verde brillante, óptimos para consumo.
* **Podrido:** Kiwis con aspecto de oxidación o manchas oscuras.

## **Tecnologías Utilizadas**
* **Lenguaje:** **Python 3.14.2**
* **Librerías:** **Ultralytics**, **OpenCV**, **Roboflow**
* **Herramientas:** Entrenamiento acelerado por **GPU T4** en **Google Colab**.

## **Estructura del Archivo**             
Clasificación de la Maduracion de un Kiwi/                
├── Kiwi-5/                   
├── runs/                     
├── sample_data/              
   └── yolo11s.pt  
### **Código del entrenamiento:**
```python
# Instalación de librerías necesarias
!pip install roboflow ultralytics

from roboflow import Roboflow
from ultralytics import YOLO

# Descarga del dataset desde Roboflow
rf = Roboflow(api_key="gaoenWXbUchgKzwR7963")
project = rf.workspace("kiwi-tpi91").project("kiwi-lmlee")
version = project.version(5)
dataset = version.download("yolov11")

# Carga del modelo entrenado (Pesos finales)
custom_model = YOLO("/content/runs/detect/train5/weights/best.pt")

# Inferencia de prueba
res = custom_model("/content/Kiwi-5/test/images")
#Grafica de los Resultados
from IPython.display import Image, display
display(Image(filename='/content/runs/detect/train/results.png', width=1000))
#
```
## **Analisis**

Esta gráfica de resultados muestra el progreso del entrenamiento de tu modelo YOLO para la clasificación de kiwis durante aproximadamente 112 épocas. 
Se observa una convergencia saludable en las funciones de pérdida de entrenamiento (train/box_loss, cls_loss, dfl_loss), las cuales descienden de manera constante, 
indicando que el modelo está aprendiendo a localizar y clasificar los objetos correctamente. Aunque las métricas de precisión y recall presentan oscilaciones , 
la tendencia general de mAP50 y mAP50-95 es ascendente, estabilizándose por encima del 50% y 25% respectivamente; esto sugiere que el modelo ha alcanzado un nivel de aprendizaje sólido, 
aunque el ruido en la validación podría indicar que el dataset es pequeño o que la variedad de las imágenes requiere un ligero ajuste en los hiperparámetros para mayor estabilidad.

<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/3168466d-a30e-4b08-83f2-b620e3926f42" />


## **Conclusion**

Este proyecto consolida una arquitectura de software robusta para la implementación de modelos de visión artificial, integrando de manera eficiente la gestión de entornos virtuales, 
el procesamiento de datasets con Kiwi-5 y la ejecución de scripts en tiempo real. La estructura organizada del repositorio asegura que el flujo de trabajo sea escalable y profesional,
facilitando la transición desde la fase de entrenamiento con YOLO hasta la puesta en marcha de una aplicación de clasificación funcional.

La integración de estas herramientas demuestra la capacidad de las TICs para resolver problemas de identificación mediante el uso de redes neuronales profundas y procesamiento de imágenes.
Al finalizar este desarrollo, se cuenta con una infraestructura técnica completa que permite la detección sistemática de objetos, cumpliendo con los estándares actuales de desarrollo de software 
e inteligencia artificial aplicada.
## **Anexos**

**Colab**

<img width="640" height="640" alt="image" src="https://github.com/user-attachments/assets/a7c7ed4b-c2ab-44ad-9c29-30bfb3fc77de" />

**Camara web**

<img width="336" height="511" alt="image" src="https://github.com/user-attachments/assets/159deaae-473f-436f-bd95-775120fae298" />

<img width="430" height="480" alt="image" src="https://github.com/user-attachments/assets/9554a8d7-8188-42ec-a59d-b21e2913e44f" />

<img width="554" height="466" alt="image" src="https://github.com/user-attachments/assets/e5df5da1-fb00-4dc1-9fea-9a618898efb2" />






