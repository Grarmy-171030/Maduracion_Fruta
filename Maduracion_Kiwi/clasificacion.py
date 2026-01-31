from ultralytics import YOLO
import cv2
# Cargamos el modelo YOLO
model = YOLO("best (1).pt")
# Cargamos el video de entrada
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # Leemos el frame del video
    ret, frame = cap.read()
    if not ret:
        break
    # Realizamos la inferencia de YOLO sobre el frame
    results = model(frame, conf=0.25, iou=0.45, imgsz=640, classes=[0, 1, 2])
    # Extraemos los resultados
    annotated_frame = results[0].plot()
    #print(annotated_frame)
    # Visualizamos los resultados
    cv2.imshow(" Maduracion del Kiwi ", annotated_frame)
    # El ciclo se rompe al presionar "Esc"
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()