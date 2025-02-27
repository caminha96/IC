from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/train18/weights/best.pt")

#Para testar com vídeo:
tempo_em_pe = 0
tempo_sentado = 0
tempo_deitado = 0
tempo_nao_detectado = 0

video_path = './data/video_final/video_teste.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
tempo_por_frame = 1 / fps
CLASSES_ESPERADAS = ["de_pe", "sentado", "deitado"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.25, imgsz=640, max_det=1, agnostic_nms=True)
    detections = results[0].boxes
    classes_filtradas = [box for box in detections if model.names[int(box.cls)] in CLASSES_ESPERADAS]
    if classes_filtradas:
        classe_mais_confiavel = max(classes_filtradas, key=lambda x: x.conf)
        classe_atual = model.names[int(classe_mais_confiavel.cls)]
    else:
        classe_atual = "não detectado"
    if classe_atual == "de_pe":
        tempo_em_pe += tempo_por_frame
    elif classe_atual == "sentado":
        tempo_sentado += tempo_por_frame
    elif classe_atual == "deitado":
        tempo_deitado += tempo_por_frame
    else:
        tempo_nao_detectado += tempo_por_frame

tempo_total = tempo_em_pe + tempo_sentado + tempo_deitado + tempo_nao_detectado
print(f"Tempo em pé: {tempo_em_pe:.2f} segundos ({(tempo_em_pe / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo sentado: {tempo_sentado:.2f} segundos ({(tempo_sentado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo deitado: {tempo_deitado:.2f} segundos ({(tempo_deitado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo com usuário não detectado: {tempo_nao_detectado:.2f} segundos ({(tempo_nao_detectado / tempo_total * 100):.2f}% do tempo total)")
print(f"Duração total: {tempo_total:.2f} segundos")


#Para testar com webcam:
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model(frame, conf=0.3)
#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()
