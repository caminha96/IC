from ultralytics import YOLO
import cv2
import time

model = YOLO("./runs/detect/train18/weights/best.pt")

# #Para testar com vídeo:
# tempo_em_pe = 0
# tempo_sentado = 0
# tempo_deitado = 0
# tempo_nao_detectado = 0
#
# video_path = './data/video_final/video_teste.mp4'
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# tempo_por_frame = 1 / fps
# CLASSES_ESPERADAS = ["de_pe", "sentado", "deitado"]
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model(frame, conf=0.25, imgsz=640, max_det=1, agnostic_nms=True)
#     detections = results[0].boxes
#     classes_filtradas = [box for box in detections if model.names[int(box.cls)] in CLASSES_ESPERADAS]
#     if classes_filtradas:
#         classe_mais_confiavel = max(classes_filtradas, key=lambda x: x.conf)
#         classe_atual = model.names[int(classe_mais_confiavel.cls)]
#     else:
#         classe_atual = "não detectado"
#     if classe_atual == "de_pe":
#         tempo_em_pe += tempo_por_frame
#     elif classe_atual == "sentado":
#         tempo_sentado += tempo_por_frame
#     elif classe_atual == "deitado":
#         tempo_deitado += tempo_por_frame
#     else:
#         tempo_nao_detectado += tempo_por_frame
#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLOv8 - Monitoramento de Idosos", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     cap.release()
#     cv2.destroyAllWindows()

# Para testar com webcam:
cap = cv2.VideoCapture(0)  # 0 representa a webcam padrão
cap.set(3, 640)  # Largura
cap.set(4, 480)  # Altura

tempo_em_pe = 0
tempo_sentado = 0
tempo_deitado = 0
tempo_nao_detectado = 0

classe_anterior = None

tempo_anterior = time.time()

CLASSES_ESPERADAS = ["de_pe", "sentado", "deitado"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tempo_atual = time.time()
    tempo_por_frame = tempo_atual - tempo_anterior
    tempo_anterior = tempo_atual
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
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Em pé: {tempo_em_pe:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Sentado: {tempo_sentado:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"Deitado: {tempo_deitado:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Nao detectado: {tempo_nao_detectado:.1f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("YOLOv8 - Monitoramento em Tempo Real", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

tempo_total = tempo_em_pe + tempo_sentado + tempo_deitado + tempo_nao_detectado
print(f"Tempo em pe: {tempo_em_pe:.2f} segundos ({(tempo_em_pe / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo sentado: {tempo_sentado:.2f} segundos ({(tempo_sentado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo deitado: {tempo_deitado:.2f} segundos ({(tempo_deitado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo não detectado: {tempo_nao_detectado:.2f} segundos ({(tempo_nao_detectado / tempo_total * 100):.2f}% do tempo total)")
print(f"Duração total: {tempo_total:.2f} segundos")
