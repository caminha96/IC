from ultralytics import YOLO
import cv2
import time

model = YOLO("./runs/detect/train11/weights/best.pt")

tempo_em_pe = 0
tempo_sentado = 0
tempo_deitado = 0
tempo_nao_detectado = 0
estado_atual = "não detectado"
tempo_inicial = time.time()

def atualizar_tempo(estado, duracao):
    global tempo_em_pe, tempo_sentado, tempo_deitado, tempo_nao_detectado
    if estado == "em pé":
        tempo_em_pe += duracao
    elif estado == "sentado":
        tempo_sentado += duracao
    elif estado == "deitado":
        tempo_deitado += duracao
    else:
        tempo_nao_detectado += duracao

video_path = './data/video_final/video_teste.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)


    detections = results[0].boxes
    classes_detectadas = [model.names[int(box.cls)] for box in detections]


    if classes_detectadas:


        if "em pé" in classes_detectadas:
            classe_atual = "em pé"
        elif "sentado" in classes_detectadas:
            classe_atual = "sentado"
        elif "deitado" in classes_detectadas:
            classe_atual = "deitado"
        else:
            classe_atual = "não detectado"
    else:
        classe_atual = "não detectado"


    tempo_atual = time.time()
    duracao = tempo_atual - tempo_inicial

    if classe_atual != estado_atual:

        atualizar_tempo(estado_atual, duracao)

        estado_atual = classe_atual

        tempo_inicial = time.time()

atualizar_tempo(estado_atual, time.time() - tempo_inicial)

tempo_total = tempo_em_pe + tempo_sentado + tempo_deitado + tempo_nao_detectado

print(f"Tempo em pé: {tempo_em_pe:.2f} segundos ({(tempo_em_pe / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo sentado: {tempo_sentado:.2f} segundos ({(tempo_sentado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo deitado: {tempo_deitado:.2f} segundos ({(tempo_deitado / tempo_total * 100):.2f}% do tempo total)")
print(f"Tempo com usuário não detectado: {tempo_nao_detectado:.2f} segundos ({(tempo_nao_detectado / tempo_total * 100):.2f}% do tempo total)")
print(f"Duração total: {tempo_total:.2f} segundos")

cap.release()
cv2.destroyAllWindows()