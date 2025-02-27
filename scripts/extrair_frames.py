import cv2
import os

video_folder = "data/videos"
output_folder = "data/frames"
frame_rate = 30

os.makedirs(output_folder, exist_ok=True)

videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video in videos:
    video_path = os.path.join(video_folder, video)
    video_name = os.path.splitext(video)[0]
    video_output_folder = os.path.join(output_folder, video_name)

    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            filename = os.path.join(video_output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extração concluída para {video}: {saved_count} frames salvos em '{video_output_folder}'")
print("Todos os vídeos foram processados!")