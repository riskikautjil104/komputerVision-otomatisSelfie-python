import cv2
import mediapipe as mp
import time
import random
import string

# Membaca file cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menginisialisasi detektor tangan dari MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Menginisialisasi webcam
cap = cv2.VideoCapture(0)

is_index_finger_raised = False  # Flag untuk menandakan apakah jari telunjuk diangkat
start_time = None  # Waktu mulai penghitungan
count = 0  # Hitungan

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()

    # Mengubah frame menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Menandai kotak dan garis-garis di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x, y), (x+w, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y), (x, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x+w, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x, y+h), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x, y+h//2), (x+w, y+h//2), (255, 0, 0), 2)

    # Konversi frame BGR menjadi RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan menggunakan MediaPipe
    results = hands.process(rgb)

    # Menggambar garis-garis pada jari-jari tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Menggambar garis-garis antara titik-titik jari
            if len(hand_landmarks.landmark) == 21:
                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    x0, y0 = int(hand_landmarks.landmark[connection[0]].x * w), int(hand_landmarks.landmark[connection[0]].y * h)
                    x1, y1 = int(hand_landmarks.landmark[connection[1]].x * w), int(hand_landmarks.landmark[connection[1]].y * h)
                    cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Menghitung jumlah jari yang ditunjukkan
            finger_state = 0
            if cy < hand_landmarks.landmark[4].y * h:
                finger_state += 1
            if cy < hand_landmarks.landmark[8].y * h:
                finger_state += 1
            if cy < hand_landmarks.landmark[12].y * h:
                finger_state += 1
            if cy < hand_landmarks.landmark[16].y * h:
                finger_state += 1
            if cy < hand_landmarks.landmark[20].y * h:
                finger_state += 1

            # Mengubah jumlah jari menjadi teks dan menampilkan di frame
            if finger_state == 1:
                if not is_index_finger_raised:
                    count += 1
                    is_index_finger_raised = True
                    start_time = time.time()

                if count >= 1:
                    elapsed_time = int(time.time() - start_time)
                    cv2.putText(frame, f"Siap siap! Waktu: {elapsed_time} detik", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if elapsed_time >= 10:
                        count = 0
                        start_time = None
                        cv2.putText(frame, "Mengambil gambar...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
                        cv2.imwrite(f"captured_image_{random_string}.jpg", frame)
            else:
                is_index_finger_raised = False

    # Menampilkan frame yang telah ditandai
    cv2.imshow('Deteksi Wajah dan Jari', frame)

    # Menghentikan program jika tombol "q" ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan
cap.release()
cv2.destroyAllWindows()
