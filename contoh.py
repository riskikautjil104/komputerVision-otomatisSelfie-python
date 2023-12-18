import cv2
import mediapipe as mp

# Fungsi untuk mendeteksi anggukan jari telunjuk
def detect_finger_gesture(landmarks, image_shape):
    tip_of_index_finger = landmarks[8]  # Koordinat landmark ujung jari telunjuk (indeks 8)

    # Dapatkan koordinat y dari landmark jari telunjuk
    y = int(tip_of_index_finger.y * image_shape[0])

    return y < 250  # Jika koordinat y jari telunjuk lebih rendah dari threshold, mengembalikan True

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

countdown = 10
photo_taken = False

while True:
    # Baca setiap frame dari webcam
    ret, frame = cap.read()

    # Ubah frame menjadi RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan dalam frame
    results = mp_hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Deteksi anggukan jari telunjuk
            finger_up = detect_finger_gesture(hand_landmarks.landmark, frame.shape)

            # Jika jari telunjuk terangkat, mulai hitung mundur
            if finger_up:
                countdown -= 1
                if countdown == 0:
                    # Ambil foto
                    cv2.imwrite('selfie.jpg', frame)
                    print("Foto berhasil diambil!")
                    photo_taken = True

    if not photo_taken:
        # Tampilkan pesan "Siap-siap"
        cv2.putText(frame, 'Siap-siap', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Tampilkan hitungan mundur
        cv2.putText(frame, str(countdown), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('Finger Gesture Detection', frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) == ord('q'):
        break

# Tutup webcam dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
