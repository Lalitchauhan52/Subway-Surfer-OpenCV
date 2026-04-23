import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # Fastest
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_x, prev_y = 0, 0
move_threshold = 35
cooldown = 0.25
last_action_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Index finger tip = landmark 8
        index_finger = hand_landmarks.landmark[0]
        cx, cy = int(index_finger.x * w), int(index_finger.y * h)

        current_time = time.time()

        dx = cx - prev_x
        dy = cy - prev_y

        if current_time - last_action_time > cooldown:

            if dx > move_threshold:
                pyautogui.press("right")
                print("RIGHT")
                last_action_time = current_time

            elif dx < -move_threshold:
                pyautogui.press("left")
                print("LEFT")
                last_action_time = current_time

            elif dy < -move_threshold:
                pyautogui.press("up")
                print("JUMP")
                last_action_time = current_time

            elif dy > move_threshold:
                pyautogui.press("down")
                print("SLIDE")
                last_action_time = current_time

        prev_x, prev_y = cx, cy

        # Draw index finger point
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

    cv2.imshow("Index Finger Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
