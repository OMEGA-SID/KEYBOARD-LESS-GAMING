import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from pynput.keyboard import Controller, Key

mp_hands = mp.solutions.hands
draw_hands = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
keyboard = Controller()
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        h, w, _ = frame.shape  # Get height and width

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmlist = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_landmarks.landmark)]
                
                draw_hands.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label

                if len(lmlist) >= 21:  # Ensure all 21 landmarks exist
                    if hand_label == "Right":
                        if lmlist[8][2] < lmlist[6][2]:  # Index finger up
                            keyboard.press("w")
                        else:
                            keyboard.release("w")

                        if abs(lmlist[4][1] - lmlist[11][1]) > w * 0.05:  # Scaled thumb open detection
                            keyboard.press("a")
                        else:
                            keyboard.release("a")

                        if lmlist[12][2] < lmlist[10][2]:  # Middle finger up
                            keyboard.press("d")
                        else:
                            keyboard.release("d")

                        if lmlist[20][2] < lmlist[18][2]:  # Pinky finger up
                            keyboard.press("s")
                        else:
                            keyboard.release("s")

                    elif hand_label == "Left":
                        if (lmlist[8][2] < lmlist[6][2]) and (lmlist[12][2] < lmlist[10][2]):  # Both fingers up
                            keyboard.press(Key.shift)
                        else:
                            keyboard.release(Key.shift)

                        if lmlist[20][2] < lmlist[18][2]:  # Pinky finger up
                            keyboard.press("f")
                        else:
                            keyboard.release("f")

                        if abs(lmlist[4][1] - lmlist[11][1]) > 60:  # thumb open 
                            keyboard.press(Key.space)
                        else:
                            keyboard.release(Key.space)

        cv2.imshow('Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    for key in ["w", "a", "s", "d", "f"]:
        keyboard.release(key)
    keyboard.release(Key.shift)
    keyboard.release(Key.space)
    cap.release()
    cv2.destroyAllWindows()
