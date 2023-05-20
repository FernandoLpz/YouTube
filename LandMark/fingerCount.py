import cv2
import mediapipe as mp
import math

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def finger_count():
    cap = cv2.VideoCapture(1)

     # Initialize holistic model
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Transform frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference
        results = holistic.process(rgb_frame)

        # Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Right Hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            fingers = 0

            if results.left_hand_landmarks:
                fingers += count_valid_fingers(results.left_hand_landmarks.landmark)

            if results.right_hand_landmarks:
                fingers += count_valid_fingers(results.right_hand_landmarks.landmark)

            cv2.putText(frame, "Fingers: " + str(fingers), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        
        # Display
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def calculate_distance(point1, point2):
    x1, y1, z1 = point1.x, point1.y, point1.z
    x2, y2, z2 = point2.x, point2.y, point2.z

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance


def count_valid_fingers(hand_landmarks):
    finger_indices = [8, 12, 16, 20]

    wrist_landmark = hand_landmarks[0]
    finger_count = 0

    # Check if thumb is up
    thumb_wrist_distance = calculate_distance(hand_landmarks[4], wrist_landmark)
    if thumb_wrist_distance > 0.25:
        finger_count += 1

    for finger_index in finger_indices:
        finger_landmark = hand_landmarks[finger_index]
        finger_wrist_distance = calculate_distance(finger_landmark, wrist_landmark)

        if finger_wrist_distance > 0.3:
            finger_count += 1

    return finger_count




if __name__ == '__main__':
    finger_count()