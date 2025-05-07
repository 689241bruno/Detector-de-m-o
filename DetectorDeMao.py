import cv2
import mediapipe as mp

video = cv2.VideoCapture(1)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
photo_count = 0
while True:
    check, img = video.read()
    results = Hand.process(img)
    handsPoints = results.multi_hand_landmarks
    if handsPoints:
        for points in handsPoints:
            print(points)
        
    cv2.namedWindow('images', cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('images', 800, 600)
    cv2.imshow("images", img),
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()