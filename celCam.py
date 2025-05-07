import cv2

video = cv2.VideoCapture(1)


photo_count = 0
while True:
    check, img = video.read()
    cv2.namedWindow('images', cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('images', 800, 600)
    cv2.imshow("images", img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        photo_name = f"./imgs/photo_{photo_count}.jpg"
        cv2.imwrite(photo_name, img)
        print(f"Foto salva como {photo_name}")
        photo_count += 1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()