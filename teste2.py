import cv2
import mediapipe as mp
import threading

# Configuração do MediaPipe para detecção de gestos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Substitua '0' pelo índice correto da sua webcam
webcam_index = 1
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

# Variáveis para sincronização entre threads
frame_to_analyze = None
result_from_thread = None
lock = threading.Lock()
terminate_program = False

def is_thumbs_up(hand_landmarks):
    """
    Verifica se o gesto de 'joinha' (polegar para cima) está sendo feito.
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Verifica se o polegar está levantado e os outros dedos estão abaixados
    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    fingers_down = (
        index_tip.y > index_mcp.y
    )  # Pode adicionar condições para os outros dedos

    return thumb_up and fingers_down

def analyze_frame():
    """
    Função que será executada em uma thread separada para analisar frames.
    """
    global frame_to_analyze, result_from_thread, terminate_program

    while True:
        # Bloqueia o acesso ao frame compartilhado
        lock.acquire()
        if frame_to_analyze is None:
            lock.release()
            continue

        # Processa o frame
        frame_rgb = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        result_from_thread = result

        # Verifica o gesto de "joinha"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if is_thumbs_up(hand_landmarks):
                    print("Joinha detectado! Encerrando o programa...")
                    terminate_program = True
                    lock.release()
                    return

        frame_to_analyze = None  # Reseta o frame após a análise
        lock.release()

# Inicializa a thread de análise
analysis_thread = threading.Thread(target=analyze_frame, daemon=True)
analysis_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo")
        break

    # Exibe o vídeo ao vivo
    cv2.imshow("Vídeo ao vivo", frame)

    # Envia o frame para análise se não houver nenhum em processamento
    lock.acquire()
    if frame_to_analyze is None:
        frame_to_analyze = frame.copy()
    lock.release()

    # Mostra resultados no vídeo (se disponíveis)
    if result_from_thread and result_from_thread.multi_hand_landmarks:
        for hand_landmarks in result_from_thread.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Detecção de gestos", frame)

    # Pressione 'q' para sair manualmente
    if cv2.waitKey(1) & 0xFF == ord('q') or terminate_program:
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
