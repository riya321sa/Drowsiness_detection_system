import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from twilio.rest import Client
import time

# Initialize mixer for alert sound
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds and consecutive frame checks for triggering alerts
EYE_THRESH = 0.25
MOUTH_THRESH = 0.7
EYE_FRAME_CHECK = 20
MOUTH_FRAME_CHECK = 15

# Load dlib's face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get the indexes of the facial landmarks for the left and right eye and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Initialize variables
eye_flag = 0
mouth_flag = 0
alert_sent = False  # To track if an alert has been sent

# Twilio setup
account_sid = 'AC4c1d456d3749cb18b37479d5312728a0'
auth_token = '5f0d283680ad8cba46a599eb9d346e56'
client = Client(account_sid, auth_token)

def send_alert_sms(message_body, to_phone):
    message = client.messages.create(
        body=message_body,
        from_='+17372500415',  # My Twilio number
        to=to_phone      # My phone number
    )
    print(f"Alert SMS sent: {message.sid}")

start_time = time.time()
max_driving_duration = 2 * 60 * 60  # 2 hours in seconds

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract eye and mouth coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate eye and mouth aspect ratios
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the blink threshold
        if ear < EYE_THRESH:
            eye_flag += 1
            if eye_flag >= EYE_FRAME_CHECK and not alert_sent:
                cv2.putText(frame, "ALERT! DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
                send_alert_sms("ALERT! Drowsiness detected!", '+916203498729')  # Family Phone Number
                alert_sent = True  # Alert has been sent
        else:
            if alert_sent and eye_flag >= EYE_FRAME_CHECK:
                send_alert_sms("Relax, Driver is now awake.", '+916203498729')  # Family phone number
                alert_sent = False  # Alert sent status changed
            eye_flag = 0

        # Check if the mouth aspect ratio is above the yawn threshold
        if mar > MOUTH_THRESH:
            mouth_flag += 1
            if mouth_flag >= MOUTH_FRAME_CHECK:
                cv2.putText(frame, "ALERT! YAWN DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
                send_alert_sms("ALERT! Yawn detected!", '+916203498729')  # Family Phone Number
        else:
            mouth_flag = 0

    elapsed_time = time.time() - start_time
    if elapsed_time > max_driving_duration:
        send_alert_sms("ALERT! You have been driving for over 2 hours. Please take a break.", '+916203498729')  # Driver's phone number
        start_time = time.time()  # Reset the timer after sending the alert

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
