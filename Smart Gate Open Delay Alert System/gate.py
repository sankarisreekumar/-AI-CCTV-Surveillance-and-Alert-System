import cv2
import numpy as np
import time
import winsound
import os
import datetime
import requests

# ---------------- TELEGRAM ---------------- #
TOKEN = "8515976739:AAG7mX_1ACz1p4iqBFkklx_AHGIFcBmxF28"
CHAT_ID = "-1003660693269"

# ---------------- THRESHOLDS ---------------- #
CLOSED_THRESH = 0.65
OPEN_THRESH = 0.35

PARTIAL_TIME_THRESH = 2
FULL_TIME_THRESH = 5


# -------- SEND TELEGRAM ALERT -------- #
def send_telegram_alert(image_path, message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    try:

        with open(image_path, "rb") as img:

            files = {
                "photo": img
            }

            data = {
                "chat_id": CHAT_ID,
                "caption": message
            }

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=15
            )

        print("✅ Telegram Alert Sent")
        print(response.text)

    except requests.exceptions.Timeout:
        print("❌ Telegram Timeout")

    except requests.exceptions.ConnectionError:
        print("❌ Internet Connection Error")

    except Exception as e:
        print("❌ Telegram Error:", e)


# -------- SAVE ALERT IMAGE -------- #
def save_alert_image(frame, state):

    now = datetime.datetime.now()

    date_folder = now.strftime("%Y-%m-%d")
    time_name = now.strftime("%H-%M-%S")

    folder_path = os.path.join("alerts", date_folder)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{time_name}.jpg"
    )

    cv2.imwrite(file_path, frame)

    print(f"📸 Saved Alert: {file_path}")

    send_telegram_alert(
        file_path,
        f"🚨 ALERT: {state}\n🕒 Time: {time_name}"
    )


class DynamicGateAlertSystem:

    def __init__(self, video_source):

        self.video_source = video_source

        self.roi_pts = []
        self.temp_pts = []

        self.bboxes = []
        self.templates = []

        self.ref_frame = None

        self.current_state = "UNKNOWN"

        self.partial_start = None
        self.full_start = None

    # -------- ALERT SOUND -------- #
    def trigger_alert(self, state):

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"[{timestamp}] ALERT: {state}")

        try:

            if state in ["GATE IS OPEN", "GATE FULL OPEN"]:
                winsound.Beep(1000, 500)

            elif state == "CLOSED":
                winsound.Beep(500, 300)

        except:
            pass

    # -------- PREPROCESS -------- #
    def get_preprocessed(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        return clahe.apply(gray)

    # -------- ROI SELECTION -------- #
    def select_rois(self, frame):

        def mouse_cb(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_pts.append((x, y))

        window_name = "Select LEFT (4 pts) then RIGHT (4 pts)"

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(
            window_name,
            mouse_cb
        )

        print(
            "Click LEFT gate (4 pts), "
            "then RIGHT gate (4 pts). "
            "Press 's' to save."
        )

        while True:

            img = frame.copy()

            for p in self.temp_pts:

                cv2.circle(
                    img,
                    p,
                    5,
                    (0,0,255),
                    -1
                )

            for i in range(0, len(self.temp_pts), 4):

                pts_chunk = self.temp_pts[i:i+4]

                if len(pts_chunk) > 1:

                    for j in range(len(pts_chunk)-1):

                        cv2.line(
                            img,
                            pts_chunk[j],
                            pts_chunk[j+1],
                            (255,0,0),
                            2
                        )

                    if len(pts_chunk) == 4:

                        cv2.line(
                            img,
                            pts_chunk[3],
                            pts_chunk[0],
                            (255,0,0),
                            2
                        )

            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and len(self.temp_pts) >= 8:
                break

            elif key == ord('r'):
                self.temp_pts = []

        cv2.destroyWindow(window_name)

        self.ref_frame = self.get_preprocessed(frame)

        for i in range(0, 8, 4):

            pts = np.array(
                self.temp_pts[i:i+4],
                dtype=np.int32
            )

            self.roi_pts.append(pts)

            x, y, w, h = cv2.boundingRect(pts)

            self.bboxes.append((x, y, w, h))

            template = self.ref_frame[y:y+h, x:x+w]

            self.templates.append(
                template.astype(np.float32)
            )

    # -------- MAIN SYSTEM -------- #
    def run(self):

        cap = cv2.VideoCapture(self.video_source)

        ret, frame = cap.read()

        if not ret:
            print("❌ Error loading source")
            return

        self.select_rois(frame)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            curr_frame = self.get_preprocessed(frame)

            gate_states = []

            # -------- DETECTION -------- #
            for i in range(2):

                x, y, w, h = self.bboxes[i]

                roi = curr_frame[y:y+h, x:x+w]

                template = self.templates[i].astype(np.uint8)

                score = cv2.matchTemplate(
                    roi,
                    template,
                    cv2.TM_CCOEFF_NORMED
                )[0][0]

                if score > CLOSED_THRESH:
                    gate_states.append("CLOSED")

                elif score < OPEN_THRESH:
                    gate_states.append("OPEN")

                else:
                    gate_states.append("PARTIAL")

            current_time = time.time()

            new_state = self.current_state

            any_open = "OPEN" in gate_states

            both_open = (
                gate_states[0] == "OPEN" and
                gate_states[1] == "OPEN"
            )

            both_closed = (
                gate_states[0] == "CLOSED" and
                gate_states[1] == "CLOSED"
            )

            # -------- PARTIAL OPEN -------- #
            if any_open:

                if self.partial_start is None:
                    self.partial_start = current_time

                partial_elapsed = (
                    current_time -
                    self.partial_start
                )

                if partial_elapsed >= PARTIAL_TIME_THRESH:
                    new_state = "GATE IS OPEN"

                else:
                    new_state = "OPENING"

            else:
                self.partial_start = None

            # -------- FULL OPEN -------- #
            if both_open:

                if self.full_start is None:
                    self.full_start = current_time

                full_elapsed = (
                    current_time -
                    self.full_start
                )

                if full_elapsed >= FULL_TIME_THRESH:
                    new_state = "GATE IS OPEN"

            else:
                self.full_start = None

            # -------- CLOSED -------- #
            if both_closed:

                self.partial_start = None
                self.full_start = None

                new_state = "CLOSED"

            # -------- ALERT -------- #
            if new_state != self.current_state:

                self.current_state = new_state

                self.trigger_alert(
                    self.current_state
                )

                if self.current_state in [
                    "GATE IS OPEN",
                    "GATE FULL OPEN"
                ]:

                    save_alert_image(
                        frame,
                        self.current_state
                    )

            # -------- DRAW -------- #
            for i in range(2):

                if gate_states[i] == "CLOSED":
                    color = (0,255,0)

                else:
                    color = (255,0,0)

                cv2.polylines(
                    frame,
                    [self.roi_pts[i]],
                    True,
                    color,
                    2
                )

            # -------- ALERT BOX -------- #
            overlay = frame.copy()

            if self.current_state == "GATE FULL OPEN":

                if int(time.time()*2) % 2 == 0:
                    bg = (0,0,255)
                else:
                    bg = (0,0,0)

            elif self.current_state == "GATE IS OPEN":
                bg = (0,0,255)

            elif self.current_state == "OPENING":
                bg = (255,0,0)

            else:
                bg = (0,255,0)

            cv2.rectangle(
                overlay,
                (10,10),
                (520,90),
                bg,
                -1
            )

            cv2.addWeighted(
                overlay,
                0.8,
                frame,
                0.2,
                0,
                frame
            )

            cv2.putText(
                frame,
                self.current_state,
                (20,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255,255,255),
                3
            )

            cv2.imshow(
                "Gate Alert System",
                frame
            )

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()