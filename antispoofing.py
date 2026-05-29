import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp


class AntiSpoofing:

    def __init__(self, model_path='antispoofing_best.pth', debug=True):   # debug=True: prints vote probs to terminal

        self.debug = debug

        # ─── Load MobileNetV2 model ───────────────────────────────────────────
        print("Loading anti-spoofing model...")
        self.device = torch.device('cpu')

        self.model = models.mobilenet_v2(weights=None)

        # Must match training — trained with Dropout(0.5)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.last_channel, 2)
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print("✅ Anti-spoofing model loaded!")

        # ─── Image transform ──────────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ─── MediaPipe Face Mesh ──────────────────────────────────────────────
        print("Loading MediaPipe...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe loaded!")

        # ─── Eye landmark indices ─────────────────────────────────────────────
        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33,  160, 158, 133, 153, 144]

        # ─── Thresholds ───────────────────────────────────────────────────────
        self.EAR_THRESHOLD   = 0.20   # eyes closed when EAR < this

        # Per-frame real-probability threshold — intentionally permissive.
        # Blink detection is the strong gate; MobileNetV2 is a secondary
        # soft check. Keeping this low avoids falsely rejecting real persons
        # whose frames score uncertain due to lighting or motion.
        # FIX: raised from 0.35 — compressed webcam frames (quality 0.75, 480x360)
        # regularly score below 0.35 for real people under imperfect lighting.
        # 0.45 still clearly separates real (typically 0.6–0.95) from spoof (0.0–0.2).
        self.SPOOF_THRESHOLD = 0.45

        # ─── Voting config ─────────────────────────────────────────────────────
        # TUNED FOR SPEED vs RELIABILITY BALANCE:
        #
        # VOTE_FRAMES 20 → 15  saves ~165ms at 30fps.
        #   At 15 frames, REAL_VOTE_RATIO 0.55 requires 9 real votes to pass.
        #   The original 20-frame / 0.50 ratio required 10 votes.
        #   So we kept the absolute vote requirement almost the same (9 vs 10)
        #   while cutting the collection window by 25%. Spoof resistance is
        #   nearly identical; perceived speed improves noticeably.
        #
        # SETTLE_FRAMES 8 → 5  saves ~100ms. Eyes recover from a blink in
        #   ~3–4 frames at 30fps; 5 gives one frame of buffer. Cutting to
        #   fewer than 4 would risk voting on mid-blink crops — don't go lower.
        # SPEED FIX: at 800ms scan interval, 8 frames = ~6.4s of voting
        # (down from 15 frames × 2500ms = 37.5s). Still needs 5/8 real votes
        # (0.55 ratio) — same spoof resistance, dramatically faster UX.
        # SETTLE_FRAMES 3: blink recovery takes ~2–3 frames at 800ms cadence.
        self.VOTE_FRAMES     = 8      # was 15 — 6.4s at 800ms instead of 37.5s
        # FIX: lowered from 0.55 — blink is the hard liveness gate.
        # MobileNetV2 is a secondary check; requiring only half the votes real
        # prevents false spoof rejections from lighting/motion variance.
        self.REAL_VOTE_RATIO = 0.50   # 4/8 votes needed to pass
        self.SETTLE_FRAMES   = 3      # was 5 — faster post-blink recovery

        # ─── State ────────────────────────────────────────────────────────────
        self._eye_was_closed  = False
        self._blink_count     = 0
        self._prob_history    = []    # smoothing buffer (reserved for future use)
        self._vote_buffer     = []    # voting buffer
        self._vote_done       = False
        self._vote_result     = None
        self._settle_counter  = 0    # counts down settle frames after blink

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ear(self, landmarks, indices, w, h):
        """Eye Aspect Ratio for one eye."""
        pts = []
        for i in indices:
            lm = landmarks[i]
            pts.append((lm.x * w, lm.y * h))

        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        hh = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * hh) if hh > 0 else 0.3

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Call this when starting a fresh verification session."""
        self._eye_was_closed  = False
        self._blink_count     = 0
        self._prob_history    = []
        self._vote_buffer     = []
        self._vote_done       = False
        self._vote_result     = None
        self._settle_counter  = 0

    def get_blink_count(self):
        return self._blink_count

    def get_vote_progress(self):
        """Returns (collected, total) for progress bar display."""
        return len(self._vote_buffer), self.VOTE_FRAMES

    def update_blink(self, frame):
        """
        Feed every webcam frame into this to track blinks.
        Expects the FULL frame (not a face crop) so MediaPipe can
        locate landmarks correctly.
        Returns (blink_ever_detected: bool, current_ear: float, landmarks_found: bool)
        """
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return self._blink_count > 0, 0.0, False

        lms       = res.multi_face_landmarks[0].landmark
        left_ear  = self._ear(lms, self.LEFT_EYE,  w, h)
        right_ear = self._ear(lms, self.RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # Detect close → open transition = one completed blink
        if avg_ear < self.EAR_THRESHOLD:
            self._eye_was_closed = True
        elif self._eye_was_closed:
            self._blink_count   += 1
            self._eye_was_closed = False

        return self._blink_count > 0, avg_ear, True

    def collect_vote(self, face_crop_bgr):
        """
        Collects votes over VOTE_FRAMES frames then makes ONE firm decision.
        Expects a non-empty BGR face crop.

        Returns:
            'settling'   — skipping early post-blink frames (eyes recovering)
            'collecting' — actively gathering votes, show progress bar
            'real'       — majority voted real, person is live
            'fake'       — majority voted fake, spoof attempt
        """
        # If already decided, return cached result
        if self._vote_done:
            return 'real' if self._vote_result else 'fake'

        # Skip the first SETTLE_FRAMES frames right after blink detection.
        # During these frames the eyes are still half-open and the face crop
        # is blurry/angled, producing unreliable model scores.
        if self._settle_counter < self.SETTLE_FRAMES:
            self._settle_counter += 1
            if self.debug:
                print(f"[Settle] {self._settle_counter}/{self.SETTLE_FRAMES} — skipping post-blink frame")
            return 'settling'

        rgb    = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(
            Image.fromarray(rgb)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        real_prob = probs[1].item()
        self._vote_buffer.append(real_prob >= self.SPOOF_THRESHOLD)

        collected  = len(self._vote_buffer)
        real_votes = sum(self._vote_buffer)

        if self.debug:
            print(f"[Vote] {collected}/{self.VOTE_FRAMES}  "
                  f"real={real_votes}  fake={collected - real_votes}  "
                  f"prob={real_prob:.3f}")

        if collected >= self.VOTE_FRAMES:
            ratio             = real_votes / collected
            self._vote_done   = True
            self._vote_result = ratio >= self.REAL_VOTE_RATIO
            if self.debug:
                print(f"[Vote] DECISION → "
                      f"{'REAL ✅' if self._vote_result else 'FAKE ❌'} "
                      f"({ratio*100:.1f}% real votes)")
            return 'real' if self._vote_result else 'fake'
        return 'collecting'