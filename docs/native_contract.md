# Native Wire Contract — `flutter_face_recognition` (`liveness-v3`)

This is the single source of truth for the method-channel contract that BOTH the
Android (Kotlin) and iOS (Swift) plugins must implement. Channel name:
`flutter_face_recognition`.

The typed Dart side lives in `lib/src/models.dart`,
`lib/flutter_face_recognition_method_channel.dart`, and
`lib/flutter_face_recognition_platform_interface.dart`. Keep every key below in
sync with those files.

## Image input convention

Every single-image method accepts **either**:
- `bytes` — `Uint8List` (encoded JPEG/PNG), preferred (zero-copy), OR
- `base64` — `String` (optionally a `data:...;base64,` URL).

Native must normalize EXIF orientation before detection. Multi-frame methods use
`frames` = `List<Uint8List>` (encoded images) — see `verifyFaceSequence`.

## Methods

### `initialize(args)` → `null`
- args: `{ enableInsecureFramePath: bool }`
- Stores the plugin-level flag used by `verifyFaceSequence`'s release guard.

### `extractFaceFeature(args)` → `{ embedding: number[192] }`
- args: image input. Uses `pickLargestFace`. L2-normalized 192-dim embedding.

### `checkLiveness(args)` → see `checkLiveness` result below
- args: image input + optional `threshold: double` (maps to `liveThreshold`).
- Runs the dual-model ensemble (`2.7` + `4.0`), argmax + veto.

**`checkLiveness` result (flat):**
```
{ isLive, score, confidence, label, liveScore, printScore, replayScore,
  spoofScore, screenSuspectScore, thresholdVersion }
```
- `score` == `liveScore` (backward-compat alias).
- `confidence` ∈ `HIGH|MEDIUM|LOW` from `liveScore`.
- `label` ∈ `LIVE|PRINT|REPLAY`.
- `thresholdVersion` = `"liveness-v3"`.

### `detectFaces(args)` → `{ count, faces: [face] }`
- args: image input. Uses the FAST meta detector.
- `face`: `{ x, y, width, height, headEulerAngleX, headEulerAngleY,
  headEulerAngleZ, leftEyeOpenProbability?, rightEyeOpenProbability? }`
- `headEulerAngleY` MUST be normalized via `YawConvention`
  (positive = user turns toward their anatomical right).

### `verifyFace(args)` → see `verifyFace` result below
- args: image input + optional `referenceEmbedding: number[]` +
  optional `config: GateConfig`.
- Single frame: Quality Gate → dual anti-spoof → embedding on the SAME bbox.
- Embedding/match only present when `isLive == true`.

**`verifyFace` result:**
```
{ isLive, thresholdVersion,
  quality: { passed, reason?, faceRatio, blur },
  liveness: { isLive, label, liveScore, printScore, replayScore, spoofScore,
              screenSuspectScore } | null,
  embedding?: number[192],
  match?: { isMatch, score, threshold, thresholdVersion },
  debug?: { probs27: number[3], probs40: number[3] }  // debug builds only
}
```
- When the quality gate fails: `isLive=false`, `liveness=null`, no `embedding`,
  and `quality.reason` set.

### `createChallengeSession(args)` → `{ challengeId, challenge, expiresAt }`
- args: none. Issues a random `LOOK_LEFT`/`LOOK_RIGHT`, single-use, TTL 60s.
- `expiresAt` = epoch ms (number).
- Rate-limited (see below).

### `verifyFaceSequence(args)` → see `sequence` result below
- args: `frames: List<Uint8List>` (≥ 2, recommend ≥ 5) + `challengeId: string` +
  optional `referenceEmbedding` + optional `config`.
- **Release guard:** reject with code `INSECURE_FRAME_PATH_DISABLED` when the
  build is NOT debuggable AND `enableInsecureFramePath` is false.
- Challenge direction is read from the native session (NOT from args); the
  session is consumed on first use. Errors: `CHALLENGE_NOT_FOUND`,
  `CHALLENGE_EXPIRED`, `CHALLENGE_CONSUMED`.
- Also accepts `images: List<base64 string>` as an alternative to `frames`
  (kept for reference/back-compat), but the Dart SDK sends `frames`.

### `captureLivenessSession(args)` → `sequence` result + native extras
- args: optional `referenceEmbedding` + optional `config`.
- Opens native camera UI (`LivenessCaptureActivity` / `LivenessCaptureViewController`),
  warms up 1200ms, captures 12 frames @ 120ms to a private cache dir, evaluates
  in-memory, deletes the frames. Returns result with `native: true`.
- Rate-limited. Requires camera permission (request if needed).
- Extra result keys: `native: true`, `deviceIntegrity: {...}`, `auditId: string|null`.

### `getDeviceIntegrity(args)` → `{ deviceIntegrity: {...} }`
- `deviceIntegrity`: `{ isEmulator, isRooted, isDebuggable, hooksDetected }`.

### `dispose(args)` → `null`
- Releases interpreters + detectors.

## `sequence` result (verifyFaceSequence / captureLivenessSession)

```
{ passed, reason?, challenge, medianLiveScore, liveVotes, frameCount,
  maxYaw, minYaw, frontalFrames, identityConsistent,
  minIdentityCosine: number|null,
  trajectory: {
    rulesPassed: { frontalStart, deviation, returnFrontal, singleDirection,
                   smoothMotion },
    startYaw, endYaw, extremeYaw, extremeFrame, wrongWayMax, deltaVariance,
    tooSmooth, durationMs
  },
  screenSuspect, thresholdVersion,
  challengeId, native,
  match?: { isMatch, score, threshold, thresholdVersion },
  deviceIntegrity?: {...},   // captureLivenessSession only
  auditId?: string|null }    // captureLivenessSession only
```

## `GateConfig` argument map
All optional; native falls back to defaults:
`liveThreshold` (0.80), `spoofVeto` (0.20), `minFrameSide` (480),
`minFaceRatio` (0.25), `edgeMargin` (0.05), `maxPose` (20), `blurMin` (50),
`exposureMin` (60), `exposureMax` (200), `identityThreshold` (0.70),
`yawSignOverride` (1), `minLiveVoteRatio` (1.0), `screenSuspectVeto` (null),
`debug` (false — only honored on debuggable builds).

## Error codes (result.error / PlatformException.code)
`MODEL_NOT_LOADED`, `NO_FACE_DETECTED`, `INVALID_IMAGE_INPUT`,
`IMAGE_DECODE_FAILED`, `EMPTY_EMBEDDING`, `RATE_LIMITED`,
`INSECURE_FRAME_PATH_DISABLED`, `CHALLENGE_NOT_FOUND`, `CHALLENGE_EXPIRED`,
`CHALLENGE_CONSUMED`, `CAMERA_PERMISSION_DENIED`.

## Rate limiting
`createChallengeSession` & `captureLivenessSession`: max 10 attempts per 5-minute
window → `RATE_LIMITED`. Guard the attempt list with a lock
(`synchronized` on Android, `NSLock` on iOS).

## Models (assets)
Loaded from Flutter assets via the platform asset lookup:
- `mobile_face_net.tflite` (112×112, 192-dim, RGB normalized `(x/128)-1`)
- `anti_spoof_2_7.tflite` (crop scale 2.7, 80×80, raw BGR `[0..255]`)
- `anti_spoof_4_0.tflite` (crop scale 4.0, 80×80, raw BGR `[0..255]`)
- `anti_spoof.tflite` (legacy fallback for `anti_spoof_2_7`)

Anti-spoof output: 3 logits `[print, live, replay]`. Ensemble:
`p = softmax(z27) + softmax(z40)` (range 0..2);
`printScore=p[0]/2`, `liveScore=p[1]/2`, `replayScore=p[2]/2`,
`spoofScore=max(p[0],p[2])/2`, `label=argmax(p)`;
`isLive = label==LIVE && liveScore >= liveThreshold && spoofScore < spoofVeto`.

Anti-spoof crop uses `getNewBox` (boundary-shifting, width/height scaled
SEPARATELY, no squaring, no `1.0` scale). See the reference Java
`FaceRecognitionPlugin.getNewBox`.
