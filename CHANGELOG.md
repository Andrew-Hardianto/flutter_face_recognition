## 1.0.0 — `liveness-v3`

Major upgrade to the enterprise, defense-in-depth liveness pipeline. **Breaking:**
the liveness score semantics changed (now `argmax`-based ensemble); `score` is
NOT comparable with 1.x. `checkLiveness`'s `threshold` is now optional (defaults
to the native `liveness-v3` `0.80`).

* **Dual-model silent-face anti-spoofing:** replaced the single-model `1.0x` +
  `2.7x` crops with the official ensemble `anti_spoof_2_7.tflite` (MiniFASNet
  scale 2.7) + `anti_spoof_4_0.tflite` (MiniFASNet scale 4.0), the `getNewBox`
  boundary-shifting crop, and an `argmax` + hard-veto decision
  (`liveThreshold` 0.80, `spoofVeto` 0.20). `LivenessResult` now reports
  `label`, `liveScore`, `printScore`, `replayScore`, `spoofScore`, and
  `thresholdVersion` (`liveness-v3`).
* **Pre-liveness quality gate** (`QualityResult`): `NO_FACE`, `MULTIPLE_FACES`,
  `LOW_RESOLUTION`, `FACE_TOO_SMALL`, `FACE_AT_EDGE`, `BAD_POSE`, `BLURRY`
  (variance-of-Laplacian), `BAD_LIGHTING` (mean luminance).
* **Anti-TOCTOU `verifyFace`:** quality gate + dual anti-spoof + MobileFaceNet
  embedding on a single frame & bounding box; embedding/match only returned when
  live.
* **Active temporal challenge:** `createChallengeSession` (randomized, single-use,
  60s TTL) + `verifyFaceSequence` evaluated by the R1–R5 `ChallengeTrajectory`,
  zero-tolerance spoof veto, and pose-aware identity consistency.
* **Anti-frame-injection:** `captureLivenessSession` native camera capture +
  `getDeviceIntegrity` (emulator / root / debuggable / hooks) + encrypted
  `AuditStore`. The external frame path is blocked in release unless
  `initialize(enableInsecureFramePath: true)`.
* **Single-source asset bundling:** all four `.tflite` models load from
  `assets/models/` via the Flutter asset lookup on both platforms.
* **Senior-review fixes:** `NSLock` around the iOS rate limiter & challenge
  session; `.recycle()` on intermediate bitmaps in the Android Laplacian /
  luminance metrics; configurable `minLiveVoteRatio` alongside the hard veto.
* New typed models: `GateConfig`, `QualityResult`, `VerifyFaceResult`,
  `ChallengeSession`, `TrajectoryResult`, `SequenceVerifyResult`,
  `DeviceIntegrityResult`, `LivenessLabel`, `ChallengeDirection`.
* **Android:** fixed manifest-merge failure in consumer apps on AGP 8.x+
  (`Namespace 'org.tensorflow.lite' is used in multiple modules and/or
  libraries`) by upgrading `org.tensorflow:tensorflow-lite` `2.14.0` →
  `2.16.1` — the transitive `tensorflow-lite-api` declares its own namespace
  from that release — and removing the unused `tensorflow-lite-support:0.4.4`
  (its `-api` split had the same duplicate-namespace problem).
* **Android:** migrated to Flutter's **built-in Kotlin** (no longer applies
  `org.jetbrains.kotlin.android` itself, and no longer declares its own AGP /
  Kotlin classpaths). Requires Flutter `>=3.44.0` / Dart `>=3.12.0`.

## 0.0.1

* Initial release of `flutter_face_recognition`.
* Added offline Face Detection (`detectFaces`) using Google ML Kit on Android and Apple Vision (`VNDetectFaceLandmarksRequest` with normalized Eye Aspect Ratio) on iOS.
* Added 192-dimensional Face Embedding Extraction (`extractFaceFeature`) powered by MobileFaceNet (`mobile_face_net.tflite`) with L2-normalization.
* Added Multi-Scale Passive Liveness / Anti-Spoofing Detection (`checkLiveness`) powered by MiniFASNet (`anti_spoof.tflite`) using `1.0x` and `2.7x` crops.
* Added pure-Dart Cosine Similarity verification (`compareFaces`) with zero platform-channel overhead.
* Added support for both `Uint8List` raw encoded bytes and `base64` image inputs, plus `dispose()` resource cleanup.

