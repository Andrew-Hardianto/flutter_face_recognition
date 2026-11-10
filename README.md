# flutter_face_recognition

[![pub package](https://img.shields.io/pub/v/flutter_face_recognition.svg)](https://pub.dev/packages/flutter_face_recognition)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

High-performance, **100% offline** Face Detection, **192-D Face Embedding Recognition** (`MobileFaceNet`), **Cosine Similarity** matching, and **Enterprise Liveness Detection** (`liveness-v3`) with dual-model anti-spoofing (`MiniFASNet`) for Flutter.

---

## ✨ Key Features

- **Zero-Copy `Uint8List` Input**: Send raw encoded image bytes (`JPEG` / `PNG`) directly over `MethodChannel` without Base64 memory overhead (Base64 string input is also supported for backward compatibility).
- **192-Dimensional Face Embeddings (`extractFaceFeature`)**: Uses `mobile_face_net.tflite` ($112 \times 112$, RGB normalized to $[-1.0, 1.0]$) with built-in **L2-normalization**.
- **Dual-Model Silent-Face Anti-Spoofing (`checkLiveness`)**: Uses `anti_spoof_2_7.tflite` (scale `2.7`) + `anti_spoof_4_0.tflite` (scale `4.0`) with a boundary-shifting crop (`getNewBox`) and an `argmax` + hard-veto decision to detect printed photos, digital screens, and video replay attacks.
- **Pre-Liveness Quality Gate (`QualityResult`)**: deterministic checks for `NO_FACE`, `MULTIPLE_FACES`, `LOW_RESOLUTION`, `FACE_TOO_SMALL`, `FACE_AT_EDGE`, `BAD_POSE`, `BLURRY`, and `BAD_LIGHTING` before inference.
- **Anti-TOCTOU Verification (`verifyFace`)**: quality gate + dual anti-spoof + embedding on a **single frame & bounding box**; the embedding (and optional match) is returned only when the face is confirmed live.
- **Active Temporal Challenge (`createChallengeSession` + `verifyFaceSequence`)**: a randomized, single-use `LOOK_LEFT` / `LOOK_RIGHT` challenge evaluated by the R1–R5 trajectory rules, a zero-tolerance spoof veto, and pose-aware identity consistency.
- **Anti-Frame-Injection Native Capture (`captureLivenessSession`)**: frames are captured and evaluated entirely on-device, with `getDeviceIntegrity` (emulator / root / debuggable / hooks) and an encrypted audit trail.
- **Cross-Platform Face & Eye-Blink Detection (`detectFaces`)**:
  - **Android**: Google ML Kit `FaceDetection` (`CLASSIFICATION_MODE_ALL` & `LANDMARK_MODE_ALL`).
  - **iOS**: Apple Vision `VNDetectFaceRectanglesRequestRevision3` with continuous head angles and 6-point Euclidean Eye Aspect Ratio (EAR) mapped to `[0.0, 1.0]`.
- **Pure-Dart Cosine Similarity (`compareFaces`)**: Instant `< 0.1 ms` vector comparison directly in Dart without platform-channel latency.
- **Single-Source Bundled Models**: all four `.tflite` models are bundled once inside the plugin's Flutter assets and loaded natively on background threads (`Dispatchers.Default` on Android, `DispatchQueue.global` on iOS).

---

## 📱 Platform Requirements

| Platform | Minimum Version | Native Engines |
| :--- | :--- | :--- |
| **Android** | `minSdk 24` (Android 7.0+) | Google ML Kit Face Detection `16.1.7` + TensorFlow Lite `2.16.1` |
| **iOS** | `iOS 15.0+` | Apple Vision Framework + `TensorFlowLiteSwift 2.14.0` |

---

## 🚀 Quickstart

### 1. Installation

Add `flutter_face_recognition` to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_face_recognition: ^1.0.0
```

### 2. Enroll & Verify with Liveness (Anti-TOCTOU)

```dart
import 'dart:typed_data';
import 'package:flutter_face_recognition/flutter_face_recognition.dart';

final faceRecognition = const FaceRecognition();

Future<void> verifyAttendance(
  Uint8List enrolledPhotoBytes,
  Uint8List liveCameraBytes,
) async {
  // 1. Extract 192-D embedding during enrollment
  final enrolledVector = await faceRecognition.extractFaceFeature(
    bytes: enrolledPhotoBytes,
  );

  // 2. Unified verify: quality gate + dual anti-spoof + embedding on ONE frame
  final verify = await faceRecognition.verifyFace(
    bytes: liveCameraBytes,
    referenceEmbedding: enrolledVector,
    config: const GateConfig(liveThreshold: 0.80),
  );

  if (!verify.quality.passed) {
    print('Quality gate failed: ${verify.quality.reason}');
    return;
  }
  if (!verify.isLive) {
    print('Spoof detected (label: ${verify.liveness?.label.value})');
    return;
  }

  print('Match: ${verify.match?.isMatch} '
      '(${verify.match?.similarityPercentage.toStringAsFixed(1)}%)');
}
```

### 3. Active Challenge (production, anti frame-injection)

```dart
// 1. Issue a randomized, single-use challenge
final session = await faceRecognition.createChallengeSession();
// Show `session.challenge` (LOOK_LEFT / LOOK_RIGHT) to the user...

// 2. Run the native capture session end to end
final result = await faceRecognition.captureLivenessSession(
  referenceEmbedding: enrolledVector,
);

print('Passed: ${result.passed}, reason: ${result.reason}');
print('Trajectory R1-R5: ${result.trajectory.rules}');
print('Audit ID: ${result.auditId}, integrity: ${result.deviceIntegrity}');
```

> In release builds the external frame path (`verifyFaceSequence`) is disabled
> unless you call `initialize(enableInsecureFramePath: true)`. Prefer
> `captureLivenessSession` in production.

### 4. Face & Eye-Blink Detection (`detectFaces`)

```dart
final detection = await faceRecognition.detectFaces(bytes: imageBytes);

if (detection.hasSingleFace) {
  final face = detection.firstFace!;
  print('Bounding Box: (${face.x}, ${face.y}) ${face.width}x${face.height}');
  print('Pitch: ${face.headEulerAngleX}°, Yaw: ${face.headEulerAngleY}°, '
      'Roll: ${face.headEulerAngleZ}°');
  print('Eyes Open: ${face.isBothEyesOpen()}, Blinking: ${face.isBlinking()}');
}
```

### 5. Cleanup Resources

```dart
await faceRecognition.dispose();
```

---

## ⚙️ Recommended Thresholds

| Use Case | Cosine Similarity Threshold | Liveness `liveThreshold` |
| :--- | :---: | :---: |
| **General Consumer App** | `0.70` | `0.80` |
| **Standard Verification (Default)** | `0.75` | `0.80` |
| **High-Security / HRIS Attendance** | `0.80` – `0.85` | `0.80` – `0.85` |

> **Note (`liveness-v3`):** the liveness `score` is `argmax`-based and is **not**
> comparable with 1.x scores. The decision is
> `label == LIVE && liveScore >= liveThreshold && spoofScore < spoofVeto`.

---

## 📄 License

MIT License — Copyright (c) 2026 Andrew Hardianto.
