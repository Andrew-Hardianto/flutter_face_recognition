# flutter_face_recognition

[![pub package](https://img.shields.io/pub/v/flutter_face_recognition.svg)](https://pub.dev/packages/flutter_face_recognition)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

High-performance, **100% offline** Face Detection, **192-D Face Embedding Recognition** (`MobileFaceNet`), **Cosine Similarity** matching, and **Multi-Scale Passive Liveness / Anti-Spoofing** (`MiniFASNet`) plugin for Flutter.

---

## ✨ Key Features

- **Zero-Copy `Uint8List` Input**: Send raw encoded image bytes (`JPEG` / `PNG`) directly over `MethodChannel` without Base64 memory overhead (Base64 string input is also supported for backward compatibility).
- **192-Dimensional Face Embeddings (`extractFaceFeature`)**: Uses `mobile_face_net.tflite` ($112 \times 112$, RGB normalized to $[-1.0, 1.0]$) with built-in **L2-normalization**.
- **Multi-Scale Passive Anti-Spoofing (`checkLiveness`)**: Uses `anti_spoof.tflite` (`MiniFASNet`) with dual-scale square cropping (**`1.0x` tight face crop** + **`2.7x` context crop**) to detect printed photos, digital screens, and video replay attacks.
- **Cross-Platform Face & Eye-Blink Detection (`detectFaces`)**:
  - **Android**: Google ML Kit `FaceDetection` (`CLASSIFICATION_MODE_ALL` & `LANDMARK_MODE_ALL`).
  - **iOS**: Apple Vision `VNDetectFaceLandmarksRequest` with normalized Eye Aspect Ratio (EAR) mapped to `[0.0, 1.0]`.
- **Pure-Dart Cosine Similarity (`compareFaces`)**: Instant `< 0.1 ms` vector comparison directly in Dart without platform-channel latency.
- **Single-Source Bundled Models**: Both `.tflite` models are bundled once inside the plugin's Flutter assets and loaded natively on background threads (`Dispatchers.Default` on Android, `DispatchQueue.global` on iOS).

---

## 📱 Platform Requirements

| Platform | Minimum Version | Native Engines |
| :--- | :--- | :--- |
| **Android** | `minSdk 24` (Android 7.0+) | Google ML Kit Face Detection `16.1.7` + TensorFlow Lite `2.14.0` |
| **iOS** | `iOS 15.0+` | Apple Vision Framework + `TensorFlowLiteSwift 2.14.0` |

---

## 🚀 Quickstart

### 1. Installation

Add `flutter_face_recognition` to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_face_recognition: ^1.0.0
```

### 2. Enroll & Verify a Face

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

  // 2. Check passive liveness (Multi-Scale MiniFASNet 1.0x & 2.7x)
  final liveness = await faceRecognition.checkLiveness(
    bytes: liveCameraBytes,
    threshold: 0.75,
  );

  if (!liveness.isLive) {
    print('Spoof detected! Score: ${liveness.score} (${liveness.confidence.label})');
    return;
  }

  // 3. Extract candidate embedding & compare in pure Dart
  final candidateVector = await faceRecognition.extractFaceFeature(
    bytes: liveCameraBytes,
  );

  final match = faceRecognition.compareFaces(
    enrolledVector,
    candidateVector,
    threshold: 0.75,
  );

  print('Match: ${match.isMatch} (${match.similarityPercentage.toStringAsFixed(1)}%)');
}
```

### 3. Face & Eye-Blink Detection (`detectFaces`)

```dart
final detection = await faceRecognition.detectFaces(bytes: imageBytes);

if (detection.hasSingleFace) {
  final face = detection.firstFace!;
  print('Bounding Box: (${face.x}, ${face.y}) ${face.width}x${face.height}');
  print('Head Yaw: ${face.headEulerAngleY}°, Roll: ${face.headEulerAngleZ}°');
  print('Eyes Open: ${face.isBothEyesOpen()}, Blinking: ${face.isBlinking()}');
}
```

### 4. Cleanup Resources

```dart
await faceRecognition.dispose();
```

---

## ⚙️ Recommended Thresholds

| Use Case | Cosine Similarity Threshold | Liveness Threshold |
| :--- | :---: | :---: |
| **General Consumer App** | `0.70` | `0.70` |
| **Standard Verification (Default)** | `0.75` | `0.75` |
| **High-Security / HRIS Attendance** | `0.80` – `0.85` | `0.75` – `0.80` |

---

## 📄 License

MIT License — Copyright (c) 2026 Andrew Hardianto.
