## Unreleased

* **Android:** fixed manifest-merge failure in consumer apps on AGP 8.x+
  (`Namespace 'org.tensorflow.lite' is used in multiple modules and/or
  libraries`) by upgrading `org.tensorflow:tensorflow-lite` `2.14.0` →
  `2.16.1` — the transitive `tensorflow-lite-api` declares its own namespace
  from that release — and removing the unused `tensorflow-lite-support:0.4.4`
  (its `-api` split had the same duplicate-namespace problem).
* **Android:** migrated to Flutter's **built-in Kotlin** (no longer applies
  `org.jetbrains.kotlin.android` itself, and no longer declares its own AGP /
  Kotlin classpaths). Requires Flutter `>=3.44.0` / Dart `>=3.12.0`.

## 1.0.0

* Initial release of `flutter_face_recognition`.
* Added offline Face Detection (`detectFaces`) using Google ML Kit on Android and Apple Vision (`VNDetectFaceLandmarksRequest` with normalized Eye Aspect Ratio) on iOS.
* Added 192-dimensional Face Embedding Extraction (`extractFaceFeature`) powered by MobileFaceNet (`mobile_face_net.tflite`) with L2-normalization.
* Added Multi-Scale Passive Liveness / Anti-Spoofing Detection (`checkLiveness`) powered by MiniFASNet (`anti_spoof.tflite`) using `1.0x` and `2.7x` crops.
* Added pure-Dart Cosine Similarity verification (`compareFaces`) with zero platform-channel overhead.
* Added support for both `Uint8List` raw encoded bytes and `base64` image inputs, plus `dispose()` resource cleanup.

