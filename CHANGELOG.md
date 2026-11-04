## 1.0.0

* Initial release of `flutter_face_recognition`.
* Added offline Face Detection (`detectFaces`) using Google ML Kit on Android and Apple Vision (`VNDetectFaceLandmarksRequest` with normalized Eye Aspect Ratio) on iOS.
* Added 192-dimensional Face Embedding Extraction (`extractFaceFeature`) powered by MobileFaceNet (`mobile_face_net.tflite`) with L2-normalization.
* Added Multi-Scale Passive Liveness / Anti-Spoofing Detection (`checkLiveness`) powered by MiniFASNet (`anti_spoof.tflite`) using `1.0x` and `2.7x` crops.
* Added pure-Dart Cosine Similarity verification (`compareFaces`) with zero platform-channel overhead.
* Added support for both `Uint8List` raw encoded bytes and `base64` image inputs, plus `dispose()` resource cleanup.

