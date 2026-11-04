import 'dart:typed_data';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_face_recognition_method_channel.dart';
import 'src/models.dart';

/// The interface that implementations of `flutter_face_recognition` must implement.
abstract class FlutterFaceRecognitionPlatform extends PlatformInterface {
  /// Constructs a [FlutterFaceRecognitionPlatform].
  FlutterFaceRecognitionPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterFaceRecognitionPlatform _instance =
      MethodChannelFlutterFaceRecognition();

  /// The default instance of [FlutterFaceRecognitionPlatform] to use.
  static FlutterFaceRecognitionPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterFaceRecognitionPlatform] when
  /// they register themselves.
  static set instance(FlutterFaceRecognitionPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Extracts a 192-dimensional L2-normalized face embedding from an image.
  Future<List<double>> extractFaceFeature({
    Uint8List? bytes,
    String? base64,
  }) {
    throw UnimplementedError('extractFaceFeature() has not been implemented.');
  }

  /// Performs passive multi-scale anti-spoofing liveness detection on the image.
  Future<LivenessResult> checkLiveness({
    Uint8List? bytes,
    String? base64,
    double threshold = 0.75,
  }) {
    throw UnimplementedError('checkLiveness() has not been implemented.');
  }

  /// Detects all faces, bounding boxes, head Euler angles, and eye open probabilities.
  Future<FaceDetectionResult> detectFaces({
    Uint8List? bytes,
    String? base64,
  }) {
    throw UnimplementedError('detectFaces() has not been implemented.');
  }

  /// Releases native TFLite interpreters and face detector resources.
  Future<void> dispose() {
    throw UnimplementedError('dispose() has not been implemented.');
  }
}

