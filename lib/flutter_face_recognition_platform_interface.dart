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

  /// Initializes the plugin and configures plugin-level security flags.
  ///
  /// Set [enableInsecureFramePath] to `true` only for legacy/debug integrations
  /// that must call [verifyFaceSequence] with externally supplied frames. In
  /// release builds the frame path stays disabled otherwise, and production
  /// callers should use [captureLivenessSession].
  Future<void> initialize({bool enableInsecureFramePath = false}) {
    throw UnimplementedError('initialize() has not been implemented.');
  }

  /// Extracts a 192-dimensional L2-normalized face embedding from an image.
  Future<List<double>> extractFaceFeature({
    Uint8List? bytes,
    String? base64,
  }) {
    throw UnimplementedError('extractFaceFeature() has not been implemented.');
  }

  /// Performs passive dual-model anti-spoofing liveness detection on the image.
  ///
  /// [threshold] maps to the liveness `liveThreshold`. When `null`, the native
  /// `liveness-v3` default (`0.80`) is used.
  Future<LivenessResult> checkLiveness({
    Uint8List? bytes,
    String? base64,
    double? threshold,
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

  /// Runs the unified quality gate + liveness + embedding pipeline on one frame.
  Future<VerifyFaceResult> verifyFace({
    Uint8List? bytes,
    String? base64,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    throw UnimplementedError('verifyFace() has not been implemented.');
  }

  /// Issues a single-use, randomized active challenge session.
  Future<ChallengeSession> createChallengeSession() {
    throw UnimplementedError(
        'createChallengeSession() has not been implemented.');
  }

  /// Verifies a multi-frame session against a previously issued challenge.
  Future<SequenceVerifyResult> verifyFaceSequence({
    required List<Uint8List> frames,
    required String challengeId,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    throw UnimplementedError(
        'verifyFaceSequence() has not been implemented.');
  }

  /// Opens the native camera capture session and evaluates it end to end.
  Future<SequenceVerifyResult> captureLivenessSession({
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    throw UnimplementedError(
        'captureLivenessSession() has not been implemented.');
  }

  /// Returns client-side device risk signals.
  Future<DeviceIntegrityResult> getDeviceIntegrity() {
    throw UnimplementedError('getDeviceIntegrity() has not been implemented.');
  }

  /// Releases native TFLite interpreters and face detector resources.
  Future<void> dispose() {
    throw UnimplementedError('dispose() has not been implemented.');
  }
}
