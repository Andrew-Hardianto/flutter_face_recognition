import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_face_recognition_platform_interface.dart';
import 'src/models.dart';

/// An implementation of [FlutterFaceRecognitionPlatform] that uses method channels.
///
/// This class defines the wire contract that the Android (Kotlin) and iOS
/// (Swift) plugins implement. Keep the method names and argument/result keys in
/// sync with both native sides.
class MethodChannelFlutterFaceRecognition
    extends FlutterFaceRecognitionPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_face_recognition');

  Map<String, dynamic> _buildImageArgs({
    Uint8List? bytes,
    String? base64,
    Map<String, dynamic>? extra,
  }) {
    if ((bytes == null || bytes.isEmpty) &&
        (base64 == null || base64.trim().isEmpty)) {
      throw ArgumentError(
        'Either non-empty `bytes` (Uint8List) or `base64` (String) must be provided.',
      );
    }

    return <String, dynamic>{
      if (bytes != null && bytes.isNotEmpty) 'bytes': bytes,
      if (base64 != null && base64.isNotEmpty) 'base64': base64,
      if (extra != null) ...extra,
    };
  }

  Map<String, dynamic> _optionalRefConfig({
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    return <String, dynamic>{
      if (referenceEmbedding != null) 'referenceEmbedding': referenceEmbedding,
      if (config != null) 'config': config.toMap(),
    };
  }

  @override
  Future<void> initialize({bool enableInsecureFramePath = false}) async {
    await methodChannel.invokeMethod<void>('initialize', <String, dynamic>{
      'enableInsecureFramePath': enableInsecureFramePath,
    });
  }

  @override
  Future<List<double>> extractFaceFeature({
    Uint8List? bytes,
    String? base64,
  }) async {
    final args = _buildImageArgs(bytes: bytes, base64: base64);
    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'extractFaceFeature',
      args,
    );

    if (result == null || result['embedding'] == null) {
      throw PlatformException(
        code: 'EMPTY_EMBEDDING',
        message: 'Native platform returned an empty face embedding.',
      );
    }

    final rawList = result['embedding'] as List<dynamic>;
    return rawList.map((e) => (e as num).toDouble()).toList(growable: false);
  }

  @override
  Future<LivenessResult> checkLiveness({
    Uint8List? bytes,
    String? base64,
    double? threshold,
  }) async {
    final args = _buildImageArgs(
      bytes: bytes,
      base64: base64,
      extra: <String, dynamic>{
        if (threshold != null) 'threshold': threshold,
      },
    );

    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'checkLiveness',
      args,
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_LIVENESS_RESULT',
        message: 'Native platform returned null for checkLiveness.',
      );
    }

    return LivenessResult.fromMap(result);
  }

  @override
  Future<FaceDetectionResult> detectFaces({
    Uint8List? bytes,
    String? base64,
  }) async {
    final args = _buildImageArgs(bytes: bytes, base64: base64);
    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'detectFaces',
      args,
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_DETECTION_RESULT',
        message: 'Native platform returned null for detectFaces.',
      );
    }

    return FaceDetectionResult.fromMap(result);
  }

  @override
  Future<VerifyFaceResult> verifyFace({
    Uint8List? bytes,
    String? base64,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) async {
    final args = _buildImageArgs(
      bytes: bytes,
      base64: base64,
      extra: _optionalRefConfig(
        referenceEmbedding: referenceEmbedding,
        config: config,
      ),
    );

    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'verifyFace',
      args,
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_VERIFY_RESULT',
        message: 'Native platform returned null for verifyFace.',
      );
    }

    return VerifyFaceResult.fromMap(result);
  }

  @override
  Future<ChallengeSession> createChallengeSession() async {
    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'createChallengeSession',
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_CHALLENGE_SESSION',
        message: 'Native platform returned null for createChallengeSession.',
      );
    }

    return ChallengeSession.fromMap(result);
  }

  @override
  Future<SequenceVerifyResult> verifyFaceSequence({
    required List<Uint8List> frames,
    required String challengeId,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) async {
    if (frames.isEmpty) {
      throw ArgumentError('`frames` must contain at least one frame.');
    }

    final args = <String, dynamic>{
      'frames': frames,
      'challengeId': challengeId,
      ..._optionalRefConfig(
        referenceEmbedding: referenceEmbedding,
        config: config,
      ),
    };

    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'verifyFaceSequence',
      args,
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_SEQUENCE_RESULT',
        message: 'Native platform returned null for verifyFaceSequence.',
      );
    }

    return SequenceVerifyResult.fromMap(result);
  }

  @override
  Future<SequenceVerifyResult> captureLivenessSession({
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) async {
    final args = _optionalRefConfig(
      referenceEmbedding: referenceEmbedding,
      config: config,
    );

    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'captureLivenessSession',
      args,
    );

    if (result == null) {
      throw PlatformException(
        code: 'EMPTY_CAPTURE_RESULT',
        message: 'Native platform returned null for captureLivenessSession.',
      );
    }

    return SequenceVerifyResult.fromMap(result);
  }

  @override
  Future<DeviceIntegrityResult> getDeviceIntegrity() async {
    final result = await methodChannel.invokeMapMethod<String, dynamic>(
      'getDeviceIntegrity',
    );

    final integrity = result?['deviceIntegrity'];
    if (integrity is! Map<Object?, Object?>) {
      throw PlatformException(
        code: 'EMPTY_INTEGRITY_RESULT',
        message: 'Native platform returned no device integrity snapshot.',
      );
    }

    return DeviceIntegrityResult.fromMap(integrity);
  }

  @override
  Future<void> dispose() async {
    await methodChannel.invokeMethod<void>('dispose');
  }
}
