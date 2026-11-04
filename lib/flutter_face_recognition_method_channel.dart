import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_face_recognition_platform_interface.dart';
import 'src/models.dart';

/// An implementation of [FlutterFaceRecognitionPlatform] that uses method channels.
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
    double threshold = 0.75,
  }) async {
    final args = _buildImageArgs(
      bytes: bytes,
      base64: base64,
      extra: <String, dynamic>{'threshold': threshold},
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
  Future<void> dispose() async {
    await methodChannel.invokeMethod<void>('dispose');
  }
}

