import 'dart:math' as math;
import 'dart:typed_data';

import 'flutter_face_recognition_platform_interface.dart';
import 'src/models.dart';

export 'flutter_face_recognition_platform_interface.dart';
export 'src/models.dart';

/// High-performance offline Face Recognition, Cosine Similarity matching,
/// and Multi-Scale Passive Liveness Detection (Anti-Spoofing) using TFLite,
/// Google ML Kit (Android), and Apple Vision (iOS).
class FaceRecognition {
  /// Creates a [FaceRecognition] instance.
  const FaceRecognition();

  /// Extracts a 192-dimensional L2-normalized face embedding from an image.
  ///
  /// Provide either [bytes] (`Uint8List` of a JPEG/PNG image, recommended for
  /// zero-copy memory efficiency) or [base64] (for backward compatibility).
  Future<List<double>> extractFaceFeature({
    Uint8List? bytes,
    String? base64,
  }) {
    return FlutterFaceRecognitionPlatform.instance.extractFaceFeature(
      bytes: bytes,
      base64: base64,
    );
  }

  /// Checks whether the face in the image is a real live person or a spoof
  /// attack (printed photo, digital screen replay, or mask).
  ///
  /// Uses Multi-Scale MiniFASNet (`1.0x` and `2.7x` crops) and compares the
  /// averaged score against [threshold] (default `0.75`).
  Future<LivenessResult> checkLiveness({
    Uint8List? bytes,
    String? base64,
    double threshold = 0.75,
  }) {
    return FlutterFaceRecognitionPlatform.instance.checkLiveness(
      bytes: bytes,
      base64: base64,
      threshold: threshold,
    );
  }

  /// Detects all faces in the image along with bounding boxes, head rotation
  /// Euler angles, and left/right eye open probabilities (`0.0` to `1.0`).
  Future<FaceDetectionResult> detectFaces({
    Uint8List? bytes,
    String? base64,
  }) {
    return FlutterFaceRecognitionPlatform.instance.detectFaces(
      bytes: bytes,
      base64: base64,
    );
  }

  /// Compares two face embedding vectors ([vector1] and [vector2]) in pure Dart
  /// using Cosine Similarity without any platform-channel overhead.
  ///
  /// Default [threshold] is `0.75` (`0.70` for tolerant matching, `0.80`–`0.85`
  /// for strict enterprise/HRIS attendance verification).
  MatchResult compareFaces(
    List<double> vector1,
    List<double> vector2, {
    double threshold = 0.75,
  }) {
    return compareEmbeddings(vector1, vector2, threshold: threshold);
  }

  /// Static helper to compare two face embedding vectors in pure Dart using
  /// Cosine Similarity.
  static MatchResult compareEmbeddings(
    List<double> vector1,
    List<double> vector2, {
    double threshold = 0.75,
  }) {
    if (vector1.isEmpty || vector1.length != vector2.length) {
      throw ArgumentError(
        'Vectors must be non-empty and have the same dimension '
        '(got ${vector1.length} and ${vector2.length}).',
      );
    }

    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (var i = 0; i < vector1.length; i++) {
      final v1 = vector1[i];
      final v2 = vector2[i];
      dotProduct += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }

    // Zero-norm guard to prevent division by zero / NaN
    if (norm1 <= 0.0 || norm2 <= 0.0) {
      return const MatchResult(
        isMatch: false,
        score: 0.0,
        similarityPercentage: 0.0,
      );
    }

    final cosineSimilarity =
        (dotProduct / (math.sqrt(norm1) * math.sqrt(norm2))).clamp(-1.0, 1.0);
    final isMatch = cosineSimilarity > threshold;
    final similarityPercentage =
        math.max(0.0, math.min(100.0, cosineSimilarity * 100.0));

    return MatchResult(
      isMatch: isMatch,
      score: cosineSimilarity,
      similarityPercentage: similarityPercentage,
    );
  }

  /// L2-normalizes an embedding vector so its Euclidean norm equals `1.0`.
  static List<double> l2Normalize(List<double> vector) {
    if (vector.isEmpty) return const <double>[];
    double normSq = 0.0;
    for (final v in vector) {
      normSq += v * v;
    }
    final norm = math.sqrt(normSq);
    if (norm <= 0.0) {
      return List<double>.from(vector, growable: false);
    }
    return vector.map((v) => v / norm).toList(growable: false);
  }

  /// Releases native TFLite interpreters and ML Kit / Vision detector resources.
  Future<void> dispose() {
    return FlutterFaceRecognitionPlatform.instance.dispose();
  }
}

