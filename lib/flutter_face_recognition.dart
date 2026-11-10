import 'dart:math' as math;
import 'dart:typed_data';

import 'flutter_face_recognition_platform_interface.dart';
import 'src/models.dart';

export 'flutter_face_recognition_platform_interface.dart';
export 'src/models.dart';

/// High-performance offline Face Recognition, Cosine Similarity matching,
/// and Enterprise Liveness Detection (`liveness-v3`) using TFLite,
/// Google ML Kit (Android), and Apple Vision (iOS).
///
/// The `liveness-v3` pipeline is a 4-phase, defense-in-depth design:
///  1. Dual-model silent-face anti-spoofing (`2.7x` + `4.0x` ensemble).
///  2. Accurate detector + pre-liveness quality gate.
///  3. Anti-TOCTOU [`verifyFace`] and an active temporal challenge
///     ([`createChallengeSession`] + [`verifyFaceSequence`]).
///  4. Anti-frame-injection native capture ([`captureLivenessSession`]),
///     device integrity, and an encrypted audit trail.
class FaceRecognition {
  /// Creates a [FaceRecognition] instance.
  const FaceRecognition();

  /// Initializes the plugin and configures plugin-level security flags.
  ///
  /// Set [enableInsecureFramePath] to `true` only for legacy/debug integrations
  /// that must call [verifyFaceSequence] with externally supplied frames. In
  /// release builds the frame path stays disabled otherwise — prefer
  /// [captureLivenessSession] for production.
  Future<void> initialize({bool enableInsecureFramePath = false}) {
    return FlutterFaceRecognitionPlatform.instance
        .initialize(enableInsecureFramePath: enableInsecureFramePath);
  }

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
  /// Uses the dual-model silent-face ensemble (`2.7x` + `4.0x` crops) with the
  /// `argmax` + veto decision of `liveness-v3`. [threshold] maps to the liveness
  /// `liveThreshold`; when `null` the native default (`0.80`) applies. The
  /// returned [LivenessResult.score] is NOT comparable with pre-`liveness-v3`
  /// SDKs.
  Future<LivenessResult> checkLiveness({
    Uint8List? bytes,
    String? base64,
    double? threshold,
  }) {
    return FlutterFaceRecognitionPlatform.instance.checkLiveness(
      bytes: bytes,
      base64: base64,
      threshold: threshold,
    );
  }

  /// Detects all faces in the image along with bounding boxes, head rotation
  /// Euler angles (pitch / yaw / roll), and left/right eye open probabilities.
  Future<FaceDetectionResult> detectFaces({
    Uint8List? bytes,
    String? base64,
  }) {
    return FlutterFaceRecognitionPlatform.instance.detectFaces(
      bytes: bytes,
      base64: base64,
    );
  }

  /// Runs the unified, anti-TOCTOU verification on a single frame.
  ///
  /// Quality gate, dual-model anti-spoofing, and MobileFaceNet embedding all
  /// execute on the same frame and bounding box. The [VerifyFaceResult.embedding]
  /// (and optional [match] against [referenceEmbedding]) is only returned when
  /// the quality gate passes and liveness is confirmed.
  Future<VerifyFaceResult> verifyFace({
    Uint8List? bytes,
    String? base64,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    return FlutterFaceRecognitionPlatform.instance.verifyFace(
      bytes: bytes,
      base64: base64,
      referenceEmbedding: referenceEmbedding,
      config: config,
    );
  }

  /// Issues a single-use, randomized active challenge session.
  ///
  /// The challenge direction is chosen natively and cannot be influenced by the
  /// caller. Display the returned [ChallengeSession.challenge] to the user, then
  /// pass its `challengeId` to [verifyFaceSequence] or [captureLivenessSession].
  Future<ChallengeSession> createChallengeSession() {
    return FlutterFaceRecognitionPlatform.instance.createChallengeSession();
  }

  /// Verifies a multi-frame session against a previously issued challenge.
  ///
  /// [frames] are captured sequentially. The session is evaluated against the
  /// temporal trajectory (R1–R5), a zero-tolerance spoof veto, and identity
  /// consistency. The challenge direction is read from [challengeId]; the
  /// session is single-use.
  ///
  /// The external frame path is blocked in release builds unless
  /// [`initialize(enableInsecureFramePath: true)`] was called — use
  /// [captureLivenessSession] for production.
  Future<SequenceVerifyResult> verifyFaceSequence({
    required List<Uint8List> frames,
    required String challengeId,
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    return FlutterFaceRecognitionPlatform.instance.verifyFaceSequence(
      frames: frames,
      challengeId: challengeId,
      referenceEmbedding: referenceEmbedding,
      config: config,
    );
  }

  /// Opens the native camera capture session and evaluates it end to end.
  ///
  /// Frames are captured and evaluated entirely on-device without crossing the
  /// Dart/JS boundary, resisting frame injection. Returns a
  /// [SequenceVerifyResult] that also carries a [DeviceIntegrityResult] and an
  /// encrypted `auditId`.
  Future<SequenceVerifyResult> captureLivenessSession({
    List<double>? referenceEmbedding,
    GateConfig? config,
  }) {
    return FlutterFaceRecognitionPlatform.instance.captureLivenessSession(
      referenceEmbedding: referenceEmbedding,
      config: config,
    );
  }

  /// Returns client-side device risk signals (emulator, root/jailbreak,
  /// debuggable build, and hooking frameworks).
  ///
  /// These are heuristic risk signals, not proof of compromise.
  Future<DeviceIntegrityResult> getDeviceIntegrity() {
    return FlutterFaceRecognitionPlatform.instance.getDeviceIntegrity();
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
      threshold: threshold,
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
