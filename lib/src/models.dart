/// Confidence level of the passive liveness check.
enum LivenessConfidence {
  /// High confidence (`score > 0.85` or `score < 0.15`).
  high('HIGH'),

  /// Medium confidence (`0.60 < score <= 0.85` or `0.15 <= score < 0.40`).
  medium('MEDIUM'),

  /// Borderline / low confidence (`0.40 <= score <= 0.60`).
  low('LOW');

  /// Raw uppercase string representation returned by the native platform.
  final String label;

  const LivenessConfidence(this.label);

  /// Parses a native string (`HIGH`, `MEDIUM`, `LOW`) into [LivenessConfidence].
  factory LivenessConfidence.fromString(String? value) {
    switch (value?.toUpperCase()) {
      case 'HIGH':
        return LivenessConfidence.high;
      case 'MEDIUM':
        return LivenessConfidence.medium;
      default:
        return LivenessConfidence.low;
    }
  }
}

/// Represents a single detected face with bounding box, head rotation, and eye probabilities.
class DetectedFace {
  /// Left coordinate of the face bounding box in pixels.
  final int x;

  /// Top coordinate of the face bounding box in pixels.
  final int y;

  /// Width of the face bounding box in pixels.
  final int width;

  /// Height of the face bounding box in pixels.
  final int height;

  /// Left-to-right head rotation (yaw) in degrees.
  final double headEulerAngleY;

  /// Head tilt rotation (roll) in degrees.
  final double headEulerAngleZ;

  /// Probability that the left eye is open (`0.0` = closed, `1.0` = open), or `null` if unavailable.
  final double? leftEyeOpenProbability;

  /// Probability that the right eye is open (`0.0` = closed, `1.0` = open), or `null` if unavailable.
  final double? rightEyeOpenProbability;

  /// Creates a [DetectedFace].
  const DetectedFace({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.headEulerAngleY,
    required this.headEulerAngleZ,
    this.leftEyeOpenProbability,
    this.rightEyeOpenProbability,
  });

  /// Returns `true` when both eyes are detected as open above [threshold] (default `0.5`).
  bool isBothEyesOpen({double threshold = 0.5}) {
    final left = leftEyeOpenProbability;
    final right = rightEyeOpenProbability;
    if (left == null || right == null) return false;
    return left >= threshold && right >= threshold;
  }

  /// Returns `true` when either or both eyes are closed below [threshold] (default `0.3`).
  bool isBlinking({double threshold = 0.3}) {
    final left = leftEyeOpenProbability;
    final right = rightEyeOpenProbability;
    if (left == null || right == null) return false;
    return left < threshold && right < threshold;
  }

  /// Creates a [DetectedFace] from a raw platform map.
  factory DetectedFace.fromMap(Map<Object?, Object?> map) {
    return DetectedFace(
      x: (map['x'] as num?)?.toInt() ?? 0,
      y: (map['y'] as num?)?.toInt() ?? 0,
      width: (map['width'] as num?)?.toInt() ?? 0,
      height: (map['height'] as num?)?.toInt() ?? 0,
      headEulerAngleY: (map['headEulerAngleY'] as num?)?.toDouble() ?? 0.0,
      headEulerAngleZ: (map['headEulerAngleZ'] as num?)?.toDouble() ?? 0.0,
      leftEyeOpenProbability:
          (map['leftEyeOpenProbability'] as num?)?.toDouble(),
      rightEyeOpenProbability:
          (map['rightEyeOpenProbability'] as num?)?.toDouble(),
    );
  }

  /// Converts this [DetectedFace] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'x': x,
      'y': y,
      'width': width,
      'height': height,
      'headEulerAngleY': headEulerAngleY,
      'headEulerAngleZ': headEulerAngleZ,
      'leftEyeOpenProbability': leftEyeOpenProbability,
      'rightEyeOpenProbability': rightEyeOpenProbability,
    };
  }

  @override
  String toString() =>
      'DetectedFace(x: $x, y: $y, width: $width, height: $height, '
      'yaw: ${headEulerAngleY.toStringAsFixed(1)}, '
      'roll: ${headEulerAngleZ.toStringAsFixed(1)}, '
      'leftEye: ${leftEyeOpenProbability?.toStringAsFixed(2)}, '
      'rightEye: ${rightEyeOpenProbability?.toStringAsFixed(2)})';
}

/// Result returned by `detectFaces`.
class FaceDetectionResult {
  /// Number of faces detected in the image.
  final int count;

  /// List of detected faces and their metadata.
  final List<DetectedFace> faces;

  /// Creates a [FaceDetectionResult].
  const FaceDetectionResult({
    required this.count,
    required this.faces,
  });

  /// Returns `true` if exactly one face is detected.
  bool get hasSingleFace => count == 1 && faces.length == 1;

  /// Returns the first detected face, or `null` if no faces were found.
  DetectedFace? get firstFace => faces.isNotEmpty ? faces.first : null;

  /// Creates a [FaceDetectionResult] from a raw platform map.
  factory FaceDetectionResult.fromMap(Map<Object?, Object?> map) {
    final rawFaces = (map['faces'] as List<dynamic>?) ?? const <dynamic>[];
    final parsedFaces = rawFaces
        .whereType<Map<Object?, Object?>>()
        .map(DetectedFace.fromMap)
        .toList(growable: false);

    return FaceDetectionResult(
      count: (map['count'] as num?)?.toInt() ?? parsedFaces.length,
      faces: parsedFaces,
    );
  }

  /// Converts this [FaceDetectionResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'count': count,
      'faces': faces.map((f) => f.toMap()).toList(growable: false),
    };
  }

  @override
  String toString() => 'FaceDetectionResult(count: $count, faces: $faces)';
}

/// Result returned by `checkLiveness` (Multi-Scale MiniFASNet Anti-Spoofing).
class LivenessResult {
  /// Whether the face is classified as a real live person (`score > threshold`).
  final bool isLive;

  /// Averaged liveness probability score between `0.0` (spoof) and `1.0` (live).
  final double score;

  /// Confidence level (`HIGH`, `MEDIUM`, or `LOW`).
  final LivenessConfidence confidence;

  /// Creates a [LivenessResult].
  const LivenessResult({
    required this.isLive,
    required this.score,
    required this.confidence,
  });

  /// Creates a [LivenessResult] from a raw platform map.
  factory LivenessResult.fromMap(Map<Object?, Object?> map) {
    return LivenessResult(
      isLive: (map['isLive'] as bool?) ?? false,
      score: (map['score'] as num?)?.toDouble() ?? 0.0,
      confidence: LivenessConfidence.fromString(map['confidence'] as String?),
    );
  }

  /// Converts this [LivenessResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isLive': isLive,
      'score': score,
      'confidence': confidence.label,
    };
  }

  @override
  String toString() =>
      'LivenessResult(isLive: $isLive, score: ${score.toStringAsFixed(4)}, '
      'confidence: ${confidence.label})';
}

/// Result returned by `compareFaces` (Cosine Similarity).
class MatchResult {
  /// Whether the two face embeddings match (`score > threshold`).
  final bool isMatch;

  /// Raw cosine similarity score in the range `[-1.0, 1.0]`.
  final double score;

  /// Similarity percentage clamped to `[0.0, 100.0]`.
  final double similarityPercentage;

  /// Creates a [MatchResult].
  const MatchResult({
    required this.isMatch,
    required this.score,
    required this.similarityPercentage,
  });

  /// Converts this [MatchResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isMatch': isMatch,
      'score': score,
      'similarityPercentage': similarityPercentage,
    };
  }

  @override
  String toString() =>
      'MatchResult(isMatch: $isMatch, score: ${score.toStringAsFixed(4)}, '
      'similarityPercentage: ${similarityPercentage.toStringAsFixed(2)}%)';
}

