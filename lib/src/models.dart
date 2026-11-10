/// Typed models for the `flutter_face_recognition` (`liveness-v3`) API.
///
/// These models mirror the maps returned by the native Android (Kotlin) and
/// iOS (Swift) pipelines over the method channel. Keeping them strongly typed
/// makes the 4-phase defense-in-depth result surface (Quality Gate → Dual
/// Anti-Spoof → Embedding → Temporal Challenge) reusable and reviewable.
library;

// ============================
// Enumerations
// ============================

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

/// The winning class label from the dual-model anti-spoof ensemble.
///
/// `argmax(p)` where `p = softmax(z_2.7) + softmax(z_4.0)` over the 3 classes
/// `[print_spoof, live, replay_spoof]`.
enum LivenessLabel {
  /// A real, live human face.
  live('LIVE'),

  /// A printed photo / paper spoof.
  print('PRINT'),

  /// A digital screen replay / video spoof.
  replay('REPLAY');

  /// Raw uppercase string representation returned by the native platform.
  final String value;

  const LivenessLabel(this.value);

  /// Parses a native string (`LIVE`, `PRINT`, `REPLAY`) into [LivenessLabel].
  factory LivenessLabel.fromString(String? value) {
    switch (value?.toUpperCase()) {
      case 'LIVE':
        return LivenessLabel.live;
      case 'PRINT':
        return LivenessLabel.print;
      case 'REPLAY':
        return LivenessLabel.replay;
      default:
        // Unknown → treat as the safe (non-live) default.
        return LivenessLabel.print;
    }
  }
}

/// Direction of an active head-pose challenge.
enum ChallengeDirection {
  /// User must turn toward their own left (negative normalized yaw).
  lookLeft('LOOK_LEFT'),

  /// User must turn toward their own right (positive normalized yaw).
  lookRight('LOOK_RIGHT');

  /// Raw uppercase string representation returned by the native platform.
  final String value;

  const ChallengeDirection(this.value);

  /// Parses a native string (`LOOK_LEFT`, `LOOK_RIGHT`) into [ChallengeDirection].
  factory ChallengeDirection.fromString(String? value) {
    switch (value?.toUpperCase()) {
      case 'LOOK_RIGHT':
        return ChallengeDirection.lookRight;
      case 'LOOK_LEFT':
      default:
        return ChallengeDirection.lookLeft;
    }
  }
}

// ============================
// Quality Gate
// ============================

/// Per-call configuration for the quality gate and liveness decision.
///
/// Every field is optional and falls back to the SDK default when omitted.
/// This maps to the native `GateConfig` on both platforms.
class GateConfig {
  /// Minimum live score required to classify a frame as live (default `0.80`).
  final double liveThreshold;

  /// Hard veto ceiling for the spoof score (default `0.20`).
  final double spoofVeto;

  /// Minimum shortest-frame side in pixels (default `480`).
  final int minFrameSide;

  /// Minimum face-width / frame-width ratio (default `0.25`).
  final double minFaceRatio;

  /// Minimum margin from the frame edge as a fraction (default `0.05`).
  final double edgeMargin;

  /// Maximum absolute yaw/pitch/roll in degrees (default `20`).
  final double maxPose;

  /// Minimum variance-of-Laplacian to pass the blur check (default `50`).
  final double blurMin;

  /// Minimum mean luminance (default `60`).
  final double exposureMin;

  /// Maximum mean luminance (default `200`).
  final double exposureMax;

  /// Minimum average cosine similarity across comparable frames (default `0.70`).
  final double identityThreshold;

  /// Per-device yaw sign override (`1` or `-1`) for flipped camera buffers.
  final int yawSignOverride;

  /// Minimum fraction of frames that must be live (default `1.0`).
  final double minLiveVoteRatio;

  /// Optional screen-suspect veto; `null` records a soft signal only.
  final double? screenSuspectVeto;

  /// Request debug detail (only honored on debuggable builds).
  final bool debug;

  /// Creates a [GateConfig].
  const GateConfig({
    this.liveThreshold = 0.80,
    this.spoofVeto = 0.20,
    this.minFrameSide = 480,
    this.minFaceRatio = 0.25,
    this.edgeMargin = 0.05,
    this.maxPose = 20,
    this.blurMin = 50,
    this.exposureMin = 60,
    this.exposureMax = 200,
    this.identityThreshold = 0.70,
    this.yawSignOverride = 1,
    this.minLiveVoteRatio = 1.0,
    this.screenSuspectVeto,
    this.debug = false,
  });

  /// Converts this [GateConfig] to a serializable map for the native side.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'liveThreshold': liveThreshold,
      'spoofVeto': spoofVeto,
      'minFrameSide': minFrameSide,
      'minFaceRatio': minFaceRatio,
      'edgeMargin': edgeMargin,
      'maxPose': maxPose,
      'blurMin': blurMin,
      'exposureMin': exposureMin,
      'exposureMax': exposureMax,
      'identityThreshold': identityThreshold,
      'yawSignOverride': yawSignOverride,
      'minLiveVoteRatio': minLiveVoteRatio,
      if (screenSuspectVeto != null) 'screenSuspectVeto': screenSuspectVeto,
      'debug': debug,
    };
  }

  @override
  String toString() =>
      'GateConfig(liveThreshold: $liveThreshold, spoofVeto: $spoofVeto, '
      'minFrameSide: $minFrameSide, minFaceRatio: $minFaceRatio, '
      'edgeMargin: $edgeMargin, maxPose: $maxPose, blurMin: $blurMin, '
      'exposure: [$exposureMin, $exposureMax], '
      'identityThreshold: $identityThreshold, yawSignOverride: $yawSignOverride, '
      'minLiveVoteRatio: $minLiveVoteRatio, screenSuspectVeto: $screenSuspectVeto, '
      'debug: $debug)';
}

/// Result of the pre-liveness quality gate.
class QualityResult {
  /// Whether the frame passed every quality check.
  final bool passed;

  /// Failure reason when [passed] is `false`
  /// (e.g. `NO_FACE`, `MULTIPLE_FACES`, `LOW_RESOLUTION`, `FACE_TOO_SMALL`,
  /// `FACE_AT_EDGE`, `BAD_POSE`, `BLURRY`, `BAD_LIGHTING`).
  final String? reason;

  /// Face-width / frame-width ratio at evaluation time.
  final double faceRatio;

  /// Variance-of-Laplacian blur metric (higher is sharper).
  final double blur;

  /// Creates a [QualityResult].
  const QualityResult({
    required this.passed,
    this.reason,
    this.faceRatio = 0.0,
    this.blur = 0.0,
  });

  /// Creates a [QualityResult] from a raw platform map.
  factory QualityResult.fromMap(Map<Object?, Object?> map) {
    return QualityResult(
      passed: (map['passed'] as bool?) ?? false,
      reason: map['reason'] as String?,
      faceRatio: (map['faceRatio'] as num?)?.toDouble() ?? 0.0,
      blur: (map['blur'] as num?)?.toDouble() ?? 0.0,
    );
  }

  /// Converts this [QualityResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'passed': passed,
      if (reason != null) 'reason': reason,
      'faceRatio': faceRatio,
      'blur': blur,
    };
  }

  @override
  String toString() =>
      'QualityResult(passed: $passed, reason: $reason, '
      'faceRatio: ${faceRatio.toStringAsFixed(3)}, '
      'blur: ${blur.toStringAsFixed(1)})';
}

// ============================
// Face Detection
// ============================

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

  /// Up-and-down head rotation (pitch) in degrees (`0` when unavailable).
  final double headEulerAngleX;

  /// Left-to-right head rotation (yaw) in degrees, normalized so that
  /// positive = user turns toward their anatomical right (see `YawConvention`).
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
    this.headEulerAngleX = 0.0,
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
      headEulerAngleX: (map['headEulerAngleX'] as num?)?.toDouble() ?? 0.0,
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
      'headEulerAngleX': headEulerAngleX,
      'headEulerAngleY': headEulerAngleY,
      'headEulerAngleZ': headEulerAngleZ,
      'leftEyeOpenProbability': leftEyeOpenProbability,
      'rightEyeOpenProbability': rightEyeOpenProbability,
    };
  }

  @override
  String toString() =>
      'DetectedFace(x: $x, y: $y, width: $width, height: $height, '
      'pitch: ${headEulerAngleX.toStringAsFixed(1)}, '
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

// ============================
// Liveness (Dual-Model Anti-Spoof)
// ============================

/// Result returned by `checkLiveness` and embedded in `verifyFace`.
///
/// The `liveness-v3` pipeline runs two MiniFASNet models (crop scale `2.7` and
/// `4.0`), sums their softmax outputs `p` (range `0..2`), then derives the
/// per-class scores below. `score` is kept as a backward-compatible alias for
/// [liveScore]; its semantics are NOT comparable with pre-`liveness-v3` SDKs.
class LivenessResult {
  /// Whether the face is classified as a real live person.
  final bool isLive;

  /// Backward-compatible alias for [liveScore] (range `0.0`..`1.0`).
  final double score;

  /// Confidence level (`HIGH`, `MEDIUM`, or `LOW`) derived from [liveScore].
  final LivenessConfidence confidence;

  /// Winning class label of the ensemble (`LIVE`, `PRINT`, or `REPLAY`).
  final LivenessLabel label;

  /// Mean live probability `p[1] / 2` (range `0.0`..`1.0`).
  final double liveScore;

  /// Mean print-spoof probability `p[0] / 2` (range `0.0`..`1.0`).
  final double printScore;

  /// Mean replay-spoof probability `p[2] / 2` (range `0.0`..`1.0`).
  final double replayScore;

  /// `max(p[0], p[2]) / 2` — the spoof score used for the hard veto.
  final double spoofScore;

  /// Screen-artifact (moiré / bezel) soft signal in `0.0`..`1.0`.
  final double screenSuspectScore;

  /// Threshold semantic version (e.g. `liveness-v3`).
  final String thresholdVersion;

  /// Optional quality-gate result (populated by `verifyFace`).
  final QualityResult? quality;

  /// Creates a [LivenessResult].
  const LivenessResult({
    required this.isLive,
    required this.score,
    required this.confidence,
    this.label = LivenessLabel.live,
    this.liveScore = 0.0,
    this.printScore = 0.0,
    this.replayScore = 0.0,
    this.spoofScore = 0.0,
    this.screenSuspectScore = 0.0,
    this.thresholdVersion = 'liveness-v3',
    this.quality,
  });

  /// Creates a [LivenessResult] from a raw platform map.
  ///
  /// Handles both the flat `checkLiveness` shape and the nested `liveness`
  /// object returned by `verifyFace`.
  factory LivenessResult.fromMap(Map<Object?, Object?> map) {
    final liveScore = (map['liveScore'] as num?)?.toDouble() ??
        (map['score'] as num?)?.toDouble() ??
        0.0;
    final qualityMap = map['quality'];
    return LivenessResult(
      isLive: (map['isLive'] as bool?) ?? false,
      score: (map['score'] as num?)?.toDouble() ?? liveScore,
      confidence: LivenessConfidence.fromString(map['confidence'] as String?),
      label: LivenessLabel.fromString(map['label'] as String?),
      liveScore: liveScore,
      printScore: (map['printScore'] as num?)?.toDouble() ?? 0.0,
      replayScore: (map['replayScore'] as num?)?.toDouble() ?? 0.0,
      spoofScore: (map['spoofScore'] as num?)?.toDouble() ?? 0.0,
      screenSuspectScore: (map['screenSuspectScore'] as num?)?.toDouble() ?? 0.0,
      thresholdVersion:
          (map['thresholdVersion'] as String?) ?? 'liveness-v3',
      quality: qualityMap is Map<Object?, Object?>
          ? QualityResult.fromMap(qualityMap)
          : null,
    );
  }

  /// Converts this [LivenessResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isLive': isLive,
      'score': score,
      'confidence': confidence.label,
      'label': label.value,
      'liveScore': liveScore,
      'printScore': printScore,
      'replayScore': replayScore,
      'spoofScore': spoofScore,
      'screenSuspectScore': screenSuspectScore,
      'thresholdVersion': thresholdVersion,
      if (quality != null) 'quality': quality!.toMap(),
    };
  }

  @override
  String toString() =>
      'LivenessResult(isLive: $isLive, label: ${label.value}, '
      'liveScore: ${liveScore.toStringAsFixed(4)}, '
      'printScore: ${printScore.toStringAsFixed(4)}, '
      'replayScore: ${replayScore.toStringAsFixed(4)}, '
      'spoofScore: ${spoofScore.toStringAsFixed(4)}, '
      'screenSuspect: ${screenSuspectScore.toStringAsFixed(3)}, '
      'confidence: ${confidence.label}, version: $thresholdVersion)';
}

// ============================
// Face Matching
// ============================

/// Result returned by `compareFaces` (Cosine Similarity).
class MatchResult {
  /// Whether the two face embeddings match (`score > threshold`).
  final bool isMatch;

  /// Raw cosine similarity score in the range `[-1.0, 1.0]`.
  final double score;

  /// Similarity percentage clamped to `[0.0, 100.0]`.
  final double similarityPercentage;

  /// The threshold used for the decision, when reported by the native side.
  final double? threshold;

  /// The threshold semantic version (e.g. `match-v1`).
  final String? thresholdVersion;

  /// Creates a [MatchResult].
  const MatchResult({
    required this.isMatch,
    required this.score,
    required this.similarityPercentage,
    this.threshold,
    this.thresholdVersion,
  });

  /// Creates a [MatchResult] from a raw platform map.
  factory MatchResult.fromMap(Map<Object?, Object?> map) {
    final score = (map['score'] as num?)?.toDouble() ?? 0.0;
    final threshold = (map['threshold'] as num?)?.toDouble();
    return MatchResult(
      isMatch: (map['isMatch'] as bool?) ?? false,
      score: score,
      similarityPercentage: (map['similarityPercentage'] as num?)?.toDouble() ??
          (score.clamp(-1.0, 1.0) * 100).clamp(0.0, 100.0),
      threshold: threshold,
      thresholdVersion: map['thresholdVersion'] as String?,
    );
  }

  /// Converts this [MatchResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isMatch': isMatch,
      'score': score,
      'similarityPercentage': similarityPercentage,
      if (threshold != null) 'threshold': threshold,
      if (thresholdVersion != null) 'thresholdVersion': thresholdVersion,
    };
  }

  @override
  String toString() => 'MatchResult(isMatch: $isMatch, score: ${score.toStringAsFixed(4)}, '
      'similarityPercentage: ${similarityPercentage.toStringAsFixed(2)}%)';
}

// ============================
// Active Challenge & Trajectory
// ============================

/// A single-use active challenge session issued by the native side.
class ChallengeSession {
  /// Opaque session identifier to pass back to `verifyFaceSequence`.
  final String challengeId;

  /// The randomized direction the user must perform.
  final ChallengeDirection challenge;

  /// Expiry timestamp (milliseconds since epoch).
  final DateTime expiresAt;

  /// Creates a [ChallengeSession].
  const ChallengeSession({
    required this.challengeId,
    required this.challenge,
    required this.expiresAt,
  });

  /// Creates a [ChallengeSession] from a raw platform map.
  factory ChallengeSession.fromMap(Map<Object?, Object?> map) {
    final expiryMs = (map['expiresAt'] as num?)?.toDouble() ?? 0.0;
    return ChallengeSession(
      challengeId: (map['challengeId'] as String?) ?? '',
      challenge: ChallengeDirection.fromString(map['challenge'] as String?),
      expiresAt: DateTime.fromMillisecondsSinceEpoch(expiryMs.round()),
    );
  }

  /// Converts this [ChallengeSession] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'challengeId': challengeId,
      'challenge': challenge.value,
      'expiresAt': expiresAt.millisecondsSinceEpoch.toDouble(),
    };
  }

  @override
  String toString() =>
      'ChallengeSession(id: $challengeId, challenge: ${challenge.value}, '
      'expiresAt: ${expiresAt.toIso8601String()})';
}

/// Per-rule outcome of the temporal challenge trajectory (R1–R5).
class TrajectoryRules {
  /// R1: the leading frames are frontal.
  final bool frontalStart;

  /// R2: a deviation toward the challenge direction was reached.
  final bool deviation;

  /// R3: the trailing frames return to frontal.
  final bool returnFrontal;

  /// R4: no sustained deviation in the wrong direction (anti head-shake).
  final bool singleDirection;

  /// R5: a smooth, unidirectional motion run is present.
  final bool smoothMotion;

  /// Creates a [TrajectoryRules].
  const TrajectoryRules({
    required this.frontalStart,
    required this.deviation,
    required this.returnFrontal,
    required this.singleDirection,
    required this.smoothMotion,
  });

  /// Creates a [TrajectoryRules] from a raw platform map.
  factory TrajectoryRules.fromMap(Map<Object?, Object?> map) {
    return TrajectoryRules(
      frontalStart: (map['frontalStart'] as bool?) ?? false,
      deviation: (map['deviation'] as bool?) ?? false,
      returnFrontal: (map['returnFrontal'] as bool?) ?? false,
      singleDirection: (map['singleDirection'] as bool?) ?? false,
      smoothMotion: (map['smoothMotion'] as bool?) ?? false,
    );
  }

  /// Converts this [TrajectoryRules] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'frontalStart': frontalStart,
      'deviation': deviation,
      'returnFrontal': returnFrontal,
      'singleDirection': singleDirection,
      'smoothMotion': smoothMotion,
    };
  }

  @override
  String toString() =>
      'TrajectoryRules(R1: $frontalStart, R2: $deviation, R3: $returnFrontal, '
      'R4: $singleDirection, R5: $smoothMotion)';
}

/// Summary of the temporal challenge trajectory (R1–R5 + yaw statistics).
class TrajectoryResult {
  /// The per-rule outcome.
  final TrajectoryRules rules;

  /// Yaw of the first frame (normalized).
  final double startYaw;

  /// Yaw of the last frame (normalized).
  final double endYaw;

  /// The extreme (largest deviation) yaw observed.
  final double extremeYaw;

  /// Index of the extreme frame (`-1` when unknown).
  final int extremeFrame;

  /// Maximum observed deviation toward the wrong direction.
  final double wrongWayMax;

  /// Variance of yaw-delta magnitudes (near-zero ⇒ suspiciously smooth).
  final double deltaVariance;

  /// Whether the trajectory is suspiciously smooth (video-loop signal).
  final bool tooSmooth;

  /// Session duration in milliseconds (`0` when timestamps are unknown).
  final double durationMs;

  /// Creates a [TrajectoryResult].
  const TrajectoryResult({
    required this.rules,
    required this.startYaw,
    required this.endYaw,
    required this.extremeYaw,
    required this.extremeFrame,
    required this.wrongWayMax,
    required this.deltaVariance,
    required this.tooSmooth,
    required this.durationMs,
  });

  /// Creates a [TrajectoryResult] from a raw platform map.
  factory TrajectoryResult.fromMap(Map<Object?, Object?> map) {
    final rulesMap = map['rulesPassed'];
    return TrajectoryResult(
      rules: rulesMap is Map<Object?, Object?>
          ? TrajectoryRules.fromMap(rulesMap)
          : const TrajectoryRules(
              frontalStart: false,
              deviation: false,
              returnFrontal: false,
              singleDirection: false,
              smoothMotion: false,
            ),
      startYaw: (map['startYaw'] as num?)?.toDouble() ?? 0.0,
      endYaw: (map['endYaw'] as num?)?.toDouble() ?? 0.0,
      extremeYaw: (map['extremeYaw'] as num?)?.toDouble() ?? 0.0,
      extremeFrame: (map['extremeFrame'] as num?)?.toInt() ?? -1,
      wrongWayMax: (map['wrongWayMax'] as num?)?.toDouble() ?? 0.0,
      deltaVariance: (map['deltaVariance'] as num?)?.toDouble() ?? 0.0,
      tooSmooth: (map['tooSmooth'] as bool?) ?? false,
      durationMs: (map['durationMs'] as num?)?.toDouble() ?? 0.0,
    );
  }

  /// Converts this [TrajectoryResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'rulesPassed': rules.toMap(),
      'startYaw': startYaw,
      'endYaw': endYaw,
      'extremeYaw': extremeYaw,
      'extremeFrame': extremeFrame,
      'wrongWayMax': wrongWayMax,
      'deltaVariance': deltaVariance,
      'tooSmooth': tooSmooth,
      'durationMs': durationMs,
    };
  }

  @override
  String toString() =>
      'TrajectoryResult(rules: $rules, startYaw: ${startYaw.toStringAsFixed(1)}, '
      'endYaw: ${endYaw.toStringAsFixed(1)}, '
      'extremeYaw: ${extremeYaw.toStringAsFixed(1)}, '
      'tooSmooth: $tooSmooth, durationMs: ${durationMs.toStringAsFixed(0)})';
}

// ============================
// Unified verifyFace (Anti-TOCTOU)
// ============================

/// Result of `verifyFace`: quality gate + liveness + embedding on one frame.
///
/// The [embedding] (and optional [match]) are only populated when both the
/// quality gate passes and liveness is confirmed, closing the TOCTOU gap.
class VerifyFaceResult {
  /// Whether the face is live and the frame passed the quality gate.
  final bool isLive;

  /// Threshold semantic version (e.g. `liveness-v3`).
  final String thresholdVersion;

  /// The quality-gate outcome (always present).
  final QualityResult quality;

  /// The liveness detail (absent when the quality gate failed).
  final LivenessResult? liveness;

  /// The 192-dim L2-normalized embedding, or `null` when not live.
  final List<double>? embedding;

  /// Optional comparison against a provided reference embedding.
  final MatchResult? match;

  /// Raw ensemble probabilities (only on debuggable builds when requested).
  final Map<String, dynamic>? debug;

  /// Creates a [VerifyFaceResult].
  const VerifyFaceResult({
    required this.isLive,
    required this.thresholdVersion,
    required this.quality,
    this.liveness,
    this.embedding,
    this.match,
    this.debug,
  });

  /// Creates a [VerifyFaceResult] from a raw platform map.
  factory VerifyFaceResult.fromMap(Map<Object?, Object?> map) {
    final qualityMap = map['quality'];
    final livenessMap = map['liveness'];
    final matchMap = map['match'];
    final embeddingRaw = map['embedding'];
    final debugMap = map['debug'];

    return VerifyFaceResult(
      isLive: (map['isLive'] as bool?) ?? false,
      thresholdVersion:
          (map['thresholdVersion'] as String?) ?? 'liveness-v3',
      quality: qualityMap is Map<Object?, Object?>
          ? QualityResult.fromMap(qualityMap)
          : const QualityResult(passed: false, reason: 'UNKNOWN'),
      liveness: livenessMap is Map<Object?, Object?>
          ? LivenessResult.fromMap(livenessMap)
          : null,
      embedding: embeddingRaw is List<dynamic>
          ? embeddingRaw
              .map((e) => (e as num).toDouble())
              .toList(growable: false)
          : null,
      match: matchMap is Map<Object?, Object?>
          ? MatchResult.fromMap(matchMap)
          : null,
      debug: debugMap is Map<Object?, Object?>
          ? Map<String, dynamic>.from(debugMap)
          : null,
    );
  }

  /// Converts this [VerifyFaceResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isLive': isLive,
      'thresholdVersion': thresholdVersion,
      'quality': quality.toMap(),
      if (liveness != null) 'liveness': liveness!.toMap(),
      if (embedding != null) 'embedding': embedding,
      if (match != null) 'match': match!.toMap(),
      if (debug != null) 'debug': debug,
    };
  }

  @override
  String toString() =>
      'VerifyFaceResult(isLive: $isLive, quality: $quality, '
      'liveness: $liveness, hasEmbedding: ${embedding != null}, '
      'match: $match, version: $thresholdVersion)';
}

// ============================
// Multi-frame sequence verification
// ============================

/// Result of `verifyFaceSequence` / `captureLivenessSession`.
///
/// Aggregates the per-frame pipeline over a temporal challenge: quality voting,
/// zero-tolerance spoof veto, identity consistency, and the R1–R5 trajectory.
class SequenceVerifyResult {
  /// Whether the whole session passed every gate.
  final bool passed;

  /// Failure reason when [passed] is `false`
  /// (e.g. `SPOOF_FRAME_DETECTED`, `SCREEN_SUSPECTED`, `TIMING_SUSPECT`,
  /// `CHALLENGE_WRONG_DIRECTION`, `CHALLENGE_NO_RETURN`, `NOT_LIVE_ENOUGH`,
  /// `IDENTITY_INCONSISTENT`, `QUALITY_FAILED`).
  final String? reason;

  /// The challenge direction that was evaluated.
  final ChallengeDirection challenge;

  /// Median live score across frames.
  final double medianLiveScore;

  /// Number of frames classified live.
  final int liveVotes;

  /// Total number of frames processed.
  final int frameCount;

  /// Maximum normalized yaw observed.
  final double maxYaw;

  /// Minimum normalized yaw observed.
  final double minYaw;

  /// Number of near-frontal frames.
  final int frontalFrames;

  /// Whether identity remained consistent across comparable frames.
  final bool identityConsistent;

  /// Minimum cosine similarity across comparable frames, or `null` if fewer
  /// than two comparable frames were available.
  final double? minIdentityCosine;

  /// The temporal trajectory summary (R1–R5).
  final TrajectoryResult trajectory;

  /// Aggregated screen-artifact soft signal (max across frames).
  final double screenSuspect;

  /// Threshold semantic version (e.g. `liveness-v3`).
  final String thresholdVersion;

  /// The challenge session identifier used.
  final String? challengeId;

  /// Whether frames were captured by the native camera session.
  final bool native;

  /// Optional comparison against a provided reference embedding.
  final MatchResult? match;

  /// Device integrity snapshot (only for the native capture session).
  final DeviceIntegrityResult? deviceIntegrity;

  /// Encrypted audit-trail identifier (only for the native capture session).
  final String? auditId;

  /// Creates a [SequenceVerifyResult].
  const SequenceVerifyResult({
    required this.passed,
    this.reason,
    required this.challenge,
    required this.medianLiveScore,
    required this.liveVotes,
    required this.frameCount,
    required this.maxYaw,
    required this.minYaw,
    required this.frontalFrames,
    required this.identityConsistent,
    this.minIdentityCosine,
    required this.trajectory,
    required this.screenSuspect,
    required this.thresholdVersion,
    this.challengeId,
    required this.native,
    this.match,
    this.deviceIntegrity,
    this.auditId,
  });

  /// Creates a [SequenceVerifyResult] from a raw platform map.
  factory SequenceVerifyResult.fromMap(Map<Object?, Object?> map) {
    final trajectoryMap = map['trajectory'];
    final matchMap = map['match'];
    final integrityMap = map['deviceIntegrity'];
    final minCos = map['minIdentityCosine'];

    return SequenceVerifyResult(
      passed: (map['passed'] as bool?) ?? false,
      reason: map['reason'] as String?,
      challenge: ChallengeDirection.fromString(map['challenge'] as String?),
      medianLiveScore: (map['medianLiveScore'] as num?)?.toDouble() ?? 0.0,
      liveVotes: (map['liveVotes'] as num?)?.toInt() ?? 0,
      frameCount: (map['frameCount'] as num?)?.toInt() ?? 0,
      maxYaw: (map['maxYaw'] as num?)?.toDouble() ?? 0.0,
      minYaw: (map['minYaw'] as num?)?.toDouble() ?? 0.0,
      frontalFrames: (map['frontalFrames'] as num?)?.toInt() ?? 0,
      identityConsistent: (map['identityConsistent'] as bool?) ?? false,
      minIdentityCosine: minCos is num ? minCos.toDouble() : null,
      trajectory: trajectoryMap is Map<Object?, Object?>
          ? TrajectoryResult.fromMap(trajectoryMap)
          : const TrajectoryResult(
              rules: TrajectoryRules(
                frontalStart: false,
                deviation: false,
                returnFrontal: false,
                singleDirection: false,
                smoothMotion: false,
              ),
              startYaw: 0,
              endYaw: 0,
              extremeYaw: 0,
              extremeFrame: -1,
              wrongWayMax: 0,
              deltaVariance: 0,
              tooSmooth: false,
              durationMs: 0,
            ),
      screenSuspect: (map['screenSuspect'] as num?)?.toDouble() ?? 0.0,
      thresholdVersion:
          (map['thresholdVersion'] as String?) ?? 'liveness-v3',
      challengeId: map['challengeId'] as String?,
      native: (map['native'] as bool?) ?? false,
      match: matchMap is Map<Object?, Object?>
          ? MatchResult.fromMap(matchMap)
          : null,
      deviceIntegrity: integrityMap is Map<Object?, Object?>
          ? DeviceIntegrityResult.fromMap(integrityMap)
          : null,
      auditId: map['auditId'] as String?,
    );
  }

  /// Converts this [SequenceVerifyResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'passed': passed,
      if (reason != null) 'reason': reason,
      'challenge': challenge.value,
      'medianLiveScore': medianLiveScore,
      'liveVotes': liveVotes,
      'frameCount': frameCount,
      'maxYaw': maxYaw,
      'minYaw': minYaw,
      'frontalFrames': frontalFrames,
      'identityConsistent': identityConsistent,
      'minIdentityCosine': minIdentityCosine,
      'trajectory': trajectory.toMap(),
      'screenSuspect': screenSuspect,
      'thresholdVersion': thresholdVersion,
      if (challengeId != null) 'challengeId': challengeId,
      'native': native,
      if (match != null) 'match': match!.toMap(),
      if (deviceIntegrity != null) 'deviceIntegrity': deviceIntegrity!.toMap(),
      if (auditId != null) 'auditId': auditId,
    };
  }

  @override
  String toString() =>
      'SequenceVerifyResult(passed: $passed, reason: $reason, '
      'challenge: ${challenge.value}, medianLiveScore: '
      '${medianLiveScore.toStringAsFixed(4)}, liveVotes: $liveVotes/'
      '$frameCount, identityConsistent: $identityConsistent, '
      'native: $native, version: $thresholdVersion)';
}

// ============================
// Device Integrity
// ============================

/// Client-side device risk signals (heuristic — not proof of compromise).
class DeviceIntegrityResult {
  /// Whether the app appears to run on an emulator.
  final bool isEmulator;

  /// Whether the device appears rooted / jailbroken.
  final bool isRooted;

  /// Whether the build is debuggable (development build).
  final bool isDebuggable;

  /// Whether hooking frameworks (Frida / Xposed / Substrate) were detected.
  final bool hooksDetected;

  /// Creates a [DeviceIntegrityResult].
  const DeviceIntegrityResult({
    required this.isEmulator,
    required this.isRooted,
    required this.isDebuggable,
    required this.hooksDetected,
  });

  /// Creates a [DeviceIntegrityResult] from a raw platform map.
  factory DeviceIntegrityResult.fromMap(Map<Object?, Object?> map) {
    return DeviceIntegrityResult(
      isEmulator: (map['isEmulator'] as bool?) ?? false,
      isRooted: (map['isRooted'] as bool?) ?? false,
      isDebuggable: (map['isDebuggable'] as bool?) ?? false,
      hooksDetected: (map['hooksDetected'] as bool?) ?? false,
    );
  }

  /// Converts this [DeviceIntegrityResult] to a serializable map.
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'isEmulator': isEmulator,
      'isRooted': isRooted,
      'isDebuggable': isDebuggable,
      'hooksDetected': hooksDetected,
    };
  }

  /// Returns `true` when at least one risk signal is present.
  bool get hasRiskSignal =>
      isEmulator || isRooted || isDebuggable || hooksDetected;

  @override
  String toString() =>
      'DeviceIntegrityResult(emulator: $isEmulator, rooted: $isRooted, '
      'debuggable: $isDebuggable, hooks: $hooksDetected)';
}
