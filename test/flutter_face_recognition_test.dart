import 'package:flutter/services.dart';
import 'package:flutter_face_recognition/flutter_face_recognition.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel('flutter_face_recognition');
  const plugin = FaceRecognition();

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  Map<Object?, Object?> captureArgs(MethodCall call) =>
      call.arguments as Map<Object?, Object?>;

  group('Pure-Dart Cosine Similarity & L2 Normalization', () {
    test('identical vectors return score 1.0 and 100% similarity', () {
      final v1 = List<double>.generate(192, (i) => (i + 1).toDouble());
      final v2 = List<double>.generate(192, (i) => (i + 1).toDouble());

      final result = plugin.compareFaces(v1, v2);
      expect(result.isMatch, isTrue);
      expect(result.score, closeTo(1.0, 1e-9));
      expect(result.similarityPercentage, closeTo(100.0, 1e-6));
    });

    test('orthogonal vectors return score 0.0 and isMatch false', () {
      final v1 = <double>[1.0, 0.0, 0.0];
      final v2 = <double>[0.0, 1.0, 0.0];

      final result = plugin.compareFaces(v1, v2);
      expect(result.isMatch, isFalse);
      expect(result.score, closeTo(0.0, 1e-9));
      expect(result.similarityPercentage, closeTo(0.0, 1e-9));
    });

    test('zero-norm vector is guarded and returns 0.0 instead of NaN', () {
      final zeroVec = <double>[0.0, 0.0, 0.0];
      final normalVec = <double>[0.5, 0.5, 0.5];

      final result = plugin.compareFaces(zeroVec, normalVec);
      expect(result.isMatch, isFalse);
      expect(result.score.isNaN, isFalse);
      expect(result.score, equals(0.0));
    });

    test('l2Normalize produces a unit vector with norm 1.0', () {
      final raw = <double>[3.0, 4.0];
      final normalized = FaceRecognition.l2Normalize(raw);

      expect(normalized[0], closeTo(0.6, 1e-9));
      expect(normalized[1], closeTo(0.8, 1e-9));
    });
  });

  group('Model serialization (liveness-v3)', () {
    test('GateConfig.toMap carries every tunable key', () {
      const config = GateConfig(
        liveThreshold: 0.85,
        spoofVeto: 0.15,
        yawSignOverride: -1,
        minLiveVoteRatio: 0.8,
        screenSuspectVeto: 0.6,
        debug: true,
      );
      final map = config.toMap();
      expect(map['liveThreshold'], 0.85);
      expect(map['spoofVeto'], 0.15);
      expect(map['yawSignOverride'], -1);
      expect(map['minLiveVoteRatio'], 0.8);
      expect(map['screenSuspectVeto'], 0.6);
      expect(map['debug'], isTrue);
      expect(map.containsKey('minFrameSide'), isTrue);
    });

    test('QualityResult round-trips pass and failure shapes', () {
      final pass = QualityResult.fromMap(
          const <Object?, Object?>{'passed': true, 'faceRatio': 0.4, 'blur': 120.0});
      expect(pass.passed, isTrue);
      expect(pass.reason, isNull);
      expect(pass.faceRatio, closeTo(0.4, 1e-9));

      final fail = QualityResult.fromMap(
          const <Object?, Object?>{'passed': false, 'reason': 'BLURRY'});
      expect(fail.passed, isFalse);
      expect(fail.reason, 'BLURRY');
    });

    test('LivenessResult parses the dual-model ensemble scores', () {
      final liveness = LivenessResult.fromMap(const <Object?, Object?>{
        'isLive': true,
        'score': 0.91,
        'confidence': 'HIGH',
        'label': 'LIVE',
        'liveScore': 0.91,
        'printScore': 0.05,
        'replayScore': 0.04,
        'spoofScore': 0.05,
        'screenSuspectScore': 0.12,
        'thresholdVersion': 'liveness-v3',
      });
      expect(liveness.isLive, isTrue);
      expect(liveness.label, LivenessLabel.live);
      expect(liveness.liveScore, closeTo(0.91, 1e-9));
      expect(liveness.printScore, closeTo(0.05, 1e-9));
      expect(liveness.replayScore, closeTo(0.04, 1e-9));
      expect(liveness.spoofScore, closeTo(0.05, 1e-9));
      expect(liveness.screenSuspectScore, closeTo(0.12, 1e-9));
      expect(liveness.thresholdVersion, 'liveness-v3');
    });

    test('LivenessLabel and ChallengeDirection parse case-insensitively', () {
      expect(LivenessLabel.fromString('replay'), LivenessLabel.replay);
      expect(LivenessLabel.fromString('print'), LivenessLabel.print);
      expect(LivenessLabel.fromString(null), LivenessLabel.print);
      expect(ChallengeDirection.fromString('LOOK_RIGHT'),
          ChallengeDirection.lookRight);
      expect(ChallengeDirection.fromString('look_left'),
          ChallengeDirection.lookLeft);
    });

    test('VerifyFaceResult exposes embedding + match only when live', () {
      final live = VerifyFaceResult.fromMap(const <Object?, Object?>{
        'isLive': true,
        'thresholdVersion': 'liveness-v3',
        'quality': <Object?, Object?>{'passed': true, 'faceRatio': 0.5, 'blur': 90.0},
        'liveness': <Object?, Object?>{
          'isLive': true,
          'label': 'LIVE',
          'liveScore': 0.93,
          'spoofScore': 0.07,
        },
        'embedding': <double>[0.1, 0.2, 0.3],
        'match': <Object?, Object?>{
          'isMatch': true,
          'score': 0.82,
          'threshold': 0.75,
          'thresholdVersion': 'match-v1',
        },
      });
      expect(live.isLive, isTrue);
      expect(live.embedding, hasLength(3));
      expect(live.match!.isMatch, isTrue);
      expect(live.match!.thresholdVersion, 'match-v1');

      final notLive = VerifyFaceResult.fromMap(const <Object?, Object?>{
        'isLive': false,
        'thresholdVersion': 'liveness-v3',
        'quality': <Object?, Object?>{'passed': false, 'reason': 'BAD_POSE'},
        'liveness': null,
      });
      expect(notLive.isLive, isFalse);
      expect(notLive.embedding, isNull);
      expect(notLive.quality.reason, 'BAD_POSE');
    });

    test('TrajectoryResult parses R1-R5 rules and yaw statistics', () {
      final trajectory = TrajectoryResult.fromMap(const <Object?, Object?>{
        'rulesPassed': <Object?, Object?>{
          'frontalStart': true,
          'deviation': true,
          'returnFrontal': true,
          'singleDirection': true,
          'smoothMotion': true,
        },
        'startYaw': 1.0,
        'endYaw': -0.5,
        'extremeYaw': -22.0,
        'extremeFrame': 6,
        'wrongWayMax': 3.0,
        'deltaVariance': 0.21,
        'tooSmooth': false,
        'durationMs': 1440.0,
      });
      expect(trajectory.rules.frontalStart, isTrue);
      expect(trajectory.rules.smoothMotion, isTrue);
      expect(trajectory.extremeYaw, closeTo(-22.0, 1e-9));
      expect(trajectory.extremeFrame, 6);
      expect(trajectory.tooSmooth, isFalse);
    });

    test('SequenceVerifyResult parses a full session and nullable minIdentityCosine',
        () {
      final seq = SequenceVerifyResult.fromMap(const <Object?, Object?>{
        'passed': true,
        'challenge': 'LOOK_LEFT',
        'medianLiveScore': 0.9,
        'liveVotes': 5,
        'frameCount': 5,
        'maxYaw': 2.0,
        'minYaw': -19.0,
        'frontalFrames': 3,
        'identityConsistent': true,
        'minIdentityCosine': 0.71,
        'trajectory': <Object?, Object?>{
          'rulesPassed': <Object?, Object?>{
            'frontalStart': true,
            'deviation': true,
            'returnFrontal': true,
            'singleDirection': true,
            'smoothMotion': true,
          },
        },
        'screenSuspect': 0.1,
        'thresholdVersion': 'liveness-v3',
        'challengeId': 'abc-123',
        'native': false,
      });
      expect(seq.passed, isTrue);
      expect(seq.challenge, ChallengeDirection.lookLeft);
      expect(seq.minIdentityCosine, closeTo(0.71, 1e-9));
      expect(seq.challengeId, 'abc-123');
      expect(seq.native, isFalse);
      expect(seq.trajectory.rules.deviation, isTrue);

      final nullCos = SequenceVerifyResult.fromMap(const <Object?, Object?>{
        'passed': false,
        'reason': 'CHALLENGE_NO_RETURN',
        'challenge': 'LOOK_RIGHT',
        'minIdentityCosine': null,
      });
      expect(nullCos.minIdentityCosine, isNull);
      expect(nullCos.reason, 'CHALLENGE_NO_RETURN');
    });

    test('ChallengeSession & DeviceIntegrityResult parse', () {
      final session = ChallengeSession.fromMap(const <Object?, Object?>{
        'challengeId': 'id-1',
        'challenge': 'LOOK_RIGHT',
        'expiresAt': 1700000000000,
      });
      expect(session.challenge, ChallengeDirection.lookRight);
      expect(session.challengeId, 'id-1');

      final integrity = DeviceIntegrityResult.fromMap(const <Object?, Object?>{
        'isEmulator': false,
        'isRooted': true,
        'isDebuggable': false,
        'hooksDetected': false,
      });
      expect(integrity.isRooted, isTrue);
      expect(integrity.hasRiskSignal, isTrue);
    });
  });

  group('MethodChannel communication', () {
    test('initialize passes enableInsecureFramePath', () async {
      late bool captured;
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'initialize');
        captured = captureArgs(call)['enableInsecureFramePath'] as bool;
        return null;
      });

      await plugin.initialize(enableInsecureFramePath: true);
      expect(captured, isTrue);
    });

    test('extractFaceFeature sends Uint8List bytes and parses 192-D embedding',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'extractFaceFeature');
        expect(captureArgs(call)['bytes'], isA<Uint8List>());
        return <String, dynamic>{
          'embedding': List<double>.filled(192, 0.1),
        };
      });

      final embedding =
          await plugin.extractFaceFeature(bytes: Uint8List.fromList([1, 2, 3]));
      expect(embedding, hasLength(192));
      expect(embedding.first, closeTo(0.1, 1e-9));
    });

    test('checkLiveness parses the dual-model ensemble result', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'checkLiveness');
        // threshold is forwarded as the liveThreshold when supplied.
        expect(captureArgs(call)['threshold'], 0.8);
        return <String, dynamic>{
          'isLive': true,
          'score': 0.92,
          'confidence': 'HIGH',
          'label': 'LIVE',
          'liveScore': 0.92,
          'printScore': 0.04,
          'replayScore': 0.04,
          'spoofScore': 0.04,
          'screenSuspectScore': 0.1,
          'thresholdVersion': 'liveness-v3',
        };
      });

      final liveness = await plugin.checkLiveness(
        bytes: Uint8List.fromList([10, 20, 30]),
        threshold: 0.8,
      );
      expect(liveness.isLive, isTrue);
      expect(liveness.label, LivenessLabel.live);
      expect(liveness.score, closeTo(0.92, 1e-9));
      expect(liveness.confidence, LivenessConfidence.high);
    });

    test('detectFaces parses faces, pitch/yaw/roll, and eye probabilities',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'detectFaces');
        return <String, dynamic>{
          'count': 1,
          'faces': <Map<String, dynamic>>[
            <String, dynamic>{
              'x': 24,
              'y': 48,
              'width': 200,
              'height': 220,
              'headEulerAngleX': -1.5,
              'headEulerAngleY': -3.5,
              'headEulerAngleZ': 1.2,
              'leftEyeOpenProbability': 0.95,
              'rightEyeOpenProbability': 0.91,
            },
          ],
        };
      });

      final result = await plugin.detectFaces(base64: 'aGVsbG8=');
      expect(result.hasSingleFace, isTrue);
      expect(result.firstFace!.isBothEyesOpen(), isTrue);
      expect(result.firstFace!.isBlinking(), isFalse);
      expect(result.firstFace!.headEulerAngleX, closeTo(-1.5, 1e-9));
      expect(result.firstFace!.x, 24);
    });

    test('verifyFace returns embedding + match on the same frame', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'verifyFace');
        final args = captureArgs(call);
        expect(args['referenceEmbedding'], isA<List<dynamic>>());
        expect(args['config'], isA<Map<Object?, Object?>>());
        return <String, dynamic>{
          'isLive': true,
          'thresholdVersion': 'liveness-v3',
          'quality': <String, dynamic>{'passed': true, 'faceRatio': 0.5, 'blur': 100.0},
          'liveness': <String, dynamic>{
            'isLive': true,
            'label': 'LIVE',
            'liveScore': 0.95,
            'spoofScore': 0.05,
          },
          'embedding': List<double>.filled(192, 0.05),
          'match': <String, dynamic>{
            'isMatch': true,
            'score': 0.88,
            'threshold': 0.75,
            'thresholdVersion': 'match-v1',
          },
        };
      });

      final verify = await plugin.verifyFace(
        bytes: Uint8List.fromList([1, 2, 3]),
        referenceEmbedding: List<double>.filled(192, 0.05),
        config: const GateConfig(liveThreshold: 0.8),
      );
      expect(verify.isLive, isTrue);
      expect(verify.embedding, hasLength(192));
      expect(verify.match!.isMatch, isTrue);
      expect(verify.quality.passed, isTrue);
    });

    test('createChallengeSession parses the single-use session', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'createChallengeSession');
        return <String, dynamic>{
          'challengeId': 'sess-1',
          'challenge': 'LOOK_LEFT',
          'expiresAt': 1700000000000,
        };
      });

      final session = await plugin.createChallengeSession();
      expect(session.challengeId, 'sess-1');
      expect(session.challenge, ChallengeDirection.lookLeft);
    });

    test('verifyFaceSequence forwards frames + challengeId and parses result',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'verifyFaceSequence');
        final args = captureArgs(call);
        expect(args['frames'], isA<List<dynamic>>());
        expect((args['frames'] as List<dynamic>).length, 2);
        expect(args['challengeId'], 'sess-2');
        return <String, dynamic>{
          'passed': true,
          'challenge': 'LOOK_LEFT',
          'medianLiveScore': 0.9,
          'liveVotes': 2,
          'frameCount': 2,
          'maxYaw': 1.0,
          'minYaw': -18.0,
          'frontalFrames': 1,
          'identityConsistent': true,
          'minIdentityCosine': 0.8,
          'trajectory': <String, dynamic>{
            'rulesPassed': <String, dynamic>{
              'frontalStart': true,
              'deviation': true,
              'returnFrontal': true,
              'singleDirection': true,
              'smoothMotion': true,
            },
            'startYaw': 1.0,
            'endYaw': -1.0,
            'extremeYaw': -18.0,
            'extremeFrame': 1,
            'wrongWayMax': 0.0,
            'deltaVariance': 0.3,
            'tooSmooth': false,
            'durationMs': 240.0,
          },
          'screenSuspect': 0.05,
          'thresholdVersion': 'liveness-v3',
          'challengeId': 'sess-2',
          'native': false,
        };
      });

      final seq = await plugin.verifyFaceSequence(
        frames: [
          Uint8List.fromList([1, 2]),
          Uint8List.fromList([3, 4]),
        ],
        challengeId: 'sess-2',
      );
      expect(seq.passed, isTrue);
      expect(seq.challenge, ChallengeDirection.lookLeft);
      expect(seq.trajectory.rules.deviation, isTrue);
      expect(seq.native, isFalse);
    });

    test('captureLivenessSession parses native result + integrity + auditId',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'captureLivenessSession');
        return <String, dynamic>{
          'passed': true,
          'challenge': 'LOOK_RIGHT',
          'medianLiveScore': 0.91,
          'liveVotes': 12,
          'frameCount': 12,
          'maxYaw': 20.0,
          'minYaw': -1.0,
          'frontalFrames': 4,
          'identityConsistent': true,
          'minIdentityCosine': 0.77,
          'trajectory': <String, dynamic>{
            'rulesPassed': <String, dynamic>{
              'frontalStart': true,
              'deviation': true,
              'returnFrontal': true,
              'singleDirection': true,
              'smoothMotion': true,
            },
          },
          'screenSuspect': 0.2,
          'thresholdVersion': 'liveness-v3',
          'challengeId': 'sess-3',
          'native': true,
          'deviceIntegrity': <String, dynamic>{
            'isEmulator': false,
            'isRooted': false,
            'isDebuggable': false,
            'hooksDetected': false,
          },
          'auditId': 'session_20260101_120000',
        };
      });

      final seq = await plugin.captureLivenessSession();
      expect(seq.passed, isTrue);
      expect(seq.native, isTrue);
      expect(seq.auditId, 'session_20260101_120000');
      expect(seq.deviceIntegrity!.isEmulator, isFalse);
    });

    test('getDeviceIntegrity parses the nested snapshot', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall call) async {
        expect(call.method, 'getDeviceIntegrity');
        return <String, dynamic>{
          'deviceIntegrity': <String, dynamic>{
            'isEmulator': true,
            'isRooted': false,
            'isDebuggable': true,
            'hooksDetected': false,
          },
        };
      });

      final integrity = await plugin.getDeviceIntegrity();
      expect(integrity.isEmulator, isTrue);
      expect(integrity.isDebuggable, isTrue);
      expect(integrity.hasRiskSignal, isTrue);
    });
  });
}
