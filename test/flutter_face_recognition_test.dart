import 'package:flutter/services.dart';
import 'package:flutter_face_recognition/flutter_face_recognition.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel('flutter_face_recognition');
  final plugin = const FaceRecognition();

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

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

  group('MethodChannel communication', () {
    test('extractFaceFeature sends Uint8List bytes and parses 192-D embedding',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        expect(methodCall.method, 'extractFaceFeature');
        final args = methodCall.arguments as Map<Object?, Object?>;
        expect(args['bytes'], isA<Uint8List>());
        return <String, dynamic>{
          'embedding': List<double>.filled(192, 0.1),
        };
      });

      final dummyBytes = Uint8List.fromList(<int>[1, 2, 3, 4]);
      final embedding = await plugin.extractFaceFeature(bytes: dummyBytes);
      expect(embedding.length, 192);
      expect(embedding.first, closeTo(0.1, 1e-9));
    });

    test('checkLiveness parses score, isLive, and LivenessConfidence',
        () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        expect(methodCall.method, 'checkLiveness');
        return <String, dynamic>{
          'isLive': true,
          'score': 0.92,
          'confidence': 'HIGH',
        };
      });

      final dummyBytes = Uint8List.fromList(<int>[10, 20, 30]);
      final liveness = await plugin.checkLiveness(bytes: dummyBytes);
      expect(liveness.isLive, isTrue);
      expect(liveness.score, closeTo(0.92, 1e-9));
      expect(liveness.confidence, LivenessConfidence.high);
    });

    test('detectFaces parses faces and eye open probabilities', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        expect(methodCall.method, 'detectFaces');
        return <String, dynamic>{
          'count': 1,
          'faces': <Map<String, dynamic>>[
            <String, dynamic>{
              'x': 24,
              'y': 48,
              'width': 200,
              'height': 220,
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
      expect(result.firstFace!.x, 24);
    });
  });
}

