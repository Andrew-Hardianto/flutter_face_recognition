import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_face_recognition/flutter_face_recognition.dart';
import 'package:image_picker/image_picker.dart';

import 'camera_view.dart';

void main() {
  runApp(const FaceRecognitionDemoApp());
}

class FaceRecognitionDemoApp extends StatelessWidget {
  const FaceRecognitionDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Face Recognition Demo (liveness-v3)',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2563EB),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const FaceRecognitionHomePage(),
    );
  }
}

class FaceRecognitionHomePage extends StatefulWidget {
  const FaceRecognitionHomePage({super.key});

  @override
  State<FaceRecognitionHomePage> createState() =>
      _FaceRecognitionHomePageState();
}

class _FaceRecognitionHomePageState extends State<FaceRecognitionHomePage> {
  final FaceRecognition _faceRecognition = const FaceRecognition();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _currentImageBytes;
  FaceDetectionResult? _detectionResult;

  // liveness-v3 state
  VerifyFaceResult? _verifyResult;
  SequenceVerifyResult? _sequenceResult;
  ChallengeSession? _challengeSession;
  DeviceIntegrityResult? _deviceIntegrity;

  List<double>? _enrolledEmbedding;
  double _threshold = 0.80;
  bool _isProcessing = false;
  String _statusMessage = 'Siap. Silakan daftarkan wajah terlebih dahulu.';

  @override
  void initState() {
    super.initState();
    // In production you would NOT enable the insecure frame path; the active
    // challenge below uses the native capture session instead. This flag is
    // only so the demo can also exercise verifyFaceSequence with picked frames.
    _faceRecognition.initialize(enableInsecureFramePath: true).then((_) {
      return _faceRecognition.getDeviceIntegrity();
    }).then((integrity) {
      if (mounted) setState(() => _deviceIntegrity = integrity);
    }).catchError((_) {
      // Ignore — integrity is optional in the demo.
    });
  }

  @override
  void dispose() {
    _faceRecognition.dispose();
    super.dispose();
  }

  Future<Uint8List?> _pickImage(ImageSource source) async {
    try {
      final file = await _picker.pickImage(
        source: source,
        preferredCameraDevice: CameraDevice.front,
      );
      if (file == null) return null;
      final bytes = await file.readAsBytes();
      setState(() {
        _currentImageBytes = bytes;
        _detectionResult = null;
        _verifyResult = null;
      });
      return bytes;
    } catch (e) {
      setState(() => _statusMessage = 'Gagal mengambil gambar: $e');
      return null;
    }
  }

  Future<void> _handleEnroll() async {
    final bytes = _currentImageBytes;
    if (bytes == null) {
      setState(() => _statusMessage = 'Pilih atau ambil foto terlebih dahulu.');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Mendeteksi wajah & mengekstrak 192-dim embedding...';
    });

    try {
      final detection = await _faceRecognition.detectFaces(bytes: bytes);
      if (detection.count != 1) {
        setState(() {
          _detectionResult = detection;
          _statusMessage = detection.count == 0
              ? 'Tidak ada wajah yang terdeteksi pada foto.'
              : 'Terdeteksi ${detection.count} wajah. Gunakan foto dengan 1 wajah saja.';
        });
        return;
      }

      final embedding = await _faceRecognition.extractFaceFeature(bytes: bytes);
      setState(() {
        _detectionResult = detection;
        _enrolledEmbedding = embedding;
        _statusMessage =
            'Wajah berhasil didaftarkan (${embedding.length} dimensi, L2-normalized).';
      });
    } on PlatformException catch (e) {
      setState(() => _statusMessage = 'Error [${e.code}]: ${e.message}');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  /// Anti-TOCTOU: quality gate + liveness + embedding on ONE frame & bbox.
  Future<void> _handleVerifyFace() async {
    final bytes = _currentImageBytes;
    if (bytes == null) {
      setState(() => _statusMessage = 'Pilih atau ambil foto terlebih dahulu.');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Menjalankan verifyFace (Quality Gate + Dual Anti-Spoof)...';
    });

    try {
      final verify = await _faceRecognition.verifyFace(
        bytes: bytes,
        referenceEmbedding: _enrolledEmbedding,
        config: GateConfig(liveThreshold: _threshold),
      );

      String msg;
      if (!verify.quality.passed) {
        msg = 'QUALITY GATE GAGAL: ${verify.quality.reason}';
      } else if (!verify.isLive) {
        final lv = verify.liveness;
        msg = 'SPOOF TERDETEKSI (label: ${lv?.label.value}).\n'
            'live=${_pct(lv?.liveScore)} print=${_pct(lv?.printScore)} '
            'replay=${_pct(lv?.replayScore)} spoof=${_pct(lv?.spoofScore)}';
      } else if (verify.match != null) {
        msg = verify.match!.isMatch
            ? 'VERIFIKASI BERHASIL (match). '
                'Cosine: ${verify.match!.similarityPercentage.toStringAsFixed(1)}%'
            : 'WAJAH TIDAK COCOK. '
                'Cosine: ${verify.match!.similarityPercentage.toStringAsFixed(1)}%';
      } else {
        msg = 'LIVE terkonfirmasi & embedding diekstrak (tanpa referensi).';
      }

      setState(() {
        _verifyResult = verify;
        _statusMessage = msg;
      });
    } on PlatformException catch (e) {
      setState(() => _statusMessage = 'Error [${e.code}]: ${e.message}');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  /// Active challenge: native capture session (anti frame-injection).
  Future<void> _handleActiveChallenge() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Membuka sesi capture native...';
    });

    try {
      // 1. Issue a single-use, randomized challenge (LOOK_LEFT / LOOK_RIGHT).
      final session = await _faceRecognition.createChallengeSession();
      setState(() {
        _challengeSession = session;
        _statusMessage =
            'Challenge: ${session.challenge.value}. Ikuti instruksi di kamera.';
      });

      // 2. Run the native capture session end to end.
      final result = await _faceRecognition.captureLivenessSession(
        referenceEmbedding: _enrolledEmbedding,
        config: GateConfig(liveThreshold: _threshold),
      );

      setState(() {
        _sequenceResult = result;
        _deviceIntegrity = result.deviceIntegrity ?? _deviceIntegrity;
        _statusMessage = result.passed
            ? 'CHALLENGE LOLOS (native=${result.native}). '
                'medianLive=${_pct(result.medianLiveScore)} '
                'votes=${result.liveVotes}/${result.frameCount} '
                'audit=${result.auditId ?? "-"}'
            : 'CHALLENGE GAGAL: ${result.reason} '
                '(trajectory R1-R5: ${result.trajectory.rules})';
      });
    } on PlatformException catch (e) {
      setState(() => _statusMessage = 'Error [${e.code}]: ${e.message}');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _handleBlinkAndFaceCheck() async {
    final bytes = _currentImageBytes;
    if (bytes == null) {
      setState(() => _statusMessage = 'Pilih atau ambil foto terlebih dahulu.');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Mendeteksi koordinat wajah, sudut kepala & mata...';
    });

    try {
      final detection = await _faceRecognition.detectFaces(bytes: bytes);
      final firstFace = detection.firstFace;
      setState(() {
        _detectionResult = detection;
        if (firstFace == null) {
          _statusMessage = 'Tidak ada wajah terdeteksi.';
        } else {
          final blinkStatus = firstFace.isBlinking()
              ? 'MATA BERKEDIP / TERTUTUP'
              : (firstFace.isBothEyesOpen() ? 'MATA TERBUKA' : 'SEBAGIAN TERTUTUP');
          _statusMessage =
              'Terdeteksi ${detection.count} wajah | Status mata: $blinkStatus';
        }
      });
    } on PlatformException catch (e) {
      setState(() => _statusMessage = 'Error [${e.code}]: ${e.message}');
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  String _pct(double? v) =>
      v == null ? '-' : '${(v * 100).toStringAsFixed(1)}%';

  @override
  Widget build(BuildContext context) {
    final face = _detectionResult?.firstFace;

    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Flutter Face Recognition • liveness-v3'),
          bottom: const TabBar(
            isScrollable: true,
            tabs: [
              Tab(icon: Icon(Icons.person_add_alt_1), text: '1. Enroll'),
              Tab(icon: Icon(Icons.verified_user), text: '2. verifyFace'),
              Tab(icon: Icon(Icons.sensors), text: '3. Challenge'),
              Tab(icon: Icon(Icons.remove_red_eye), text: '4. Blink'),
            ],
          ),
        ),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                CameraFaceOverlayView(
                  imageBytes: _currentImageBytes,
                  detectionResult: _detectionResult,
                  isProcessing: _isProcessing,
                  onPickCamera: () => _pickImage(ImageSource.camera),
                  onPickGallery: () => _pickImage(ImageSource.gallery),
                ),
                const SizedBox(height: 16),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Icon(
                          _enrolledEmbedding != null
                              ? Icons.check_circle
                              : Icons.info_outline,
                          color: _enrolledEmbedding != null
                              ? Colors.greenAccent
                              : Colors.amberAccent,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _statusMessage,
                            style: const TextStyle(fontSize: 13),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  height: 360,
                  child: TabBarView(
                    children: [
                      _buildEnrollTab(),
                      _buildVerifyTab(),
                      _buildChallengeTab(),
                      _buildBlinkTab(face),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEnrollTab() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Pendaftaran Wajah (Enrollment)',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              _enrolledEmbedding == null
                  ? 'Status: Belum ada embedding wajah yang tersimpan.'
                  : 'Status: Embedding terdaftar (${_enrolledEmbedding!.length} dimensi).\n'
                      'Sample[0..2]: ${_enrolledEmbedding!.take(3).map((v) => v.toStringAsFixed(4)).join(', ')}',
              style: const TextStyle(color: Colors.white70),
            ),
            const Spacer(),
            FilledButton.icon(
              onPressed: _isProcessing ? null : _handleEnroll,
              icon: const Icon(Icons.fingerprint),
              label: const Text('Ekstrak & Simpan Wajah (192-D)'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVerifyTab() {
    final verify = _verifyResult;
    final quality = verify?.quality;
    final liveness = verify?.liveness;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Live Threshold',
                    style: TextStyle(fontWeight: FontWeight.bold)),
                Text(
                  _threshold.toStringAsFixed(2),
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.lightBlueAccent,
                  ),
                ),
              ],
            ),
            Slider(
              value: _threshold,
              min: 0.60,
              max: 0.95,
              divisions: 35,
              label: _threshold.toStringAsFixed(2),
              onChanged: (val) => setState(() => _threshold = val),
            ),
            if (quality != null)
              Text(
                'Quality Gate: ${quality.passed ? "PASS" : "FAIL (${quality.reason})"} '
                '| faceRatio=${quality.faceRatio.toStringAsFixed(2)} '
                '| blur=${quality.blur.toStringAsFixed(0)}',
                style: TextStyle(
                  color: quality.passed ? Colors.greenAccent : Colors.redAccent,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
            if (liveness != null) ...[
              const SizedBox(height: 4),
              Text(
                'Label: ${liveness.label.value}  (${liveness.isLive ? "LIVE" : "SPOOF"})',
                style: TextStyle(
                  color: liveness.isLive ? Colors.greenAccent : Colors.redAccent,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                'live=${_pct(liveness.liveScore)}  print=${_pct(liveness.printScore)}  '
                'replay=${_pct(liveness.replayScore)}  spoof=${_pct(liveness.spoofScore)}',
                style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
              ),
            ],
            if (verify?.match != null) ...[
              const SizedBox(height: 6),
              Text(
                'Cosine Match: ${verify!.match!.similarityPercentage.toStringAsFixed(2)}% '
                '(${verify.match!.isMatch ? "MATCH" : "NO MATCH"})',
                style: TextStyle(
                  color: verify.match!.isMatch
                      ? Colors.greenAccent
                      : Colors.orangeAccent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
            const Spacer(),
            FilledButton.icon(
              onPressed: _isProcessing ? null : _handleVerifyFace,
              icon: const Icon(Icons.shield_outlined),
              label: const Text('verifyFace (Anti-TOCTOU)'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChallengeTab() {
    final seq = _sequenceResult;
    final session = _challengeSession;
    final traj = seq?.trajectory;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Active Challenge (Native Capture)',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              session == null
                  ? 'Belum ada challenge. Tekan tombol untuk memulai sesi native.'
                  : 'Challenge: ${session.challenge.value} '
                      '(berlaku single-use, TTL 60s)',
              style: const TextStyle(color: Colors.white70),
            ),
            if (seq != null) ...[
              const SizedBox(height: 8),
              Text(
                '${seq.passed ? "LOLOS" : "GAGAL (${seq.reason})"} '
                '| medianLive=${_pct(seq.medianLiveScore)} '
                '| votes=${seq.liveVotes}/${seq.frameCount} '
                '| native=${seq.native}',
                style: TextStyle(
                  color: seq.passed ? Colors.greenAccent : Colors.redAccent,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
              if (traj != null)
                Text(
                  'Trajectory R1-R5: ${traj.rules.toMap().entries.map((e) => "${e.key}=${e.value ? "✓" : "✗"}").join(" ")}',
                  style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
                ),
              if (seq.auditId != null)
                Text('Audit ID: ${seq.auditId}',
                    style: const TextStyle(fontSize: 11)),
            ],
            if (_deviceIntegrity != null) ...[
              const SizedBox(height: 8),
              Text(
                'Device Integrity: emulator=${_deviceIntegrity!.isEmulator} '
                'rooted=${_deviceIntegrity!.isRooted} '
                'debuggable=${_deviceIntegrity!.isDebuggable} '
                'hooks=${_deviceIntegrity!.hooksDetected}',
                style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
              ),
            ],
            const Spacer(),
            FilledButton.icon(
              onPressed: _isProcessing ? null : _handleActiveChallenge,
              icon: const Icon(Icons.sensors),
              label: const Text('Mulai Sesi Liveness Native'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBlinkTab(DetectedFace? face) {
    final leftProb = face?.leftEyeOpenProbability ?? 0.0;
    final rightProb = face?.rightEyeOpenProbability ?? 0.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Deteksi Mata Berkedip & Rotasi Kepala',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            if (face != null)
              Text(
                'Pitch=${face.headEulerAngleX.toStringAsFixed(1)}° '
                'Yaw=${face.headEulerAngleY.toStringAsFixed(1)}° '
                'Roll=${face.headEulerAngleZ.toStringAsFixed(1)}°',
                style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
              ),
            Text('Mata Kiri Terbuka: ${(leftProb * 100).toStringAsFixed(0)}%'),
            const SizedBox(height: 4),
            LinearProgressIndicator(value: leftProb.clamp(0.0, 1.0)),
            const SizedBox(height: 12),
            Text('Mata Kanan Terbuka: ${(rightProb * 100).toStringAsFixed(0)}%'),
            const SizedBox(height: 4),
            LinearProgressIndicator(value: rightProb.clamp(0.0, 1.0)),
            const Spacer(),
            FilledButton.icon(
              onPressed: _isProcessing ? null : _handleBlinkAndFaceCheck,
              icon: const Icon(Icons.visibility_outlined),
              label: const Text('Deteksi Wajah & Probabilitas Mata'),
            ),
          ],
        ),
      ),
    );
  }
}
