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
      title: 'Flutter Face Recognition Demo',
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
  LivenessResult? _livenessResult;
  MatchResult? _matchResult;

  List<double>? _enrolledEmbedding;
  double _threshold = 0.75;
  bool _isProcessing = false;
  String _statusMessage = 'Siap. Silakan daftarkan wajah terlebih dahulu.';

  @override
  void dispose() {
    _faceRecognition.dispose();
    super.dispose();
  }

  Future<Uint8List?> _pickImage(ImageSource source) async {
    try {
      final file = await _picker.pickImage(
        source: source,
        maxWidth: 1280,
        maxHeight: 1280,
        imageQuality: 90,
        preferredCameraDevice: CameraDevice.front,
      );
      if (file == null) return null;
      final bytes = await file.readAsBytes();
      setState(() {
        _currentImageBytes = bytes;
        _detectionResult = null;
        _livenessResult = null;
        _matchResult = null;
      });
      return bytes;
    } catch (e) {
      setState(() {
        _statusMessage = 'Gagal mengambil gambar: $e';
      });
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
      if (detection.count == 0) {
        setState(() {
          _detectionResult = detection;
          _statusMessage = 'Tidak ada wajah yang terdeteksi pada foto.';
        });
        return;
      }
      if (detection.count > 1) {
        setState(() {
          _detectionResult = detection;
          _statusMessage =
              'Terdeteksi ${detection.count} wajah. Gunakan foto dengan 1 wajah saja.';
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

  Future<void> _handleVerifyWithLiveness() async {
    final bytes = _currentImageBytes;
    if (bytes == null) {
      setState(() => _statusMessage = 'Pilih atau ambil foto terlebih dahulu.');
      return;
    }
    if (_enrolledEmbedding == null) {
      setState(() {
        _statusMessage =
            'Belum ada wajah terdaftar. Lakukan Enrollment terlebih dahulu.';
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Memeriksa Multi-Scale Passive Liveness (1.0x & 2.7x)...';
    });

    try {
      final detection = await _faceRecognition.detectFaces(bytes: bytes);
      final liveness = await _faceRecognition.checkLiveness(
        bytes: bytes,
        threshold: _threshold,
      );

      if (!liveness.isLive) {
        setState(() {
          _detectionResult = detection;
          _livenessResult = liveness;
          _matchResult = null;
          _statusMessage =
              'SPOOF TERDETEKSI! Skor Liveness: ${(liveness.score * 100).toStringAsFixed(1)}% '
              '(${liveness.confidence.label})';
        });
        return;
      }

      final candidateEmbedding =
          await _faceRecognition.extractFaceFeature(bytes: bytes);
      final match = _faceRecognition.compareFaces(
        _enrolledEmbedding!,
        candidateEmbedding,
        threshold: _threshold,
      );

      setState(() {
        _detectionResult = detection;
        _livenessResult = liveness;
        _matchResult = match;
        _statusMessage = match.isMatch
            ? 'VERIFIKASI BERHASIL! Kemiripan: ${match.similarityPercentage.toStringAsFixed(1)}%'
            : 'WAJAH TIDAK COCOK! Kemiripan: ${match.similarityPercentage.toStringAsFixed(1)}%';
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

  @override
  Widget build(BuildContext context) {
    final face = _detectionResult?.firstFace;

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Flutter Face Recognition'),
          bottom: const TabBar(
            tabs: [
              Tab(icon: Icon(Icons.person_add_alt_1), text: '1. Enroll'),
              Tab(icon: Icon(Icons.verified_user), text: '2. Verify & Live'),
              Tab(icon: Icon(Icons.remove_red_eye), text: '3. Blink Check'),
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
                  height: 310,
                  child: TabBarView(
                    children: [
                      _buildEnrollTab(),
                      _buildVerifyTab(),
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
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Threshold Verifikasi & Liveness',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
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
              max: 0.90,
              divisions: 30,
              label: _threshold.toStringAsFixed(2),
              onChanged: (val) => setState(() => _threshold = val),
            ),
            if (_livenessResult != null)
              Text(
                'Liveness: ${_livenessResult!.isLive ? "LIVE" : "SPOOF"} '
                '(${(_livenessResult!.score * 100).toStringAsFixed(1)}% - ${_livenessResult!.confidence.label})',
                style: TextStyle(
                  color: _livenessResult!.isLive
                      ? Colors.greenAccent
                      : Colors.redAccent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            if (_matchResult != null) ...[
              const SizedBox(height: 6),
              Text(
                'Cosine Similarity: ${_matchResult!.similarityPercentage.toStringAsFixed(2)}% '
                '(${_matchResult!.isMatch ? "MATCH" : "NO MATCH"})',
                style: TextStyle(
                  color: _matchResult!.isMatch
                      ? Colors.greenAccent
                      : Colors.orangeAccent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
            const Spacer(),
            FilledButton.icon(
              onPressed: _isProcessing ? null : _handleVerifyWithLiveness,
              icon: const Icon(Icons.shield_outlined),
              label: const Text('Cek Liveness & Verifikasi Wajah'),
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

