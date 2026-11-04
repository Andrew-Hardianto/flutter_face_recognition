import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_face_recognition/flutter_face_recognition.dart';

/// Displays the selected/captured image along with face detection bounding box
/// and eye open probability indicators.
class CameraFaceOverlayView extends StatelessWidget {
  final Uint8List? imageBytes;
  final FaceDetectionResult? detectionResult;
  final VoidCallback onPickCamera;
  final VoidCallback onPickGallery;
  final bool isProcessing;

  const CameraFaceOverlayView({
    super.key,
    required this.imageBytes,
    required this.detectionResult,
    required this.onPickCamera,
    required this.onPickGallery,
    this.isProcessing = false,
  });

  @override
  Widget build(BuildContext context) {
    final face = detectionResult?.firstFace;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          height: 260,
          decoration: BoxDecoration(
            color: Colors.grey.shade900,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.grey.shade800),
          ),
          clipBehavior: Clip.antiAlias,
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (imageBytes != null)
                Image.memory(imageBytes!, fit: BoxFit.contain)
              else
                const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.face_retouching_natural,
                        size: 64,
                        color: Colors.white54,
                      ),
                      SizedBox(height: 12),
                      Text(
                        'Ambil foto kamera atau pilih dari galeri',
                        style: TextStyle(color: Colors.white70),
                      ),
                    ],
                  ),
                ),
              if (face != null)
                Positioned(
                  left: 12,
                  bottom: 12,
                  right: 12,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      'Wajah: (${face.x}, ${face.y}) ${face.width}x${face.height}px | '
                      'Yaw: ${face.headEulerAngleY.toStringAsFixed(1)}° | '
                      'Roll: ${face.headEulerAngleZ.toStringAsFixed(1)}°',
                      style: const TextStyle(
                        color: Colors.greenAccent,
                        fontSize: 12,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                ),
              if (isProcessing)
                Container(
                  color: Colors.black54,
                  child: const Center(child: CircularProgressIndicator()),
                ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: isProcessing ? null : onPickCamera,
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Kamera'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: isProcessing ? null : onPickGallery,
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Galeri'),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
