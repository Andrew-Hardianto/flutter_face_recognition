import AVFoundation
import CoreImage
import Flutter
import Foundation
import TensorFlowLite
import UIKit
import Vision

/// Flutter plugin — port of the tested `liveness-v3` native pipeline
/// (capacitor-face-recognition/ios) ke Flutter method channel.
///
/// Wire contract: docs/native_contract.md (channel `flutter_face_recognition`).
/// Pipeline 4 fase: Quality Gate → Dual Anti-Spoof (2.7 + 4.0) → Embedding →
/// Temporal Challenge (R1–R5). Semua konstanta & perilaku mengikuti referensi.
public class FlutterFaceRecognitionPlugin: NSObject, FlutterPlugin, LivenessCaptureViewControllerDelegate {

    // ============================
    // Configuration — Face Recognition (MobileFaceNet)
    // ============================
    private let packageName = "flutter_face_recognition"
    private let modelName = "mobile_face_net"
    private let inputSize = 112
    private let embeddingSize = 192
    /// Threshold cosine similarity untuk face match.
    private let matchThreshold: Double = 0.75

    // ============================
    // Configuration — Anti-Spoofing (Silent-Face / minivision-ai)
    // ============================
    // anti_spoof_2_7.tflite = spoof_model_scale_2_7.tflite (crop 2.7x)
    // anti_spoof_4_0.tflite = spoof_model_scale_4_0.tflite (crop 4.0x)
    private let antiSpoofModelName27 = "anti_spoof_2_7.tflite"
    private let antiSpoofModelName40 = "anti_spoof_4_0.tflite"
    // Fallback legacy untuk bundle lama yang masih memakai anti_spoof.tflite
    private let antiSpoofModelNameLegacy = "anti_spoof.tflite"
    // MiniFASNet input 80x80, output 3 kelas [print, live, replay]
    private let antiSpoofInputSize = 80
    private let antiSpoofOutputSize = 3
    // Crop scale per model (referensi Silent-Face; TIDAK ada scale 1.0)
    private let cropScale27: CGFloat = 2.7
    private let cropScale40: CGFloat = 4.0

    // ============================
    // Threshold — SATU sumber kebenaran (liveness-v3)
    // ============================
    // Keputusan live:
    //   p = softmax(out27) + softmax(out40)        (rentang 0..2)
    //   label = argmax(p); liveScore = p[1]/2; spoofScore = max(p[0],p[2])/2
    //   isLive = label==LIVE && liveScore >= liveThreshold && spoofScore < spoofVeto
    private let thresholdVersion = "liveness-v3"
    private let matchThresholdVersion = "match-v1"

    // Konsistensi identitas antar frame (verifyFaceSequence)
    private let identityMaxPose: Float = 12
    // Pelaporan field frontalFrames
    private let challengeFrontMax: Float = 15
    // Challenge session: single-use + expiry
    private let challengeTTL: TimeInterval = 60

    // ============================
    // Pertahanan berlapis (Fase 4)
    // ============================
    private let maxSessionAttempts = 10
    private let sessionAttemptWindow: TimeInterval = 5 * 60
    // Veto per-frame: satu frame spoof → sesi gagal, tanpa toleransi voting.
    private let spoofHardVeto: Float = 0.5
    // Rentang durasi sesi yang wajar untuk gerakan manusia (12 frame @ ~120ms).
    private let sessionDurationMinMs: Double = 1000
    private let sessionDurationMaxMs: Double = 5000

    /// Output debug hanya diizinkan pada build DEBUG.
    private var debugAllowed: Bool {
        return DeviceIntegrity.isDebuggable
    }

    // ============================
    // State
    // ============================
    private let inferenceQueue = DispatchQueue(
        label: "com.andrew.face.inferenceQueue",
        qos: .userInitiated
    )

    private var registrar: FlutterPluginRegistrar?

    // Interpreter TIDAK thread-safe → semua akses via inferenceQueue + lock per model
    private var faceInterpreter: Interpreter?
    private var antiSpoof27: Interpreter?
    private var antiSpoof40: Interpreter?
    private let antiSpoof27Lock = NSLock()
    private let antiSpoof40Lock = NSLock()
    private let embeddingLock = NSLock()

    // Challenge session (single-use + expiry) — dijaga NSLock (thread-safety)
    private struct ChallengeSession {
        let id: String
        let challenge: String
        let expiresAt: Date
        var consumed: Bool
    }

    private enum ChallengeLookup {
        case ok(ChallengeSession)
        case missing
        case expired
        case consumed
    }

    private var activeChallenge: ChallengeSession?
    private let challengeLock = NSLock()

    // Rate limiter — dijaga NSLock (thread-safety)
    private var sessionAttempts: [Date] = []
    private let rateLock = NSLock()

    // Flag enableInsecureFramePath dari initialize() — dijaga NSLock
    private var enableInsecureFramePath = false
    private let stateLock = NSLock()

    // Sesi capture native yang sedang berjalan (modal, satu per waktu)
    private struct PendingCapture {
        let result: FlutterResult
        let challenge: String
        let config: GateConfig
        let referenceEmbedding: [Double]?
    }

    private var pendingCapture: PendingCapture?
    private var activeCaptureVC: LivenessCaptureViewController?

    // ============================
    // Registration & dispatch
    // ============================
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "flutter_face_recognition",
            binaryMessenger: registrar.messenger()
        )
        let instance = FlutterFaceRecognitionPlugin(registrar: registrar)
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    init(registrar: FlutterPluginRegistrar) {
        self.registrar = registrar
        super.init()
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize":
            handleInitialize(call: call, result: result)
        case "extractFaceFeature", "checkLiveness", "detectFaces", "verifyFace", "verifyFaceSequence":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                switch call.method {
                case "extractFaceFeature":
                    self.handleExtractFaceFeature(call: call, result: result)
                case "checkLiveness":
                    self.handleCheckLiveness(call: call, result: result)
                case "detectFaces":
                    self.handleDetectFaces(call: call, result: result)
                case "verifyFace":
                    self.handleVerifyFace(call: call, result: result)
                default:
                    self.handleVerifyFaceSequence(call: call, result: result)
                }
            }
        case "createChallengeSession":
            handleCreateChallengeSession(call: call, result: result)
        case "captureLivenessSession":
            inferenceQueue.async { [weak self] in
                self?.handleCaptureLivenessSession(call: call, result: result)
            }
        case "getDeviceIntegrity":
            replySuccess(result, data: ["deviceIntegrity": DeviceIntegrity.snapshot()])
        case "dispose":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                self.faceInterpreter = nil
                self.antiSpoof27 = nil
                self.antiSpoof40 = nil
                // Vision request bersifat sinkron per-call — tidak ada yang persisten untuk dibatalkan.
                self.replySuccess(result, data: nil)
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // ============================
    // initialize
    // ============================
    /// Menyimpan flag `enableInsecureFramePath` — gerbang release verifyFaceSequence.
    private func handleInitialize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as? [String: Any]
        let enabled = (args?["enableInsecureFramePath"] as? NSNumber)?.boolValue ?? false
        stateLock.lock()
        enableInsecureFramePath = enabled
        stateLock.unlock()
        replySuccess(result, data: nil)
    }

    /// P2.4: jalur frame dari Dart (verifyFaceSequence) adalah celah injeksi
    /// frame — hanya untuk build DEBUG, ATAU saat app secara eksplisit
    /// meng-enable flag `enableInsecureFramePath` di initialize().
    private func isInsecureFramePathAllowed() -> Bool {
        if DeviceIntegrity.isDebuggable { return true }
        stateLock.lock()
        defer { stateLock.unlock() }
        return enableInsecureFramePath
    }

    // ============================
    // extractFaceFeature
    // ============================
    private func handleExtractFaceFeature(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard getOrCreateFaceInterpreter() != nil else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model TFLite '\(modelName).tflite' gagal dimuat dari Flutter assets."
            )
            return
        }

        let image: UIImage
        switch decodeImage(from: call) {
        case .success(let decoded):
            image = decoded
        case .failure(let error):
            replyError(result, error: error)
            return
        }
        guard let cgImage = image.cgImage else {
            replyError(result, code: "IMAGE_DECODE_FAILED", message: "Gagal men-decode data gambar menjadi UIImage.")
            return
        }

        // Deteksi wajah (Rev3) — pakai wajah TERBESAR, embedding dari frame & bbox yang sama
        let request = makeFaceRectanglesRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            guard let results = req.results as? [VNFaceObservation],
                  let face = self.pickLargestFace(results)
            else {
                self.replyError(
                    result,
                    code: "NO_FACE_DETECTED",
                    message: "Tidak ada wajah yang terdeteksi"
                )
                return
            }

            let faceRect = self.pixelRect(
                from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            guard let embeddings = self.extractEmbedding(cgImage: cgImage, faceRect: faceRect) else {
                self.replyError(
                    result,
                    code: "EMPTY_EMBEDDING",
                    message: "Gagal menjalankan inference face recognition"
                )
                return
            }

            // Kontrak: embedding 192-dim ternormalisasi L2
            self.replySuccess(result, data: ["embedding": self.l2Normalize(embeddings)])
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_ERROR",
                message: "Gagal menjalankan Apple Vision Request: \(error.localizedDescription)"
            )
        }
    }

    // ============================
    // checkLiveness — wrapper backward-compatible
    // ============================
    /// Ensemble dual-model (2.7 + 4.0), argmax + veto. `threshold` (opsional)
    /// dipetakan ke `liveThreshold`. Hasil FLAT sesuai kontrak.
    private func handleCheckLiveness(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard getOrCreateAntiSpoof27() != nil, getOrCreateAntiSpoof40() != nil else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model anti-spoofing '\(antiSpoofModelName27)' / '\(antiSpoofModelName40)' tidak ditemukan di Flutter assets."
            )
            return
        }

        let image: UIImage
        switch decodeImage(from: call) {
        case .success(let decoded):
            image = decoded
        case .failure(let error):
            replyError(result, error: error)
            return
        }
        guard let cgImage = image.cgImage else {
            replyError(result, code: "IMAGE_DECODE_FAILED", message: "Gagal men-decode data gambar menjadi UIImage.")
            return
        }

        let args = call.arguments as? [String: Any]
        var config = GateConfig.from(nil, debugAllowed: debugAllowed)
        if let threshold = (args?["threshold"] as? NSNumber)?.floatValue {
            config.liveThreshold = threshold
        }

        let request = makeFaceRectanglesRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            guard let results = req.results as? [VNFaceObservation],
                  let face = self.pickLargestFace(results)
            else {
                self.replyError(
                    result,
                    code: "NO_FACE_DETECTED",
                    message: "Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness"
                )
                return
            }

            let baseFaceRect = self.pixelRect(
                from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            guard let liveness = self.runAntiSpoofPipeline(
                cgImage: cgImage,
                baseFaceRect: baseFaceRect,
                imgWidth: image.size.width,
                imgHeight: image.size.height,
                config: config
            ) else {
                self.replyError(
                    result,
                    code: "TFLITE_INFERENCE_ERROR",
                    message: "Gagal menjalankan anti-spoof inference"
                )
                return
            }

            // Tingkat kepercayaan (kompatibel response lama) dari liveScore
            let confidence: String
            if liveness.liveScore > 0.85 || liveness.liveScore < 0.15 {
                confidence = "HIGH"
            } else if liveness.liveScore > 0.6 || liveness.liveScore < 0.4 {
                confidence = "MEDIUM"
            } else {
                confidence = "LOW"
            }

            // Hasil FLAT sesuai kontrak: score == liveScore (alias backward-compat)
            self.replySuccess(result, data: [
                "isLive": liveness.isLive,
                "score": Double(liveness.liveScore),
                "confidence": confidence,
                "label": liveness.label,
                "liveScore": Double(liveness.liveScore),
                "printScore": Double(liveness.printScore),
                "replayScore": Double(liveness.replayScore),
                "spoofScore": Double(liveness.spoofScore),
                "screenSuspectScore": Double(liveness.screenSuspectScore),
                "thresholdVersion": self.thresholdVersion,
            ])
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_ERROR",
                message: "Gagal menjalankan Apple Vision Request untuk liveness: \(error.localizedDescription)"
            )
        }
    }

    // ============================
    // detectFaces — jalur FAST (landmarks)
    // ============================
    /// Deteksi semua wajah + bounding box + euler angles (derajat) + eye open
    /// probability. `headEulerAngleY` ternormalisasi (YawConvention): positif =
    /// user menoleh ke kanan anatomisnya.
    private func handleDetectFaces(call: FlutterMethodCall, result: @escaping FlutterResult) {
        let image: UIImage
        switch decodeImage(from: call) {
        case .success(let decoded):
            image = decoded
        case .failure(let error):
            replyError(result, error: error)
            return
        }
        guard let cgImage = image.cgImage else {
            replyError(result, code: "IMAGE_DECODE_FAILED", message: "Gagal men-decode data gambar menjadi UIImage.")
            return
        }

        // VNDetectFaceLandmarksRequest mencakup deteksi wajah + landmark (mata, dll)
        let request = VNDetectFaceLandmarksRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            let results = req.results as? [VNFaceObservation] ?? []
            let imgWidth = image.size.width
            let imgHeight = image.size.height

            var facesArray: [[String: Any]] = []
            for face in results {
                let bb = face.boundingBox

                // Konversi bounding box Vision (kiri-bawah, normalized) → piksel (kiri-atas)
                let x = bb.origin.x * imgWidth
                let y = (1 - bb.origin.y - bb.size.height) * imgHeight
                let width = bb.size.width * imgWidth
                let height = bb.size.height * imgHeight

                // Konvensi sama dengan ML Kit Android: X = pitch, Y = yaw, Z = roll.
                // Sudut dikonversi radian → derajat; yaw ternormalisasi (YawConvention).
                let angles = self.headAnglesDeg(face)

                // Estimasi eye open probability dari Eye Aspect Ratio (EAR) 6 titik
                var leftEyeOpenProb: Double = -1.0
                var rightEyeOpenProb: Double = -1.0
                if let landmarks = face.landmarks {
                    if let leftEye = landmarks.leftEye {
                        leftEyeOpenProb = self.eyeOpenProbability(from: leftEye.normalizedPoints)
                    }
                    if let rightEye = landmarks.rightEye {
                        rightEyeOpenProb = self.eyeOpenProbability(from: rightEye.normalizedPoints)
                    }
                }

                var faceDict: [String: Any] = [
                    "x": Int(max(0, x)),
                    "y": Int(max(0, y)),
                    "width": Int(min(width, imgWidth - max(0, x))),
                    "height": Int(min(height, imgHeight - max(0, y))),
                    "headEulerAngleX": Double(angles.pitch),
                    "headEulerAngleY": Double(angles.yaw),
                    "headEulerAngleZ": Double(angles.roll),
                ]
                // Nullable — hanya set jika berhasil dihitung
                if leftEyeOpenProb >= 0 { faceDict["leftEyeOpenProbability"] = leftEyeOpenProb }
                if rightEyeOpenProb >= 0 { faceDict["rightEyeOpenProbability"] = rightEyeOpenProb }

                facesArray.append(faceDict)
            }

            self.replySuccess(result, data: [
                "count": results.count,
                "faces": facesArray,
            ])
        }
        // Pin revision 3 (iOS 15+) agar yaw/pitch/roll kontinu, bukan bin ±45°
        request.revision = VNDetectFaceLandmarksRequestRevision3

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_ERROR",
                message: "Gagal menjalankan Apple Vision Request: \(error.localizedDescription)"
            )
        }
    }

    // ============================
    // verifyFace — API terpadu (anti-TOCTOU)
    // ============================
    /// Quality gate + anti-spoof + embedding dari FRAME DAN BBOX YANG SAMA.
    /// Embedding hanya dikembalikan jika isLive = true.
    private func handleVerifyFace(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard getOrCreateFaceInterpreter() != nil,
              getOrCreateAntiSpoof27() != nil,
              getOrCreateAntiSpoof40() != nil
        else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model liveness atau face recognition tidak ditemukan di Flutter assets"
            )
            return
        }

        let image: UIImage
        switch decodeImage(from: call) {
        case .success(let decoded):
            image = decoded
        case .failure(let error):
            replyError(result, error: error)
            return
        }
        guard let cgImage = image.cgImage else {
            replyError(result, code: "IMAGE_DECODE_FAILED", message: "Gagal men-decode data gambar menjadi UIImage.")
            return
        }

        let args = call.arguments as? [String: Any]
        let config = GateConfig.from(args?["config"] as? [String: Any], debugAllowed: debugAllowed)
        let referenceEmbedding = (args?["referenceEmbedding"] as? [NSNumber])?.map { $0.doubleValue }

        let request = makeFaceRectanglesRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            let results = req.results as? [VNFaceObservation] ?? []

            // ---- Quality gate ----
            let quality = self.runQualityGate(
                cgImage: cgImage, faces: results,
                imgWidth: image.size.width, imgHeight: image.size.height, config: config)
            if !quality.passed {
                // Kontrak: quality gagal → isLive=false, liveness=null, tanpa embedding
                self.replySuccess(result, data: [
                    "isLive": false,
                    "thresholdVersion": self.thresholdVersion,
                    "quality": quality.toJson(),
                    "liveness": NSNull(),
                ])
                return
            }

            guard let face = self.pickLargestFace(results) else {
                self.replyError(
                    result,
                    code: "NO_FACE_DETECTED",
                    message: "Tidak ada wajah yang terdeteksi"
                )
                return
            }
            let baseFaceRect = self.pixelRect(
                from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            // ---- Anti-spoof (dual model) ----
            guard let liveness = self.runAntiSpoofPipeline(
                cgImage: cgImage,
                baseFaceRect: baseFaceRect,
                imgWidth: image.size.width,
                imgHeight: image.size.height,
                config: config
            ) else {
                self.replyError(
                    result,
                    code: "TFLITE_INFERENCE_ERROR",
                    message: "Gagal menjalankan anti-spoof inference"
                )
                return
            }

            var ret: [String: Any] = [
                "isLive": liveness.isLive,
                "thresholdVersion": self.thresholdVersion,
                "quality": quality.toJson(),
                "liveness": liveness.toJson(),
            ]

            if config.debug {
                ret["debug"] = liveness.debugToJson()
            }

            // ---- Embedding HANYA jika live, dari frame & bbox yang sama ----
            if liveness.isLive {
                guard let embeddings = self.extractEmbedding(cgImage: cgImage, faceRect: baseFaceRect) else {
                    self.replyError(
                        result,
                        code: "EMPTY_EMBEDDING",
                        message: "Gagal mengekstrak embedding dari frame yang sama"
                    )
                    return
                }
                ret["embedding"] = self.l2Normalize(embeddings)

                if let ref = referenceEmbedding {
                    guard let score = self.cosineSimilarity(ref, embeddings.map { Double($0) }) else {
                        self.replyError(
                            result,
                            code: "EMPTY_EMBEDDING",
                            message: "referenceEmbedding tidak valid (norma nol)"
                        )
                        return
                    }
                    ret["match"] = [
                        "isMatch": score > self.matchThreshold,
                        "score": score,
                        "threshold": self.matchThreshold,
                        "thresholdVersion": self.matchThresholdVersion,
                    ]
                }
            }

            self.replySuccess(result, data: ret)
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_ERROR",
                message: "Gagal menjalankan Apple Vision Request untuk verifyFace: \(error.localizedDescription)"
            )
        }
    }

    // ============================
    // Challenge session (liveness-v3)
    // ============================
    /// Membuat challenge session (acak, single-use, expiry 60 detik).
    /// Challenge TIDAK bisa dipilih dari Dart.
    private func handleCreateChallengeSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard checkRateLimit() else {
            replyError(
                result,
                code: "RATE_LIMITED",
                message: "Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit."
            )
            return
        }
        let session = issueChallengeSession()
        replySuccess(result, data: [
            "challengeId": session.id,
            "challenge": session.challenge,
            "expiresAt": session.expiresAt.timeIntervalSince1970 * 1000,
        ])
    }

    /// Membuat challenge session baru (acak) — menggantikan yang lama.
    private func issueChallengeSession() -> ChallengeSession {
        let session = ChallengeSession(
            id: UUID().uuidString,
            challenge: Int.random(in: 0...1) == 0 ? "LOOK_LEFT" : "LOOK_RIGHT",
            expiresAt: Date().addingTimeInterval(challengeTTL),
            consumed: false)
        challengeLock.lock()
        activeChallenge = session
        challengeLock.unlock()
        return session
    }

    /// Mengambil SEKALIGUS mengonsumsi challenge session untuk `id`.
    /// Percobaan pertama (gagal maupun lulus) memakai sesi — brute-force retry
    /// dengan challenge yang sama tidak mungkin.
    private func consumeChallengeSession(id: String?) -> ChallengeLookup {
        guard let id = id, !id.isEmpty else { return .missing }
        challengeLock.lock()
        defer { challengeLock.unlock() }
        guard var session = activeChallenge, session.id == id else { return .missing }
        guard Date() < session.expiresAt else { return .expired }
        guard !session.consumed else { return .consumed }
        session.consumed = true
        activeChallenge = session
        return .ok(session)
    }

    /// Rate limit: maksimal 10 percobaan per jendela 5 menit (anti brute-force).
    private func checkRateLimit() -> Bool {
        let now = Date()
        rateLock.lock()
        defer { rateLock.unlock() }
        sessionAttempts.removeAll { now.timeIntervalSince($0) > sessionAttemptWindow }
        if sessionAttempts.count >= maxSessionAttempts { return false }
        sessionAttempts.append(now)
        return true
    }

    // ============================
    // verifyFaceSequence
    // ============================
    /// Verifikasi sesi multi-frame dengan active challenge.
    /// Menerima `frames: [Uint8List]` (atau `images: [base64]` untuk back-compat).
    private func handleVerifyFaceSequence(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard getOrCreateFaceInterpreter() != nil,
              getOrCreateAntiSpoof27() != nil,
              getOrCreateAntiSpoof40() != nil
        else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model liveness atau face recognition tidak ditemukan di Flutter assets"
            )
            return
        }
        // P2.4: jalur frame dari Dart = celah injeksi frame — hanya untuk build
        // debug ATAU saat app meng-enable flag enableInsecureFramePath (initialize).
        guard isInsecureFramePathAllowed() else {
            replyError(
                result,
                code: "INSECURE_FRAME_PATH_DISABLED",
                message: "verifyFaceSequence (frame dari Dart) hanya untuk build debug atau app dengan flag enableInsecureFramePath — gunakan captureLivenessSession di produksi"
            )
            return
        }

        let args = call.arguments as? [String: Any]

        // frames: [Uint8List] (preferred) ATAU images: [base64] (back-compat)
        var frameData: [Data] = []
        if let typedFrames = args?["frames"] as? [FlutterStandardTypedData], !typedFrames.isEmpty {
            frameData = typedFrames.map { $0.data }
        } else if let images = args?["images"] as? [String], !images.isEmpty {
            for base64 in images {
                let clean: String
                if let commaRange = base64.range(of: ",") {
                    clean = String(base64[commaRange.upperBound...])
                } else {
                    clean = base64
                }
                guard let data = Data(base64Encoded: clean, options: .ignoreUnknownCharacters),
                      !data.isEmpty
                else {
                    replyError(
                        result,
                        code: "IMAGE_DECODE_FAILED",
                        message: "Gagal decode base64 frame menjadi data gambar"
                    )
                    return
                }
                frameData.append(data)
            }
        } else {
            replyError(
                result,
                code: "INVALID_IMAGE_INPUT",
                message: "frames (List<Uint8List>) minimal berisi 2 frame (disarankan ≥ 5)"
            )
            return
        }
        guard frameData.count >= 2 else {
            replyError(
                result,
                code: "INVALID_IMAGE_INPUT",
                message: "frames minimal berisi 2 frame (disarankan ≥ 5)"
            )
            return
        }

        // Challenge dibaca dari session native (acak, single-use, expiry) —
        // BUKAN dari input Dart. Percobaan pertama langsung mengonsumsi sesi.
        let challengeId = args?["challengeId"] as? String ?? ""
        let challenge: String
        switch consumeChallengeSession(id: challengeId) {
        case .ok(let session):
            challenge = session.challenge
        case .missing:
            replyError(
                result,
                code: "CHALLENGE_NOT_FOUND",
                message: "challengeId tidak dikenal — panggil createChallengeSession dulu"
            )
            return
        case .expired:
            replyError(
                result,
                code: "CHALLENGE_EXPIRED",
                message: "Challenge session kedaluwarsa — buat challenge baru"
            )
            return
        case .consumed:
            replyError(
                result,
                code: "CHALLENGE_CONSUMED",
                message: "Challenge session sudah dipakai — buat challenge baru"
            )
            return
        }

        let config = GateConfig.from(args?["config"] as? [String: Any], debugAllowed: debugAllowed)
        let referenceEmbedding = (args?["referenceEmbedding"] as? [NSNumber])?.map { $0.doubleValue }

        // Frame diproses SEKUENSIAL (VNImageRequestHandler sinkron)
        guard let frames = processFrames(frameData: frameData, config: config) else {
            replyError(
                result,
                code: "IMAGE_DECODE_FAILED",
                message: "Gagal memproses frame (decode / Vision)"
            )
            return
        }

        var ret = finishSequence(frames: frames, config: config,
                                 challenge: challenge, referenceEmbedding: referenceEmbedding)
        ret["challengeId"] = challengeId
        ret["native"] = false
        if config.debug {
            ret["debug"] = ["yawTrace": yawTraceJson(frames: frames, config: config)]
        }
        replySuccess(result, data: ret)
    }

    private func yawTraceJson(frames: [FrameResult], config: GateConfig) -> [[String: Any]] {
        return frames.map { fr -> [String: Any] in
            var entry: [String: Any] = [
                "index": fr.index,
                "yaw": Double(YawConvention.applyOverride(fr.yaw, overrideSign: config.yawSignOverride)),
            ]
            if let ts = fr.timestampMs { entry["timestampMs"] = ts }
            return entry
        }
    }

    // ============================
    // Native capture (Fase 4) — captureLivenessSession
    // ============================
    /// Sesi liveness native (anti injeksi frame): kamera dibuka oleh view
    /// controller plugin, frame disimpan ke cache privat TANPA melewati Dart,
    /// lalu pipeline verifikasi dijalankan. Yang kembali ke Dart hanya HASIL.
    private func handleCaptureLivenessSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard getOrCreateFaceInterpreter() != nil,
              getOrCreateAntiSpoof27() != nil,
              getOrCreateAntiSpoof40() != nil
        else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model liveness atau face recognition tidak ditemukan di Flutter assets"
            )
            return
        }
        guard checkRateLimit() else {
            replyError(
                result,
                code: "RATE_LIMITED",
                message: "Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit."
            )
            return
        }

        // Challenge SELALU acak dari native — menutup brute-force
        // "ulangi sesi sampai challenge cocok dengan video yang disiapkan".
        let challenge = Int.random(in: 0...1) == 0 ? "LOOK_LEFT" : "LOOK_RIGHT"

        let args = call.arguments as? [String: Any]
        let config = GateConfig.from(args?["config"] as? [String: Any], debugAllowed: debugAllowed)
        let referenceEmbedding = (args?["referenceEmbedding"] as? [NSNumber])?.map { $0.doubleValue }

        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self = self else { return }
                guard granted else {
                    self.replyError(
                        result,
                        code: "CAMERA_PERMISSION_DENIED",
                        message: "Izin kamera ditolak"
                    )
                    return
                }
                guard self.activeCaptureVC == nil else {
                    self.replyError(
                        result,
                        code: "CAPTURE_FAILED",
                        message: "Sesi capture lain sedang berjalan"
                    )
                    return
                }
                guard let presenter = self.topPresenter() else {
                    self.replyError(
                        result,
                        code: "CAPTURE_FAILED",
                        message: "ViewController tidak tersedia untuk membuka kamera"
                    )
                    return
                }

                let captureVC = LivenessCaptureViewController(
                    challenge: challenge, frameCount: 12, interval: 0.12, warmup: 1.2
                )
                captureVC.delegate = self
                captureVC.modalPresentationStyle = .fullScreen

                self.stateLock.lock()
                self.pendingCapture = PendingCapture(
                    result: result,
                    challenge: challenge,
                    config: config,
                    referenceEmbedding: referenceEmbedding)
                self.activeCaptureVC = captureVC
                self.stateLock.unlock()

                presenter.present(captureVC, animated: true)
            }
        }
    }

    /// Verifikasi frame hasil native capture + jejak audit terenkripsi.
    private func processCapturedSession(
        result: @escaping FlutterResult,
        framePaths: [String],
        timestamps: [Double],
        challenge: String,
        config: GateConfig,
        referenceEmbedding: [Double]?
    ) {
        var frameBytes: [Data] = []
        var frameTimestamps: [Double] = []
        for (index, path) in framePaths.enumerated() {
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                deleteFrameFiles(framePaths)
                replyError(
                    result,
                    code: "IMAGE_DECODE_FAILED",
                    message: "Gagal membaca frame capture"
                )
                return
            }
            frameBytes.append(data)
            if timestamps.indices.contains(index) {
                frameTimestamps.append(timestamps[index])
            }
        }
        // Frame sementara dihapus — salinan terenkripsi disimpan oleh AuditStore
        deleteFrameFiles(framePaths)

        guard let frames = processFrames(
            frameData: frameBytes, config: config, timestamps: frameTimestamps) else {
            replyError(
                result,
                code: "IMAGE_DECODE_FAILED",
                message: "Gagal memproses frame capture (decode / Vision)"
            )
            return
        }

        var ret = finishSequence(frames: frames, config: config,
                                 challenge: challenge, referenceEmbedding: referenceEmbedding)
        ret["challengeId"] = NSNull()
        ret["native"] = true
        ret["deviceIntegrity"] = DeviceIntegrity.snapshot()

        if let json = try? JSONSerialization.data(withJSONObject: ret, options: [.sortedKeys]) {
            do {
                ret["auditId"] = try AuditStore.storeSession(json: json, frames: frameBytes)
            } catch {
                print("[FlutterFaceRecognition] Audit store gagal: \(error)")
                ret["auditId"] = NSNull()
            }
        } else {
            ret["auditId"] = NSNull()
        }

        replySuccess(result, data: ret)
    }

    private func deleteFrameFiles(_ paths: [String]) {
        for path in paths {
            try? FileManager.default.removeItem(atPath: path)
        }
    }

    /// ViewController teratas untuk presentasi modal (key window → presented chain).
    private func topPresenter() -> UIViewController? {
        let windows = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap { $0.windows }
        let root = (windows.first(where: { $0.isKeyWindow }) ?? windows.first)?.rootViewController
        var top = root
        while let presented = top?.presentedViewController {
            top = presented
        }
        return top
    }

    // ============================
    // LivenessCaptureViewControllerDelegate
    // ============================
    func livenessCaptureViewController(
        _ controller: LivenessCaptureViewController,
        didFinishWithChallenge challenge: String,
        framePaths: [String],
        timestamps: [Double]
    ) {
        stateLock.lock()
        let pending = pendingCapture
        pendingCapture = nil
        activeCaptureVC = nil
        stateLock.unlock()

        guard let pending = pending else { return }
        inferenceQueue.async { [weak self] in
            self?.processCapturedSession(
                result: pending.result,
                framePaths: framePaths,
                timestamps: timestamps,
                challenge: challenge,
                config: pending.config,
                referenceEmbedding: pending.referenceEmbedding)
        }
    }

    func livenessCaptureViewController(
        _ controller: LivenessCaptureViewController,
        didFailWithError error: Error
    ) {
        stateLock.lock()
        let pending = pendingCapture
        pendingCapture = nil
        activeCaptureVC = nil
        stateLock.unlock()

        guard let pending = pending else { return }
        replyError(
            pending.result,
            code: "CAPTURE_FAILED",
            message: "Capture gagal: \(error.localizedDescription)"
        )
    }

    // ============================
    // Sequence pipeline (frame sekuensial + agregasi)
    // ============================

    /// Proses frame sekuensial (VNImageRequestHandler bersifat sinkron).
    /// Mengembalikan nil jika ada frame gagal di-decode atau Vision error.
    private func processFrames(frameData: [Data], config: GateConfig,
                               timestamps: [Double]? = nil) -> [FrameResult]? {
        var frames: [FrameResult] = []

        for (index, data) in frameData.enumerated() {
            guard let image = decodeOrientedImage(from: data), let cgImage = image.cgImage else {
                print("[verifyFaceSequence] Frame \(index) gagal di-decode")
                return nil
            }

            var captured: FrameResult?
            let request = makeFaceRectanglesRequest { [weak self] req, error in
                guard let self = self else { return }

                var fr = FrameResult()
                fr.index = index
                if let ts = timestamps, ts.indices.contains(index) {
                    fr.timestampMs = ts[index]
                }

                if let err = error {
                    fr.quality = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
                    print("[verifyFaceSequence] Frame \(index) gagal: \(err.localizedDescription)")
                } else {
                    let results = req.results as? [VNFaceObservation] ?? []
                    if results.isEmpty {
                        fr.quality = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
                    } else {
                        fr.quality = self.runQualityGate(
                            cgImage: cgImage, faces: results,
                            imgWidth: image.size.width, imgHeight: image.size.height, config: config)
                        if fr.quality.passed, let face = self.pickLargestFace(results) {
                            let baseFaceRect = self.pixelRect(
                                from: face, imgWidth: image.size.width, imgHeight: image.size.height)
                            let angles = self.headAnglesDeg(face)
                            fr.yaw = angles.yaw
                            fr.pitch = angles.pitch
                            fr.roll = angles.roll
                            fr.liveness = self.runAntiSpoofPipeline(
                                cgImage: cgImage,
                                baseFaceRect: baseFaceRect,
                                imgWidth: image.size.width,
                                imgHeight: image.size.height,
                                config: config)
                            fr.embedding = self.extractEmbedding(cgImage: cgImage, faceRect: baseFaceRect)
                        }
                    }
                }
                captured = fr
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("[verifyFaceSequence] Vision error frame \(index): \(error)")
                return nil
            }

            guard let fr = captured else { return nil }
            frames.append(fr)
        }

        return frames
    }

    /// Agregasi sesi (median + voting, konsistensi identitas, challenge).
    /// Mengembalikan dictionary hasil — TIDAK menyelesaikan call, agar bisa
    /// dipakai ulang oleh captureLivenessSession.
    private func finishSequence(frames: [FrameResult], config: GateConfig,
                                challenge: String, referenceEmbedding: [Double]?) -> [String: Any] {
        var failureReason: String? = nil

        // SEMUA frame harus lolos quality gate
        for fr in frames {
            if !fr.quality.passed {
                failureReason = "QUALITY_FAILED"
                break
            }
        }

        // Referensi identitas = frame PALING FRONTAL (|yaw| terkecil), bukan frame
        // pertama — embedding paling frontal paling stabil untuk pembanding.
        let refFrame: FrameResult? = failureReason == nil
            ? frames.filter { $0.embedding != nil }.min(by: { abs($0.yaw) < abs($1.yaw) })
            : nil
        let refEmbedding = refFrame?.embedding

        // Konsistensi identitas: hanya bandingkan frame "sebanding" (|yaw| <=
        // identityMaxPose). Frame dengan pose berat tugasnya membuktikan challenge;
        // embedding-nya tidak pose-invariant dan tidak boleh mematikan sesi.
        var identityConsistent = true
        var minIdentityCosine: Double? = nil
        if failureReason == nil, let ref = refEmbedding, let refIndex = refFrame?.index {
            var comparableCosines: [Double] = []
            for fr in frames {
                guard let embedding = fr.embedding, fr.index != refIndex else { continue }
                guard abs(fr.yaw) <= identityMaxPose else { continue }
                if let cos = cosineSimilarity(ref.map { Double($0) }, embedding.map { Double($0) }) {
                    comparableCosines.append(cos)
                    if minIdentityCosine == nil || cos < minIdentityCosine! {
                        minIdentityCosine = cos
                    }
                }
            }
            if !comparableCosines.isEmpty {
                let avg = comparableCosines.reduce(0, +) / Double(comparableCosines.count)
                let minCos = minIdentityCosine ?? 1.0
                if minCos < 0.55 || avg < config.identityThreshold {
                    identityConsistent = false
                }
            }
        }

        // ---- Agregasi (median + voting, BUKAN all-live) ----
        let liveScores: [Float] = frames.map { $0.liveness?.liveScore ?? 0 }
        let liveVotes = frames.filter { $0.liveness?.isLive == true }.count
        // Yaw sudah ternormalisasi di sumber (YawConvention): positif = menoleh
        // ke kanan anatomis user. yawSignOverride hanya untuk perangkat dengan
        // buffer kamera ter-flip — diterapkan SEBELUM agregasi.
        let yawSamples = frames.filter { $0.quality.passed }
            .map {
                ChallengeTrajectory.Sample(
                    yaw: YawConvention.applyOverride($0.yaw, overrideSign: config.yawSignOverride),
                    timestampMs: $0.timestampMs)
            }
        let yawValues = yawSamples.map { $0.yaw }
        let maxYaw = yawValues.max() ?? 0
        let minYaw = yawValues.min() ?? 0
        let frontalFrames = yawValues.filter { abs($0) < challengeFrontMax }.count

        let medianLiveScore = median(liveScores)
        // P2.1: voting sesi diperketat — default SEMUA frame harus isLive.
        let requiredVotes = Int(ceil(Double(frames.count) * Double(config.minLiveVoteRatio)))

        // Challenge (P1): verifikasi LINTASAN TEMPORAL (R1–R5) —
        // lihat ChallengeTrajectory.
        let trajectory = ChallengeTrajectory.evaluate(samples: yawSamples, challenge: challenge)
        let challengeSatisfied = trajectory.passed

        // ---- Anti-replay (P2) ----
        // P2.1: veto per-frame — satu frame dengan label spoof (PRINT/REPLAY)
        // atau spoofScore >= spoofHardVeto langsung menggagalkan sesi,
        // TANPA toleransi voting.
        var spoofFrameDetected = false
        var maxScreenSuspect: Float = 0
        for fr in frames {
            guard let live = fr.liveness else { continue }
            if live.label != "LIVE" || live.spoofScore >= spoofHardVeto {
                spoofFrameDetected = true
            }
            maxScreenSuspect = max(maxScreenSuspect, live.screenSuspectScore)
        }
        // P2.2: veto screen-suspect hanya jika di-set eksplisit (mode ketat opt-in);
        // default hanya dicatat sebagai soft signal.
        let screenVetoTriggered = config.screenSuspectVeto.map { maxScreenSuspect >= $0 } ?? false
        // P2.3: durasi sesi harus masuk rentang gerakan manusia
        // (hanya diperiksa jika timestamp diketahui — jalur native capture).
        let timingOk = trajectory.durationMs <= 0
            || (trajectory.durationMs >= sessionDurationMinMs
                && trajectory.durationMs <= sessionDurationMaxMs)

        let passed = failureReason == nil
            && !spoofFrameDetected
            && !screenVetoTriggered
            && timingOk
            && medianLiveScore >= config.liveThreshold
            && liveVotes >= requiredVotes
            && identityConsistent
            && challengeSatisfied

        var ret = buildSequenceResult(
            passed: passed, reason: nil, challenge: challenge,
            medianLiveScore: medianLiveScore, liveVotes: liveVotes, frameCount: frames.count,
            maxYaw: maxYaw, minYaw: minYaw, frontalFrames: frontalFrames,
            identityConsistent: identityConsistent, minIdentityCosine: minIdentityCosine,
            trajectory: trajectory.toJson(), maxScreenSuspect: maxScreenSuspect)

        if passed, let ref = referenceEmbedding, let best = refEmbedding {
            if let score = cosineSimilarity(ref, best.map { Double($0) }) {
                ret["match"] = [
                    "isMatch": score > matchThreshold,
                    "score": score,
                    "threshold": matchThreshold,
                    "thresholdVersion": matchThresholdVersion,
                ]
            }
        }

        if !passed {
            var reason = failureReason
            if reason == nil {
                // Prioritas alasan gagal: spoof > screen > timing > challenge > votes > identitas
                if spoofFrameDetected {
                    reason = "SPOOF_FRAME_DETECTED"
                } else if screenVetoTriggered {
                    reason = "SCREEN_SUSPECTED"
                } else if !timingOk {
                    reason = "TIMING_SUSPECT"
                } else if !challengeSatisfied {
                    reason = trajectory.failureReason ?? "CHALLENGE_FAILED"
                } else if liveVotes < requiredVotes || medianLiveScore < config.liveThreshold {
                    reason = "NOT_LIVE_ENOUGH"
                } else if !identityConsistent {
                    reason = "IDENTITY_INCONSISTENT"
                }
            }
            ret["reason"] = reason ?? "UNKNOWN"
        }

        return ret
    }

    private func buildSequenceResult(passed: Bool, reason: String?, challenge: String,
                                     medianLiveScore: Float, liveVotes: Int, frameCount: Int,
                                     maxYaw: Float, minYaw: Float, frontalFrames: Int,
                                     identityConsistent: Bool,
                                     minIdentityCosine: Double?,
                                     trajectory: [String: Any],
                                     maxScreenSuspect: Float) -> [String: Any] {
        var ret: [String: Any] = [
            "passed": passed,
            "challenge": challenge,
            "medianLiveScore": Double(medianLiveScore),
            "liveVotes": liveVotes,
            "frameCount": frameCount,
            "maxYaw": Double(maxYaw),
            "minYaw": Double(minYaw),
            "frontalFrames": frontalFrames,
            "identityConsistent": identityConsistent,
            // Diagnostik kalibrasi: cosine minimum antar frame sebanding (null jika < 2)
            "minIdentityCosine": minIdentityCosine.map { $0 as Any } ?? NSNull(),
            // Ringkasan lintasan challenge: R1–R5 + statistik yaw
            "trajectory": trajectory,
            // Sinyal artefak layar agregat (P2.2) — soft signal kecuali screenSuspectVeto di-set
            "screenSuspect": Double(maxScreenSuspect),
            "thresholdVersion": thresholdVersion,
        ]
        if let reason = reason {
            ret["reason"] = reason
        }
        return ret
    }

    // ============================
    // Quality Gate
    // ============================
    private func runQualityGate(cgImage: CGImage, faces: [VNFaceObservation],
                                imgWidth: CGFloat, imgHeight: CGFloat,
                                config: GateConfig) -> QualityResult {
        guard let largest = pickLargestFace(faces) else {
            return QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
        }

        // MULTIPLE_FACES: tolak jika ada > 1 wajah yang cukup besar
        let largestArea = max(0.0001, largest.boundingBox.width * largest.boundingBox.height)
        var significantFaces = 0
        for f in faces {
            let area = f.boundingBox.width * f.boundingBox.height
            if area >= largestArea * 0.3 { significantFaces += 1 }
        }
        if significantFaces > 1 {
            return QualityResult.fail(reason: "MULTIPLE_FACES", faceRatio: 0, blur: 0)
        }

        // LOW_RESOLUTION
        if min(imgWidth, imgHeight) < CGFloat(config.minFrameSide) {
            return QualityResult.fail(reason: "LOW_RESOLUTION", faceRatio: 0, blur: 0)
        }

        // FACE_TOO_SMALL
        let faceRect = pixelRect(from: largest, imgWidth: imgWidth, imgHeight: imgHeight)
        let faceRatio = Float(faceRect.width / imgWidth)
        if faceRatio < config.minFaceRatio {
            return QualityResult.fail(reason: "FACE_TOO_SMALL", faceRatio: faceRatio, blur: 0)
        }

        // FACE_AT_EDGE: bbox menyentuh tepi frame (margin 5%)
        let marginX = imgWidth * CGFloat(config.edgeMargin)
        let marginY = imgHeight * CGFloat(config.edgeMargin)
        if faceRect.minX <= marginX || faceRect.minY <= marginY
                || faceRect.maxX >= imgWidth - marginX || faceRect.maxY >= imgHeight - marginY {
            return QualityResult.fail(reason: "FACE_AT_EDGE", faceRatio: faceRatio, blur: 0)
        }

        // BAD_POSE
        let angles = headAnglesDeg(largest)
        if abs(angles.yaw) > config.maxPose || abs(angles.pitch) > config.maxPose
                || abs(angles.roll) > config.maxPose {
            return QualityResult.fail(reason: "BAD_POSE", faceRatio: faceRatio, blur: 0)
        }

        // BLURRY + BAD_LIGHTING dari crop wajah
        guard let faceCrop = cgImage.cropping(to: faceRect) else {
            return QualityResult.fail(reason: "BLURRY", faceRatio: faceRatio, blur: 0)
        }
        let blur = varianceOfLaplacian(faceCrop)
        let exposure = meanLuminance(faceCrop)
        if blur < config.blurMin {
            return QualityResult.fail(reason: "BLURRY", faceRatio: faceRatio, blur: blur)
        }
        if exposure < config.exposureMin || exposure > config.exposureMax {
            return QualityResult.fail(reason: "BAD_LIGHTING", faceRatio: faceRatio, blur: blur)
        }

        return QualityResult.pass(faceRatio: faceRatio, blur: blur)
    }

    // ============================
    // Anti-Spoof Pipeline — port referensi Silent-Face
    // ============================
    private struct AntiSpoofResult {
        var label: String
        var liveScore: Float
        var printScore: Float
        var replayScore: Float
        var spoofScore: Float
        var isLive: Bool
        var probs27: [Float]
        var probs40: [Float]
        /// Sinyal artefak layar per frame (P2.2) — soft signal, lihat ScreenSuspect.
        var screenSuspectScore: Float = 0

        /// Bentuk nested `liveness` sesuai kontrak.
        func toJson() -> [String: Any] {
            return [
                "isLive": isLive,
                "label": label,
                "liveScore": Double(liveScore),
                "printScore": Double(printScore),
                "replayScore": Double(replayScore),
                "spoofScore": Double(spoofScore),
                "screenSuspectScore": Double(screenSuspectScore),
            ]
        }

        func debugToJson() -> [String: Any] {
            return [
                "probs27": probs27.map { Double($0) },
                "probs40": probs40.map { Double($0) },
            ]
        }
    }

    /// Pipeline referensi:
    ///   crop 2.7x (shift) → model 2.7 ─┐
    ///                                    ├→ sum softmax → argmax + veto → keputusan
    ///   crop 4.0x (shift) → model 4.0 ─┘
    private func runAntiSpoofPipeline(
        cgImage: CGImage,
        baseFaceRect: CGRect,
        imgWidth: CGFloat,
        imgHeight: CGFloat,
        config: GateConfig? = nil
    ) -> AntiSpoofResult? {
        guard let interp27 = antiSpoof27, let interp40 = antiSpoof40 else { return nil }
        let cfg = config ?? GateConfig.from(nil, debugAllowed: false)

        guard let logits27 = runAntiSpoofModel(
                interpreter: interp27, lock: antiSpoof27Lock, cgImage: cgImage,
                baseFaceRect: baseFaceRect, cropScale: cropScale27,
                imgWidth: imgWidth, imgHeight: imgHeight),
              let logits40 = runAntiSpoofModel(
                interpreter: interp40, lock: antiSpoof40Lock, cgImage: cgImage,
                baseFaceRect: baseFaceRect, cropScale: cropScale40,
                imgWidth: imgWidth, imgHeight: imgHeight)
        else {
            return nil
        }

        let probs27 = softmax(logits27)
        let probs40 = softmax(logits40)

        // p = softmax(out27) + softmax(out40) → rentang 0..2
        var p = [Float](repeating: 0, count: antiSpoofOutputSize)
        for i in 0..<antiSpoofOutputSize {
            p[i] = probs27[i] + probs40[i]
        }

        // argmax 3 kelas: [print, live, replay]
        var argmax = 0
        for i in 1..<p.count where p[i] > p[argmax] {
            argmax = i
        }

        var result = AntiSpoofResult(
            label: argmax == 1 ? "LIVE" : (argmax == 0 ? "PRINT" : "REPLAY"),
            liveScore: p[1] / 2,
            printScore: p[0] / 2,
            replayScore: p[2] / 2,
            spoofScore: max(p[0], p[2]) / 2,
            isLive: false,
            probs27: probs27,
            probs40: probs40
        )
        result.isLive = argmax == 1
            && result.liveScore >= cfg.liveThreshold
            && result.spoofScore < cfg.spoofVeto
        // Sinyal artefak layar per frame (P2.2) — soft signal untuk audit/tuning
        result.screenSuspectScore = analyzeScreenSuspect(cgImage: cgImage, baseFaceRect: baseFaceRect)
        return result
    }

    /// Downsample crop wajah ke 32x32 lalu analisis artefak layar (P2.2).
    private func analyzeScreenSuspect(cgImage: CGImage, baseFaceRect: CGRect) -> Float {
        let size = 32
        let imageBounds = CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height)
        let clamped = baseFaceRect.integral.intersection(imageBounds)
        guard !clamped.isEmpty, let crop = cgImage.cropping(to: clamped) else { return 0 }

        var rawPixels = [UInt8](repeating: 0, count: size * size * 4)
        guard let context = CGContext(
            data: &rawPixels, width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return 0 }
        context.interpolationQuality = .low
        context.draw(crop, in: CGRect(x: 0, y: 0, width: size, height: size))

        var argb = [UInt32](repeating: 0, count: size * size)
        for i in 0..<(size * size) {
            let r = UInt32(rawPixels[i * 4])
            let g = UInt32(rawPixels[i * 4 + 1])
            let b = UInt32(rawPixels[i * 4 + 2])
            argb[i] = (r << 16) | (g << 8) | b
        }
        return ScreenSuspect.analyze(argb: argb, width: size, height: size).score
    }

    /// Menjalankan satu inferensi anti-spoofing untuk satu model dengan crop scale-nya.
    /// Interpreter TFLite TIDAK thread-safe → akses disinkronkan per model.
    private func runAntiSpoofModel(
        interpreter: Interpreter,
        lock: NSLock,
        cgImage: CGImage,
        baseFaceRect: CGRect,
        cropScale: CGFloat,
        imgWidth: CGFloat,
        imgHeight: CGFloat
    ) -> [Float]? {
        // Crop sesuai referensi (_get_new_box): scale diturunkan jika tidak muat,
        // w & h terpisah, SHIFT bukan clip, tanpa square-ifikasi
        let faceRect = getNewBox(baseFaceRect, cropScale: cropScale, imgWidth: imgWidth, imgHeight: imgHeight)

        guard let croppedCgImage = cgImage.cropping(to: faceRect) else { return nil }

        // Resize ke 80x80
        let targetSize = CGSize(width: antiSpoofInputSize, height: antiSpoofInputSize)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let scaledImage = renderer.image { _ in
            UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let inputData = imageToInputDataAntiSpoof(scaledImage, size: antiSpoofInputSize) else {
            return nil
        }

        do {
            lock.lock()
            defer { lock.unlock() }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var scores = [Float](repeating: 0, count: antiSpoofOutputSize)
            _ = scores.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }

            // Model mengeluarkan raw logits — softmax dilakukan di pemanggil
            return scores
        } catch {
            print("[AntiSpoof] Inference error (scale \(cropScale)): \(error)")
            return nil
        }
    }

    /// Port logika crop referensi (_get_new_box):
    ///   scale diturunkan jika crop tidak muat di gambar,
    ///   lebar dan tinggi diskalakan TERPISAH (tanpa square),
    ///   jika keluar batas maka DIGESER (shift), bukan dipotong (clip).
    private func getNewBox(_ bounds: CGRect, cropScale: CGFloat,
                           imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let boxW = bounds.width
        let boxH = bounds.height
        let cx = bounds.midX
        let cy = bounds.midY

        // Turunkan scale jika crop tidak muat
        let maxScaleW = (imgWidth - 1) / boxW
        let maxScaleH = (imgHeight - 1) / boxH
        let s = min(cropScale, min(maxScaleW, maxScaleH))

        let newW = boxW * s
        let newH = boxH * s

        var left = cx - newW / 2
        var top = cy - newH / 2
        var right = left + newW
        var bottom = top + newH

        // SHIFT, bukan clip
        if left < 0 { right -= left; left = 0 }
        if top < 0 { bottom -= top; top = 0 }
        if right > imgWidth - 1 { left -= (right - (imgWidth - 1)); right = imgWidth - 1 }
        if bottom > imgHeight - 1 { top -= (bottom - (imgHeight - 1)); bottom = imgHeight - 1 }

        // Penjagaan akhir untuk kasus ekstrem
        left = max(0, left)
        top = max(0, top)

        let x = (left).rounded()
        let y = (top).rounded()
        let w = max(1, min(imgWidth - x, (right - left).rounded()))
        let h = max(1, min(imgHeight - y, (bottom - top).rounded()))
        return CGRect(x: x, y: y, width: w, height: h)
    }

    // ============================
    // Face Recognition helpers
    // ============================

    /// Ekstrak embedding MobileFaceNet dari frame dan bbox yang diberikan.
    /// Crop memakai logika yang SAMA dengan pipeline enrollment (bbox langsung,
    /// di-clip ke batas gambar) agar embedding enroll dan verify konsisten.
    private func extractEmbedding(cgImage: CGImage, faceRect: CGRect) -> [Float]? {
        guard let interpreter = getOrCreateFaceInterpreter() else { return nil }

        let clamped = faceRect.intersection(CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        guard !clamped.isNull, clamped.width > 0, clamped.height > 0,
              let croppedCgImage = cgImage.cropping(to: clamped) else { return nil }

        let targetSize = CGSize(width: inputSize, height: inputSize)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let scaledImage = renderer.image { _ in
            UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let inputData = imageToInputData(scaledImage, size: inputSize) else { return nil }

        do {
            embeddingLock.lock()
            defer { embeddingLock.unlock() }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var embeddings = [Float](repeating: 0, count: outputData.count / MemoryLayout<Float>.size)
            _ = embeddings.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }
            return embeddings
        } catch {
            print("[FaceRecognition] extractEmbedding error: \(error)")
            return nil
        }
    }

    private func pickLargestFace(_ faces: [VNFaceObservation]) -> VNFaceObservation? {
        return faces.max { a, b in
            (a.boundingBox.width * a.boundingBox.height) < (b.boundingBox.width * b.boundingBox.height)
        }
    }

    /// Konversi bounding box Vision (kiri-bawah, normalized) → piksel (kiri-atas)
    private func pixelRect(from face: VNFaceObservation,
                           imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let bb = face.boundingBox
        return CGRect(
            x: bb.origin.x * imgWidth,
            y: (1 - bb.origin.y - bb.size.height) * imgHeight,
            width: bb.size.width * imgWidth,
            height: bb.size.height * imgHeight
        )
    }

    // ============================
    // Vision request factory
    // ============================
    /// Pin revision 3 (iOS 15+). CATATAN PENTING: revision ≤2 mengembalikan
    /// yaw/roll DISKRIT (bin ±45°) dan pitch selalu nil — mematikan akurasi
    /// challenge 15°. Revision 3 memberi nilai kontinu + pitch.
    private func makeFaceRectanglesRequest(
        completionHandler: @escaping VNRequestCompletionHandler
    ) -> VNDetectFaceRectanglesRequest {
        let request = VNDetectFaceRectanglesRequest(completionHandler: completionHandler)
        request.revision = VNDetectFaceRectanglesRequestRevision3
        return request
    }

    /// Konvensi sama dengan ML Kit Android: X = pitch, Y = yaw, Z = roll.
    /// Yaw TERNORMALISASI (YawConvention): hasil POSITIF = user menoleh ke
    /// kanan anatomisnya — dikoreksi dari yaw mentah Vision.
    private func headAnglesDeg(_ face: VNFaceObservation) -> (pitch: Float, yaw: Float, roll: Float) {
        let rad2deg = Float(180.0 / .pi)
        let pitch = face.pitch.map { Float(truncating: $0) } ?? 0
        let yaw = YawConvention.normalize(rawYaw: face.yaw.map { Float(truncating: $0) } ?? 0)
        let roll = face.roll.map { Float(truncating: $0) } ?? 0
        return (pitch: pitch * rad2deg, yaw: yaw * rad2deg, roll: roll * rad2deg)
    }

    /// Eye Aspect Ratio (EAR) 6 titik:
    ///   EAR = (|p1-p5| + |p2-p4|) / (2 * |p3-p0|)
    /// dinormalisasi ke probabilitas 0..1 (clamp(ear / 0.3, 0, 1)).
    private func eyeOpenProbability(from points: [CGPoint]) -> Double {
        guard points.count >= 6 else { return 1.0 }
        let eyeWidth = hypot(points[3].x - points[0].x, points[3].y - points[0].y)
        guard eyeWidth > 0 else { return 1.0 }

        let height1 = hypot(points[1].x - points[5].x, points[1].y - points[5].y)
        let height2 = hypot(points[2].x - points[4].x, points[2].y - points[4].y)
        let ear = Double((height1 + height2) / (2.0 * eyeWidth))

        return min(1.0, max(0.0, ear / 0.3))
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var dot = 0.0, norm1 = 0.0, norm2 = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            norm1 += a[i] * a[i]
            norm2 += b[i] * b[i]
        }
        guard norm1 > 0, norm2 > 0 else { return nil }
        return dot / (sqrt(norm1) * sqrt(norm2))
    }

    // ============================
    // Quality gate helpers — blur, exposure
    // ============================

    /// Variance of Laplacian pada grayscale — ukuran blur (semakin tinggi semakin tajam).
    private func varianceOfLaplacian(_ cgImage: CGImage) -> Float {
        let size = 160 // ukuran kerja tetap agar skor konsisten antar frame
        guard let gray = grayscalePixels(cgImage, size: size) else { return 0 }

        // Kernel Laplacian 3x3: [0 1 0; 1 -4 1; 0 1 0]
        var sum: Double = 0, sumSq: Double = 0
        var count = 0
        for y in 1..<(size - 1) {
            for x in 1..<(size - 1) {
                let idx = y * size + x
                let lap = Double(gray[idx - size]) + Double(gray[idx + size])
                        + Double(gray[idx - 1]) + Double(gray[idx + 1])
                        - 4 * Double(gray[idx])
                sum += lap
                sumSq += lap * lap
                count += 1
            }
        }
        guard count > 0 else { return 0 }
        let mean = sum / Double(count)
        return Float(sumSq / Double(count) - mean * mean)
    }

    /// Mean luminance crop wajah (0..255) — untuk gate exposure.
    private func meanLuminance(_ cgImage: CGImage) -> Float {
        let size = 64
        guard let gray = grayscalePixels(cgImage, size: size) else { return 0 }
        let sum = gray.reduce(0.0) { $0 + Double($1) }
        return Float(sum / Double(gray.count))
    }

    /// Render CGImage ke grayscale (8-bit) berukuran size x size.
    private func grayscalePixels(_ cgImage: CGImage, size: Int) -> [UInt8]? {
        var pixels = [UInt8](repeating: 0, count: size * size)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.linearGray),
              let context = CGContext(
                  data: &pixels,
                  width: size,
                  height: size,
                  bitsPerComponent: 8,
                  bytesPerRow: size,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.none.rawValue
              )
        else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        return pixels
    }

    // ============================
    // Image / preprocessing helpers
    // ============================

    private enum PluginError: Error {
        case invalidImageInput
        case imageDecodeFailed

        var code: String {
            switch self {
            case .invalidImageInput: return "INVALID_IMAGE_INPUT"
            case .imageDecodeFailed: return "IMAGE_DECODE_FAILED"
            }
        }

        var message: String {
            switch self {
            case .invalidImageInput:
                return "Input gambar tidak valid — sediakan `bytes` (Uint8List) atau `base64` (String)."
            case .imageDecodeFailed:
                return "Gagal men-decode data gambar (bytes/base64) menjadi UIImage."
            }
        }
    }

    /// Baca kontrak input gambar (`bytes` = Uint8List ATAU `base64` = String,
    /// opsional dengan prefix `data:...;base64,`).
    private func readImageData(_ args: [String: Any]?) -> Result<Data, PluginError> {
        guard let args = args else { return .failure(.invalidImageInput) }

        if let typedData = args["bytes"] as? FlutterStandardTypedData, !typedData.data.isEmpty {
            return .success(typedData.data)
        }
        if let base64Str = (args["base64"] as? String) ?? (args["imageBase64"] as? String),
           !base64Str.isEmpty {
            let cleanBase64: String
            if let commaRange = base64Str.range(of: ",") {
                cleanBase64 = String(base64Str[commaRange.upperBound...])
            } else {
                cleanBase64 = base64Str
            }
            guard let data = Data(base64Encoded: cleanBase64, options: .ignoreUnknownCharacters),
                  !data.isEmpty
            else {
                return .failure(.imageDecodeFailed)
            }
            return .success(data)
        }
        return .failure(.invalidImageInput)
    }

    /// Decode input gambar dari call → UIImage dengan orientasi dinormalisasi.
    private func decodeImage(from call: FlutterMethodCall) -> Result<UIImage, PluginError> {
        switch readImageData(call.arguments as? [String: Any]) {
        case .failure(let error):
            return .failure(error)
        case .success(let data):
            guard let image = decodeOrientedImage(from: data) else {
                return .failure(.imageDecodeFailed)
            }
            return .success(image)
        }
    }

    /// Decode Data → UIImage dengan orientasi sudah dinormalisasi
    /// (EXIF/UIImage.orientation) agar piksel benar-benar tersetel.
    private func decodeOrientedImage(from data: Data) -> UIImage? {
        guard let image = UIImage(data: data) else { return nil }
        return normalizeImageOrientationIfNeeded(image)
    }

    private func normalizeImageOrientationIfNeeded(_ image: UIImage) -> UIImage {
        // Jika orientasi sudah .up, kembalikan langsung tanpa re-render
        if image.imageOrientation == .up { return image }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.preferredRange = .standard // sRGB standar, bukan Display P3
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }

    // ============================
    // Helper: UIImage → ByteData TFLite — Face Recognition (MobileFaceNet)
    // Normalisasi ke [-1, 1]
    // ============================
    /// Konversi UIImage ke Data berformat [R, G, B, R, G, B, ...]
    /// dengan normalisasi ke [-1, 1]: normalized = (pixel / 128.0) - 1.0
    private func imageToInputData(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        var rawPixels = [UInt8](repeating: 0, count: totalBytes)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rawPixels,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Buffer float: [R, G, B] per pixel, ternormalisasi ke [-1, 1]
        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            floatBuffer[floatIndex]     = (r / 128.0) - 1.0
            floatBuffer[floatIndex + 1] = (g / 128.0) - 1.0
            floatBuffer[floatIndex + 2] = (b / 128.0) - 1.0
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }

    // ============================
    // Helper: UIImage → ByteData TFLite — Anti-Spoofing (Silent-Face)
    // ============================
    private func imageToInputDataAntiSpoof(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        var rawPixels = [UInt8](repeating: 0, count: totalBytes)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rawPixels,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Model Silent-Face mengekspektasikan nilai piksel raw [0, 255] dalam
        // urutan BGR (normalisasi ada di dalam graf model). Jangan paksa ke [-1, 1].
        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            // BGR order (unnormalized)
            floatBuffer[floatIndex]     = b
            floatBuffer[floatIndex + 1] = g
            floatBuffer[floatIndex + 2] = r
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }

    // ============================
    // Softmax / statistik
    // ============================
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sumExps = exps.reduce(0, +)
        guard sumExps > 0 else { return logits.map { _ in 0 } }
        return exps.map { $0 / sumExps }
    }

    private func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2
    }

    /// L2-normalisasi embedding (kontrak: embedding = number[192] ternormalisasi).
    private func l2Normalize(_ embedding: [Float]) -> [Double] {
        var normSq: Double = 0.0
        for v in embedding {
            let d = Double(v)
            normSq += d * d
        }
        let norm = normSq.squareRoot()
        guard norm > 0.0 else {
            return embedding.map { Double($0) }
        }
        return embedding.map { Double($0) / norm }
    }

    // ============================
    // Konfigurasi quality gate yang bisa di-override per panggilan
    // ============================
    private struct GateConfig {
        var liveThreshold: Float = 0.80
        var spoofVeto: Float = 0.20
        var minFrameSide: Int = 480
        var minFaceRatio: Float = 0.25
        var edgeMargin: Float = 0.05
        var maxPose: Float = 20
        var blurMin: Float = 50
        var exposureMin: Float = 60
        var exposureMax: Float = 200
        var identityThreshold: Double = 0.70
        /// Override tanda yaw per-perangkat (1 atau -1) untuk kasus buffer
        /// kamera ter-flip. Default 1 — konvensi normal ditangani YawConvention.
        var yawSignOverride: Int = 1
        /// Rasio minimum frame yang harus isLive per sesi (P2.1). Default 1.0 =
        /// SEMUA frame harus isLive (paling ketat).
        var minLiveVoteRatio: Float = 1.0
        /// Veto screen-suspect per sesi (P2.2) — nil = soft signal saja (default).
        var screenSuspectVeto: Float? = nil
        var debug: Bool = false

        static func from(_ config: [String: Any]?) -> GateConfig {
            return from(config, debugAllowed: false)
        }

        /// - Parameter debugAllowed: hanya true pada build DEBUG — output debug
        ///   tidak pernah dikembalikan dari build rilis (Fase 4)
        static func from(_ config: [String: Any]?, debugAllowed: Bool) -> GateConfig {
            var c = GateConfig()
            c.debug = false
            guard let config = config else { return c }
            if let v = config["liveThreshold"] as? NSNumber { c.liveThreshold = v.floatValue }
            if let v = config["spoofVeto"] as? NSNumber { c.spoofVeto = v.floatValue }
            if let v = config["minFrameSide"] as? NSNumber { c.minFrameSide = v.intValue }
            if let v = config["minFaceRatio"] as? NSNumber { c.minFaceRatio = v.floatValue }
            if let v = config["edgeMargin"] as? NSNumber { c.edgeMargin = v.floatValue }
            if let v = config["maxPose"] as? NSNumber { c.maxPose = v.floatValue }
            if let v = config["blurMin"] as? NSNumber { c.blurMin = v.floatValue }
            if let v = config["exposureMin"] as? NSNumber { c.exposureMin = v.floatValue }
            if let v = config["exposureMax"] as? NSNumber { c.exposureMax = v.floatValue }
            if let v = config["identityThreshold"] as? NSNumber { c.identityThreshold = v.doubleValue }
            if let v = config["yawSignOverride"] as? NSNumber { c.yawSignOverride = v.intValue }
            if let v = config["minLiveVoteRatio"] as? NSNumber { c.minLiveVoteRatio = v.floatValue }
            if let v = config["screenSuspectVeto"] as? NSNumber { c.screenSuspectVeto = v.floatValue }
            if let v = config["debug"] as? NSNumber { c.debug = v.boolValue && debugAllowed }
            return c
        }
    }

    private struct QualityResult {
        var passed: Bool
        var reason: String?
        var faceRatio: Float
        var blur: Float

        static func pass(faceRatio: Float, blur: Float) -> QualityResult {
            return QualityResult(passed: true, reason: nil, faceRatio: faceRatio, blur: blur)
        }

        static func fail(reason: String, faceRatio: Float, blur: Float) -> QualityResult {
            return QualityResult(passed: false, reason: reason, faceRatio: faceRatio, blur: blur)
        }

        func toJson() -> [String: Any] {
            var o: [String: Any] = [
                "passed": passed,
                "faceRatio": Double(faceRatio),
                "blur": Double(blur),
            ]
            if let reason = reason {
                o["reason"] = reason
            }
            return o
        }
    }

    private struct FrameResult {
        var index: Int = 0
        var yaw: Float = 0
        var pitch: Float = 0
        var roll: Float = 0
        /// Epoch ms saat frame di-capture (native capture) — nil untuk jalur Dart.
        var timestampMs: Double? = nil
        var quality: QualityResult = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
        var liveness: AntiSpoofResult?
        var embedding: [Float]?
    }

    // ============================
    // Model Lookup & Interpreter Lifecycle
    // ============================

    /// Lookup file `.tflite`:
    /// 1. Flutter asset lookup (model dikirim sebagai Flutter assets via pubspec)
    /// 2. Fallback ke resource bundle CocoaPods / host app
    private func resolveModelPath(fileName: String) -> String? {
        let assetPath = "assets/models/\(fileName)"

        // Flutter asset lookup (FlutterDartProject.lookupKey + registrar lookupKey)
        let assetKeys = [
            FlutterDartProject.lookupKey(forAsset: assetPath),
            FlutterDartProject.lookupKey(forAsset: assetPath, fromPackage: packageName),
        ]
        for key in assetKeys {
            if let path = Bundle.main.path(forResource: key, ofType: nil) {
                return path
            }
        }
        if let registrar = registrar {
            for key in [
                registrar.lookupKey(forAsset: assetPath),
                registrar.lookupKey(forAsset: assetPath, fromPackage: packageName),
            ] {
                if let path = Bundle.main.path(forResource: key, ofType: nil) {
                    return path
                }
            }
        }

        // Fallback: resource bundle plugin / bundle host app
        let name = (fileName as NSString).deletingPathExtension
        let bundles = [Bundle(for: FlutterFaceRecognitionPlugin.self), Bundle.main]
        for bundle in bundles {
            if let path = bundle.path(forResource: name, ofType: "tflite") {
                return path
            }
        }
        return nil
    }

    private func getOrCreateFaceInterpreter() -> Interpreter? {
        if let existing = faceInterpreter {
            return existing
        }
        guard let modelPath = resolveModelPath(fileName: "\(modelName).tflite") else {
            return nil
        }
        do {
            let newInterpreter = try Interpreter(modelPath: modelPath)
            try newInterpreter.allocateTensors()
            faceInterpreter = newInterpreter
            return newInterpreter
        } catch {
            print("[FlutterFaceRecognition] ERROR: Gagal membuat Interpreter: \(error)")
            return nil
        }
    }

    /// Muat model anti-spoof — `fallbackName` dipakai jika `fileName` tidak ada
    /// (27 → legacy `anti_spoof.tflite`; 40 → tanpa fallback).
    private func loadAntiSpoofModel(fileName: String, fallbackName: String? = nil) -> Interpreter? {
        var candidates = [fileName]
        if let fallbackName = fallbackName, !fallbackName.isEmpty {
            candidates.append(fallbackName)
        }
        for name in candidates {
            if let modelPath = resolveModelPath(fileName: name) {
                do {
                    let newInterpreter = try Interpreter(modelPath: modelPath)
                    try newInterpreter.allocateTensors()
                    return newInterpreter
                } catch {
                    print("[FlutterFaceRecognition] ERROR: Gagal membuat antiSpoofInterpreter (\(name)): \(error)")
                }
            }
        }
        print("[FlutterFaceRecognition] INFO: Model anti-spoofing \(candidates) tidak ditemukan. checkLiveness tidak akan tersedia.")
        return nil
    }

    private func getOrCreateAntiSpoof27() -> Interpreter? {
        if let existing = antiSpoof27 {
            return existing
        }
        let loaded = loadAntiSpoofModel(
            fileName: antiSpoofModelName27,
            fallbackName: antiSpoofModelNameLegacy)
        antiSpoof27 = loaded
        return loaded
    }

    private func getOrCreateAntiSpoof40() -> Interpreter? {
        if let existing = antiSpoof40 {
            return existing
        }
        let loaded = loadAntiSpoofModel(fileName: antiSpoofModelName40)
        antiSpoof40 = loaded
        return loaded
    }

    // ============================
    // Main-Thread Result Helpers
    // ============================

    private func replySuccess(_ result: @escaping FlutterResult, data: Any?) {
        DispatchQueue.main.async {
            result(data)
        }
    }

    private func replyError(
        _ result: @escaping FlutterResult,
        code: String,
        message: String
    ) {
        DispatchQueue.main.async {
            result(FlutterError(code: code, message: message, details: nil))
        }
    }

    private func replyError(
        _ result: @escaping FlutterResult,
        error: PluginError
    ) {
        replyError(result, code: error.code, message: error.message)
    }
}
