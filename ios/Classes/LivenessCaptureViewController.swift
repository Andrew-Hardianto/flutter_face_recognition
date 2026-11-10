import AVFoundation
import UIKit

/// Callback untuk hasil capture `LivenessCaptureViewController` (Fase 4 — native capture).
///
/// Frame diambil DI DALAM plugin dan disimpan ke file privat, TIDAK pernah
/// melewati method channel — menutup celah injeksi frame. Yang kembali ke Dart
/// hanya HASIL verifikasi, bukan frame mentah.
protocol LivenessCaptureViewControllerDelegate: AnyObject {
    /// Capture selesai — frame JPEG tersimpan di cache privat.
    /// - Parameters:
    ///   - challenge: arah challenge sesi ini (`LOOK_LEFT` / `LOOK_RIGHT`)
    ///   - framePaths: path file frame (urutan capture)
    ///   - timestamps: epoch ms per frame (sejajar `framePaths`)
    func livenessCaptureViewController(
        _ controller: LivenessCaptureViewController,
        didFinishWithChallenge challenge: String,
        framePaths: [String],
        timestamps: [Double]
    )

    /// Capture gagal / dibatalkan.
    func livenessCaptureViewController(
        _ controller: LivenessCaptureViewController,
        didFailWithError error: Error
    )
}

/// View controller kamera untuk captureLivenessSession (Fase 4 — native capture).
final class LivenessCaptureViewController: UIViewController, AVCapturePhotoCaptureDelegate {

    weak var delegate: LivenessCaptureViewControllerDelegate?

    private let challenge: String
    private let frameCount: Int
    private let interval: TimeInterval
    private let warmup: TimeInterval

    private let captureSession = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let instructionLabel = UILabel()
    private let progressLabel = UILabel()

    private var framePaths: [String] = []
    private var frameTimestamps: [Double] = []
    private var warmupTimer: Timer?
    private var captureTimer: Timer?
    private var finished = false

    /// Payload sukses: challenge + path file frame + timestamp (epoch ms).
    typealias CaptureSuccess = (challenge: String, framePaths: [String], timestamps: [Double])

    enum CaptureError: LocalizedError {
        case cameraUnavailable
        case photoFailed
        case writeFailed
        case invalidChallenge

        var errorDescription: String? {
            switch self {
            case .cameraUnavailable: return "Kamera depan tidak tersedia"
            case .photoFailed: return "Gagal mengambil frame"
            case .writeFailed: return "Gagal menyimpan frame"
            case .invalidChallenge: return "Challenge tidak dikenal"
            }
        }
    }

    init(challenge: String, frameCount: Int, interval: TimeInterval, warmup: TimeInterval) {
        self.challenge = challenge
        self.frameCount = max(2, frameCount)
        self.interval = interval
        self.warmup = warmup
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        // FAIL-CLOSED: challenge tak dikenal tidak boleh diam-diam memakai
        // instruksi default — sesi dibatalkan sebelum kamera dibuka.
        guard challenge == "LOOK_LEFT" || challenge == "LOOK_RIGHT" else {
            DispatchQueue.main.async { [weak self] in
                self?.finish(.failure(CaptureError.invalidChallenge))
            }
            return
        }
        setupLabels()
        configureSession()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    private func setupLabels() {
        // FAIL-CLOSED: mapping eksplisit per challenge — nilai tak dikenal sudah
        // ditolak di viewDidLoad, TIDAK ada default diam-diam ke "ke kanan".
        let instruction: String
        switch challenge {
        case "LOOK_LEFT":
            instruction = "TOLEH KE KIRI ANDA (dagu ke bahu kiri) perlahan, lalu kembali lurus"
        case "LOOK_RIGHT":
            instruction = "TOLEH KE KANAN ANDA (dagu ke bahu kanan) perlahan, lalu kembali lurus"
        default:
            instruction = ""
        }

        instructionLabel.text = instruction
        instructionLabel.textColor = .white
        instructionLabel.font = .systemFont(ofSize: 18, weight: .semibold)
        instructionLabel.textAlignment = .center
        instructionLabel.numberOfLines = 0
        instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        instructionLabel.translatesAutoresizingMaskIntoConstraints = false

        progressLabel.text = "Bersiap..."
        progressLabel.textColor = .white
        progressLabel.font = .systemFont(ofSize: 16)
        progressLabel.textAlignment = .center
        progressLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        progressLabel.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(instructionLabel)
        view.addSubview(progressLabel)

        NSLayoutConstraint.activate([
            instructionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            instructionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            instructionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            instructionLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 64),

            progressLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            progressLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            progressLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            progressLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
        ])
    }

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input)
        else {
            captureSession.commitConfiguration()
            finish(.failure(CaptureError.cameraUnavailable))
            return
        }
        captureSession.addInput(input)

        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        }
        captureSession.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(layer, at: 0)
        previewLayer = layer

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }

        // Beri waktu user memosisikan wajah, lalu capture frame berkala
        warmupTimer = Timer.scheduledTimer(withTimeInterval: warmup, repeats: false) { [weak self] _ in
            self?.captureNext()
        }
    }

    private func captureNext() {
        let settings = AVCapturePhotoSettings()
        if photoOutput.supportedFlashModes.contains(.off) {
            settings.flashMode = .off
        }
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    // MARK: - AVCapturePhotoCaptureDelegate

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        guard error == nil, let data = photo.fileDataRepresentation() else {
            finish(.failure(CaptureError.photoFailed))
            return
        }

        // Cache privat "liveness_frames" — file frame TIDAK pernah melewati Dart.
        guard let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            finish(.failure(CaptureError.writeFailed))
            return
        }
        let dir = cacheDir.appendingPathComponent("liveness_frames", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent("frame_\(framePaths.count).jpg")
            try data.write(to: url)
            framePaths.append(url.path)
            frameTimestamps.append(Date().timeIntervalSince1970 * 1000)
        } catch {
            finish(.failure(CaptureError.writeFailed))
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.progressLabel.text = "Mengambil frame \(self.framePaths.count)/\(self.frameCount)"
            if self.framePaths.count >= self.frameCount {
                self.finish(.success((challenge: self.challenge,
                                      framePaths: self.framePaths,
                                      timestamps: self.frameTimestamps)))
            } else {
                self.captureTimer = Timer.scheduledTimer(
                    withTimeInterval: self.interval, repeats: false) { [weak self] _ in
                    self?.captureNext()
                }
            }
        }
    }

    private func finish(_ result: Result<CaptureSuccess, Error>) {
        guard !finished else { return }
        finished = true
        DispatchQueue.main.async {
            self.warmupTimer?.invalidate()
            self.captureTimer?.invalidate()
            self.captureSession.stopRunning()
            self.dismiss(animated: true) {
                switch result {
                case .success(let session):
                    self.delegate?.livenessCaptureViewController(
                        self,
                        didFinishWithChallenge: session.challenge,
                        framePaths: session.framePaths,
                        timestamps: session.timestamps)
                case .failure(let error):
                    self.delegate?.livenessCaptureViewController(self, didFailWithError: error)
                }
            }
        }
    }
}
