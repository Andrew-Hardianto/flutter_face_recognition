import CoreImage
import Flutter
import Foundation
import TensorFlowLite
import UIKit
import Vision

public class FlutterFaceRecognitionPlugin: NSObject, FlutterPlugin {

    // ============================
    // Configuration — Face Recognition (MobileFaceNet)
    // ============================
    private let packageName = "flutter_face_recognition"
    private let modelName = "mobile_face_net"
    private let inputSize = 112
    private let embeddingSize = 192

    // ============================
    // Configuration — Anti-Spoofing (MiniFASNet)
    // ============================
    private let antiSpoofModelName = "anti_spoof"
    private let antiSpoofInputSize = 80
    private let antiSpoofOutputSize = 3
    private let defaultLivenessThreshold: Float = 0.75
    private let antiSpoofScales: [CGFloat] = [1.0, 2.7]

    private let maxImageDimension: CGFloat = 1920.0

    private let inferenceQueue = DispatchQueue(
        label: "com.andrew.face.inferenceQueue",
        qos: .userInitiated
    )

    private var registrar: FlutterPluginRegistrar?
    private var interpreter: Interpreter?
    private var antiSpoofInterpreter: Interpreter?

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
        case "extractFaceFeature":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                self.handleExtractFaceFeature(call: call, result: result)
            }
        case "checkLiveness":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                self.handleCheckLiveness(call: call, result: result)
            }
        case "detectFaces":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                self.handleDetectFaces(call: call, result: result)
            }
        case "dispose":
            inferenceQueue.async { [weak self] in
                self?.interpreter = nil
                self?.antiSpoofInterpreter = nil
                DispatchQueue.main.async {
                    result(nil)
                }
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // ============================
    // Method Handlers
    // ============================

    private func handleExtractFaceFeature(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let interpreter = getOrCreateFaceInterpreter() else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model TFLite '\(modelName).tflite' gagal dimuat dari Flutter assets."
            )
            return
        }

        guard let image = decodeImage(from: call),
              let cgImage = image.cgImage
        else {
            replyError(
                result,
                code: "IMAGE_DECODE_FAILED",
                message: "Gagal men-decode data gambar (bytes/base64) menjadi UIImage."
            )
            return
        }

        let request = VNDetectFaceRectanglesRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            guard let observations = req.results as? [VNFaceObservation],
                  let face = observations.first
            else {
                self.replyError(
                    result,
                    code: "NO_FACE_DETECTED",
                    message: "Tidak ada wajah yang terdeteksi"
                )
                return
            }

            let imgWidth = CGFloat(cgImage.width)
            let imgHeight = CGFloat(cgImage.height)
            let bb = face.boundingBox

            let rawRect = CGRect(
                x: bb.origin.x * imgWidth,
                y: (1.0 - bb.origin.y - bb.size.height) * imgHeight,
                width: bb.size.width * imgWidth,
                height: bb.size.height * imgHeight
            )
            let clampedRect = rawRect.intersection(
                CGRect(x: 0, y: 0, width: imgWidth, height: imgHeight)
            )

            guard !clampedRect.isNull,
                  clampedRect.width > 0,
                  clampedRect.height > 0,
                  let croppedCgImage = cgImage.cropping(to: clampedRect)
            else {
                self.replyError(
                    result,
                    code: "CROP_FAILED",
                    message: "Gagal memotong gambar wajah"
                )
                return
            }

            let targetSize = CGSize(width: self.inputSize, height: self.inputSize)
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1.0
            let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
            let scaledImage = renderer.image { _ in
                UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
            }

            guard let inputData = self.imageToInputData(scaledImage, size: self.inputSize) else {
                self.replyError(
                    result,
                    code: "PREPROCESS_FAILED",
                    message: "Gagal mengonversi gambar ke buffer input TFLite"
                )
                return
            }

            do {
                try interpreter.copy(inputData, toInputAt: 0)
                try interpreter.invoke()

                let outputTensor = try interpreter.output(at: 0)
                let outputData = outputTensor.data

                let floatCount = outputData.count / MemoryLayout<Float>.size
                var embeddings = [Float](repeating: 0, count: floatCount)
                _ = embeddings.withUnsafeMutableBytes { ptr in
                    outputData.copyBytes(to: ptr)
                }

                let normalized = self.l2Normalize(embeddings)
                self.replySuccess(result, data: ["embedding": normalized])
            } catch {
                self.replyError(
                    result,
                    code: "TFLITE_INFERENCE_ERROR",
                    message: "TFLite inference gagal: \(error.localizedDescription)"
                )
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_EXECUTION_ERROR",
                message: "Gagal menjalankan Apple Vision Request: \(error.localizedDescription)"
            )
        }
    }

    private func handleCheckLiveness(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let antiSpoofInterpreter = getOrCreateAntiSpoofInterpreter() else {
            replyError(
                result,
                code: "MODEL_NOT_LOADED",
                message: "Model anti-spoofing '\(antiSpoofModelName).tflite' tidak ditemukan di Flutter assets."
            )
            return
        }

        guard let image = decodeImage(from: call),
              let cgImage = image.cgImage
        else {
            replyError(
                result,
                code: "IMAGE_DECODE_FAILED",
                message: "Gagal men-decode data gambar menjadi UIImage."
            )
            return
        }

        let args = call.arguments as? [String: Any]
        let threshold = (args?["threshold"] as? NSNumber)?.floatValue ?? defaultLivenessThreshold

        let request = VNDetectFaceRectanglesRequest { [weak self] req, error in
            guard let self = self else { return }

            if let err = error {
                self.replyError(
                    result,
                    code: "VISION_ERROR",
                    message: "Apple Vision gagal mendeteksi wajah: \(err.localizedDescription)"
                )
                return
            }

            guard let observations = req.results as? [VNFaceObservation],
                  let face = observations.first
            else {
                self.replyError(
                    result,
                    code: "NO_FACE_DETECTED",
                    message: "Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness"
                )
                return
            }

            let imgWidth = CGFloat(cgImage.width)
            let imgHeight = CGFloat(cgImage.height)
            let bb = face.boundingBox

            let baseFaceRect = CGRect(
                x: bb.origin.x * imgWidth,
                y: (1.0 - bb.origin.y - bb.size.height) * imgHeight,
                width: bb.size.width * imgWidth,
                height: bb.size.height * imgHeight
            )

            var scaleScores: [Float] = []
            for scaleFactor in self.antiSpoofScales {
                if let score = self.runAntiSpoofInference(
                    interpreter: antiSpoofInterpreter,
                    cgImage: cgImage,
                    baseFaceRect: baseFaceRect,
                    scaleFactor: scaleFactor,
                    imgWidth: imgWidth,
                    imgHeight: imgHeight
                ) {
                    scaleScores.append(score)
                }
            }

            guard !scaleScores.isEmpty else {
                self.replyError(
                    result,
                    code: "LIVENESS_INFERENCE_FAILED",
                    message: "Gagal menjalankan anti-spoof inference"
                )
                return
            }

            let livenessScore = scaleScores.reduce(0, +) / Float(scaleScores.count)
            let isLive = livenessScore > threshold

            let confidence: String
            if livenessScore > 0.85 || livenessScore < 0.15 {
                confidence = "HIGH"
            } else if livenessScore > 0.60 || livenessScore < 0.40 {
                confidence = "MEDIUM"
            } else {
                confidence = "LOW"
            }

            self.replySuccess(
                result,
                data: [
                    "isLive": isLive,
                    "score": Double(livenessScore),
                    "confidence": confidence,
                ]
            )
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_EXECUTION_ERROR",
                message: "Gagal menjalankan Apple Vision Request untuk liveness: \(error.localizedDescription)"
            )
        }
    }

    private func handleDetectFaces(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let image = decodeImage(from: call),
              let cgImage = image.cgImage
        else {
            replyError(
                result,
                code: "IMAGE_DECODE_FAILED",
                message: "Gagal men-decode data gambar menjadi UIImage."
            )
            return
        }

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
            let imgWidth = CGFloat(cgImage.width)
            let imgHeight = CGFloat(cgImage.height)

            var facesArray: [[String: Any]] = []
            for face in results {
                let bb = face.boundingBox
                let x = bb.origin.x * imgWidth
                let y = (1.0 - bb.origin.y - bb.size.height) * imgHeight
                let width = bb.size.width * imgWidth
                let height = bb.size.height * imgHeight

                // Convert Vision radians to degrees to match Android ML Kit Euler angles
                let yawDegrees = (face.yaw?.doubleValue ?? 0.0) * 180.0 / Double.pi
                let rollDegrees = (face.roll?.doubleValue ?? 0.0) * 180.0 / Double.pi

                var leftEyeOpenProb: Double?
                var rightEyeOpenProb: Double?

                if let landmarks = face.landmarks {
                    if let leftEye = landmarks.leftEye {
                        leftEyeOpenProb = self.calculateNormalizedEyeOpenProbability(region: leftEye)
                    }
                    if let rightEye = landmarks.rightEye {
                        rightEyeOpenProb = self.calculateNormalizedEyeOpenProbability(region: rightEye)
                    }
                }

                let clampedX = Int(max(0, x))
                let clampedY = Int(max(0, y))
                let clampedWidth = Int(max(0, min(width, imgWidth - max(0, x))))
                let clampedHeight = Int(max(0, min(height, imgHeight - max(0, y))))

                var faceDict: [String: Any] = [
                    "x": clampedX,
                    "y": clampedY,
                    "width": clampedWidth,
                    "height": clampedHeight,
                    "headEulerAngleY": yawDegrees,
                    "headEulerAngleZ": rollDegrees,
                ]

                if let leftProb = leftEyeOpenProb {
                    faceDict["leftEyeOpenProbability"] = leftProb
                }
                if let rightProb = rightEyeOpenProb {
                    faceDict["rightEyeOpenProbability"] = rightProb
                }

                facesArray.append(faceDict)
            }

            self.replySuccess(
                result,
                data: [
                    "count": results.count,
                    "faces": facesArray,
                ]
            )
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            replyError(
                result,
                code: "VISION_EXECUTION_ERROR",
                message: "Gagal menjalankan Apple Vision Request: \(error.localizedDescription)"
            )
        }
    }

    // ============================
    // Multi-Scale Anti-Spoofing Helper
    // ============================

    private func runAntiSpoofInference(
        interpreter: Interpreter,
        cgImage: CGImage,
        baseFaceRect: CGRect,
        scaleFactor: CGFloat,
        imgWidth: CGFloat,
        imgHeight: CGFloat
    ) -> Float? {
        let cx = baseFaceRect.origin.x + baseFaceRect.width / 2.0
        let cy = baseFaceRect.origin.y + baseFaceRect.height / 2.0
        let side = max(baseFaceRect.width, baseFaceRect.height)

        let newSide = side * scaleFactor
        let expandW = newSide / 2.0
        let expandH = newSide / 2.0

        let expandedX = max(0, cx - expandW)
        let expandedY = max(0, cy - expandH)
        let expandedW = min(imgWidth - expandedX, cx + expandW - expandedX)
        let expandedH = min(imgHeight - expandedY, cy + expandH - expandedY)

        guard expandedW > 0, expandedH > 0 else { return nil }
        let faceRect = CGRect(x: expandedX, y: expandedY, width: expandedW, height: expandedH)

        guard let croppedCgImage = cgImage.cropping(to: faceRect) else { return nil }

        let targetSize = CGSize(width: antiSpoofInputSize, height: antiSpoofInputSize)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let scaledImage = renderer.image { _ in
            UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let inputData = imageToInputDataAntiSpoof(scaledImage, size: antiSpoofInputSize) else {
            return nil
        }

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var scores = [Float](repeating: 0, count: antiSpoofOutputSize)
            _ = scores.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }

            let probs = softmax(scores)
            // Index 1 = Real Face
            return probs[1]
        } catch {
            return nil
        }
    }

    // ============================
    // Preprocessing & Math Helpers
    // ============================

    private func calculateNormalizedEyeOpenProbability(region: VNFaceLandmarkRegion2D) -> Double? {
        let pts = region.normalizedPoints
        guard pts.count >= 4 else { return nil }
        let ys = pts.map { $0.y }
        let xs = pts.map { $0.x }
        guard let minY = ys.min(), let maxY = ys.max(),
              let minX = xs.min(), let maxX = xs.max()
        else { return nil }

        let eyeHeight = maxY - minY
        let eyeWidth = maxX - minX
        guard eyeWidth > 0 else { return nil }

        let ear = Double(eyeHeight / eyeWidth)
        // Map Eye Aspect Ratio (~0.08 closed -> ~0.28 open) into [0.0, 1.0] probability
        let normalized = (ear - 0.08) / 0.20
        return min(1.0, max(0.0, normalized))
    }

    private func imageToInputData(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
        let bytesPerPixel = 4
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

        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            floatBuffer[floatIndex] = (r / 128.0) - 1.0
            floatBuffer[floatIndex + 1] = (g / 128.0) - 1.0
            floatBuffer[floatIndex + 2] = (b / 128.0) - 1.0
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }

    private func imageToInputDataAntiSpoof(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
        let bytesPerPixel = 4
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

        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            // MiniFASNet expects raw BGR [0, 255]
            floatBuffer[floatIndex] = b
            floatBuffer[floatIndex + 1] = g
            floatBuffer[floatIndex + 2] = r
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sumExps = exps.reduce(0, +)
        guard sumExps > 0 else { return logits.map { _ in 0 } }
        return exps.map { $0 / sumExps }
    }

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
    // Image Decoding & Orientation
    // ============================

    private func decodeImage(from call: FlutterMethodCall) -> UIImage? {
        guard let args = call.arguments as? [String: Any] else { return nil }

        var imageData: Data?
        if let typedData = args["bytes"] as? FlutterStandardTypedData {
            imageData = typedData.data
        } else if let base64Str = (args["base64"] as? String) ?? (args["imageBase64"] as? String),
                  !base64Str.isEmpty {
            let cleanBase64: String
            if let commaRange = base64Str.range(of: ",") {
                cleanBase64 = String(base64Str[commaRange.upperBound...])
            } else {
                cleanBase64 = base64Str
            }
            imageData = Data(base64Encoded: cleanBase64, options: .ignoreUnknownCharacters)
        }

        guard let data = imageData, !data.isEmpty,
              let rawImage = UIImage(data: data)
        else { return nil }

        return normalizeAndDownsampleImage(rawImage)
    }

    private func normalizeAndDownsampleImage(_ image: UIImage) -> UIImage {
        let width = image.size.width
        let height = image.size.height
        let maxDim = max(width, height)

        let scaleRatio: CGFloat = maxDim > maxImageDimension ? (maxImageDimension / maxDim) : 1.0
        if image.imageOrientation == .up && scaleRatio == 1.0 {
            return image
        }

        let targetSize = CGSize(
            width: floor(width * scaleRatio),
            height: floor(height * scaleRatio)
        )
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    // ============================
    // Model Lookup & Interpreter Lifecycle
    // ============================

    private func resolveModelPath(named name: String) -> String? {
        if let registrar = registrar {
            let packageKey = registrar.lookupKey(
                forAsset: "assets/models/\(name).tflite",
                fromPackage: packageName
            )
            if let path = Bundle.main.path(forResource: packageKey, ofType: nil) {
                return path
            }

            let directKey = registrar.lookupKey(forAsset: "assets/models/\(name).tflite")
            if let path = Bundle.main.path(forResource: directKey, ofType: nil) {
                return path
            }
        }

        if let path = Bundle(for: FlutterFaceRecognitionPlugin.self).path(
            forResource: name,
            ofType: "tflite"
        ) {
            return path
        }

        return Bundle.main.path(forResource: name, ofType: "tflite")
    }

    private func getOrCreateFaceInterpreter() -> Interpreter? {
        if let existing = interpreter {
            return existing
        }
        guard let modelPath = resolveModelPath(named: modelName) else {
            return nil
        }
        do {
            let newInterpreter = try Interpreter(modelPath: modelPath)
            try newInterpreter.allocateTensors()
            interpreter = newInterpreter
            return newInterpreter
        } catch {
            return nil
        }
    }

    private func getOrCreateAntiSpoofInterpreter() -> Interpreter? {
        if let existing = antiSpoofInterpreter {
            return existing
        }
        guard let modelPath = resolveModelPath(named: antiSpoofModelName) else {
            return nil
        }
        do {
            let newInterpreter = try Interpreter(modelPath: modelPath)
            try newInterpreter.allocateTensors()
            antiSpoofInterpreter = newInterpreter
            return newInterpreter
        } catch {
            return nil
        }
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
}

