package com.andrew.face

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Base64
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class FlutterFaceRecognitionPlugin : FlutterPlugin, MethodCallHandler {

    companion object {
        private const val CHANNEL_NAME = "flutter_face_recognition"
        private const val PACKAGE_NAME = "flutter_face_recognition"

        // MobileFaceNet configuration
        private const val MODEL_FILE = "mobile_face_net.tflite"
        private const val INPUT_SIZE = 112
        private const val PIXEL_CHANNELS = 3
        private const val EMBEDDING_SIZE = 192
        private const val BYTES_PER_FLOAT = 4

        // Anti-Spoofing (MiniFASNet) configuration — identical to capacitor-face-recognition
        private const val ANTI_SPOOF_MODEL_FILE = "anti_spoof.tflite"
        private const val ANTI_SPOOF_INPUT_SIZE = 80
        private const val ANTI_SPOOF_OUTPUT_SIZE = 3
        private const val DEFAULT_LIVENESS_THRESHOLD = 0.75f
        private val ANTI_SPOOF_SCALES = floatArrayOf(1.0f, 2.7f)
    }

    private lateinit var channel: MethodChannel
    private lateinit var applicationContext: Context
    private lateinit var flutterAssets: FlutterPlugin.FlutterAssets

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val interpreterLock = Any()
    private val antiSpoofLock = Any()

    private var tfliteInterpreter: Interpreter? = null
    private var antiSpoofInterpreter: Interpreter? = null
    private var faceDetector: FaceDetector? = null

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        applicationContext = flutterPluginBinding.applicationContext
        flutterAssets = flutterPluginBinding.flutterAssets
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, CHANNEL_NAME)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        scope.cancel()
        releaseResources()
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "extractFaceFeature" -> {
                scope.launch {
                    try {
                        val response = handleExtractFaceFeature(call)
                        withContext(Dispatchers.Main) {
                            result.success(response)
                        }
                    } catch (e: FaceRecognitionException) {
                        withContext(Dispatchers.Main) {
                            result.error(e.code, e.message, null)
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            result.error("EXTRACT_FEATURE_ERROR", e.message ?: "Unknown error", null)
                        }
                    }
                }
            }
            "checkLiveness" -> {
                scope.launch {
                    try {
                        val response = handleCheckLiveness(call)
                        withContext(Dispatchers.Main) {
                            result.success(response)
                        }
                    } catch (e: FaceRecognitionException) {
                        withContext(Dispatchers.Main) {
                            result.error(e.code, e.message, null)
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            result.error("CHECK_LIVENESS_ERROR", e.message ?: "Unknown error", null)
                        }
                    }
                }
            }
            "detectFaces" -> {
                scope.launch {
                    try {
                        val response = handleDetectFaces(call)
                        withContext(Dispatchers.Main) {
                            result.success(response)
                        }
                    } catch (e: FaceRecognitionException) {
                        withContext(Dispatchers.Main) {
                            result.error(e.code, e.message, null)
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            result.error("DETECT_FACES_ERROR", e.message ?: "Unknown error", null)
                        }
                    }
                }
            }
            "dispose" -> {
                releaseResources()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    // ============================
    // Core Method Handlers
    // ============================

    private suspend fun handleExtractFaceFeature(call: MethodCall): Map<String, Any> {
        val bitmap = decodeBitmapFromCall(call)
        val interpreter = getOrCreateFaceInterpreter()
            ?: throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Failed to load TFLite model '$MODEL_FILE' from Flutter assets."
            )

        val inputImage = InputImage.fromBitmap(bitmap, 0)
        val faces = getOrCreateDetector().process(inputImage).await()
        if (faces.isEmpty()) {
            throw FaceRecognitionException("NO_FACE_DETECTED", "Tidak ada wajah yang terdeteksi")
        }

        val bounds = faces[0].boundingBox
        val left = max(0, bounds.left)
        val top = max(0, bounds.top)
        val width = min(bounds.width(), bitmap.width - left)
        val height = min(bounds.height(), bitmap.height - top)

        if (width <= 0 || height <= 0) {
            throw FaceRecognitionException("INVALID_FACE_BOUNDS", "Area wajah tidak valid")
        }

        val croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height)
        val scaledFace = Bitmap.createScaledBitmap(croppedFace, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = bitmapToByteBuffer(scaledFace, INPUT_SIZE)

        val outputArray = Array(1) { FloatArray(EMBEDDING_SIZE) }
        synchronized(interpreterLock) {
            interpreter.run(inputBuffer, outputArray)
        }

        val embedding = outputArray[0].map { it.toDouble() }
        return mapOf("embedding" to embedding)
    }

    private suspend fun handleCheckLiveness(call: MethodCall): Map<String, Any> {
        val bitmap = decodeBitmapFromCall(call)
        val interpreter = getOrCreateAntiSpoofInterpreter()
            ?: throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Failed to load anti-spoofing TFLite model '$ANTI_SPOOF_MODEL_FILE' from Flutter assets."
            )

        val threshold = (call.argument<Number>("threshold")?.toFloat()) ?: DEFAULT_LIVENESS_THRESHOLD

        val inputImage = InputImage.fromBitmap(bitmap, 0)
        val faces = getOrCreateDetector().process(inputImage).await()
        if (faces.isEmpty()) {
            throw FaceRecognitionException(
                "NO_FACE_DETECTED",
                "Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness"
            )
        }

        val bounds = faces[0].boundingBox
        var scoreSum = 0f
        var scaleCount = 0

        for (scaleFactor in ANTI_SPOOF_SCALES) {
            val score = runAntiSpoofInference(interpreter, bitmap, bounds, scaleFactor)
            if (score >= 0f) {
                scoreSum += score
                scaleCount++
            }
        }

        if (scaleCount == 0) {
            throw FaceRecognitionException(
                "LIVENESS_INFERENCE_FAILED",
                "Gagal menjalankan anti-spoof inference di semua scale"
            )
        }

        val livenessScore = scoreSum / scaleCount
        val isLive = livenessScore > threshold
        val confidence = when {
            livenessScore > 0.85f || livenessScore < 0.15f -> "HIGH"
            livenessScore > 0.60f || livenessScore < 0.40f -> "MEDIUM"
            else -> "LOW"
        }

        return mapOf(
            "isLive" to isLive,
            "score" to livenessScore.toDouble(),
            "confidence" to confidence
        )
    }

    private suspend fun handleDetectFaces(call: MethodCall): Map<String, Any> {
        val bitmap = decodeBitmapFromCall(call)
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        val faces = getOrCreateDetector().process(inputImage).await()

        val facesList = faces.map { face ->
            val bounds = face.boundingBox
            val x = max(0, bounds.left)
            val y = max(0, bounds.top)
            val width = min(bounds.width(), bitmap.width - x)
            val height = min(bounds.height(), bitmap.height - y)

            val faceMap = mutableMapOf<String, Any?>(
                "x" to x,
                "y" to y,
                "width" to max(0, width),
                "height" to max(0, height),
                "headEulerAngleY" to face.headEulerAngleY.toDouble(),
                "headEulerAngleZ" to face.headEulerAngleZ.toDouble(),
                "leftEyeOpenProbability" to face.leftEyeOpenProbability?.toDouble(),
                "rightEyeOpenProbability" to face.rightEyeOpenProbability?.toDouble()
            )
            faceMap
        }

        return mapOf(
            "count" to faces.size,
            "faces" to facesList
        )
    }

    // ============================
    // Anti-Spoofing Multi-Scale Helper
    // ============================

    private fun runAntiSpoofInference(
        interpreter: Interpreter,
        bitmap: Bitmap,
        bounds: Rect,
        scaleFactor: Float
    ): Float {
        return try {
            // 1. Buat bounding box menjadi BUJUR SANGKAR (square) terlebih dahulu
            // Persis seperti di capacitor-face-recognition
            val cx = bounds.left + bounds.width() / 2
            val cy = bounds.top + bounds.height() / 2
            val side = max(bounds.width(), bounds.height())

            // 2. Perbesar bujur sangkar berdasarkan scaleFactor (1.0x dan 2.7x)
            val newSide = (side * scaleFactor).toInt()
            val expandW = newSide / 2
            val expandH = newSide / 2

            // 3. Hitung koordinat crop dan pastikan tidak keluar dari batas gambar
            val left = max(0, cx - expandW)
            val top = max(0, cy - expandH)
            val right = min(bitmap.width, cx + expandW)
            val bottom = min(bitmap.height, cy + expandH)
            val width = right - left
            val height = bottom - top

            if (width <= 0 || height <= 0) return -1f

            val croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height)
            val scaledFace = Bitmap.createScaledBitmap(
                croppedFace,
                ANTI_SPOOF_INPUT_SIZE,
                ANTI_SPOOF_INPUT_SIZE,
                true
            )

            val inputBuffer = bitmapToByteBufferAntiSpoof(scaledFace, ANTI_SPOOF_INPUT_SIZE)
            val outputArray = Array(1) { FloatArray(ANTI_SPOOF_OUTPUT_SIZE) }

            synchronized(antiSpoofLock) {
                interpreter.run(inputBuffer, outputArray)
            }

            val probs = softmax(outputArray[0])
            // Index 1 = Real Face
            probs[1]
        } catch (e: Exception) {
            -1f
        }
    }

    // ============================
    // Preprocessing & Math Helpers
    // ============================

    private fun bitmapToByteBuffer(bitmap: Bitmap, size: Int): ByteBuffer {
        val bufferSize = 1 * size * size * PIXEL_CHANNELS * BYTES_PER_FLOAT
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.rewind()

        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16) and 0xFF
            val g = (pixelValue shr 8) and 0xFF
            val b = pixelValue and 0xFF

            byteBuffer.putFloat((r / 128.0f) - 1.0f)
            byteBuffer.putFloat((g / 128.0f) - 1.0f)
            byteBuffer.putFloat((b / 128.0f) - 1.0f)
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun bitmapToByteBufferAntiSpoof(bitmap: Bitmap, size: Int): ByteBuffer {
        val bufferSize = 1 * size * size * PIXEL_CHANNELS * BYTES_PER_FLOAT
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.rewind()

        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16) and 0xFF
            val g = (pixelValue shr 8) and 0xFF
            val b = pixelValue and 0xFF

            // MiniFASNet expects raw BGR [0, 255]
            byteBuffer.putFloat(b.toFloat())
            byteBuffer.putFloat(g.toFloat())
            byteBuffer.putFloat(r.toFloat())
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        var sum = 0f
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += exps[i]
        }
        if (sum > 0f) {
            for (i in exps.indices) {
                exps[i] /= sum
            }
        }
        return exps
    }

    private fun l2Normalize(embedding: FloatArray): List<Double> {
        var normSq = 0.0
        for (v in embedding) {
            val d = v.toDouble()
            normSq += d * d
        }
        val norm = sqrt(normSq)
        if (norm <= 0.0) {
            return embedding.map { it.toDouble() }
        }
        return embedding.map { (it.toDouble() / norm) }
    }

    // ============================
    // Image Decoding & EXIF Orientation
    // ============================

    private fun decodeBitmapFromCall(call: MethodCall): Bitmap {
        val rawBytes: ByteArray = when {
            call.hasArgument("bytes") && call.argument<ByteArray>("bytes") != null -> {
                call.argument<ByteArray>("bytes")!!
            }
            call.hasArgument("base64") && !call.argument<String>("base64").isNullOrBlank() -> {
                cleanAndDecodeBase64(call.argument<String>("base64")!!)
            }
            call.hasArgument("imageBase64") && !call.argument<String>("imageBase64").isNullOrBlank() -> {
                cleanAndDecodeBase64(call.argument<String>("imageBase64")!!)
            }
            else -> throw FaceRecognitionException(
                "INVALID_IMAGE_INPUT",
                "Parameter 'bytes' (Uint8List) atau 'base64' tidak boleh kosong"
            )
        }

        if (rawBytes.isEmpty()) {
            throw FaceRecognitionException("INVALID_IMAGE_INPUT", "Data gambar kosong")
        }

        val decoded = BitmapFactory.decodeByteArray(rawBytes, 0, rawBytes.size)
            ?: throw FaceRecognitionException("IMAGE_DECODE_FAILED", "Gagal men-decode data gambar menjadi Bitmap")

        return applyExifOrientation(decoded, rawBytes)
    }

    private fun cleanAndDecodeBase64(input: String): ByteArray {
        val commaIndex = input.indexOf(',')
        val cleanBase64 = if (commaIndex != -1) input.substring(commaIndex + 1) else input
        return Base64.decode(cleanBase64, Base64.DEFAULT)
    }

    private fun applyExifOrientation(bitmap: Bitmap, bytes: ByteArray): Bitmap {
        return try {
            val exif = ExifInterface(ByteArrayInputStream(bytes))
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                    matrix.setRotate(180f)
                    matrix.postScale(-1f, 1f)
                }
                ExifInterface.ORIENTATION_TRANSPOSE -> {
                    matrix.setRotate(90f)
                    matrix.postScale(-1f, 1f)
                }
                ExifInterface.ORIENTATION_TRANSVERSE -> {
                    matrix.setRotate(-90f)
                    matrix.postScale(-1f, 1f)
                }
                else -> return bitmap
            }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (_: Exception) {
            bitmap
        }
    }

    // ============================
    // Lazy Resource & Asset Management
    // ============================

    private fun getOrCreateDetector(): FaceDetector {
        var current = faceDetector
        if (current == null) {
            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .build()
            current = FaceDetection.getClient(options)
            faceDetector = current
        }
        return current
    }

    private fun getOrCreateFaceInterpreter(): Interpreter? {
        synchronized(interpreterLock) {
            if (tfliteInterpreter == null) {
                tfliteInterpreter = try {
                    Interpreter(loadModelFile(MODEL_FILE))
                } catch (_: Exception) {
                    null
                }
            }
            return tfliteInterpreter
        }
    }

    private fun getOrCreateAntiSpoofInterpreter(): Interpreter? {
        synchronized(antiSpoofLock) {
            if (antiSpoofInterpreter == null) {
                antiSpoofInterpreter = try {
                    Interpreter(loadModelFile(ANTI_SPOOF_MODEL_FILE))
                } catch (_: Exception) {
                    null
                }
            }
            return antiSpoofInterpreter
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val candidatePaths = listOf(
            flutterAssets.getAssetFilePathBySubpath("assets/models/$fileName", PACKAGE_NAME),
            flutterAssets.getAssetFilePathByName("assets/models/$fileName"),
            "flutter_assets/packages/$PACKAGE_NAME/assets/models/$fileName",
            "flutter_assets/assets/models/$fileName",
            fileName
        )

        val assetManager = applicationContext.assets

        // 1. Try memory-mapping directly via openFd (fastest when uncompressed)
        for (assetPath in candidatePaths) {
            try {
                assetManager.openFd(assetPath).use { fd ->
                    FileInputStream(fd.fileDescriptor).use { inputStream ->
                        return inputStream.channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            fd.startOffset,
                            fd.declaredLength
                        )
                    }
                }
            } catch (_: Exception) {
                // Try next or fallback to stream copy
            }
        }

        // 2. Fallback: copy asset stream to cache file and memory-map it
        for (assetPath in candidatePaths) {
            try {
                val cachedFile = File(applicationContext.cacheDir, fileName)
                if (!cachedFile.exists() || cachedFile.length() == 0L) {
                    assetManager.open(assetPath).use { input ->
                        FileOutputStream(cachedFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                FileInputStream(cachedFile).use { fis ->
                    return fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, cachedFile.length())
                }
            } catch (_: Exception) {
                // Try next candidate path
            }
        }

        throw IOException("Model file '$fileName' not found in Flutter or Android assets.")
    }

    private fun releaseResources() {
        synchronized(interpreterLock) {
            tfliteInterpreter?.close()
            tfliteInterpreter = null
        }
        synchronized(antiSpoofLock) {
            antiSpoofInterpreter?.close()
            antiSpoofInterpreter = null
        }
        faceDetector?.close()
        faceDetector = null
    }

    private class FaceRecognitionException(val code: String, override val message: String) :
        Exception(message)
}

