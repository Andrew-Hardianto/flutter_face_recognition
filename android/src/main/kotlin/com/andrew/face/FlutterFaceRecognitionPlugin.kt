package com.andrew.face

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Base64
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
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
import java.util.UUID
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Flutter plugin liveness-v3 — port Android (Kotlin) dari pipeline
 * FaceRecognitionPlugin (Java) dengan kontrak wire di docs/native_contract.md.
 *
 * Pipeline: Quality Gate → Dual Anti-Spoof (crop 2.7 + 4.0) → Embedding
 * (MobileFaceNet) → Challenge Trajectory temporal (R1–R5).
 */
class FlutterFaceRecognitionPlugin : FlutterPlugin, MethodCallHandler, ActivityAware,
    PluginRegistry.ActivityResultListener, PluginRegistry.RequestPermissionsResultListener {

    companion object {
        private const val TAG = "FlutterFaceRecognition"
        private const val CHANNEL_NAME = "flutter_face_recognition"
        private const val PACKAGE_NAME = "flutter_face_recognition"

        // ============================
        // Konfigurasi — Face Recognition (MobileFaceNet)
        // ============================
        private const val MODEL_FILE = "mobile_face_net.tflite"
        private const val INPUT_SIZE = 112
        private const val PIXEL_CHANNELS = 3
        private const val EMBEDDING_SIZE = 192
        private const val BYTES_PER_FLOAT = 4

        // ============================
        // Konfigurasi — Anti-Spoofing (Silent-Face / minivision-ai)
        // ============================
        private const val ANTI_SPOOF_MODEL_FILE_27 = "anti_spoof_2_7.tflite"
        private const val ANTI_SPOOF_MODEL_FILE_40 = "anti_spoof_4_0.tflite"
        private const val ANTI_SPOOF_MODEL_FILE_LEGACY = "anti_spoof.tflite"
        private const val ANTI_SPOOF_INPUT_SIZE = 80
        private const val ANTI_SPOOF_OUTPUT_SIZE = 3
        private const val CROP_SCALE_27 = 2.7f
        private const val CROP_SCALE_40 = 4.0f

        // ============================
        // Threshold — SATU sumber kebenaran (liveness-v3)
        // ============================
        private const val LIVE_THRESHOLD = 0.80f
        private const val SPOOF_VETO = 0.20f
        private const val THRESHOLD_VERSION = "liveness-v3"
        private const val MATCH_THRESHOLD_VERSION = "match-v1"
        private const val MATCH_THRESHOLD = 0.75

        // ============================
        // Quality gate (Fase 3.1) — semua bisa di-override via config
        // ============================
        private const val MIN_FRAME_SIDE = 480
        private const val MIN_FACE_RATIO = 0.25f
        private const val EDGE_MARGIN = 0.05f
        private const val MAX_POSE_DEG = 20f
        private const val BLUR_MIN = 50f
        private const val EXPOSURE_MIN = 60f
        private const val EXPOSURE_MAX = 200f
        private const val IDENTITY_CONSISTENCY = 0.70f
        private const val IDENTITY_MAX_POSE = 12f
        private const val CHALLENGE_FRONT_MAX = 15f
        private const val CHALLENGE_TTL_MS = 60000L

        // ============================
        // Pertahanan berlapis (Fase 4)
        // ============================
        private const val MAX_SESSION_ATTEMPTS = 10
        private const val SESSION_ATTEMPT_WINDOW_MS = 300000L
        private val sessionAttempts = mutableListOf<Long>()
        private const val SPOOF_HARD_VETO = 0.5f
        private const val SESSION_DURATION_MIN_MS = 1000.0
        private const val SESSION_DURATION_MAX_MS = 5000.0

        // Native capture — captureLivenessSession
        private const val REQUEST_LIVENESS_SESSION = 4321
        private const val REQUEST_CAMERA_PERMISSION = 4320
        private const val NATIVE_FRAME_COUNT = 12
        private const val NATIVE_FRAME_INTERVAL_MS = 120L
        private const val NATIVE_WARMUP_MS = 1200L
    }

    private lateinit var channel: MethodChannel
    private lateinit var applicationContext: Context
    private lateinit var flutterAssets: FlutterPlugin.FlutterAssets

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // TFLite interpreters — TIDAK thread-safe → akses wajib melalui lock masing-masing
    private val faceInterpreterLock = Any()
    private val antiSpoof27Lock = Any()
    private val antiSpoof40Lock = Any()
    private var tfliteInterpreter: Interpreter? = null
    private var antiSpoof27: Interpreter? = null
    private var antiSpoof40: Interpreter? = null

    // ML Kit detectors
    private var faceDetector: FaceDetector? = null
    private var livenessDetector: FaceDetector? = null

    // Activity plumbing (captureLivenessSession)
    private var activityBinding: ActivityPluginBinding? = null
    private var pendingCapture: PendingCapture? = null

    // Flag inisialisasi (initialize) — penjaga jalur frame dari JS (verifyFaceSequence)
    private var enableInsecureFramePath = false

    // Challenge session aktif (acak, single-use, expiry)
    private val challengeLock = Any()
    private var activeChallenge: ChallengeSession? = null

    // ============================
    // Flutter plumbing
    // ============================

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

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activityBinding = binding
        binding.addActivityResultListener(this)
        binding.addRequestPermissionsResultListener(this)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        detachActivity()
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        onAttachedToActivity(binding)
    }

    override fun onDetachedFromActivity() {
        detachActivity()
    }

    private fun detachActivity() {
        activityBinding?.removeActivityResultListener(this)
        activityBinding?.removeRequestPermissionsResultListener(this)
        activityBinding = null
    }

    // ============================
    // Method channel dispatch
    // ============================

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initialize" -> {
                enableInsecureFramePath = call.argument<Boolean>("enableInsecureFramePath") ?: false
                result.success(null)
            }

            "dispose" -> {
                releaseResources()
                result.success(null)
            }

            "createChallengeSession" -> {
                try {
                    if (!checkRateLimit()) {
                        throw FaceRecognitionException(
                            "RATE_LIMITED",
                            "Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit."
                        )
                    }
                    val session = issueChallengeSession()
                    result.success(
                        mapOf(
                            "challengeId" to session.id,
                            "challenge" to session.challenge,
                            "expiresAt" to session.expiresAt.toDouble(),
                        )
                    )
                } catch (e: FaceRecognitionException) {
                    result.error(e.code, e.message, null)
                }
            }

            "captureLivenessSession" -> handleCaptureLivenessSession(call, result)

            "extractFaceFeature", "checkLiveness", "detectFaces", "verifyFace",
            "verifyFaceSequence" -> {
                scope.launch {
                    try {
                        val response = when (call.method) {
                            "extractFaceFeature" -> handleExtractFaceFeature(call)
                            "checkLiveness" -> handleCheckLiveness(call)
                            "detectFaces" -> handleDetectFaces(call)
                            "verifyFace" -> handleVerifyFace(call)
                            else -> handleVerifyFaceSequence(call)
                        }
                        withContext(Dispatchers.Main) { result.success(response) }
                    } catch (e: FaceRecognitionException) {
                        withContext(Dispatchers.Main) { result.error(e.code, e.message, null) }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            result.error("INTERNAL_ERROR", e.message ?: "Unknown error", null)
                        }
                    }
                }
            }

            else -> result.notImplemented()
        }
    }

    // ============================
    // Core Method Handlers
    // ============================

    private suspend fun handleExtractFaceFeature(call: MethodCall): Map<String, Any?> {
        if (getOrCreateFaceInterpreter() == null) {
            throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Failed to load TFLite model '$MODEL_FILE' from Flutter assets."
            )
        }
        val bitmap = decodeBitmapFromCall(call)

        val faces = getOrCreateLivenessDetector().process(InputImage.fromBitmap(bitmap, 0)).await()
        if (faces.isEmpty()) {
            throw FaceRecognitionException("NO_FACE_DETECTED", "Tidak ada wajah yang terdeteksi")
        }

        val bounds = pickLargestFace(faces).boundingBox
        val embedding = extractEmbedding(bitmap, bounds)
            ?: throw FaceRecognitionException(
                "EMPTY_EMBEDDING",
                "Gagal menjalankan inference face recognition"
            )
        return mapOf("embedding" to l2Normalize(embedding))
    }

    private suspend fun handleCheckLiveness(call: MethodCall): Map<String, Any?> {
        if (getOrCreateAntiSpoof27() == null || getOrCreateAntiSpoof40() == null) {
            throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Model anti-spoofing tidak ditemukan. Letakkan file " +
                        "'$ANTI_SPOOF_MODEL_FILE_27' dan '$ANTI_SPOOF_MODEL_FILE_40' " +
                        "di folder assets."
            )
        }
        val bitmap = decodeBitmapFromCall(call)

        // threshold arg (jika ada) dipetakan ke liveThreshold untuk keputusan
        val threshold = call.argument<Number>("threshold")?.toFloat() ?: LIVE_THRESHOLD
        val config = GateConfig.from(null)
        config.liveThreshold = threshold

        val faces = getOrCreateLivenessDetector().process(InputImage.fromBitmap(bitmap, 0)).await()
        if (faces.isEmpty()) {
            throw FaceRecognitionException(
                "NO_FACE_DETECTED",
                "Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness"
            )
        }

        val bounds = pickLargestFace(faces).boundingBox
        val result = runAntiSpoofPipeline(bitmap, bounds, config)

        // Tingkat kepercayaan (kompatibel dengan response lama)
        val confidence = when {
            result.liveScore > 0.85f || result.liveScore < 0.15f -> "HIGH"
            result.liveScore > 0.6f || result.liveScore < 0.4f -> "MEDIUM"
            else -> "LOW"
        }

        // Hasil FLAT sesuai kontrak (score == liveScore, alias backward-compat)
        return mapOf(
            "isLive" to result.isLive,
            "score" to result.liveScore.toDouble(),
            "confidence" to confidence,
            "label" to result.label,
            "liveScore" to result.liveScore.toDouble(),
            "printScore" to result.printScore.toDouble(),
            "replayScore" to result.replayScore.toDouble(),
            "spoofScore" to result.spoofScore.toDouble(),
            "screenSuspectScore" to result.screenSuspectScore.toDouble(),
            "thresholdVersion" to THRESHOLD_VERSION,
        )
    }

    private suspend fun handleDetectFaces(call: MethodCall): Map<String, Any?> {
        val bitmap = decodeBitmapFromCall(call)

        val faces = getOrCreateFaceDetector().process(InputImage.fromBitmap(bitmap, 0)).await()

        val facesList = faces.map { face ->
            val bounds = face.boundingBox
            val x = max(0, bounds.left)
            val y = max(0, bounds.top)
            val width = min(bounds.width(), bitmap.width - x)
            val height = min(bounds.height(), bitmap.height - y)

            mapOf<String, Any?>(
                "x" to x,
                "y" to y,
                "width" to max(0, width),
                "height" to max(0, height),
                // Head euler angles (derajat, sesuai dokumentasi ML Kit)
                "headEulerAngleX" to face.headEulerAngleX.toDouble(), // pitch (atas-bawah)
                "headEulerAngleY" to YawConvention.normalize(face.headEulerAngleY)
                    .toDouble(), // yaw ternormalisasi (positif = kanan anatomis)
                "headEulerAngleZ" to face.headEulerAngleZ.toDouble(), // roll (miring)
                // Eye open probability (null jika tidak bisa dihitung)
                "leftEyeOpenProbability" to face.leftEyeOpenProbability?.toDouble(),
                "rightEyeOpenProbability" to face.rightEyeOpenProbability?.toDouble(),
            )
        }

        return mapOf(
            "count" to faces.size,
            "faces" to facesList,
        )
    }

    private suspend fun handleVerifyFace(call: MethodCall): Map<String, Any?> {
        if (getOrCreateAntiSpoof27() == null || getOrCreateAntiSpoof40() == null ||
            getOrCreateFaceInterpreter() == null
        ) {
            throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Model liveness atau face recognition tidak ditemukan di assets"
            )
        }
        val bitmap = decodeBitmapFromCall(call)
        val config = GateConfig.from(
            call.argument<Map<String, Any?>>("config"),
            DeviceIntegrity.isDebuggable(applicationContext)
        )
        val referenceEmbedding =
            call.argument<List<*>>("referenceEmbedding")?.mapNotNull { (it as? Number)?.toDouble() }

        val faces = getOrCreateLivenessDetector().process(InputImage.fromBitmap(bitmap, 0)).await()

        // ---- Quality gate (Fase 3.1) ----
        val quality = runQualityGate(bitmap, faces, config)
        if (!quality.passed) {
            return mutableMapOf<String, Any?>(
                "isLive" to false,
                "thresholdVersion" to THRESHOLD_VERSION,
                "liveness" to null,
                "quality" to quality.toMap(),
            )
        }

        val bounds = pickLargestFace(faces).boundingBox

        // ---- Anti-spoof (Fase 2) ----
        val liveness = runAntiSpoofPipeline(bitmap, bounds, config)

        val ret = mutableMapOf<String, Any?>(
            "isLive" to liveness.isLive,
            "thresholdVersion" to THRESHOLD_VERSION,
            "liveness" to liveness.toMap(),
            "quality" to quality.toMap(),
        )
        if (config.debug) {
            ret["debug"] = liveness.debugToMap()
        }

        // ---- Embedding HANYA jika live, dari frame & bbox yang sama ----
        if (liveness.isLive) {
            val embedding = extractEmbedding(bitmap, bounds)
                ?: throw FaceRecognitionException(
                    "EMPTY_EMBEDDING",
                    "Gagal mengekstrak embedding dari frame yang sama"
                )
            val embeddingList = l2Normalize(embedding)
            ret["embedding"] = embeddingList

            if (referenceEmbedding != null) {
                val score = cosineSimilarity(referenceEmbedding, embeddingList)
                if (score.isNaN()) {
                    throw FaceRecognitionException(
                        "EMPTY_EMBEDDING",
                        "referenceEmbedding tidak valid (norma nol)"
                    )
                }
                ret["match"] = matchToMap(score)
            }
        }
        return ret
    }

    private suspend fun handleVerifyFaceSequence(call: MethodCall): Map<String, Any?> {
        if (getOrCreateAntiSpoof27() == null || getOrCreateAntiSpoof40() == null ||
            getOrCreateFaceInterpreter() == null
        ) {
            throw FaceRecognitionException(
                "MODEL_NOT_LOADED",
                "Model liveness atau face recognition tidak ditemukan di assets"
            )
        }
        // P2.4: jalur frame base64 dari JS = celah injeksi frame — hanya untuk
        // build debug ATAU saat app meng-enable flag `enableInsecureFramePath`.
        if (!isInsecureFramePathAllowed()) {
            throw FaceRecognitionException(
                "INSECURE_FRAME_PATH_DISABLED",
                "verifyFaceSequence (frame base64 dari JS) hanya untuk build debug atau " +
                        "app dengan flag enableInsecureFramePath — gunakan captureLivenessSession di produksi"
            )
        }

        val framesArg = call.argument<List<*>>("frames")
        val imagesArg = call.argument<List<*>>("images")

        val frameBytes: List<ByteArray> = when {
            framesArg != null -> framesArg.mapIndexed { index, item ->
                item as? ByteArray ?: throw FaceRecognitionException(
                    "INVALID_IMAGE_INPUT",
                    "Frame ke-$index bukan data gambar yang valid"
                )
            }

            imagesArg != null -> imagesArg.mapIndexed { index, item ->
                val b64 = item as? String ?: throw FaceRecognitionException(
                    "INVALID_IMAGE_INPUT",
                    "Frame ke-$index bukan base64 yang valid"
                )
                cleanAndDecodeBase64(b64)
            }

            else -> emptyList()
        }
        if (frameBytes.size < 2) {
            throw FaceRecognitionException(
                "INVALID_IMAGE_INPUT",
                "images minimal berisi 2 frame (disarankan ≥ 5)"
            )
        }

        // Challenge dibaca dari session native (acak, single-use, expiry) —
        // BUKAN dari input JS. Percobaan pertama langsung mengonsumsi sesi.
        val challengeId = call.argument<String>("challengeId")
        val lookup = consumeChallengeSession(challengeId)
        when (lookup.status) {
            ChallengeLookup.Status.MISSING -> throw FaceRecognitionException(
                "CHALLENGE_NOT_FOUND",
                "challengeId tidak dikenal — panggil createChallengeSession dulu"
            )

            ChallengeLookup.Status.EXPIRED -> throw FaceRecognitionException(
                "CHALLENGE_EXPIRED",
                "Challenge session kedaluwarsa — buat challenge baru"
            )

            ChallengeLookup.Status.CONSUMED -> throw FaceRecognitionException(
                "CHALLENGE_CONSUMED",
                "Challenge session sudah dipakai — buat challenge baru"
            )

            ChallengeLookup.Status.OK -> {}
        }

        val config = GateConfig.from(
            call.argument<Map<String, Any?>>("config"),
            DeviceIntegrity.isDebuggable(applicationContext)
        )
        val referenceEmbedding =
            call.argument<List<*>>("referenceEmbedding")?.mapNotNull { (it as? Number)?.toDouble() }

        // Proses frame SEKUENSIAL, lalu agregasi (finishSequence).
        val frames = processFramesSequentially(frameBytes, null, config)
        return finishSequence(
            frames, config, lookup.challenge ?: "",
            referenceEmbedding, challengeId, false
        )
    }

    // ============================
    // Native capture (Fase 4) — captureLivenessSession
    // ============================

    private class PendingCapture(
        val result: Result,
        val referenceEmbedding: List<Double>?,
        val config: Map<String, Any?>?,
    )

    /**
     * Sesi liveness native (anti injeksi frame): kamera dibuka oleh activity plugin,
     * frame diambil dan disimpan ke file privat TANPA melewati JS, lalu pipeline
     * verifikasi dijalankan terhadap frame-file tersebut. Yang kembali ke JS hanya
     * HASIL — bukan frame mentah.
     */
    private fun handleCaptureLivenessSession(call: MethodCall, result: Result) {
        if (getOrCreateAntiSpoof27() == null || getOrCreateAntiSpoof40() == null ||
            getOrCreateFaceInterpreter() == null
        ) {
            result.error(
                "MODEL_NOT_LOADED",
                "Model liveness atau face recognition tidak ditemukan di assets", null
            )
            return
        }
        if (!checkRateLimit()) {
            result.error(
                "RATE_LIMITED",
                "Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit.", null
            )
            return
        }

        val activity = activityBinding?.activity
        if (activity == null) {
            result.error(
                "CAPTURE_FAILED",
                "captureLivenessSession membutuhkan Activity foreground yang terpasang", null
            )
            return
        }

        val pending = PendingCapture(
            result,
            call.argument<List<*>>("referenceEmbedding")?.mapNotNull { (it as? Number)?.toDouble() },
            call.argument<Map<String, Any?>>("config"),
        )

        if (ContextCompat.checkSelfPermission(
                activity, Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingCapture = pending
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
            return
        }
        startLivenessSession(pending)
    }

    private fun startLivenessSession(pending: PendingCapture) {
        val binding = activityBinding
        if (binding == null) {
            pending.result.error("CAPTURE_FAILED", "Activity tidak tersedia", null)
            return
        }
        // Challenge SELALU acak dari native — menutup brute-force
        // "ulangi sesi sampai challenge cocok dengan video yang disiapkan".
        val challenge =
            if (Math.random() < 0.5) LivenessCaptureActivity.CHALLENGE_LOOK_LEFT
            else LivenessCaptureActivity.CHALLENGE_LOOK_RIGHT

        pendingCapture = pending
        val intent = LivenessCaptureActivity.createIntent(
            applicationContext, challenge,
            NATIVE_FRAME_COUNT, NATIVE_FRAME_INTERVAL_MS, NATIVE_WARMUP_MS
        )
        try {
            binding.activity.startActivityForResult(intent, REQUEST_LIVENESS_SESSION)
        } catch (e: Exception) {
            pendingCapture = null
            pending.result.error("CAPTURE_FAILED", "Gagal membuka kamera: ${e.message}", null)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ): Boolean {
        if (requestCode != REQUEST_CAMERA_PERMISSION) return false
        val pending = pendingCapture ?: return true
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startLivenessSession(pending)
        } else {
            pendingCapture = null
            pending.result.error("CAMERA_PERMISSION_DENIED", "Izin kamera ditolak", null)
        }
        return true
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST_LIVENESS_SESSION) return false
        val pending = pendingCapture ?: return true
        pendingCapture = null

        if (resultCode != Activity.RESULT_OK || data == null) {
            val error = data?.getStringExtra(LivenessCaptureActivity.EXTRA_ERROR)
            pending.result.error("CAPTURE_FAILED", error ?: "Capture dibatalkan", null)
            return true
        }

        val challenge = data.getStringExtra(LivenessCaptureActivity.EXTRA_CHALLENGE)
        val framePaths = data.getStringArrayListExtra(LivenessCaptureActivity.EXTRA_FRAME_PATHS)
        val frameTimestamps = data.getDoubleArrayExtra(LivenessCaptureActivity.EXTRA_FRAME_TIMESTAMPS)

        if (framePaths.isNullOrEmpty()) {
            pending.result.error("CAPTURE_FAILED", "Tidak ada frame yang tertangkap", null)
            return true
        }

        scope.launch {
            try {
                val frameBytes = framePaths.map { readFileBytes(File(it)) }
                val config = GateConfig.from(
                    pending.config,
                    DeviceIntegrity.isDebuggable(applicationContext)
                )
                val frames = processFramesSequentially(frameBytes, frameTimestamps, config)
                val resultMap = finishSequence(
                    frames, config, challenge ?: "",
                    pending.referenceEmbedding, null, true
                )

                // Sink khusus sesi native: integrity + jejak audit terenkripsi.
                resultMap["deviceIntegrity"] =
                    jsonObjectToMap(DeviceIntegrity.snapshot(applicationContext))
                val auditJson = toJsonString(resultMap)
                resultMap["auditId"] = try {
                    AuditStore.storeSession(applicationContext, auditJson, frameBytes)
                } catch (e: Exception) {
                    Log.e(TAG, "Audit store gagal: ${e.message}")
                    null
                }
                deleteFrameFiles()
                withContext(Dispatchers.Main) { pending.result.success(resultMap) }
            } catch (e: Exception) {
                deleteFrameFiles()
                withContext(Dispatchers.Main) {
                    if (e is FaceRecognitionException) {
                        pending.result.error(e.code, e.message, null)
                    } else {
                        pending.result.error(
                            "CAPTURE_FAILED",
                            "Gagal memproses frame capture: ${e.message}", null
                        )
                    }
                }
            }
        }
        return true
    }

    private fun deleteFrameFiles() {
        val dir = File(applicationContext.cacheDir, "liveness_frames")
        dir.listFiles()?.forEach { it.delete() }
    }

    @Throws(IOException::class)
    private fun readFileBytes(file: File): ByteArray {
        FileInputStream(file).use { input ->
            val bytes = ByteArray(file.length().toInt())
            var offset = 0
            while (offset < bytes.size) {
                val read = input.read(bytes, offset, bytes.size - offset)
                if (read < 0) break
                offset += read
            }
            return bytes
        }
    }

    // ============================
    // Challenge session (acak, single-use, expiry)
    // ============================

    /** Challenge session aktif: acak, single-use, dengan expiry. */
    private class ChallengeSession(
        val id: String,
        val challenge: String,
        val expiresAt: Long,
    ) {
        var consumed = false
    }

    /** Hasil lookup challenge session — status membedakan alasan reject. */
    private class ChallengeLookup private constructor(
        val status: Status,
        val challenge: String?,
    ) {
        enum class Status { OK, MISSING, EXPIRED, CONSUMED }

        companion object {
            fun ok(challenge: String) = ChallengeLookup(Status.OK, challenge)
            fun missing() = ChallengeLookup(Status.MISSING, null)
            fun expired() = ChallengeLookup(Status.EXPIRED, null)
            fun consumed() = ChallengeLookup(Status.CONSUMED, null)
        }
    }

    /** Membuat challenge session baru (acak) — menggantikan yang lama. */
    private fun issueChallengeSession(): ChallengeSession {
        synchronized(challengeLock) {
            val challenge =
                if (Math.random() < 0.5) LivenessCaptureActivity.CHALLENGE_LOOK_LEFT
                else LivenessCaptureActivity.CHALLENGE_LOOK_RIGHT
            val session = ChallengeSession(
                UUID.randomUUID().toString(), challenge,
                System.currentTimeMillis() + CHALLENGE_TTL_MS
            )
            activeChallenge = session
            return session
        }
    }

    /**
     * Mengambil SEKALIGUS mengonsumsi challenge session untuk [id].
     * Percobaan pertama (gagal maupun lulus) memakai sesi — brute-force retry
     * dengan challenge yang sama tidak mungkin.
     */
    private fun consumeChallengeSession(id: String?): ChallengeLookup {
        synchronized(challengeLock) {
            val current = activeChallenge
            if (id.isNullOrEmpty() || current == null || current.id != id) {
                return ChallengeLookup.missing()
            }
            val now = System.currentTimeMillis()
            if (now > current.expiresAt) return ChallengeLookup.expired()
            if (current.consumed) return ChallengeLookup.consumed()
            current.consumed = true
            return ChallengeLookup.ok(current.challenge)
        }
    }

    /** Rate limiting percobaan sesi (anti brute-force). */
    private fun checkRateLimit(): Boolean {
        val now = System.currentTimeMillis()
        synchronized(sessionAttempts) {
            while (sessionAttempts.isNotEmpty() &&
                now - sessionAttempts[0] > SESSION_ATTEMPT_WINDOW_MS
            ) {
                sessionAttempts.removeAt(0)
            }
            if (sessionAttempts.size >= MAX_SESSION_ATTEMPTS) return false
            sessionAttempts.add(now)
            return true
        }
    }

    /**
     * P2.4: jalur frame base64 dari JS (verifyFaceSequence) adalah celah injeksi
     * frame — hanya untuk build DEBUG, ATAU saat app secara eksplisit meng-enable
     * flag inisialisasi `enableInsecureFramePath`.
     */
    private fun isInsecureFramePathAllowed(): Boolean {
        return DeviceIntegrity.isDebuggable(applicationContext) || enableInsecureFramePath
    }

    // ============================
    // Sequential frame processing + agregasi sesi
    // ============================

    private class FrameResult {
        var index = 0
        var bounds: Rect? = null
        var yaw = 0f
        var pitch = 0f
        var roll = 0f
        /** Epoch ms saat frame di-capture (native capture); -1 untuk jalur JS base64. */
        var timestampMs = -1.0
        var quality: QualityResult? = null
        var liveness: AntiSpoofResult? = null
        var embedding: FloatArray? = null
    }

    /** Proses frame SEKUENSIAL (aman dari deadlock, ML Kit via .await()). */
    private suspend fun processFramesSequentially(
        frameBytes: List<ByteArray>,
        timestamps: DoubleArray?,
        config: GateConfig,
    ): List<FrameResult> {
        val frames = mutableListOf<FrameResult>()
        for (index in frameBytes.indices) {
            val bytes = frameBytes[index]
            val bitmap = try {
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?.let { applyExifOrientation(it, bytes) }
            } catch (_: Exception) {
                null
            } ?: throw FaceRecognitionException(
                "IMAGE_DECODE_FAILED",
                "Frame ke-$index gagal di-decode"
            )

            val fr = FrameResult()
            fr.index = index
            if (timestamps != null && index < timestamps.size) {
                fr.timestampMs = timestamps[index]
            }

            val faces = try {
                getOrCreateLivenessDetector().process(InputImage.fromBitmap(bitmap, 0)).await()
            } catch (e: Exception) {
                fr.quality = QualityResult.fail("NO_FACE", 0f, 0f)
                frames.add(fr)
                continue
            }

            if (faces.isEmpty()) {
                fr.quality = QualityResult.fail("NO_FACE", 0f, 0f)
            } else {
                fr.quality = runQualityGate(bitmap, faces, config)
                if (fr.quality?.passed == true) {
                    val face = pickLargestFace(faces)
                    val bounds = face.boundingBox
                    fr.bounds = bounds
                    fr.yaw = YawConvention.normalize(face.headEulerAngleY)
                    fr.pitch = face.headEulerAngleX
                    fr.roll = face.headEulerAngleZ
                    fr.liveness = runAntiSpoofPipeline(bitmap, bounds, config)
                    fr.embedding = extractEmbedding(bitmap, bounds)
                }
            }
            frames.add(fr)
        }
        return frames
    }

    /**
     * Agregasi sesi (median + voting, BUKAN all-live) + veto anti-replay +
     * lintasan challenge (R1–R5). Port dari FaceRecognitionPlugin.finishSequence.
     */
    private fun finishSequence(
        frames: List<FrameResult>,
        config: GateConfig,
        challenge: String,
        referenceEmbedding: List<Double>?,
        challengeId: String?,
        nativeCapture: Boolean,
    ): MutableMap<String, Any?> {
        var failureReason: String? = null

        // SEMUA frame harus lolos quality gate
        for (fr in frames) {
            if (fr.quality?.passed != true) {
                failureReason = "QUALITY_FAILED"
                break
            }
        }

        // Referensi identitas = frame PALING FRONTAL (|yaw| terkecil), bukan frame
        // pertama — embedding paling frontal paling stabil untuk pembanding.
        var refFrame: FrameResult? = null
        if (failureReason == null) {
            for (fr in frames) {
                if (fr.embedding == null) continue
                val current = refFrame
                if (current == null || abs(fr.yaw) < abs(current.yaw)) {
                    refFrame = fr
                }
            }
        }
        val refEmbedding = refFrame?.embedding

        // Konsistensi identitas: hanya bandingkan frame "sebanding" (|yaw| <=
        // IDENTITY_MAX_POSE). Frame dengan pose berat tugasnya membuktikan challenge;
        // embedding-nya tidak pose-invariant dan tidak boleh mematikan sesi.
        var identityConsistent = true
        var minIdentityCosine: Double? = null
        val comparableCosines = mutableListOf<Double>()
        if (failureReason == null && refEmbedding != null) {
            for (fr in frames) {
                val emb = fr.embedding ?: continue
                if (fr === refFrame) continue
                if (abs(fr.yaw) > IDENTITY_MAX_POSE) continue
                val cos = cosineSimilarity(refEmbedding, emb)
                comparableCosines.add(cos)
                if (minIdentityCosine == null || cos < minIdentityCosine) {
                    minIdentityCosine = cos
                }
            }
            if (comparableCosines.isNotEmpty()) {
                val avgCosine = comparableCosines.sum() / comparableCosines.size
                val minCos = minIdentityCosine ?: Double.NaN
                // Gagal jika ada frame yang sangat jauh berbeda (< 0.55, indikasi swap
                // wajah / orang lain), ATAU rata-rata kemiripan frame di bawah threshold.
                if (minCos < 0.55 || avgCosine < config.identityThreshold) {
                    identityConsistent = false
                }
            }
        }

        // ---- Agregasi (median + voting, BUKAN all-live) ----
        val liveScores = FloatArray(frames.size)
        var liveVotes = 0
        val yawSamples = mutableListOf<ChallengeTrajectory.Sample>()
        for (i in frames.indices) {
            val fr = frames[i]
            liveScores[i] = fr.liveness?.liveScore ?: 0f
            if (fr.liveness?.isLive == true) liveVotes++
            if (fr.quality?.passed == true) {
                // Yaw sudah ternormalisasi di sumber (YawConvention): positif =
                // menoleh ke kanan anatomis user. yawSignOverride hanya untuk
                // perangkat dengan buffer kamera ter-flip.
                yawSamples.add(
                    ChallengeTrajectory.Sample(
                        YawConvention.applyOverride(fr.yaw, config.yawSignOverride),
                        fr.timestampMs
                    )
                )
            }
        }

        var maxYaw = Float.NEGATIVE_INFINITY
        var minYaw = Float.POSITIVE_INFINITY
        var frontalFrames = 0
        for (s in yawSamples) {
            maxYaw = max(maxYaw, s.yaw)
            minYaw = min(minYaw, s.yaw)
            if (abs(s.yaw) < CHALLENGE_FRONT_MAX) frontalFrames++
        }
        if (yawSamples.isEmpty()) {
            maxYaw = 0f
            minYaw = 0f
        }

        val medianLiveScore = median(liveScores)
        // P2.1: voting sesi diperketat — default SEMUA frame harus isLive.
        val requiredVotes = ceil(frames.size * config.minLiveVoteRatio.toDouble()).toInt()

        // Challenge (P1): verifikasi LINTASAN TEMPORAL, bukan min/max agregat —
        // frontal di awal → deviasi ≥ 15° ke arah challenge → kembali frontal,
        // tanpa gerakan dua arah (head-shake), dengan gerakan halus.
        // Lihat ChallengeTrajectory untuk aturan R1–R5.
        val trajectory = ChallengeTrajectory.evaluate(yawSamples, challenge)
        val challengeSatisfied = trajectory.passed

        // ---- Anti-replay (P2) ----
        // P2.1: veto per-frame — satu frame dengan label spoof (PRINT/REPLAY)
        // atau spoofScore >= SPOOF_HARD_VETO langsung menggagalkan sesi,
        // TANPA toleransi voting.
        var spoofFrameDetected = false
        var maxScreenSuspect = 0f
        for (fr in frames) {
            val liveness = fr.liveness ?: continue
            if (liveness.label != "LIVE" || liveness.spoofScore >= SPOOF_HARD_VETO) {
                spoofFrameDetected = true
            }
            maxScreenSuspect = max(maxScreenSuspect, liveness.screenSuspectScore)
        }
        // P2.2: veto screen-suspect hanya jika di-set eksplisit (mode ketat opt-in);
        // default hanya dicatat sebagai soft signal untuk pengumpulan data.
        val screenVeto = config.screenSuspectVeto
        val screenVetoTriggered = screenVeto != null && maxScreenSuspect >= screenVeto
        // P2.3: durasi sesi harus masuk rentang gerakan manusia
        // (hanya diperiksa jika timestamp diketahui — jalur native capture).
        val timingOk = trajectory.durationMs <= 0 ||
                (trajectory.durationMs >= SESSION_DURATION_MIN_MS
                        && trajectory.durationMs <= SESSION_DURATION_MAX_MS)

        val passed = failureReason == null
                && !spoofFrameDetected
                && !screenVetoTriggered
                && timingOk
                && medianLiveScore >= config.liveThreshold
                && liveVotes >= requiredVotes
                && identityConsistent
                && challengeSatisfied

        var reason = failureReason
        if (reason == null && !passed) {
            // Prioritas alasan gagal: spoof > screen > timing > challenge > votes > identitas
            reason = when {
                spoofFrameDetected -> "SPOOF_FRAME_DETECTED"
                screenVetoTriggered -> "SCREEN_SUSPECTED"
                !timingOk -> "TIMING_SUSPECT"
                !challengeSatisfied -> trajectory.failureReason ?: "CHALLENGE_FAILED"
                liveVotes < requiredVotes || medianLiveScore < config.liveThreshold -> "NOT_LIVE_ENOUGH"
                !identityConsistent -> "IDENTITY_INCONSISTENT"
                else -> null
            }
        }

        val ret = mutableMapOf<String, Any?>(
            "passed" to passed,
            "challenge" to challenge,
            "medianLiveScore" to medianLiveScore.toDouble(),
            "liveVotes" to liveVotes,
            "frameCount" to frames.size,
            "maxYaw" to maxYaw.toDouble(),
            "minYaw" to minYaw.toDouble(),
            "frontalFrames" to frontalFrames,
            "identityConsistent" to identityConsistent,
            // Diagnostik kalibrasi: cosine minimum antar frame sebanding (null jika tak ada)
            "minIdentityCosine" to minIdentityCosine,
            // Ringkasan lintasan challenge (P1): R1–R5 + statistik yaw
            "trajectory" to trajectoryToMap(trajectory),
            // Sinyal artefak layar agregat (P2.2) — soft signal kecuali screenSuspectVeto
            "screenSuspect" to maxScreenSuspect.toDouble(),
            "thresholdVersion" to THRESHOLD_VERSION,
            "challengeId" to challengeId,
            "native" to nativeCapture,
        )
        if (reason != null) ret["reason"] = reason

        // Verifikasi opsional terhadap referenceEmbedding (frame paling frontal)
        if (passed && referenceEmbedding != null && refEmbedding != null) {
            val score = cosineSimilarity(referenceEmbedding, l2Normalize(refEmbedding))
            if (score.isNaN()) {
                throw FaceRecognitionException(
                    "EMPTY_EMBEDDING",
                    "referenceEmbedding tidak valid (norma nol)"
                )
            }
            // match dilaporkan; tidak menggagalkan sesi (keputusan ada di aplikasi)
            ret["match"] = matchToMap(score)
        }

        if (config.debug) {
            ret["yawTrace"] = frames.map { fr ->
                val entry = mutableMapOf<String, Any?>(
                    "index" to fr.index,
                    "yaw" to YawConvention.applyOverride(fr.yaw, config.yawSignOverride)
                        .toDouble(),
                )
                if (fr.timestampMs >= 0) entry["timestampMs"] = fr.timestampMs
                entry
            }
        }
        return ret
    }

    private fun trajectoryToMap(t: ChallengeTrajectory.Result): Map<String, Any?> {
        val rules = mapOf(
            "frontalStart" to t.frontalStart,
            "deviation" to t.deviation,
            "returnFrontal" to t.returnFrontal,
            "singleDirection" to t.singleDirection,
            "smoothMotion" to t.smoothMotion,
        )
        return mapOf(
            "startYaw" to t.startYaw.toDouble(),
            "endYaw" to t.endYaw.toDouble(),
            "extremeYaw" to t.extremeYaw.toDouble(),
            "extremeFrame" to t.extremeFrame,
            "wrongWayMax" to t.wrongWayObserved.toDouble(),
            "deltaVariance" to t.deltaVariance.toDouble(),
            "tooSmooth" to t.tooSmooth,
            "durationMs" to t.durationMs,
            "rulesPassed" to rules,
        )
    }

    private fun matchToMap(score: Double): Map<String, Any?> = mapOf(
        "isMatch" to (score > MATCH_THRESHOLD),
        "score" to score,
        "threshold" to MATCH_THRESHOLD,
        "thresholdVersion" to MATCH_THRESHOLD_VERSION,
    )

    // ============================
    // Quality Gate (Fase 3.1)
    // ============================

    private class QualityResult private constructor(
        val passed: Boolean,
        val reason: String?,
        val faceRatio: Float,
        val blur: Float,
    ) {
        fun toMap(): Map<String, Any?> {
            val map = mutableMapOf<String, Any?>(
                "passed" to passed,
                "faceRatio" to faceRatio.toDouble(),
                "blur" to blur.toDouble(),
            )
            if (reason != null) map["reason"] = reason
            return map
        }

        companion object {
            fun pass(faceRatio: Float, blur: Float) = QualityResult(true, null, faceRatio, blur)

            fun fail(reason: String, faceRatio: Float, blur: Float) =
                QualityResult(false, reason, faceRatio, blur)
        }
    }

    private fun runQualityGate(bitmap: Bitmap, faces: List<Face>, config: GateConfig): QualityResult {
        val frameW = bitmap.width
        val frameH = bitmap.height

        // MULTIPLE_FACES: tolak jika ada > 1 wajah yang cukup besar
        val largest = pickLargestFace(faces)
        val largestArea = max(1, largest.boundingBox.width() * largest.boundingBox.height())
        var significantFaces = 0
        for (f in faces) {
            val area = f.boundingBox.width() * f.boundingBox.height()
            if (area >= largestArea * 0.3f) significantFaces++
        }
        if (significantFaces > 1) {
            return QualityResult.fail("MULTIPLE_FACES", 0f, 0f)
        }

        // LOW_RESOLUTION
        if (min(frameW, frameH) < config.minFrameSide) {
            return QualityResult.fail("LOW_RESOLUTION", 0f, 0f)
        }

        // FACE_TOO_SMALL
        val bounds = largest.boundingBox
        val faceRatio = bounds.width() / frameW.toFloat()
        if (faceRatio < config.minFaceRatio) {
            return QualityResult.fail("FACE_TOO_SMALL", faceRatio, 0f)
        }

        // FACE_AT_EDGE: bbox menyentuh tepi frame (margin 5%)
        val marginX = frameW * config.edgeMargin
        val marginY = frameH * config.edgeMargin
        if (bounds.left <= marginX || bounds.top <= marginY
            || bounds.right >= frameW - marginX || bounds.bottom >= frameH - marginY
        ) {
            return QualityResult.fail("FACE_AT_EDGE", faceRatio, 0f)
        }

        // BAD_POSE
        val yaw = largest.headEulerAngleY
        val pitch = largest.headEulerAngleX
        val roll = largest.headEulerAngleZ
        if (abs(yaw) > config.maxPose || abs(pitch) > config.maxPose || abs(roll) > config.maxPose) {
            return QualityResult.fail("BAD_POSE", faceRatio, 0f)
        }

        // BLURRY + BAD_LIGHTING dari crop wajah
        val faceCrop = cropFace(bitmap, bounds)
        val blur = varianceOfLaplacian(faceCrop)
        val exposure = meanLuminance(faceCrop)
        if (blur < config.blurMin) {
            return QualityResult.fail("BLURRY", faceRatio, blur)
        }
        if (exposure < config.exposureMin || exposure > config.exposureMax) {
            return QualityResult.fail("BAD_LIGHTING", faceRatio, blur)
        }

        return QualityResult.pass(faceRatio, blur)
    }

    private fun cropFace(bitmap: Bitmap, bounds: Rect): Bitmap {
        val left = max(0, bounds.left)
        val top = max(0, bounds.top)
        val width = min(bounds.width(), bitmap.width - left)
        val height = min(bounds.height(), bitmap.height - top)
        if (width <= 0 || height <= 0) return bitmap
        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    /** Variance of Laplacian pada grayscale — ukuran blur (semakin tinggi semakin tajam). */
    private fun varianceOfLaplacian(faceCrop: Bitmap): Float {
        val size = 160 // ukuran kerja tetap agar skor konsisten antar frame
        val work = Bitmap.createScaledBitmap(faceCrop, size, size, true)
        try {
            val pixels = IntArray(size * size)
            work.getPixels(pixels, 0, size, 0, 0, size, size)

            val gray = FloatArray(size * size)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
            }

            // Kernel Laplacian 3x3: [0 1 0; 1 -4 1; 0 1 0]
            var sum = 0.0
            var sumSq = 0.0
            var count = 0
            for (y in 1 until size - 1) {
                for (x in 1 until size - 1) {
                    val idx = y * size + x
                    val lap = gray[idx - size] + gray[idx + size] + gray[idx - 1] + gray[idx + 1] -
                            4f * gray[idx]
                    sum += lap
                    sumSq += lap * lap
                    count++
                }
            }
            if (count == 0) return 0f
            val mean = sum / count
            return (sumSq / count - mean * mean).toFloat()
        } finally {
            if (work !== faceCrop) work.recycle()
        }
    }

    /** Mean luminance crop wajah (0..255) — untuk gate exposure. */
    private fun meanLuminance(faceCrop: Bitmap): Float {
        val size = 64
        val work = Bitmap.createScaledBitmap(faceCrop, size, size, true)
        try {
            val pixels = IntArray(size * size)
            work.getPixels(pixels, 0, size, 0, 0, size, size)
            var sum = 0.0
            for (p in pixels) {
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                sum += 0.299 * r + 0.587 * g + 0.114 * b
            }
            return (sum / pixels.size).toFloat()
        } finally {
            if (work !== faceCrop) work.recycle()
        }
    }

    // ============================
    // Anti-Spoof Pipeline (Fase 2) — port referensi Silent-Face
    // ============================

    private class AntiSpoofResult {
        var label: String = "PRINT"
        var liveScore = 0f
        var printScore = 0f
        var replayScore = 0f
        var spoofScore = 0f
        /** Sinyal artefak layar per frame (P2.2) — soft signal, lihat ScreenSuspect. */
        var screenSuspectScore = 0f
        var isLive = false
        var probs27: FloatArray? = null
        var probs40: FloatArray? = null

        fun toMap(): Map<String, Any?> = mapOf(
            "isLive" to isLive,
            "label" to label,
            "liveScore" to liveScore.toDouble(),
            "printScore" to printScore.toDouble(),
            "replayScore" to replayScore.toDouble(),
            "spoofScore" to spoofScore.toDouble(),
            "screenSuspectScore" to screenSuspectScore.toDouble(),
        )

        fun debugToMap(): Map<String, Any?> = mapOf(
            "probs27" to (probs27?.map { it.toDouble() } ?: emptyList<Double>()),
            "probs40" to (probs40?.map { it.toDouble() } ?: emptyList<Double>()),
        )
    }

    private fun runAntiSpoofPipeline(bitmap: Bitmap, bounds: Rect): AntiSpoofResult =
        runAntiSpoofPipeline(bitmap, bounds, GateConfig.from(null))

    /**
     * Pipeline referensi:
     *   crop 2.7x (shift) → model 2.7 ─┐
     *                                    ├→ sum softmax → argmax + veto → keputusan
     *   crop 4.0x (shift) → model 4.0 ─┘
     */
    private fun runAntiSpoofPipeline(
        bitmap: Bitmap,
        bounds: Rect,
        config: GateConfig,
    ): AntiSpoofResult {
        val logits27 =
            runAntiSpoofModel(getOrCreateAntiSpoof27(), antiSpoof27Lock, bitmap, bounds, CROP_SCALE_27)
        val logits40 =
            runAntiSpoofModel(getOrCreateAntiSpoof40(), antiSpoof40Lock, bitmap, bounds, CROP_SCALE_40)
        if (logits27 == null || logits40 == null) {
            throw IllegalStateException("Inference anti-spoof gagal pada salah satu model")
        }

        val probs27 = softmax(logits27)
        val probs40 = softmax(logits40)

        // p = softmax(out27) + softmax(out40) → rentang 0..2
        val p = FloatArray(ANTI_SPOOF_OUTPUT_SIZE)
        for (i in 0 until ANTI_SPOOF_OUTPUT_SIZE) {
            p[i] = probs27[i] + probs40[i]
        }

        // argmax 3 kelas: [print, live, replay]
        var argmax = 0
        for (i in 1 until p.size) {
            if (p[i] > p[argmax]) argmax = i
        }

        val result = AntiSpoofResult()
        result.printScore = p[0] / 2f
        result.liveScore = p[1] / 2f
        result.replayScore = p[2] / 2f
        result.spoofScore = max(p[0], p[2]) / 2f
        result.label = if (argmax == 1) "LIVE" else if (argmax == 0) "PRINT" else "REPLAY"
        result.isLive = argmax == 1
                && result.liveScore >= config.liveThreshold
                && result.spoofScore < config.spoofVeto
        result.probs27 = probs27
        result.probs40 = probs40
        // Sinyal artefak layar per frame (P2.2) — soft signal untuk audit/tuning
        result.screenSuspectScore = analyzeScreenSuspect(bitmap, bounds)
        return result
    }

    /**
     * Menjalankan satu inferensi anti-spoofing untuk satu model dengan crop scale-nya.
     * Interpreter TFLite TIDAK thread-safe → akses disinkronkan per model.
     *
     * @return raw logits (3 kelas), atau null jika gagal
     */
    private fun runAntiSpoofModel(
        interpreter: Interpreter?,
        lock: Any,
        bitmap: Bitmap,
        bounds: Rect,
        cropScale: Float,
    ): FloatArray? {
        if (interpreter == null) return null
        return try {
            // Crop sesuai referensi (_get_new_box): scale diturunkan jika tidak muat,
            // w & h terpisah, SHIFT bukan clip, tanpa square-ifikasi
            val box = getNewBox(bounds, cropScale, bitmap.width, bitmap.height)
            val croppedFace = Bitmap.createBitmap(bitmap, box[0], box[1], box[2], box[3])

            // Resize ke 80x80 (input MiniFASNet)
            val scaledFace = Bitmap.createScaledBitmap(
                croppedFace, ANTI_SPOOF_INPUT_SIZE, ANTI_SPOOF_INPUT_SIZE, true
            )

            // Konversi ke ByteBuffer: BGR mentah 0..255 (TANPA normalisasi)
            val inputBuffer = bitmapToByteBufferAntiSpoof(scaledFace, ANTI_SPOOF_INPUT_SIZE)

            // Output model: [1, 3] → [print_spoof, live, replay_spoof] (raw logits)
            val outputArray = Array(1) { FloatArray(ANTI_SPOOF_OUTPUT_SIZE) }
            synchronized(lock) {
                interpreter.run(inputBuffer, outputArray)
            }
            outputArray[0]
        } catch (e: Exception) {
            Log.e(TAG, "Anti-spoof inference error (scale $cropScale): ${e.message}")
            null
        }
    }

    /**
     * Port logika crop referensi (_get_new_box):
     *   scale diturunkan jika crop tidak muat di gambar,
     *   lebar dan tinggi diskalakan TERPISAH (tanpa square),
     *   jika keluar batas maka DIGESER (shift), bukan dipotong (clip).
     */
    private fun getNewBox(bounds: Rect, scale: Float, imgW: Int, imgH: Int): IntArray {
        val boxW = bounds.width().toFloat()
        val boxH = bounds.height().toFloat()
        val cx = bounds.exactCenterX()
        val cy = bounds.exactCenterY()

        // Turunkan scale jika crop tidak muat
        val maxScaleW = (imgW - 1) / boxW
        val maxScaleH = (imgH - 1) / boxH
        val s = min(scale, min(maxScaleW, maxScaleH))

        val newW = boxW * s
        val newH = boxH * s

        var left = cx - newW / 2f
        var top = cy - newH / 2f
        var right = left + newW
        var bottom = top + newH

        // SHIFT, bukan clip
        if (left < 0) {
            right -= left
            left = 0f
        }
        if (top < 0) {
            bottom -= top
            top = 0f
        }
        if (right > imgW - 1) {
            left -= right - (imgW - 1)
            right = (imgW - 1).toFloat()
        }
        if (bottom > imgH - 1) {
            top -= bottom - (imgH - 1)
            bottom = (imgH - 1).toFloat()
        }

        // Penjagaan akhir untuk kasus ekstrem
        left = max(0f, left)
        top = max(0f, top)

        val x = left.roundToInt()
        val y = top.roundToInt()
        var w = min(imgW - x, (right - left).roundToInt())
        var h = min(imgH - y, (bottom - top).roundToInt())
        w = max(1, w)
        h = max(1, h)
        return intArrayOf(x, y, w, h)
    }

    /** Downsample crop wajah ke 32x32 lalu analisis artefak layar (P2.2). */
    private fun analyzeScreenSuspect(bitmap: Bitmap, bounds: Rect): Float {
        return try {
            val x = max(0, bounds.left)
            val y = max(0, bounds.top)
            val w = min(bitmap.width - x, bounds.width())
            val h = min(bitmap.height - y, bounds.height())
            if (w <= 0 || h <= 0) return 0f
            val crop = Bitmap.createBitmap(bitmap, x, y, w, h)
            val small = Bitmap.createScaledBitmap(crop, 32, 32, true)
            val pixels = IntArray(32 * 32)
            small.getPixels(pixels, 0, 32, 0, 0, 32, 32)
            if (small !== crop) small.recycle()
            if (crop !== bitmap) crop.recycle()
            ScreenSuspect.analyze(pixels, 32, 32).score
        } catch (_: Exception) {
            0f
        }
    }

    // ============================
    // Face Recognition helpers
    // ============================

    /**
     * Ekstrak embedding MobileFaceNet dari frame dan bbox yang diberikan.
     * Crop memakai logika yang SAMA dengan pipeline enrollment (bbox langsung,
     * di-clip ke batas gambar) agar embedding enroll dan verify konsisten.
     */
    private fun extractEmbedding(bitmap: Bitmap, bounds: Rect): FloatArray? {
        val interpreter = getOrCreateFaceInterpreter() ?: return null
        return try {
            val left = max(0, bounds.left)
            val top = max(0, bounds.top)
            val width = min(bounds.width(), bitmap.width - left)
            val height = min(bounds.height(), bitmap.height - top)
            if (width <= 0 || height <= 0) return null

            val croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height)
            val scaledFace = Bitmap.createScaledBitmap(croppedFace, INPUT_SIZE, INPUT_SIZE, true)
            val inputBuffer = bitmapToByteBuffer(scaledFace, INPUT_SIZE)

            val outputArray = Array(1) { FloatArray(EMBEDDING_SIZE) }
            synchronized(faceInterpreterLock) {
                interpreter.run(inputBuffer, outputArray)
            }
            outputArray[0]
        } catch (e: Exception) {
            Log.e(TAG, "extractEmbedding error: ${e.message}")
            null
        }
    }

    private fun pickLargestFace(faces: List<Face>): Face {
        var best = faces[0]
        var bestArea = best.boundingBox.width() * best.boundingBox.height()
        for (f in faces) {
            val area = f.boundingBox.width() * f.boundingBox.height()
            if (area > bestArea) {
                best = f
                bestArea = area
            }
        }
        return best
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        var norm1 = 0.0
        var norm2 = 0.0
        for (i in 0 until min(a.size, b.size)) {
            val v1 = a[i].toDouble()
            val v2 = b[i].toDouble()
            dot += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2
        }
        if (norm1 == 0.0 || norm2 == 0.0) return Double.NaN
        return dot / (sqrt(norm1) * sqrt(norm2))
    }

    private fun cosineSimilarity(a: List<Double>, b: List<Double>): Double {
        if (a.size != b.size) return Double.NaN
        var dot = 0.0
        var norm1 = 0.0
        var norm2 = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            norm1 += a[i] * a[i]
            norm2 += b[i] * b[i]
        }
        if (norm1 == 0.0 || norm2 == 0.0) return Double.NaN
        return dot / (sqrt(norm1) * sqrt(norm2))
    }

    // ============================
    // Konfigurasi quality gate yang bisa di-override per panggilan
    // ============================

    private class GateConfig {
        var liveThreshold: Float = LIVE_THRESHOLD
        var spoofVeto: Float = SPOOF_VETO
        var minFrameSide: Int = MIN_FRAME_SIDE
        var minFaceRatio: Float = MIN_FACE_RATIO
        var edgeMargin: Float = EDGE_MARGIN
        var maxPose: Float = MAX_POSE_DEG
        var blurMin: Float = BLUR_MIN
        var exposureMin: Float = EXPOSURE_MIN
        var exposureMax: Float = EXPOSURE_MAX
        var identityThreshold: Float = IDENTITY_CONSISTENCY

        /**
         * Override tanda yaw per-perangkat (1 atau -1) untuk kasus buffer kamera
         * ter-flip (mis. beberapa Samsung/OnePlus). Default 1 — konvensi normal
         * sudah ditangani YawConvention di sumber pengukuran.
         */
        var yawSignOverride: Int = 1

        /**
         * Rasio minimum frame yang harus isLive per sesi (P2.1). Default 1.0 =
         * SEMUA frame harus isLive (paling ketat).
         */
        var minLiveVoteRatio: Float = 1.0f

        /** Veto screen-suspect per sesi (P2.2) — null = soft signal saja (default). */
        var screenSuspectVeto: Float? = null

        /** Output debug hanya pada build debuggable — tidak pernah dari build rilis. */
        var debug: Boolean = false

        companion object {
            /**
             * @param debugAllowed hanya true pada build debuggable — output debug
             * tidak pernah dikembalikan dari build rilis (Fase 4)
             */
            fun from(config: Map<String, Any?>?, debugAllowed: Boolean = false): GateConfig {
                val c = GateConfig()
                c.debug = false
                if (config == null) return c
                c.liveThreshold = (config["liveThreshold"] as? Number)?.toFloat() ?: c.liveThreshold
                c.spoofVeto = (config["spoofVeto"] as? Number)?.toFloat() ?: c.spoofVeto
                c.minFrameSide = (config["minFrameSide"] as? Number)?.toInt() ?: MIN_FRAME_SIDE
                c.minFaceRatio = (config["minFaceRatio"] as? Number)?.toFloat() ?: c.minFaceRatio
                c.edgeMargin = (config["edgeMargin"] as? Number)?.toFloat() ?: c.edgeMargin
                c.maxPose = (config["maxPose"] as? Number)?.toFloat() ?: c.maxPose
                c.blurMin = (config["blurMin"] as? Number)?.toFloat() ?: c.blurMin
                c.exposureMin = (config["exposureMin"] as? Number)?.toFloat() ?: c.exposureMin
                c.exposureMax = (config["exposureMax"] as? Number)?.toFloat() ?: c.exposureMax
                c.identityThreshold =
                    (config["identityThreshold"] as? Number)?.toFloat() ?: c.identityThreshold
                c.yawSignOverride = (config["yawSignOverride"] as? Number)?.toInt() ?: 1
                c.minLiveVoteRatio = (config["minLiveVoteRatio"] as? Number)?.toFloat() ?: 1.0f
                if (config.containsKey("screenSuspectVeto")) {
                    c.screenSuspectVeto = (config["screenSuspectVeto"] as? Number)?.toFloat()
                }
                c.debug = (config["debug"] == true) && debugAllowed
                return c
            }
        }
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
            ?: throw FaceRecognitionException(
                "IMAGE_DECODE_FAILED",
                "Gagal men-decode data gambar menjadi Bitmap"
            )

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
    // Preprocessing & Math Helpers
    // ============================

    /**
     * Konversi Bitmap → ByteBuffer untuk input TFLite (face recognition).
     * Normalisasi ke rentang [-1, 1]: normalized = (pixel_value / 128.0f) - 1.0f
     * Format buffer: [1, size, size, 3] dengan FLOAT32.
     */
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

    /**
     * Konversi Bitmap → ByteBuffer untuk input TFLite (anti-spoofing).
     * Model Silent-Face mengekspektasikan nilai piksel RAW [0, 255] dalam urutan
     * BGR (normalisasi ada di dalam graf model). Jangan paksa ke [-1, 1].
     */
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

            // BGR, unnormalized [0..255]
            byteBuffer.putFloat(b.toFloat())
            byteBuffer.putFloat(g.toFloat())
            byteBuffer.putFloat(r.toFloat())
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    /** Softmax — konversi raw logits ke probability (numerically stable). */
    private fun softmax(logits: FloatArray): FloatArray {
        var maxVal = logits[0]
        for (v in logits) if (v > maxVal) maxVal = v

        var sum = 0f
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxVal).toDouble()).toFloat()
            sum += exps[i]
        }
        for (i in exps.indices) {
            exps[i] /= sum
        }
        return exps
    }

    private fun median(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2f
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
    // JSON helpers (audit serialization + integrity snapshot)
    // ============================

    private fun jsonObjectToMap(json: JSONObject): Map<String, Any?> {
        val map = mutableMapOf<String, Any?>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = json.get(key)
        }
        return map
    }

    private fun toJsonString(map: Map<*, *>): String = mapToJson(map).toString()

    private fun mapToJson(map: Map<*, *>): JSONObject {
        val json = JSONObject()
        for (entry in map.entries) {
            json.put(entry.key.toString(), anyToJson(entry.value))
        }
        return json
    }

    private fun anyToJson(value: Any?): Any? = when (value) {
        null -> JSONObject.NULL
        is Map<*, *> -> mapToJson(value)
        is FloatArray -> JSONArray(value.map { it.toDouble() })
        is DoubleArray -> JSONArray(value.toList())
        is IntArray -> JSONArray(value.toList())
        is List<*> -> JSONArray(value.map { anyToJson(it) })
        else -> value
    }

    // ============================
    // Lazy Resource & Asset Management
    // ============================

    private fun getOrCreateFaceDetector(): FaceDetector {
        faceDetector?.let { return it }
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        val detector = FaceDetection.getClient(options)
        faceDetector = detector
        return detector
    }

    private fun getOrCreateLivenessDetector(): FaceDetector {
        livenessDetector?.let { return it }
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        val detector = FaceDetection.getClient(options)
        livenessDetector = detector
        return detector
    }

    private fun getOrCreateFaceInterpreter(): Interpreter? {
        synchronized(faceInterpreterLock) {
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

    private fun getOrCreateAntiSpoof27(): Interpreter? {
        synchronized(antiSpoof27Lock) {
            if (antiSpoof27 == null) {
                antiSpoof27 = loadAntiSpoofModel(ANTI_SPOOF_MODEL_FILE_27, ANTI_SPOOF_MODEL_FILE_LEGACY)
            }
            return antiSpoof27
        }
    }

    private fun getOrCreateAntiSpoof40(): Interpreter? {
        synchronized(antiSpoof40Lock) {
            if (antiSpoof40 == null) {
                antiSpoof40 = loadAntiSpoofModel(ANTI_SPOOF_MODEL_FILE_40, null)
            }
            return antiSpoof40
        }
    }

    private fun loadAntiSpoofModel(fileName: String, fallbackName: String?): Interpreter? {
        try {
            return Interpreter(loadModelFile(fileName))
        } catch (_: Exception) {
            // coba fallback
        }
        if (fallbackName != null) {
            try {
                return Interpreter(loadModelFile(fallbackName))
            } catch (_: Exception) {
                // jatuh ke bawah
            }
        }
        // Model anti-spoofing tidak ditemukan — checkLiveness akan return error informatif
        return null
    }

    /**
     * Memuat file model TFLite dari Flutter/Android assets secara memory-mapped
     * untuk performa yang lebih baik.
     */
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
        synchronized(faceInterpreterLock) {
            tfliteInterpreter?.close()
            tfliteInterpreter = null
        }
        synchronized(antiSpoof27Lock) {
            antiSpoof27?.close()
            antiSpoof27 = null
        }
        synchronized(antiSpoof40Lock) {
            antiSpoof40?.close()
            antiSpoof40 = null
        }
        faceDetector?.close()
        faceDetector = null
        livenessDetector?.close()
        livenessDetector = null
    }

    private class FaceRecognitionException(val code: String, override val message: String) :
        Exception(message)
}
