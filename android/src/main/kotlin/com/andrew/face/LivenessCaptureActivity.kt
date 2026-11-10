package com.andrew.face

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.util.concurrent.Executor

/**
 * Activity kamera untuk captureLivenessSession (Fase 4 — native capture).
 *
 * Frame diambil DI DALAM plugin dan disimpan ke file cache privat,
 * TIDAK pernah melewati layer JS — menutup celah injeksi frame.
 * Pipeline verifikasi (quality gate + liveness + challenge) tetap dijalankan
 * oleh FlutterFaceRecognitionPlugin terhadap frame-file ini.
 *
 * Hasil dikembalikan lewat setResult():
 *   RESULT_OK     → EXTRA_FRAME_PATHS (ArrayList<String>), EXTRA_CHALLENGE
 *   RESULT_CANCELED → EXTRA_ERROR (String)
 */
class LivenessCaptureActivity : ComponentActivity() {

    private var previewView: PreviewView? = null
    private var instructionView: TextView? = null
    private var progressView: TextView? = null
    private var imageCapture: ImageCapture? = null
    private val handler = Handler(Looper.getMainLooper())

    private var challenge: String? = null
    private var frameCount = 0
    private var intervalMs = 0L
    private var warmupMs = 0L
    private var captured = 0
    private val framePaths = ArrayList<String>()
    private val frameTimestamps = ArrayList<Double>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        challenge = intent.getStringExtra(EXTRA_CHALLENGE)
        frameCount = maxOf(2, intent.getIntExtra(EXTRA_FRAME_COUNT, 5))
        intervalMs = intent.getLongExtra(EXTRA_INTERVAL_MS, 300L)
        warmupMs = intent.getLongExtra(EXTRA_WARMUP_MS, 1200L)

        // FAIL-CLOSED: challenge tak dikenal tidak boleh diam-diam memakai
        // instruksi default — sesi dibatalkan sebelum kamera dibuka.
        if (challenge != CHALLENGE_LOOK_LEFT && challenge != CHALLENGE_LOOK_RIGHT) {
            failWith("Challenge tidak dikenal")
            return
        }

        buildUi()
        startCamera()
    }

    private fun buildUi() {
        val root = FrameLayout(this)
        root.setBackgroundColor(Color.BLACK)

        val preview = PreviewView(this)
        previewView = preview
        root.addView(
            preview,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        val instruction = TextView(this)
        instructionView = instruction
        // FAIL-CLOSED: mapping eksplisit per challenge — nilai tak dikenal sudah
        // ditolak di onCreate, TIDAK ada default diam-diam ke "ke kanan".
        val text = when (challenge) {
            CHALLENGE_LOOK_LEFT ->
                "TOLEH KE KIRI ANDA (dagu ke bahu kiri) perlahan, lalu kembali lurus"

            CHALLENGE_LOOK_RIGHT ->
                "TOLEH KE KANAN ANDA (dagu ke bahu kanan) perlahan, lalu kembali lurus"

            else -> "" // tidak tercapai — onCreate sudah fail-closed
        }
        instruction.text = text
        instruction.setTextColor(Color.WHITE)
        instruction.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        instruction.gravity = Gravity.CENTER
        instruction.setBackgroundColor(Color.argb(160, 0, 0, 0))
        instruction.setPadding(24, 32, 24, 32)
        val instrParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        instrParams.gravity = Gravity.TOP
        root.addView(instruction, instrParams)

        val progress = TextView(this)
        progressView = progress
        progress.setTextColor(Color.WHITE)
        progress.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        progress.gravity = Gravity.CENTER
        progress.setBackgroundColor(Color.argb(160, 0, 0, 0))
        progress.setPadding(24, 24, 24, 24)
        val progressParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        progressParams.gravity = Gravity.BOTTOM
        root.addView(progress, progressParams)

        setContentView(root)
    }

    private fun startCamera() {
        val future: ListenableFuture<ProcessCameraProvider> =
            ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                bindCamera(provider)
            } catch (e: Exception) {
                failWith("Gagal memulai kamera: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(provider: ProcessCameraProvider) {
        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView!!.surfaceProvider)

        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageCapture = capture

        val selector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()

        provider.unbindAll()
        provider.bindToLifecycle(this, selector, preview, capture)

        // Beri waktu user memosisikan wajah, lalu capture frame berkala
        handler.postDelayed({ captureNext() }, warmupMs)
    }

    private fun captureNext() {
        val capture = imageCapture
        if (capture == null) {
            failWith("ImageCapture tidak tersedia")
            return
        }

        val dir = File(cacheDir, FRAME_DIR)
        if (!dir.exists() && !dir.mkdirs()) {
            failWith("Gagal membuat direktori frame")
            return
        }
        val outFile = File(dir, "frame_$captured.jpg")
        val options = ImageCapture.OutputFileOptions.Builder(outFile).build()

        val executor: Executor = ContextCompat.getMainExecutor(this)
        capture.takePicture(options, executor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(results: ImageCapture.OutputFileResults) {
                framePaths.add(outFile.absolutePath)
                frameTimestamps.add(System.currentTimeMillis().toDouble())
                captured++
                progressView?.text = "Mengambil frame $captured/$frameCount"

                if (captured >= frameCount) {
                    val timestamps = DoubleArray(frameTimestamps.size) { frameTimestamps[it] }
                    val data = Intent()
                    data.putStringArrayListExtra(EXTRA_FRAME_PATHS, framePaths)
                    data.putExtra(EXTRA_FRAME_TIMESTAMPS, timestamps)
                    data.putExtra(EXTRA_CHALLENGE, challenge)
                    setResult(RESULT_OK, data)
                    finish()
                } else {
                    handler.postDelayed({ captureNext() }, intervalMs)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                failWith("Gagal mengambil frame: ${exception.message}")
            }
        })
    }

    private fun failWith(message: String) {
        val data = Intent()
        data.putExtra(EXTRA_ERROR, message)
        setResult(RESULT_CANCELED, data)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
    }

    companion object {
        const val EXTRA_CHALLENGE = "challenge"
        const val EXTRA_FRAME_COUNT = "frameCount"
        const val EXTRA_INTERVAL_MS = "intervalMs"
        const val EXTRA_WARMUP_MS = "warmupMs"
        const val EXTRA_FRAME_PATHS = "framePaths"
        const val EXTRA_FRAME_TIMESTAMPS = "frameTimestamps"
        const val EXTRA_ERROR = "error"

        const val CHALLENGE_LOOK_LEFT = "LOOK_LEFT"
        const val CHALLENGE_LOOK_RIGHT = "LOOK_RIGHT"

        private const val FRAME_DIR = "liveness_frames"

        @JvmStatic
        fun createIntent(
            context: Context,
            challenge: String,
            frameCount: Int,
            intervalMs: Long,
            warmupMs: Long,
        ): Intent {
            val intent = Intent(context, LivenessCaptureActivity::class.java)
            intent.putExtra(EXTRA_CHALLENGE, challenge)
            intent.putExtra(EXTRA_FRAME_COUNT, frameCount)
            intent.putExtra(EXTRA_INTERVAL_MS, intervalMs)
            intent.putExtra(EXTRA_WARMUP_MS, warmupMs)
            return intent
        }
    }
}
