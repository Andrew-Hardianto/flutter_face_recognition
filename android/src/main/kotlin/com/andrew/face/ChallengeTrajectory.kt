package com.andrew.face

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Evaluator lintasan challenge (liveness-v3, P1) — PURE FUNCTION, dapat diuji
 * tanpa Vision/ML Kit.
 *
 * Menggantikan aturan lama berbasis min/max agregat (yang bisa dilewati oleh
 * head-shake dan tidak memvalidasi urutan gerakan). Input: deret yaw
 * TERNORMALISASI ([YawConvention]: positif = menoleh ke kanan anatomis
 * user) beserta timestamp per frame, dalam urutan capture.
 *
 * Aturan lulus (semua harus benar):
 *  - R1 `frontalStart` — frame awal harus frontal (|yaw| < 8°)
 *  - R2 `deviation` — ∃ deviasi ≥ 15° ke arah challenge, SETELAH fase R1
 *  - R3 `returnFrontal` — frame akhir kembali mendekati frontal (|yaw| < 12°)
 *  - R4 `singleDirection` — deviasi ke arah SALAH < 12° (menolak head-shake)
 *  - R5 `smoothMotion` — ∃ run ≥ 3 delta searah (pada nilai ter-filter
 *    median-3) dengan magnitudo ≥ 1°/frame (menolak spike 1 frame)
 *
 * R1/R2/R3 dihitung dari nilai mentah (frame sudah lolos quality gate);
 * R4/R5 dari nilai ter-filter median-3 agar tahan spike noise tetapi tetap
 * menolak gerakan dua arah yang berkelanjutan.
 *
 * Alasan gagal (prioritas): R4 → `CHALLENGE_WRONG_DIRECTION`;
 * R3 → `CHALLENGE_NO_RETURN`; selain itu `CHALLENGE_FAILED`.
 */
object ChallengeTrajectory {

    /** Satu sampel yaw ternormalisasi (positif = menoleh ke kanan anatomis user). */
    class Sample(
        val yaw: Float,
        /** Epoch milliseconds; -1 jika tidak diketahui (jalur JS base64). */
        val timestampMs: Double,
    )

    /** Threshold aturan lintasan — SATU sumber kebenaran per platform. */
    class Thresholds {
        var startFrontalMax = 8f      // R1: |yaw| maksimum frame awal
        var startFrontalFrames = 2    // R1: jumlah frame awal yang harus frontal
        var challengeYawMin = 15f     // R2: deviasi minimum ke arah challenge
        var returnFrontalMax = 12f    // R3: |yaw| maksimum frame akhir
        var returnFrontalFrames = 2   // R3: jumlah frame akhir yang harus kembali
        var wrongWayMax = 12f         // R4: deviasi maksimum ke arah SALAH
        var smoothRunLength = 3       // R5: run delta searah minimum
        var smoothDeltaMin = 1f       // R5: floor magnitudo delta (anti noise)
        var tooSmoothMax = 0.05f      // P2.3: ambang deltaVariance "terlalu sempurna"
    }

    /** Hasil evaluasi — semua angka yaw ternormalisasi. */
    class Result {
        var passed = false
        /** null jika lulus; selain itu CHALLENGE_WRONG_DIRECTION | CHALLENGE_NO_RETURN | CHALLENGE_FAILED */
        var failureReason: String? = null
        var frontalStart = false     // R1
        var deviation = false        // R2
        var returnFrontal = false    // R3
        var singleDirection = false  // R4
        var smoothMotion = false     // R5
        var startYaw = 0f
        var endYaw = 0f
        var extremeYaw = 0f
        var extremeFrame = -1
        /** Deviasi teramati ke arah SALAH (untuk diagnostik). */
        var wrongWayObserved = 0f
        /** Varians magnitudo delta yaw — sinyal "terlalu sempurna" untuk P2.3 (tooSmooth). */
        var deltaVariance = 0f
        /** Lintasan "terlalu sempurna" (deltaVariance ≈ 0) — soft signal dugaan video-loop (P2.3). */
        var tooSmooth = false
        /** Durasi sesi dalam ms; 0 jika timestamp tidak diketahui. */
        var durationMs = 0.0
    }

    @JvmStatic
    fun evaluate(samples: List<Sample>?, challenge: String?): Result =
        evaluate(samples, challenge, Thresholds())

    @JvmStatic
    fun evaluate(samples: List<Sample>?, challenge: String?, t: Thresholds): Result {
        val r = Result()
        val lookLeft = challenge == "LOOK_LEFT"
        val list = samples ?: emptyList()
        val n = list.size
        if (n == 0) {
            r.passed = false
            r.failureReason = "CHALLENGE_FAILED"
            return r
        }

        val raw = FloatArray(n)
        for (i in 0 until n) {
            raw[i] = list[i].yaw
        }
        val filtered = medianFilter3(raw)

        // R1 — frontalStart: frame awal harus frontal
        r.frontalStart = true
        for (i in 0 until min(t.startFrontalFrames, n)) {
            if (abs(raw[i]) >= t.startFrontalMax) {
                r.frontalStart = false
                break
            }
        }

        // R2 — deviation: ∃ deviasi ≥ challengeYawMin ke arah challenge, setelah fase R1
        var maxDev = -Float.MAX_VALUE
        for (i in 0 until n) {
            val dev = if (lookLeft) -raw[i] else raw[i]
            if (dev > maxDev) {
                maxDev = dev
                r.extremeYaw = raw[i]
                r.extremeFrame = i
            }
        }
        r.deviation = false
        for (i in t.startFrontalFrames until n) {
            val dev = if (lookLeft) -raw[i] else raw[i]
            if (dev >= t.challengeYawMin) {
                r.deviation = true
                break
            }
        }

        // R3 — returnFrontal: frame akhir kembali mendekati frontal
        r.returnFrontal = true
        val from = max(0, n - t.returnFrontalFrames)
        for (i in from until n) {
            if (abs(raw[i]) >= t.returnFrontalMax) {
                r.returnFrontal = false
                break
            }
        }

        // R4 — singleDirection: deviasi ke arah SALAH < wrongWayMax (nilai ter-filter)
        var minF = Float.MAX_VALUE
        var maxF = -Float.MAX_VALUE
        for (v in filtered) {
            minF = min(minF, v)
            maxF = max(maxF, v)
        }
        r.wrongWayObserved = if (lookLeft) max(0f, maxF) else max(0f, -minF)
        r.singleDirection = r.wrongWayObserved < t.wrongWayMax

        // R5 — smoothMotion: run delta searah pada nilai ter-filter + deltaVariance
        val deltas = FloatArray(max(0, n - 1))
        for (i in deltas.indices) {
            deltas[i] = filtered[i + 1] - filtered[i]
        }
        var run = 0
        var bestRun = 0
        for (d in deltas) {
            val searah = if (lookLeft) d <= -t.smoothDeltaMin else d >= t.smoothDeltaMin
            if (searah) {
                run++
                bestRun = max(bestRun, run)
            } else {
                run = 0
            }
        }
        r.smoothMotion = bestRun >= t.smoothRunLength
        r.deltaVariance = absVariance(deltas, t.smoothDeltaMin)
        r.tooSmooth = r.smoothMotion && r.deltaVariance < t.tooSmoothMax

        // Rangkuman
        r.startYaw = raw[0]
        r.endYaw = raw[n - 1]
        val t0 = list[0].timestampMs
        val t1 = list[n - 1].timestampMs
        r.durationMs = if (t0 >= 0 && t1 >= 0) max(0.0, t1 - t0) else 0.0

        r.passed = r.frontalStart && r.deviation && r.returnFrontal &&
                r.singleDirection && r.smoothMotion
        r.failureReason = when {
            r.passed -> null
            !r.singleDirection -> "CHALLENGE_WRONG_DIRECTION"
            !r.returnFrontal -> "CHALLENGE_NO_RETURN"
            else -> "CHALLENGE_FAILED"
        }
        return r
    }

    /** Median filter jendela 3 (edge di-replicate) — meredam spike 1 frame. */
    internal fun medianFilter3(v: FloatArray): FloatArray {
        val n = v.size
        if (n < 3) {
            return v.copyOf()
        }
        val out = FloatArray(n)
        for (i in 0 until n) {
            val a = v[max(0, i - 1)]
            val b = v[i]
            val c = v[min(n - 1, i + 1)]
            out[i] = median3(a, b, c)
        }
        return out
    }

    internal fun median3(a: Float, b: Float, c: Float): Float {
        val lo = min(a, min(b, c))
        val hi = max(a, max(b, c))
        return a + b + c - lo - hi
    }

    internal fun absVariance(v: FloatArray, minAbs: Float): Float {
        // Varians MAGNITUDO delta yang BUKAN diam/rest (|delta| >= minAbs) —
        // fase berbalik/diam dikecualikan. Gerakan video-loop sempurna
        // berkecepatan konstan menghasilkan langkah seragam (varians ≈ 0),
        // sedangkan gerakan manusia mengakselerasi/melambat (varians > 0).
        var count = 0
        var sum = 0f
        for (x in v) {
            if (abs(x) >= minAbs) {
                sum += abs(x)
                count++
            }
        }
        if (count < 2) {
            return 0f
        }
        val mean = sum / count
        var sq = 0f
        for (x in v) {
            if (abs(x) >= minAbs) {
                val d = abs(x) - mean
                sq += d * d
            }
        }
        return sq / count
    }
}
