import Foundation

/// Evaluator lintasan challenge (liveness-v3, P1) — PURE FUNCTION, dapat diuji
/// tanpa Vision. Paritas dari `ChallengeTrajectory.java`.
///
/// Input: deret yaw TERNORMALISASI (`YawConvention`: positif = menoleh ke kanan
/// anatomis user) beserta timestamp per frame, dalam urutan capture.
///
/// Aturan lulus (semua harus benar):
/// - R1 `frontalStart` — frame awal harus frontal (|yaw| < 8°)
/// - R2 `deviation` — ∃ deviasi ≥ 15° ke arah challenge, SETELAH fase R1
/// - R3 `returnFrontal` — frame akhir kembali mendekati frontal (|yaw| < 12°)
/// - R4 `singleDirection` — deviasi ke arah SALAH < 12° (menolak head-shake)
/// - R5 `smoothMotion` — ∃ run ≥ 3 delta searah (pada nilai ter-filter median-3)
///   dengan magnitudo ≥ 1°/frame (menolak spike 1 frame)
///
/// R1/R2/R3 dihitung dari nilai mentah (frame sudah lolos quality gate);
/// R4/R5 dari nilai ter-filter median-3 agar tahan spike noise tetapi tetap
/// menolak gerakan dua arah yang berkelanjutan.
///
/// Alasan gagal (prioritas): R4 → `CHALLENGE_WRONG_DIRECTION`;
/// R3 → `CHALLENGE_NO_RETURN`; selain itu `CHALLENGE_FAILED`.
enum ChallengeTrajectory {

    /// Satu sampel yaw ternormalisasi (positif = menoleh ke kanan anatomis user).
    struct Sample {
        let yaw: Float
        /// Epoch milliseconds; nil jika tidak diketahui (jalur JS base64).
        let timestampMs: Double?
    }

    /// Threshold aturan lintasan — SATU sumber kebenaran per platform.
    struct Thresholds {
        var startFrontalMax: Float = 8      // R1: |yaw| maksimum frame awal
        var startFrontalFrames: Int = 2     // R1: jumlah frame awal yang harus frontal
        var challengeYawMin: Float = 15     // R2: deviasi minimum ke arah challenge
        var returnFrontalMax: Float = 12    // R3: |yaw| maksimum frame akhir
        var returnFrontalFrames: Int = 2    // R3: jumlah frame akhir yang harus kembali
        var wrongWayMax: Float = 12         // R4: deviasi maksimum ke arah SALAH
        var smoothRunLength: Int = 3        // R5: run delta searah minimum
        var smoothDeltaMin: Float = 1       // R5: floor magnitudo delta (anti noise)
        var tooSmoothMax: Float = 0.05      // P2.3: ambang deltaVariance "terlalu sempurna"
    }

    struct RulesPassed {
        let frontalStart: Bool
        let deviation: Bool
        let returnFrontal: Bool
        let singleDirection: Bool
        let smoothMotion: Bool

        func toJson() -> [String: Any] {
            return [
                "frontalStart": frontalStart,
                "deviation": deviation,
                "returnFrontal": returnFrontal,
                "singleDirection": singleDirection,
                "smoothMotion": smoothMotion,
            ]
        }
    }

    /// Hasil evaluasi — semua angka yaw ternormalisasi.
    struct Result {
        let passed: Bool
        /// nil jika lulus; selain itu CHALLENGE_WRONG_DIRECTION | CHALLENGE_NO_RETURN | CHALLENGE_FAILED
        let failureReason: String?
        let rules: RulesPassed
        let startYaw: Float
        let endYaw: Float
        let extremeYaw: Float
        let extremeFrame: Int
        /// Deviasi teramati ke arah SALAH (untuk diagnostik).
        let wrongWayObserved: Float
        /// Varians magnitudo delta yaw — sinyal "terlalu sempurna" untuk P2.3 (tooSmooth).
        let deltaVariance: Float
        /// Lintasan "terlalu sempurna" (deltaVariance ≈ 0) — soft signal dugaan video-loop (P2.3).
        let tooSmooth: Bool
        /// Durasi sesi dalam ms; 0 jika timestamp tidak diketahui.
        let durationMs: Double

        func toJson() -> [String: Any] {
            return [
                "startYaw": startYaw,
                "endYaw": endYaw,
                "extremeYaw": extremeYaw,
                "extremeFrame": extremeFrame,
                "wrongWayMax": wrongWayObserved,
                "deltaVariance": deltaVariance,
                "tooSmooth": tooSmooth,
                "durationMs": durationMs,
                "rulesPassed": rules.toJson(),
            ]
        }
    }

    static func evaluate(samples: [Sample], challenge: String,
                         thresholds: Thresholds = Thresholds()) -> Result {
        let t = thresholds
        let lookLeft = challenge == "LOOK_LEFT"
        let n = samples.count
        guard n > 0 else {
            return Result(passed: false, failureReason: "CHALLENGE_FAILED",
                          rules: RulesPassed(frontalStart: false, deviation: false,
                                             returnFrontal: false, singleDirection: false,
                                             smoothMotion: false),
                          startYaw: 0, endYaw: 0, extremeYaw: 0, extremeFrame: -1,
                          wrongWayObserved: 0, deltaVariance: 0, tooSmooth: false, durationMs: 0)
        }

        let raw = samples.map { $0.yaw }
        let filtered = medianFilter3(raw)

        // R1 — frontalStart: frame awal harus frontal
        var frontalStart = true
        for i in 0..<min(t.startFrontalFrames, n) {
            if abs(raw[i]) >= t.startFrontalMax {
                frontalStart = false
                break
            }
        }

        // R2 — deviation: ∃ deviasi ≥ challengeYawMin ke arah challenge, setelah fase R1
        var extremeYaw: Float = 0
        var extremeFrame = -1
        var maxDev: Float = -.greatestFiniteMagnitude
        for (i, value) in raw.enumerated() {
            let dev = lookLeft ? -value : value
            if dev > maxDev {
                maxDev = dev
                extremeYaw = value
                extremeFrame = i
            }
        }
        var deviation = false
        if t.startFrontalFrames < n {
            for i in t.startFrontalFrames..<n {
                let dev = lookLeft ? -raw[i] : raw[i]
                if dev >= t.challengeYawMin {
                    deviation = true
                    break
                }
            }
        }

        // R3 — returnFrontal: frame akhir kembali mendekati frontal
        var returnFrontal = true
        for i in max(0, n - t.returnFrontalFrames)..<n {
            if abs(raw[i]) >= t.returnFrontalMax {
                returnFrontal = false
                break
            }
        }

        // R4 — singleDirection: deviasi ke arah SALAH < wrongWayMax (nilai ter-filter)
        let minF = filtered.min() ?? 0
        let maxF = filtered.max() ?? 0
        let wrongWayObserved = lookLeft ? max(0, maxF) : max(0, -minF)
        let singleDirection = wrongWayObserved < t.wrongWayMax

        // R5 — smoothMotion: run delta searah pada nilai ter-filter + deltaVariance
        var deltas: [Float] = []
        if n > 1 {
            for i in 0..<(n - 1) {
                deltas.append(filtered[i + 1] - filtered[i])
            }
        }
        var run = 0
        var bestRun = 0
        for d in deltas {
            let searah = lookLeft ? d <= -t.smoothDeltaMin : d >= t.smoothDeltaMin
            if searah {
                run += 1
                bestRun = max(bestRun, run)
            } else {
                run = 0
            }
        }
        let smoothMotion = bestRun >= t.smoothRunLength
        let deltaVariance = absVariance(deltas, minAbs: t.smoothDeltaMin)
        let tooSmooth = smoothMotion && deltaVariance < t.tooSmoothMax

        // Rangkuman
        let startYaw = raw[0]
        let endYaw = raw[n - 1]
        var durationMs: Double = 0
        if let t0 = samples.first?.timestampMs, let t1 = samples.last?.timestampMs {
            durationMs = max(0, t1 - t0)
        }

        let passed = frontalStart && deviation && returnFrontal
            && singleDirection && smoothMotion
        var failureReason: String? = nil
        if !passed {
            if !singleDirection {
                failureReason = "CHALLENGE_WRONG_DIRECTION"
            } else if !returnFrontal {
                failureReason = "CHALLENGE_NO_RETURN"
            } else {
                failureReason = "CHALLENGE_FAILED"
            }
        }

        return Result(
            passed: passed, failureReason: failureReason,
            rules: RulesPassed(frontalStart: frontalStart, deviation: deviation,
                               returnFrontal: returnFrontal, singleDirection: singleDirection,
                               smoothMotion: smoothMotion),
            startYaw: startYaw, endYaw: endYaw,
            extremeYaw: extremeYaw, extremeFrame: extremeFrame,
            wrongWayObserved: wrongWayObserved, deltaVariance: deltaVariance,
            tooSmooth: tooSmooth, durationMs: durationMs)
    }

    /// Median filter jendela 3 (edge di-replicate) — meredam spike 1 frame.
    static func medianFilter3(_ v: [Float]) -> [Float] {
        let n = v.count
        guard n >= 3 else { return v }
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let a = v[max(0, i - 1)]
            let b = v[i]
            let c = v[min(n - 1, i + 1)]
            out[i] = median3(a, b, c)
        }
        return out
    }

    private static func median3(_ a: Float, _ b: Float, _ c: Float) -> Float {
        let lo = min(a, min(b, c))
        let hi = max(a, max(b, c))
        return a + b + c - lo - hi
    }

    private static func absVariance(_ v: [Float], minAbs: Float) -> Float {
        // Varians MAGNITUDO delta yang BUKAN diam/rest (|delta| >= minAbs) —
        // fase berbalik/diam dikecualikan. Gerakan video-loop sempurna
        // berkecepatan konstan menghasilkan langkah seragam (varians ≈ 0),
        // sedangkan gerakan manusia mengakselerasi/melambat (varians > 0).
        let moving = v.map { abs($0) }.filter { $0 >= minAbs }
        guard moving.count >= 2 else { return 0 }
        let mean = moving.reduce(0, +) / Float(moving.count)
        let sq = moving.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        return sq / Float(moving.count)
    }
}
