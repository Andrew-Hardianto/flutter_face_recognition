package com.andrew.face

import kotlin.math.max
import kotlin.math.min

/**
 * Heuristik artefak layar (P2.2) — SOFT SIGNAL EKSPERIMENTAL, bukan bukti.
 *
 * Memeriksa tiga komponen pada crop wajah (downsample 32x32 ARGB):
 *  - `periodicity` — puncak autocorrelation proyeksi luma (pola grid/moiré layar)
 *  - `flatFraction` — fraksi blok 8x8 nyaris tanpa tekstur (panel datar)
 *  - `saturationFlat` — fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar)
 *
 * Skor gabungan = max ketiganya (0..1). Pada rilis pertama sinyal ini HANYA
 * dicatat ke hasil & audit (pengumpulan data) — auto-fail hanya jika
 * `config.screenSuspectVeto` di-set eksplisit (mode ketat opt-in).
 * Threshold definitif menyusul setelah ada angka kalibrasi (lihat plan P2.2).
 */
object ScreenSuspect {

    /** Hasil analisis artefak layar pada satu crop wajah. */
    class Result {
        /** Periodisitas piksel (puncak autocorrelation proyeksi luma) — 0..1 */
        var periodicity = 0f

        /** Fraksi blok 8x8 dengan varians sangat rendah (panel datar) — 0..1 */
        var flatFraction = 0f

        /** Fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar) — 0..1 */
        var saturationFlat = 0f

        /** Skor gabungan = max ketiganya — 0..1 */
        var score = 0f
    }

    /**
     * @param argb piksel ARGB (0xAARRGGBB) hasil downsample crop wajah
     */
    @JvmStatic
    fun analyze(argb: IntArray?, width: Int, height: Int): Result {
        val r = Result()
        val n = width * height
        if (argb == null || width <= 0 || height <= 0 || argb.size < n) {
            return r
        }

        val gray = FloatArray(n)
        var satFlat = 0
        for (i in 0 until n) {
            val px = argb[i]
            val red = ((px shr 16) and 0xFF).toFloat()
            val green = ((px shr 8) and 0xFF).toFloat()
            val blue = (px and 0xFF).toFloat()
            gray[i] = 0.299f * red + 0.587f * green + 0.114f * blue
            val maxC = max(red, max(green, blue))
            val minC = min(red, min(green, blue))
            val sat = if (maxC <= 0f) 0f else (maxC - minC) / maxC
            if (sat < 0.06f) {
                satFlat++
            }
        }
        r.saturationFlat = satFlat / n.toFloat()

        // Proyeksi luma: rata-rata per baris & per kolom
        val rowMean = FloatArray(height)
        for (y in 0 until height) {
            var s = 0f
            for (x in 0 until width) {
                s += gray[y * width + x]
            }
            rowMean[y] = s / width
        }
        val colMean = FloatArray(width)
        for (x in 0 until width) {
            var s = 0f
            for (y in 0 until height) {
                s += gray[y * width + x]
            }
            colMean[x] = s / height
        }
        r.periodicity = max(autocorrPeak(rowMean), autocorrPeak(colMean))

        r.flatFraction = flatFraction(gray, width, height)
        r.score = max(r.periodicity, max(r.flatFraction, r.saturationFlat))
        return r
    }

    /** Puncak autocorrelation ternormalisasi (lag 2..min(12, n/2)) dari proyeksi 1D. */
    internal fun autocorrPeak(x: FloatArray): Float {
        val n = x.size
        if (n < 6) {
            return 0f
        }
        var mean = 0f
        for (v in x) {
            mean += v
        }
        mean /= n
        var energy = 0f
        for (v in x) {
            energy += (v - mean) * (v - mean)
        }
        if (energy <= 1e-6f) {
            return 0f
        }
        var best = 0f
        val maxLag = min(12, n / 2)
        for (lag in 2..maxLag) {
            var acc = 0f
            for (i in 0 until n - lag) {
                acc += (x[i] - mean) * (x[i + lag] - mean)
            }
            val a = acc / energy
            if (a > best) {
                best = a
            }
        }
        return best
    }

    private fun flatFraction(gray: FloatArray, width: Int, height: Int): Float {
        var blocks = 0
        var flat = 0
        var by = 0
        while (by + 8 <= height) {
            var bx = 0
            while (bx + 8 <= width) {
                blocks++
                var mean = 0f
                for (y in by until by + 8) {
                    for (x in bx until bx + 8) {
                        mean += gray[y * width + x]
                    }
                }
                mean /= 64f
                var variance = 0f
                for (y in by until by + 8) {
                    for (x in bx until bx + 8) {
                        val d = gray[y * width + x] - mean
                        variance += d * d
                    }
                }
                variance /= 64f
                if (variance < 4f) {
                    flat++
                }
                bx += 8
            }
            by += 8
        }
        return if (blocks == 0) 0f else flat / blocks.toFloat()
    }
}
