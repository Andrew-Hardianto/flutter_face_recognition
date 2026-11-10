import Foundation

/// Heuristik artefak layar (P2.2) — SOFT SIGNAL EKSPERIMENTAL, bukan bukti.
/// Paritas dari `ScreenSuspect.java`.
///
/// Memeriksa tiga komponen pada crop wajah (downsample 32x32 ARGB):
/// - `periodicity` — puncak autocorrelation proyeksi luma (pola grid/moiré layar)
/// - `flatFraction` — fraksi blok 8x8 nyaris tanpa tekstur (panel datar)
/// - `saturationFlat` — fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar)
///
/// Skor gabungan = max ketiganya (0..1). Pada rilis pertama sinyal ini HANYA
/// dicatat ke hasil & audit (pengumpulan data) — auto-fail hanya jika
/// `config.screenSuspectVeto` di-set eksplisit (mode ketat opt-in).
/// Threshold definitif menyusul setelah ada angka kalibrasi (lihat plan P2.2).
enum ScreenSuspect {

    /// Hasil analisis artefak layar pada satu crop wajah.
    struct Result {
        /// Periodisitas piksel (puncak autocorrelation proyeksi luma) — 0..1
        var periodicity: Float = 0
        /// Fraksi blok 8x8 dengan varians sangat rendah (panel datar) — 0..1
        var flatFraction: Float = 0
        /// Fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar) — 0..1
        var saturationFlat: Float = 0
        /// Skor gabungan = max ketiganya — 0..1
        var score: Float = 0
    }

    /// - Parameter argb: piksel ARGB (0xAARRGGBB) hasil downsample crop wajah
    static func analyze(argb: [UInt32], width: Int, height: Int) -> Result {
        var r = Result()
        let n = width * height
        guard width > 0, height > 0, argb.count >= n else { return r }

        var gray = [Float](repeating: 0, count: n)
        var satFlat = 0
        for i in 0..<n {
            let px = argb[i]
            let red = Float((px >> 16) & 0xFF)
            let green = Float((px >> 8) & 0xFF)
            let blue = Float(px & 0xFF)
            gray[i] = 0.299 * red + 0.587 * green + 0.114 * blue
            let maxValue = max(red, green, blue)
            let minValue = min(red, green, blue)
            let sat = maxValue <= 0 ? 0 : (maxValue - minValue) / maxValue
            if sat < 0.06 {
                satFlat += 1
            }
        }
        r.saturationFlat = Float(satFlat) / Float(n)

        // Proyeksi luma: rata-rata per baris & per kolom
        var rowMean = [Float](repeating: 0, count: height)
        for y in 0..<height {
            var s: Float = 0
            for x in 0..<width {
                s += gray[y * width + x]
            }
            rowMean[y] = s / Float(width)
        }
        var colMean = [Float](repeating: 0, count: width)
        for x in 0..<width {
            var s: Float = 0
            for y in 0..<height {
                s += gray[y * width + x]
            }
            colMean[x] = s / Float(height)
        }
        r.periodicity = max(autocorrPeak(rowMean), autocorrPeak(colMean))

        r.flatFraction = flatFraction(gray, width: width, height: height)
        r.score = max(r.periodicity, r.flatFraction, r.saturationFlat)
        return r
    }

    /// Puncak autocorrelation ternormalisasi (lag 2..min(12, n/2)) dari proyeksi 1D.
    static func autocorrPeak(_ x: [Float]) -> Float {
        let n = x.count
        guard n >= 6 else { return 0 }
        let mean = x.reduce(0, +) / Float(n)
        let energy = x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        guard energy > 1e-6 else { return 0 }
        var best: Float = 0
        let maxLag = min(12, n / 2)
        for lag in 2...maxLag {
            var acc: Float = 0
            for i in 0..<(n - lag) {
                acc += (x[i] - mean) * (x[i + lag] - mean)
            }
            let a = acc / energy
            if a > best {
                best = a
            }
        }
        return best
    }

    private static func flatFraction(_ gray: [Float], width: Int, height: Int) -> Float {
        var blocks = 0
        var flat = 0
        var by = 0
        while by + 8 <= height {
            var bx = 0
            while bx + 8 <= width {
                blocks += 1
                var mean: Float = 0
                for y in by..<(by + 8) {
                    for x in bx..<(bx + 8) {
                        mean += gray[y * width + x]
                    }
                }
                mean /= 64
                var variance: Float = 0
                for y in by..<(by + 8) {
                    for x in bx..<(bx + 8) {
                        let d = gray[y * width + x] - mean
                        variance += d * d
                    }
                }
                variance /= 64
                if variance < 4 {
                    flat += 1
                }
                bx += 8
            }
            by += 8
        }
        return blocks == 0 ? 0 : Float(flat) / Float(blocks)
    }
}
