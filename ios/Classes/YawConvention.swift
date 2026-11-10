import Foundation

/// Konvensi yaw SDK (liveness-v3) — SATU kebenaran untuk iOS & Android.
///
/// Yaw TERNORMALISASI: **positif = user menoleh ke KANAN anatomisnya**,
/// negatif = ke KIRI anatomisnya. Semua aturan challenge bekerja pada konvensi
/// ini sehingga tidak lagi bergantung pada tanda mentah platform.
///
/// `platformYawSign` mengoreksi yaw MENTAH platform ke konvensi anatomis:
///   - Apple Vision (`VNFaceObservation.yaw`): yaw mentah positif = wajah
///     berbalik ke sisi KANAN gambar (frame tak termirror) = user menoleh ke
///     KIRI anatomisnya → dikalikan -1.
///   - ML Kit Android (`getHeadEulerAngleY`): konvensi mentah sama — dokumen
///     resmi: "Positive y euler angle is when the face is turned towards the
///     right side of the image" → dikalikan -1 juga (lihat `YawConvention.java`).
///
/// VERIFIKASI WAJIB via halaman "Kalibrasi Yaw" di example-app (P0-a):
/// "Dagu ke bahu KIRI Anda" → yaw ternormalisasi harus **negatif**.
/// Jika positif pada perangkat tertentu (buffer kamera ter-flip, dilaporkan
/// pada beberapa Samsung/OnePlus), set `config.yawSignOverride = -1` —
/// JANGAN mengubah konstanta ini.
public enum YawConvention {
    /// Pengali koreksi yaw mentah Apple Vision → konvensi anatomis SDK.
    /// Hasil kalibrasi P0-a; lihat dokumentasi enum ini sebelum mengubah.
    public static let platformYawSign: Float = -1

    /// Normalisasi yaw mentah platform → konvensi anatomis SDK
    /// (positif = menoleh ke kanan anatomis user).
    public static func normalize(rawYaw: Float) -> Float {
        return rawYaw * platformYawSign
    }

    /// Override per-perangkat (`config.yawSignOverride`) untuk kasus buffer
    /// ter-flip. Diterapkan ke yaw ternormalisasi SEBELUM agregasi challenge.
    public static func applyOverride(_ normalizedYaw: Float, overrideSign: Int) -> Float {
        return overrideSign >= 0 ? normalizedYaw : -normalizedYaw
    }
}
