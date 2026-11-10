package com.andrew.face

/**
 * Konvensi yaw SDK (liveness-v3) — SATU kebenaran untuk iOS & Android.
 *
 * Yaw TERNORMALISASI: **positif = user menoleh ke KANAN anatomisnya**,
 * negatif = ke KIRI anatomisnya. Semua aturan challenge bekerja pada konvensi
 * ini sehingga tidak lagi bergantung pada tanda mentah platform.
 *
 * [PLATFORM_YAW_SIGN] mengoreksi yaw MENTAH ML Kit ke konvensi anatomis:
 * `getHeadEulerAngleY` mentah positif = wajah berbalik ke sisi KANAN gambar
 * (frame tak termirror) = user menoleh ke KIRI anatomisnya (dokumen resmi ML Kit:
 * "Positive y euler angle is when the face is turned towards the right side of
 * the image") → dikalikan -1.
 *
 * VERIFIKASI WAJIB via halaman "Kalibrasi Yaw" di example-app (P0-a):
 * "Dagu ke bahu KIRI Anda" → yaw ternormalisasi harus negatif.
 * Jika positif pada perangkat tertentu (buffer kamera ter-flip, dilaporkan pada
 * beberapa Samsung/OnePlus), set `config.yawSignOverride = -1` — JANGAN
 * mengubah konstanta ini.
 */
object YawConvention {

    /** Pengali koreksi yaw mentah ML Kit → konvensi anatomis SDK. */
    const val PLATFORM_YAW_SIGN = -1f

    /**
     * Normalisasi yaw mentah platform → konvensi anatomis SDK
     * (positif = menoleh ke kanan anatomis user).
     */
    @JvmStatic
    fun normalize(rawYaw: Float): Float = rawYaw * PLATFORM_YAW_SIGN

    /**
     * Override per-perangkat (`config.yawSignOverride`) untuk kasus buffer
     * ter-flip. Diterapkan ke yaw ternormalisasi SEBELUM agregasi challenge.
     */
    @JvmStatic
    fun applyOverride(normalizedYaw: Float, overrideSign: Int): Float =
        if (overrideSign >= 0) normalizedYaw else -normalizedYaw
}
