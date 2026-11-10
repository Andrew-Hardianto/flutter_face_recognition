package com.andrew.face

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * Pemeriksaan integritas perangkat (Fase 4) — heuristik sisi klien.
 *
 * CATATAN PENTING: semua pemeriksaan di sini adalah SINYAL RISIKO, bukan bukti.
 * Deteksi root/emulator/hook yang benar-benar kuat membutuhkan device attestation
 * terverifikasi server (Play Integrity) — di luar scope plugin ini karena backend
 * hanya menyediakan penyimpanan vector.
 *
 * Hasil dipakai oleh aplikasi untuk mengambil keputusan (tolak, minta tantangan
 * tambahan, atau catat untuk audit), bukan diandalkan sebagai gerbang tunggal.
 */
object DeviceIntegrity {

    private val SU_PATHS = arrayOf(
        "/system/bin/su",
        "/system/xbin/su",
        "/sbin/su",
        "/system/app/Superuser.apk",
        "/data/local/xbin/su",
        "/data/local/bin/su",
        "/data/local/su",
        "/su/bin/su",
        "/system/bin/failsafe/su",
    )

    private val MAGISK_PATHS = arrayOf(
        "/sbin/.magisk",
        "/data/adb/magisk",
        "/cache/magisk.log",
    )

    /** Build/prop yang khas pada emulator. */
    @JvmStatic
    fun isEmulator(): Boolean {
        return Build.FINGERPRINT.startsWith("generic")
                || Build.FINGERPRINT.startsWith("unknown")
                || Build.FINGERPRINT.contains("emulator")
                || Build.FINGERPRINT.contains("sdk_gphone")
                || Build.MODEL.contains("Emulator")
                || Build.MODEL.contains("Android SDK built for x86")
                || Build.MODEL.contains("google_sdk")
                || Build.MANUFACTURER.contains("Genymotion")
                || (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic"))
                || Build.HARDWARE.contains("goldfish")
                || Build.HARDWARE.contains("ranchu")
                || Build.PRODUCT.contains("sdk_gphone")
                || Build.PRODUCT.contains("emulator")
                || Build.PRODUCT.contains("simulator")
    }

    /** Heuristik root: binary su, Magisk, atau build test-keys. */
    @JvmStatic
    fun isRooted(): Boolean {
        for (path in SU_PATHS) {
            if (File(path).exists()) return true
        }
        for (path in MAGISK_PATHS) {
            if (File(path).exists()) return true
        }
        val tags = Build.TAGS
        if (tags != null && tags.contains("test-keys")) return true

        // which su
        var process: Process? = null
        try {
            process = Runtime.getRuntime().exec(arrayOf("/system/xbin/which", "su"))
            BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                if (reader.readLine() != null) return true
            }
        } catch (_: Exception) {
            // tidak ada which/su — anggap tidak rooted dari jalur ini
        } finally {
            process?.destroy()
        }
        return false
    }

    /** Build debuggable (development) — hasil liveness dari build debug tidak untuk produksi. */
    @JvmStatic
    fun isDebuggable(context: Context): Boolean {
        return (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }

    /** Heuristik framework hook (Xposed / Frida / Substrate) via classloader dan /proc/self/maps. */
    @JvmStatic
    fun hooksDetected(): Boolean {
        val hookClasses = arrayOf(
            "de.robv.android.xposed.XposedBridge",
            "de.robv.android.xposed.XposedHelpers",
            "com.saurik.substrate.MS\$2",
            "com.topjohnwu.magisk.core",
        )
        for (className in hookClasses) {
            try {
                Class.forName(className)
                return true
            } catch (_: ClassNotFoundException) {
                // normal — kelas tidak ada
            }
        }

        // Cari pustaka mencurigakan di /proc/self/maps (Frida dll)
        try {
            BufferedReader(InputStreamReader(java.io.FileInputStream("/proc/self/maps"))).use { reader ->
                var line: String? = reader.readLine()
                while (line != null) {
                    val lower = line!!.lowercase()
                    if (lower.contains("frida") || lower.contains("xposed")
                        || lower.contains("substrate") || lower.contains("libgadget")
                    ) {
                        return true
                    }
                    line = reader.readLine()
                }
            }
        } catch (_: Exception) {
            // tidak bisa membaca maps — lewati
        }
        return false
    }

    /** Snapshot lengkap untuk disertakan di hasil verifikasi. */
    @JvmStatic
    fun snapshot(context: Context): JSONObject {
        val o = JSONObject()
        try {
            o.put("isEmulator", isEmulator())
            o.put("isRooted", isRooted())
            o.put("isDebuggable", isDebuggable(context))
            o.put("hooksDetected", hooksDetected())
        } catch (_: Exception) {
            // JSONObject.put tidak melempar untuk tipe ini
        }
        return o
    }
}
