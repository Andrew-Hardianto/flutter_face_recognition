package com.andrew.face

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.security.KeyStore
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Jejak audit lokal terenkripsi (Fase 4).
 *
 * Setiap sesi liveness native menyimpan hasil + frame yang dipakai dalam bentuk
 * AES-256-GCM dengan kunci di AndroidKeyStore (tidak pernah keluar dari perangkat,
 * tidak ikut backup). File berada di penyimpanan privat aplikasi.
 *
 * Format file: [1 byte panjang IV][IV][ciphertext + GCM tag].
 * Dipakai untuk pemeriksaan fraud HRIS — kebijakan retensi mengikuti persetujuan
 * dan UU PDP (hak hapus: cukup hapus folder liveness_audit).
 */
object AuditStore {

    private const val KEY_ALIAS = "andrew.face.audit.v1"
    private const val AUDIT_DIR = "liveness_audit"
    private const val KEYSTORE = "AndroidKeyStore"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val GCM_TAG_BITS = 128
    private const val IV_BYTES = 12

    /**
     * Simpan JSON sesi + frame yang dipakai, semuanya terenkripsi.
     * Mengembalikan sessionId.
     */
    @JvmStatic
    @Throws(Exception::class)
    fun storeSession(context: Context, sessionJson: String, frames: List<ByteArray>): String {
        val sessionId = "session_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(Date())
        val dir = File(context.filesDir, AUDIT_DIR)
        if (!dir.exists() && !dir.mkdirs()) {
            throw IllegalStateException("Gagal membuat direktori audit")
        }

        encryptToFile(
            sessionJson.toByteArray(StandardCharsets.UTF_8),
            File(dir, "$sessionId.json.enc")
        )
        for (i in frames.indices) {
            encryptToFile(frames[i], File(dir, "${sessionId}_frame_$i.jpg.enc"))
        }
        return sessionId
    }

    /** Dekripsi file audit — untuk alat pemeriksa (dijalankan di perangkat yang sama). */
    @JvmStatic
    @Throws(Exception::class)
    fun readEncryptedFile(file: File): ByteArray {
        val blob: ByteArray = FileInputStream(file).use { input ->
            val bytes = ByteArray(file.length().toInt())
            var offset = 0
            while (offset < bytes.size) {
                val read = input.read(bytes, offset, bytes.size - offset)
                if (read < 0) break
                offset += read
            }
            bytes
        }
        if (blob.size < IV_BYTES + 1) {
            throw IllegalStateException("File audit tidak valid")
        }
        val ivLen = blob[0].toInt() and 0xFF
        val iv = ByteArray(ivLen)
        System.arraycopy(blob, 1, iv, 0, ivLen)
        val ciphertext = ByteArray(blob.size - 1 - ivLen)
        System.arraycopy(blob, 1 + ivLen, ciphertext, 0, ciphertext.size)

        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(GCM_TAG_BITS, iv))
        return cipher.doFinal(ciphertext)
    }

    @Throws(Exception::class)
    private fun encryptToFile(plaintext: ByteArray, outFile: File) {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val iv = cipher.iv
        val ciphertext = cipher.doFinal(plaintext)

        FileOutputStream(outFile).use { out ->
            out.write(iv.size)
            out.write(iv)
            out.write(ciphertext)
        }
    }

    @Throws(Exception::class)
    private fun getOrCreateKey(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE)
        keyStore.load(null)
        if (keyStore.containsAlias(KEY_ALIAS)) {
            val entry = keyStore.getEntry(KEY_ALIAS, null)
            if (entry is KeyStore.SecretKeyEntry) {
                return entry.secretKey
            }
        }

        val keyGenerator =
            KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        keyGenerator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return keyGenerator.generateKey()
    }
}
