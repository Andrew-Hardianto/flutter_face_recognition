import CryptoKit
import Foundation
import Security

/// Jejak audit lokal terenkripsi (Fase 4).
///
/// Setiap sesi liveness native menyimpan hasil + frame yang dipakai dalam bentuk
/// AES-256-GCM dengan kunci simetris di Keychain
/// (kSecAttrAccessibleWhenUnlockedThisDeviceOnly — tidak ikut backup, tidak
/// bisa dipindahkan ke perangkat lain). File berada di Application Support
/// dengan proteksi file penuh (terenkripsi saat perangkat terkunci).
///
/// Dipakai untuk pemeriksaan fraud HRIS — kebijakan retensi mengikuti persetujuan
/// dan UU PDP (hak hapus: cukup hapus folder liveness_audit).
enum AuditStore {

    enum AuditError: Error {
        case keyGenerationFailed
        case keychainError(OSStatus)
    }

    private static let keychainAccount = "andrew.face.audit.key"
    private static let auditDirName = "liveness_audit"

    /// Simpan JSON sesi + frame yang dipakai, semuanya terenkripsi.
    /// Mengembalikan sessionId.
    static func storeSession(json: Data, frames: [Data]) throws -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let sessionId = "session_" + formatter.string(from: Date())

        let dir = try auditDirectory()
        let key = try loadOrCreateKey()

        try encryptToFile(json, key: key,
                          url: dir.appendingPathComponent("\(sessionId).json.enc"))
        for (index, frame) in frames.enumerated() {
            try encryptToFile(frame, key: key,
                              url: dir.appendingPathComponent("\(sessionId)_frame_\(index).jpg.enc"))
        }
        return sessionId
    }

    /// Dekripsi file audit — untuk alat pemeriksa (dijalankan di perangkat yang sama).
    static func readEncryptedFile(at url: URL) throws -> Data {
        let blob = try Data(contentsOf: url)
        let key = try loadOrCreateKey()
        let box = try AES.GCM.SealedBox(combined: blob)
        return try AES.GCM.open(box, using: key)
    }

    // MARK: - Crypto

    private static func encryptToFile(_ plaintext: Data, key: SymmetricKey, url: URL) throws {
        let sealed = try AES.GCM.seal(plaintext, using: key)
        guard let combined = sealed.combined else {
            throw AuditError.keyGenerationFailed
        }
        try combined.write(to: url)
    }

    /// Kunci AES-256 dari Keychain — dibuat sekali, tidak pernah keluar perangkat.
    private static func loadOrCreateKey() throws -> SymmetricKey {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecSuccess, let data = item as? Data, data.count == 32 {
            return SymmetricKey(data: data)
        }

        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw AuditError.keyGenerationFailed
        }
        let keyData = Data(bytes)

        // Bersihkan sisa item lama (jika ada) sebelum menambah
        SecItemDelete(query as CFDictionary)

        let attributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: keychainAccount,
            kSecValueData as String: keyData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        let addStatus = SecItemAdd(attributes as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw AuditError.keychainError(addStatus)
        }
        return SymmetricKey(data: keyData)
    }

    private static func auditDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent(auditDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Proteksi file penuh — terenkripsi saat perangkat terkunci
        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.complete], ofItemAtPath: dir.path)
        return dir
    }
}
