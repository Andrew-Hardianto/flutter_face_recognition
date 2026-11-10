import Foundation

/// Pemeriksaan integritas perangkat (Fase 4) — heuristik sisi klien.
///
/// CATATAN PENTING: semua pemeriksaan di sini adalah SINYAL RISIKO, bukan bukti.
/// Deteksi jailbreak/hook yang benar-benar kuat membutuhkan device attestation
/// terverifikasi server (App Attest) — di luar scope plugin ini karena backend
/// hanya menyediakan penyimpanan vector (lihat plan Fase 3.4).
///
/// Hasil dipakai oleh aplikasi untuk mengambil keputusan (tolak, minta tantangan
/// tambahan, atau catat untuk audit), bukan diandalkan sebagai gerbang tunggal.
enum DeviceIntegrity {

    /// Build simulator — bukan perangkat fisik.
    static var isEmulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    /// Build debuggable (development) — hasil liveness dari build debug tidak untuk produksi.
    static var isDebuggable: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    /// Heuristik jailbreak: path file umum, env DYLD, dan probe escape sandbox.
    static var isJailbroken: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        let paths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/",
            "/usr/bin/ssh",
            "/private/var/stash",
            "/usr/libexec/sftp-server",
            "/Library/Frameworks/CydiaSubstrate.framework",
        ]
        for path in paths where FileManager.default.fileExists(atPath: path) {
            return true
        }
        if injectedLibraryPresent { return true }
        return canEscapeSandbox()
        #endif
    }

    /// Heuristik framework hook (Frida / Substrate / tweak injection).
    static var hooksDetected: Bool {
        if injectedLibraryPresent { return true }
        let hookPaths = [
            "/usr/lib/frida",
            "/usr/lib/TweakInject",
            "/Library/MobileSubstrate",
            "/usr/sbin/frida-server",
        ]
        for path in hookPaths where FileManager.default.fileExists(atPath: path) {
            return true
        }
        return false
    }

    /// Snapshot lengkap untuk disertakan di hasil verifikasi.
    static func snapshot() -> [String: Any] {
        return [
            "isEmulator": isEmulator,
            "isRooted": isJailbroken,
            "isDebuggable": isDebuggable,
            "hooksDetected": hooksDetected,
        ]
    }

    private static var injectedLibraryPresent: Bool {
        guard let env = getenv("DYLD_INSERT_LIBRARIES") else { return false }
        return !String(cString: env).isEmpty
    }

    private static func canEscapeSandbox() -> Bool {
        // Jika bisa menulis ke luar sandbox → jailbreak
        let probePath = "/private/jailbreak_probe.txt"
        do {
            try "probe".write(toFile: probePath, atomically: true, encoding: .utf8)
            try? FileManager.default.removeItem(atPath: probePath)
            return true
        } catch {
            return false
        }
    }
}
