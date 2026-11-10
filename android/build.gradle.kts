group = "com.andrew.face"
version = "1.0.0"

// Catatan: tidak ada blok `buildscript`/`allprojects` di sini. AGP & Kotlin
// Gradle Plugin disediakan oleh build aplikasi konsumen (Flutter), sehingga
// plugin ini tidak "menerapkan KGP" sendiri dan tidak memicu peringatan
// "Built-in Kotlin" pada Flutter versi mendatang.
//
// Built-in Kotlin (Flutter >= 3.44): plugin cukup memakai
// `com.android.library`; blok `kotlin { compilerOptions {} }` di bawah sudah
// disediakan KGP bawaan Flutter.
plugins {
    id("com.android.library")
}

android {
    namespace = "com.andrew.face"
    compileSdk = 35

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            java.srcDirs("src/test/kotlin")
        }
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    androidResources {
        noCompress += "tflite"
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.1")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.mlkit:face-detection:16.1.7")
    // 2.16.1: `tensorflow-lite-api` transitive sudah memakai namespace unik
    // (`org.tensorflow.lite.api`). Pada 2.14.0 keduanya memakai
    // `org.tensorflow.lite` sehingga manifest merger AGP 8.x gagal di app
    // konsumen. `tensorflow-lite-support` sengaja tidak dipakai — kode plugin
    // hanya butuh `org.tensorflow.lite.Interpreter` (dan support-nya ikut
    // membawa konflik namespace serupa).
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    // LivenessCaptureActivity (native capture, captureLivenessSession) —
    // CameraX + ComponentActivity (LifecycleOwner untuk bindToLifecycle) +
    // ContextCompat. Versi sama dengan referensi capacitor-face-recognition.
    implementation("androidx.core:core:1.13.1")
    implementation("androidx.activity:activity:1.9.3")
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")

    testImplementation("org.jetbrains.kotlin:kotlin-test")
}

