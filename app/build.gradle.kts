plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "pe.demo.irismlponnx"
    compileSdk = 35

    defaultConfig {
        applicationId = "pe.demo.irismlponnx"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // Inference Android nativa; no usa WebView ni CDN.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
}

// La app siempre debe incluir el ONNX entrenado localmente.
// Si falta, explicamos cómo generarlo antes de compilar el APK.
val verifyOnnxModel by tasks.registering {
    group = "verification"
    description = "Verifica que model.onnx esté incluido en los assets Android."
    doLast {
        val modelFile = layout.projectDirectory.file("src/main/assets/model.onnx").asFile
        check(modelFile.isFile && modelFile.length() > 0L) {
            "Falta app/src/main/assets/model.onnx. Ejecuta: pip install -r requirements.txt && python train.py desde la raíz del proyecto."
        }
    }
}

tasks.named("preBuild") {
    dependsOn(verifyOnnxModel)
}
