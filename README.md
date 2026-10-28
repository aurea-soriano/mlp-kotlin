# Iris MLP → ONNX → Android APK (solo Android)

Proyecto didáctico: entrenar una red neuronal MLP **localmente en Python**, exportarla a **ONNX** e introducir el modelo dentro de una **app nativa Android (Kotlin)**. La predicción ocurre **en el teléfono, sin Internet, CDN, navegador, HTML ni backend**.

> **Este ZIP contiene el código fuente, no un APK compilado.** Hace falta ejecutar `train.py` y compilar en Android Studio; el archivo `model.onnx` se genera localmente.

## Arquitectura

```text
Iris: 150 flores, 4 medidas, 3 clases
             ↓
       Python / scikit-learn
StandardScaler → MLP(16, 8) → salida 3 clases
             ↓
       skl2onnx (opset 15)
             ↓
app/src/main/assets/model.onnx  ← se genera con train.py
             ↓
       Android Studio / Kotlin
          ONNX Runtime
             ↓
         app-debug.apk
             ↓
  Clasificación local en el dispositivo
```

### Qué se utiliza

| Componente | Tecnología | Función |
|---|---|---|
| Dataset | Iris / scikit-learn | 150 registros, 4 medidas, 3 especies |
| Entrenamiento | Python, NumPy, scikit-learn | `StandardScaler` y `MLPClassifier(hidden_layer_sizes=(16, 8))` |
| Exportación | skl2onnx y ONNX | Guarda preprocesamiento + MLP en un solo archivo |
| Verificación | onnx, onnxruntime para Python | Compara predicciones y probabilidades ONNX vs scikit-learn |
| Aplicación Android | Kotlin + XML | Cuatro campos de entrada y barras de probabilidad |
| Inferencia móvil | `com.microsoft.onnxruntime:onnxruntime-android:1.31.1` | Carga el ONNX y realiza predicciones dentro del APK |
| Build | Android Studio, Gradle, JDK | Genera el instalador Android `.apk` |

## Paso 1. Entrenar y exportar en tu equipo

Necesitas **Python 3.10+**. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python train.py
```

Esto crea **un único archivo ONNX**:

```text
app/src/main/assets/model.onnx
```

El script imprime la exactitud sobre un subconjunto de prueba y comprueba que ONNX y scikit-learn producen las mismas clases y probabilidades (dentro de tolerancia numérica).

**Entrada de la MLP:** `[longitud_sépalo, anchura_sépalo, longitud_pétalo, anchura_pétalo]`, en centímetros y sin normalizar manualmente. El propio archivo ONNX incorpora `StandardScaler`. Forma del tensor: `float32[1, 4]`.

**Salida del modelo:** índice entero `0/1/2` y vector de probabilidades (Setosa, Versicolor, Virginica). Las probabilidades son estimaciones del modelo, no certezas sobre la planta real.

## Paso 2. Abrir Android Studio

1. Instala **Android Studio**, **Android SDK 35** y usa **JDK 17 o superior** compatible con tu versión de Gradle.
2. Abre la carpeta raíz de este proyecto con **File → Open**.
3. Permite que Android Studio descargue/sincronice los plugins de Android/Kotlin y la dependencia de ONNX Runtime (requiere Internet *para preparar la compilación*, no para predecir después).
4. Este ZIP incluye scripts Gradle Kotlin DSL pero no el **Gradle Wrapper binario**. Si Android Studio te solicita un wrapper, instala Gradle 8.11.1 y ejecuta desde la raíz:

   ```bash
   gradle wrapper --gradle-version 8.11.1
   ```

   Después ya puedes compilar con `./gradlew :app:assembleDebug` (Windows: `gradlew.bat :app:assembleDebug`). También puedes compilar usando el menú **Build → Build APK(s)** en Android Studio.
5. Puedes conectar un teléfono con depuración USB activada o utilizar un emulador Android.

**Resultado previsto:**

```text
app/build/outputs/apk/debug/app-debug.apk
```

Es un APK firmado con la clave **debug**, válido para probar o compartir en clase. Para distribución de producción deberás firmar una versión **release** con tu propia clave.

### Si aparece un error al compilar

- **«Falta model.onnx»**: vuelve al paso 1 y ejecuta `python train.py` antes de compilar.
- **«No se encontró el SDK 35»**: instálalo desde Android Studio → Settings → SDK Manager.
- **«Gradle Wrapper no disponible»**: sigue la indicación de `gradle wrapper` del paso 2.
- **Errores al resolver dependencias**: conecta a Internet y vuelve a sincronizar Gradle.
- **Las predicciones no cambian tras reentrenar**: recompila e instala de nuevo el APK.

## Paso 3. Código que llama al modelo en Kotlin

Archivo: [`app/src/main/java/pe/demo/irismlponnx/IrisClassifier.kt`](app/src/main/java/pe/demo/irismlponnx/IrisClassifier.kt).

```kotlin
val env = OrtEnvironment.getEnvironment()
val bytes = context.assets.open("model.onnx").use { it.readBytes() }
val session = env.createSession(bytes)

val valores = floatArrayOf(5.1f, 3.5f, 1.4f, 0.2f)
val buffer = ByteBuffer.allocateDirect(4 * Float.SIZE_BYTES)
    .order(ByteOrder.nativeOrder()).asFloatBuffer()
buffer.put(valores)
buffer.rewind()

OnnxTensor.createTensor(env, buffer, longArrayOf(1L, 4L)).use { tensor ->
    session.run(mapOf(session.inputNames.first() to tensor)).use { salida ->
        val clase = (salida[0].value as LongArray)[0].toInt()
        val probabilidades = (salida[1].value as Array<*>)[0] as FloatArray
        println("Clase: $clase")
        println(probabilidades.contentToString())
    }
}
session.close()
```

La versión completa **abre la sesión una sola vez**, la reutiliza durante las predicciones, la cierra al salir y ejecuta el cálculo fuera del hilo principal para no bloquear la interfaz.

## Estructura del proyecto

```text
iris-mlp-onnx-apk/
├── README.md
├── requirements.txt
├── train.py                         # Entrenamiento y exportación SOLO para Android
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
└── app/
    ├── build.gradle.kts
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/
        │   ├── GENERAR_MODELO.txt
        │   └── model.onnx           # Se crea al ejecutar train.py
        ├── java/pe/demo/irismlponnx/
        │   ├── IrisClassifier.kt    # Carga y ejecuta ONNX Runtime
        │   └── MainActivity.kt      # Interfaz, validación y probabilidades
        └── res/                    # XML de interfaz, recursos y estilos
```

## Comprobar que funciona

- Abre la app y utiliza las medidas precargadas: `5.1, 3.5, 1.4, 0.2` cm.
- Pulsa **CLASIFICAR FLOR**. Deberías obtener **Setosa** si el entrenamiento y exportación se completaron correctamente.
- Activa el modo avión, vuelve a ejecutar la inferencia y verifica que funciona sin conexión.
- Si modificas `train.py`, genera el modelo de nuevo y reconstruye el APK.

## Requisitos técnicos

- Android SDK `compileSdk = 35`, `targetSdk = 35`, `minSdk = 24`.
- Android Gradle Plugin `8.9.2`, Gradle `8.11.1`, Kotlin `2.0.21`.
- ONNX Runtime Android `1.31.1` desde Maven Central.
- La app no solicita permiso `INTERNET` y no contiene WebView.

## Notas y límites

- El dataset Iris es educativo; esta app no sustituye una identificación botánica validada.
- La compilación Android **no se probó** en este entorno porque no dispone de Android SDK/Gradle. El código y la estructura XML se revisaron, pero se requiere Android Studio para producir y verificar el APK.
- Tampoco se generó aquí el `.onnx`, porque faltan las bibliotecas Python ONNX y no se pudieron instalar en este entorno. `python train.py` lo genera en tu computadora.

Referencias oficiales: [ONNX Runtime Android](https://onnxruntime.ai/docs/tutorials/mobile/) · [Android: Compilar desde la terminal](https://developer.android.com/build/building-cmdline) · [AGP 8.9 (compatibilidad)](https://developer.android.com/build/releases/agp-8-9-0-release-notes)
