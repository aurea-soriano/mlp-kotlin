package pe.demo.irismlponnx

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Ejecuta el pipeline completo exportado desde scikit-learn:
 * StandardScaler -> MLP(16, 8) -> (label, probabilities).
 * Recibe medidas originales en cm; NO volver a normalizarlas en Kotlin.
 */
class IrisClassifier(context: Context) : AutoCloseable {

    data class Prediction(val classIndex: Int, val probabilities: FloatArray)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        // El ONNX se copia a assets con: python train.py
        val modelBytes = context.assets.open("model.onnx").use { it.readBytes() }
        session = env.createSession(modelBytes)
        require(session.inputNames.size == 1) { "Se esperaba una sola entrada ONNX" }
        require(session.outputNames.size == 2) { "Se esperaban label y probabilities" }
        inputName = session.inputNames.first()
    }

    fun predict(valuesCm: FloatArray): Prediction {
        require(valuesCm.size == 4) { "Se necesitan cuatro medidas" }
        require(valuesCm.all { it.isFinite() && it > 0f }) { "Las medidas deben ser positivas" }

        // Input tensor float32 con dimensiones [1, 4].
        val floatBuffer = ByteBuffer.allocateDirect(4 * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        floatBuffer.put(valuesCm)
        floatBuffer.rewind()

        OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1L, 4L)).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { outputs ->
                // skl2onnx con zipmap=false devuelve 2 salidas:
                // [0] label: int64[1]
                // [1] probabilities: float32[1, 3]
                val indices = outputs[0].value as LongArray
                val probabilities = (outputs[1].value as Array<*>)[0] as FloatArray
                require(indices.size == 1 && probabilities.size == 3) {
                    "Dimensiones de salida ONNX inesperadas"
                }
                val classIndex = indices[0].toInt()
                require(classIndex in 0..2) { "Clase desconocida: $classIndex" }
                return Prediction(classIndex, probabilities.copyOf())
            }
        }
    }

    override fun close() {
        session.close()
        // No cerrar env: es un singleton compartido a nivel de JVM.
    }
}
