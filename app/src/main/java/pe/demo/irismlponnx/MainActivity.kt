package pe.demo.irismlponnx

import android.app.Activity
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var destroyed = false
    // Sólo acceder desde el worker, para no ejecutar inferencia en el hilo UI.
    private var model: IrisClassifier? = null

    private val classNames = arrayOf("Setosa", "Versicolor", "Virginica")
    private lateinit var inputFields: List<EditText>
    private lateinit var bars: List<ProgressBar>
    private lateinit var percentages: List<TextView>
    private lateinit var btnPredict: Button
    private lateinit var status: TextView
    private lateinit var resultCard: View
    private lateinit var classResult: TextView
    private lateinit var elapsedResult: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputFields = listOf(
            findViewById(R.id.sepalLength), findViewById(R.id.sepalWidth),
            findViewById(R.id.petalLength), findViewById(R.id.petalWidth)
        )
        bars = listOf(findViewById(R.id.barSetosa), findViewById(R.id.barVersicolor),
            findViewById(R.id.barVirginica))
        percentages = listOf(findViewById(R.id.pctSetosa), findViewById(R.id.pctVersicolor),
            findViewById(R.id.pctVirginica))
        btnPredict = findViewById(R.id.predictButton)
        status = findViewById(R.id.statusText)
        resultCard = findViewById(R.id.resultCard)
        classResult = findViewById(R.id.classResult)
        elapsedResult = findViewById(R.id.elapsedText)

        btnPredict.isEnabled = false
        status.text = getString(R.string.loading_model)
        worker.execute {
            try {
                val loaded = IrisClassifier(applicationContext)
                model = loaded
                runOnUiThread {
                    if (!destroyed) {
                        status.text = getString(R.string.model_ready)
                        btnPredict.isEnabled = true
                    }
                }
            } catch (e: Exception) {
                showStatus("No se pudo cargar model.onnx: ${e.message}")
            }
        }

        btnPredict.setOnClickListener {
            val values = inputFields.map {
                it.text.toString().trim().replace(',', '.').toFloatOrNull()
            }
            if (values.any { it == null || !it.isFinite() || it <= 0f || it >= 100f }) {
                status.text = getString(R.string.invalid_input)
                return@setOnClickListener
            }
            val input = values.map { requireNotNull(it) }.toFloatArray()
            btnPredict.isEnabled = false
            status.text = getString(R.string.running)
            worker.execute {
                try {
                    val classifier = model ?: error("Modelo no disponible")
                    val start = SystemClock.elapsedRealtimeNanos()
                    val prediction = classifier.predict(input)
                    val millis = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                    runOnUiThread {
                        if (!destroyed) {
                            render(prediction, millis)
                            btnPredict.isEnabled = true
                        }
                    }
                } catch (e: Exception) {
                    showStatus("Error en la inferencia: ${e.message}")
                    runOnUiThread { if (!destroyed) btnPredict.isEnabled = true }
                }
            }
        }
    }

    private fun render(prediction: IrisClassifier.Prediction, millis: Double) {
        classResult.text = classNames[prediction.classIndex]
        elapsedResult.text = String.format(Locale.getDefault(), "%.2f ms de inferencia", millis)
        prediction.probabilities.forEachIndexed { i, p ->
            val bounded = p.coerceIn(0f, 1f)
            bars[i].progress = (bounded * 1000).roundToInt()
            percentages[i].text = String.format(Locale.getDefault(), "%.1f %%", bounded * 100)
        }
        resultCard.visibility = View.VISIBLE
        status.text = getString(R.string.local_inference)
    }

    private fun showStatus(message: String) {
        runOnUiThread { if (!destroyed) status.text = message }
    }

    override fun onDestroy() {
        destroyed = true
        worker.execute {
            model?.close()
            model = null
        }
        worker.shutdown()
        super.onDestroy()
    }
}
