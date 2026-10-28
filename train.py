"""Entrenar la MLP Iris y exportarla exclusivamente para un APK Android."""
from pathlib import Path
import hashlib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent
APK_MODEL = ROOT / "app" / "src" / "main" / "assets" / "model.onnx"


def main() -> None:
    iris = load_iris()
    X = iris.data.astype(np.float32)  # [sepal_length, sepal_width, petal_length, petal_width]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # Preprocesamiento INCLUIDO dentro del archivo ONNX.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu",
            max_iter=1500, random_state=42, early_stopping=False
        )),
    ])
    pipeline.fit(X_train, y_train)

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("features", FloatTensorType([None, 4]))],
        options={id(pipeline.named_steps["mlp"]): {"zipmap": False}},
        target_opset=15,
    )
    onnx.checker.check_model(onnx_model)
    data = onnx_model.SerializeToString()
    APK_MODEL.parent.mkdir(parents=True, exist_ok=True)
    APK_MODEL.write_bytes(data)

    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1 and len(session.get_outputs()) == 2
    assert session.get_inputs()[0].type == "tensor(float)"
    out = session.run(None, {session.get_inputs()[0].name: X_test})
    labels = np.asarray(out[0]).ravel()
    probs = np.asarray(out[1])
    assert probs.shape == (len(X_test), 3), probs.shape
    np.testing.assert_array_equal(labels, pipeline.predict(X_test))
    np.testing.assert_allclose(
        probs, pipeline.predict_proba(X_test), atol=1e-4, rtol=1e-4
    )

    sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
    demo = session.run(None, {session.get_inputs()[0].name: sample})
    print(f"Accuracy sklearn/test: {accuracy_score(y_test, pipeline.predict(X_test)):.3f}")
    print(f"Accuracy ONNX/test:    {accuracy_score(y_test, labels):.3f}")
    print("Clases:", iris.target_names.tolist())
    print("Entrada ONNX:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
    print("Salidas ONNX:", [(x.name, x.type) for x in session.get_outputs()])
    print("Ejemplo:", sample[0].tolist())
    print("Predicción:", iris.target_names[int(demo[0][0])])
    print("Probabilidades:", np.round(demo[1][0], 4).tolist())
    print("SHA256 modelo:", hashlib.sha256(data).hexdigest())
    print("Guardado:", APK_MODEL.relative_to(ROOT))
    print("sklearn vs ONNX: correcto (clases y probabilidades)")


if __name__ == "__main__":
    main()
