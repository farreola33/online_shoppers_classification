from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, ConfusionMatrixDisplay
from imblearn.metrics import geometric_mean_score
import mlflow
from mlflow.models import infer_signature
import tempfile

def entero_o_decimal(valor):
    # Función de ayuda para que argparse lea correctamente el parámetro min_samples_leaf
    return int(valor) if valor.lstrip("-").isdigit() else float(valor)

def inicializar_parser() -> ArgumentParser:
    """Inicializar parser para permitir ejecutar el entrenamiento desde una línea de comando, sin necesitar
    modificaciones al código

    :return: Parser inicializado
    """
    parser = ArgumentParser(
        description="Ejecutar un pipeline de entrenamiento y evaluación para un modelo Random Forest")

    parser.add_argument("--file_path", type=str, default="./data/raw/online_shoppers_intention.csv",
                        help="Dirección y nombre del archivo con los datos")

    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Número de árboles en el modelo (default: 100)")

    parser.add_argument("--min_samples_split", type=int, default=5,
                        help="Número mínimo de muestras requerido para separar un nodo (default: 5)")

    parser.add_argument("--max_depth", type=int, default=None,
                        help="Profundidad máxima de cada árbol (default: 10)")

    parser.add_argument("--class_weight", type=str, choices=["balanced", "balanced_subsample"],
                        default="balanced",
                        help="Distribución de pesos para cada clase")

    parser.add_argument("--max_leaf_nodes", type=int, default=None,
                        help='Máximo de hojas permitido en un árbol (default: None)')

    parser.add_argument("--min_samples_leaf", type=entero_o_decimal, default=2,
                        help='Número mínimo de muestras en una hoja (default: 2)')

    parser.add_argument("--criterion", type=str, choices=["gini", "entropy", "log_loss"], default="gini",
                        help="Función para determinar la calidad de una separación de nodo (default: gini)")

    parser.add_argument("--random_state", type=int, default=42,
                        help="Control de aleatoriedad en la ejecución del entrenamiento (default: 42)")

    parser.add_argument("--run_name", type=str, default="Random Forest Run",
                        help="Nombre de la corrida en MLflow")

    parser.add_argument("--experiment_name", type=str, default="Actividad Semana 4",
                        help="Nombre del experimento a registrar en MLflow")

    parser.add_argument("--model_name", type=str, default="Online Shoppers Sales Classifier",
                        help="Nombre del modelo a registrar en MLflow")

    return parser


def cargar_datos(file_path: str, cols_numericas_categoricas: list | None = None,
                 cols_a_eliminar: list | None = None) -> pd.DataFrame:
    """Cargar datos a usar en el proyecto

    :param file_path: dirección del archivo .csv a leer
    :param cols_numericas_categoricas: Columnas categóricas, con categorías numéricas, que se convertirán a string
    :param cols_a_eliminar: Columnas del dataset a eliminar
    :return: un DataFrame con los datos leídos.
    """
    loaded_df = pd.read_csv(file_path)

    loaded_df = loaded_df.drop(cols_a_eliminar, axis=1) if cols_a_eliminar is not None else loaded_df

    if cols_numericas_categoricas is None:
        return loaded_df

    loaded_df[cols_numericas_categoricas] = loaded_df[cols_numericas_categoricas].astype('str')

    return loaded_df


def limpieza_de_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar datos leídos antes de pasar a los pasos de entrenamiento y prueba

    :param df: DataFrame con los datos a limpiar
    :return: un DataFrame con los datos limpiados.
    """

    # Como nota:
    # Anteriormente también se ejecutaba un agrupamiento de columnas infrecuentes como paso de limpieza a nivel de todos los datos
    # Esto corre el riesgo de provocar fugas de datos; por lo tanto, ese paso ahora se ejecuta en el ajuste de transformador

    # Se eliminan duplicados
    clean_df = df.drop_duplicates()

    return clean_df


def crear_conjuntos_de_entrenamiento_y_prueba(df: pd.DataFrame, variable_de_salida: str,
                                              tamano_conjunto_de_prueba: float, semilla: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crear conjuntos de entrenamiento y prueba para desarrollar el modelo
    :param df: DataFrame con los datos de entrada
    :param variable_de_salida: variable a clasificar
    :param tamano_conjunto_de_prueba: tamaño (en porcentaje) del conjunto de prueba respecto al total de datos
    :param semilla: valor de semilla para controlar la aleatoriedad de la creación de conjuntos
    :return: tuple de DataFrames y Series con conjuntos de entrenamiento y prueba
    """
    X = df.drop(variable_de_salida, axis=1)
    y = df[variable_de_salida]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tamano_conjunto_de_prueba, stratify=y,
                                                        random_state=semilla)

    return (X_train, X_test, y_train, y_test)


def inicializar_transformador(ohe_cols: list, yj_cols: list, max_categorias: int | None = 10) -> ColumnTransformer:
    """Inicializar transformador que se usará para codificar y estandarizar los datos
    :param ohe_cols: columnas a las que se le aplicará One-Hot Encoding
    :param yj_cols: columnas a las que se le aplicará la transformación de Yeo-Johnson
    :param max_categorias: número máximo de categorías a crear en el OneHotEncoder
    :return: transformador de tipo ColumnTransformer
    """

    # Nota: se espera que el paso de fit() se ejecute fuera de la inicialización
    # La función regresa un transformador sin ajustar

    col_transformer = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(drop=None, sparse_output=False, handle_unknown="infrequent_if_exist", max_categories=max_categorias), ohe_cols),
            ('yeo-johnson', PowerTransformer(method='yeo-johnson'), yj_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return col_transformer

def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series, transformer_config: dict, hiperparametros: dict) -> Pipeline:
    """Entrenar un modelo de RandomForestClassifier con datos e hiperparámetros especificados

    :param X_train: Datos de entrada sin preprocesar
    :param y_train: Datos de salida
    :param transformer_config: diccionario de configuración para inicializar el ColumnTransformer
    :param hiperparametros: hiperparámetros para entrenar el modelo
    :return: Pipeline con el modelo entrenado
    """

    # Se extraen los valores de configuración del transformador (sin ajustarlo)
    ohe_cols = transformer_config['ohe_cols']
    yj_cols = transformer_config['yj_cols']
    max_categorias = transformer_config['max_categorias']

    # Se inicializa el transformador
    col_transformer = inicializar_transformador(ohe_cols, yj_cols, max_categorias=max_categorias)

    pipeline = Pipeline([
        ("preprocesamiento", col_transformer),
        ("modelo", RandomForestClassifier(**hiperparametros))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

def obtener_predicciones(X: pd.DataFrame, pipeline: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    """A partir de datos de entrada y un modelo ya entrenado, se calculan predicciones

    :param X: datos de entrada que se usarán para generar la predicción
    :param pipeline: Pipeline con pasos de preprocesamiento y entrenamiento de modelo
    :return: tuple con predicciones del modelo a partir de los datos de entrada X
    """

    # Primero se valida que el pipeline ya esté ajustado
    check_is_fitted(pipeline)

    # Se realizan las predicciones
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:,1]

    return y_pred, y_pred_proba

def evaluar_predicciones(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Evaluar las predicciones hechas por un modelo con distintas métricas de clasificación

    :param y_test: Datos de referencia para la evaluación
    :param y_pred: Datos predichos por el modelo
    :param y_proba: Probabilidades predichas por el modelo
    :return: diccionario con las métricas evaluadas
    """

    # Se crea un diccionario con las métricas a evaluar
    evaluaciones = {"g_mean": geometric_mean_score(y_test, y_pred),
                 "roc_auc": roc_auc_score(y_test, y_proba),
                 "f1": f1_score(y_test, y_pred),
                 "pr_auc": average_precision_score(y_test, y_proba)
                 }

    return evaluaciones

def obtener_matriz_de_confusion(y_test: pd.Series, y_pred: np.ndarray) -> ConfusionMatrixDisplay:
    """Obtener la matriz de confusión a partir de predicciones hechas por un modelo

    :param y_test: Datos de referencia para la evaluacion
    :param y_pred: Datos predichos por el modelo
    :return: Matriz de confusión
    """
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm.ax_.set_title("Matriz de confusión")
    plt.close(cm.figure_) # Con el fin de evitar el despliegue automático de la figura
    return cm


def loggear_corrida(tracking_uri: str,
                    nombre_de_experimento: str,
                    nombre_de_corrida: str,
                    nombre_de_modelo: str,
                    hiperparametros: dict,
                    metricas: dict,
                    matriz_de_confusion: ConfusionMatrixDisplay,
                    X_train: pd.DataFrame,
                    pipeline: Pipeline):
    """Usar MLflow para loggear el entrenamiento y prueba del modelo
    :param tracking_uri: Tracking URI usado por mlflow
    :param nombre_de_experimento: Nombre del experimento
    :param nombre_de_corrida: Nombre de la corrida
    :param nombre_de_modelo: Nombre del modelo a registrar en MLflow
    :param hiperparametros: Diccionario de hiperparámetros usados
    :param metricas: Diccionario de métricas de evaluación
    :param matriz_de_confusion: Gráfica de la matriz de confusión
    :param X_train: Datos de entrenamiento usados
    :param pipeline: Pipeline con el preprocesamiento y modelo entrenado
    """

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(nombre_de_experimento)

    # Se obtiene la firma del modelo a partir de los datos de entrenamiento
    firma = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name=nombre_de_corrida):
        mlflow.log_params(hiperparametros)
        mlflow.log_metrics(metricas)

        mlflow.log_figure(matriz_de_confusion.figure_, artifact_file="plots/confusion_matrix.png")
        model_info = mlflow.sklearn.log_model(pipeline, signature=firma, input_example=X_train.head(10),
                                              registered_model_name=nombre_de_modelo)

        with tempfile.TemporaryDirectory() as d:
            artifacts = mlflow.artifacts.download_artifacts(artifact_uri=model_info.model_uri, dst_path=d)
            mlflow.log_artifacts(artifacts, artifact_path="model")


def ejecutar_pipeline(data_file_path: str,
                         cols_numericas_categoricas: list,
                         variable_de_salida: str,
                         tamano_conjunto_de_prueba: float,
                         transformer_config: dict,
                         cols_a_eliminar: list,
                         hiperparametros: dict,
                         tracking_uri: str,
                         nombre_de_experimento: str,
                         nombre_de_corrida: str,
                         nombre_de_modelo: str,
                         semilla: int):
    """Ejecutar pipeline completo: lectura de datos, preprocesamiento, entrenamiento, evaluación y registro en MLflow

    :param data_file_path: Nombre del archivo con los datos
    :param cols_numericas_categoricas: Columnas categoricas con categorías numéricas
    :param variable_de_salida: Nombre de la variable de salida
    :param tamano_conjunto_de_prueba: Tamaño proporcional del conjunto de prueba
    :param transformer_config: Diccionario de configuración para el transformador
    :param cols_a_eliminar: Columnas de los datos a eliminar
    :param hiperparametros: Diccionario de hiperparametros usados para entrenar el modelo
    :param tracking_uri: Tracking URI usado por mlflow
    :param nombre_de_experimento: Nombre del experimento
    :param nombre_de_corrida: Nombre de la corrida
    :param nombre_de_modelo: Nombre del modelo a registrar en MLflow
    :param semilla: Semilla para controlar aleatoriedad
    """

    df = cargar_datos(data_file_path, cols_numericas_categoricas, cols_a_eliminar)

    clean_df = limpieza_de_datos(df)

    X_train, X_test, y_train, y_test = crear_conjuntos_de_entrenamiento_y_prueba(clean_df, variable_de_salida=variable_de_salida,
                                                                                 tamano_conjunto_de_prueba=tamano_conjunto_de_prueba,
                                                                                 semilla=semilla)

    rf_pipeline = entrenar_modelo(X_train, y_train, transformer_config=transformer_config,
                                                hiperparametros=hiperparametros)

    y_pred, y_proba = obtener_predicciones(X_test, rf_pipeline)

    evaluaciones = evaluar_predicciones(y_test, y_pred, y_proba)
    matriz_de_confusion = obtener_matriz_de_confusion(y_test, y_pred)

    loggear_corrida(tracking_uri=tracking_uri,
                    nombre_de_experimento=nombre_de_experimento,
                    nombre_de_corrida=nombre_de_corrida,
                    nombre_de_modelo=nombre_de_modelo,
                    hiperparametros=hiperparametros,
                    metricas=evaluaciones,
                    matriz_de_confusion=matriz_de_confusion,
                    X_train=X_train,
                    pipeline=rf_pipeline)


if __name__ == "__main__":
    parser = inicializar_parser()
    args = parser.parse_args()

    file_path = args.file_path
    cols_numericas_categoricas = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
    variable_de_salida = "Revenue"
    tamano_conjunto_de_prueba = 0.2
    transformer_config = {
        "ohe_cols": ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType'],
        "yj_cols": ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'PageValues', 'SpecialDay'],
        "max_categorias": 10
    }
    cols_a_eliminar = ['ExitRates']
    semilla = args.random_state

    rf_params = {
        "n_estimators": args.n_estimators,
        "min_samples_split": args.min_samples_split,
        "max_depth": args.max_depth,
        "class_weight": args.class_weight,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "criterion": args.criterion,
        "random_state": semilla
    }

    tracking_uri = "http://localhost:5000"
    experiment_name = args.experiment_name
    model_name = args.model_name
    run_name = args.run_name

    ejecutar_pipeline(data_file_path=file_path,
                         cols_numericas_categoricas=cols_numericas_categoricas,
                         variable_de_salida=variable_de_salida,
                         tamano_conjunto_de_prueba=tamano_conjunto_de_prueba,
                         transformer_config=transformer_config,
                         cols_a_eliminar=cols_a_eliminar,
                         hiperparametros=rf_params,
                         tracking_uri=tracking_uri,
                         nombre_de_experimento=experiment_name,
                         nombre_de_corrida=run_name,
                         nombre_de_modelo=model_name,
                         semilla=semilla)
