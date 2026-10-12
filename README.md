# Online Shoppers Sales Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Clasificación de sesiones de pagina en linea a partir de si resultaron en una compra. La clasificación se hace mediante
un RandomForestClassifier de sklearn, y se registra en MLflow. El ejecutable fue probado usando Python 3.13.3.

El proyecto está hecho para ejecutarse usando uv. Se asume en el estado actual del proyecto, que MLflow está corriendo
localmente en localhost:5000. Una vez se cumpla esta condición, el proyecto se puede ejecutar con el siguiente comando:

```bash
uv run ./online_shoppers_sales_classification/modeling/train.py
```

Para ver todas las opciones:

```bash
uv run ./online_shoppers_sales_classification/modeling/train.py --help
```

### Argumentos
 
| Argumento | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `--file_path` | texto | `./data/raw/online_shoppers_intention.csv` | Ruta del archivo CSV |
| `--n_estimators` | entero | `100` | Número de árboles |
| `--min_samples_split` | entero | `5` | Mínimo de muestras para separar un nodo |
| `--max_depth` | entero | `10` | Profundidad máxima de cada árbol |
| `--class_weight` | texto | `balanced` | Pesos por clase (`balanced` o `balanced_subsample`) |
| `--max_leaf_nodes` | entero | sin límite | Máximo de hojas por árbol |
| `--min_samples_leaf` | entero o decimal | `2` | Entero = número mínimo de muestras en una hoja; decimal (por ejemplo `0.01`) = fracción de las muestras |
| `--criterion` | `gini`, `entropy`, `log_loss` | `gini` | Función para medir la calidad de una separación |
| `--random_state` | entero | `42` | Semilla del split y del modelo |
| `--run_name` | texto | `Random Forest Run` | Nombre de la corrida en MLflow |
| `--experiment_name` | texto | `Actividad Semana 4` | Experimento de MLflow |
| `--model_name` | texto | `Online Shoppers Sales Classifier` | Nombre del modelo registrado |


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         online_shoppers_sales_classification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── online_shoppers_sales_classification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes online_shoppers_sales_classification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

