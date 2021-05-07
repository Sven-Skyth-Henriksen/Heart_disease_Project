{
 "cells": [
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00000-ecf67c49-271d-4de3-b94d-6c7524881e40",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "8ad57f32",
    "execution_start": 1620375813931,
    "execution_millis": 3,
    "deepnote_cell_type": "code"
   },
   "source": "# Start writing code here...",
   "execution_count": 1,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00001-afdce475-1e04-4c52-b60f-3a97aa0c9aff",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "dc6e7c2",
    "execution_start": 1620375813941,
    "execution_millis": 1933,
    "deepnote_cell_type": "code"
   },
   "source": "import time\nfrom IPython.display import clear_output\nimport numpy    as np\nimport pandas   as pd\nimport seaborn  as sb\nimport matplotlib.pyplot as plt\nimport sklearn  as skl\n\nfrom sklearn import pipeline      # Pipeline\nfrom sklearn import preprocessing # OrdinalEncoder, LabelEncoder\nfrom sklearn import impute\nfrom sklearn import compose\nfrom sklearn import model_selection # train_test_split\nfrom sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix\nfrom sklearn import set_config\n\nset_config(display='diagram') # Useful for display the pipeline",
   "execution_count": 2,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00002-08f85c71-df5c-4d20-a6b4-42236c7ef10b",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "296427eb",
    "execution_start": 1620380912504,
    "execution_millis": 10848,
    "deepnote_cell_type": "code"
   },
   "source": "!pip install xgboost==1.4.1",
   "execution_count": 29,
   "outputs": [
    {
     "name": "stdout",
     "text": "Collecting xgboost==1.4.1\n  Downloading xgboost-1.4.1-py3-none-manylinux2010_x86_64.whl (166.7 MB)\n\u001b[K     |████████████████████████████████| 166.7 MB 76.6 MB/s \n\u001b[?25hRequirement already satisfied: numpy in /shared-libs/python3.7/py/lib/python3.7/site-packages (from xgboost==1.4.1) (1.19.5)\nRequirement already satisfied: scipy in /shared-libs/python3.7/py/lib/python3.7/site-packages (from xgboost==1.4.1) (1.6.3)\nInstalling collected packages: xgboost\nSuccessfully installed xgboost-1.4.1\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00003-0ee92283-976a-442b-b514-b8436703f812",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "16b9bd07",
    "execution_start": 1620380954923,
    "execution_millis": 3070,
    "deepnote_cell_type": "code"
   },
   "source": "!pip install lightgbm==3.2.1",
   "execution_count": 31,
   "outputs": [
    {
     "name": "stdout",
     "text": "Collecting lightgbm==3.2.1\n  Downloading lightgbm-3.2.1-py3-none-manylinux1_x86_64.whl (2.0 MB)\n\u001b[K     |████████████████████████████████| 2.0 MB 18.9 MB/s \n\u001b[?25hRequirement already satisfied: numpy in /shared-libs/python3.7/py/lib/python3.7/site-packages (from lightgbm==3.2.1) (1.19.5)\nRequirement already satisfied: scikit-learn!=0.22.0 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from lightgbm==3.2.1) (0.24.2)\nRequirement already satisfied: scipy in /shared-libs/python3.7/py/lib/python3.7/site-packages (from lightgbm==3.2.1) (1.6.3)\nRequirement already satisfied: wheel in /root/venv/lib/python3.7/site-packages (from lightgbm==3.2.1) (0.36.2)\nRequirement already satisfied: joblib>=0.11 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from scikit-learn!=0.22.0->lightgbm==3.2.1) (1.0.1)\nRequirement already satisfied: threadpoolctl>=2.0.0 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from scikit-learn!=0.22.0->lightgbm==3.2.1) (2.1.0)\nInstalling collected packages: lightgbm\nSuccessfully installed lightgbm-3.2.1\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00004-a16169fc-97b0-47ed-b384-c4ba210d9be6",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "f3be0113",
    "execution_start": 1620380983936,
    "execution_millis": 5770,
    "deepnote_cell_type": "code"
   },
   "source": "!pip install catboost==0.25.1",
   "execution_count": 33,
   "outputs": [
    {
     "name": "stdout",
     "text": "Collecting catboost==0.25.1\n  Downloading catboost-0.25.1-cp37-none-manylinux1_x86_64.whl (67.3 MB)\n\u001b[K     |████████████████████████████████| 67.3 MB 40 kB/s \n\u001b[?25hRequirement already satisfied: numpy>=1.16.0 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from catboost==0.25.1) (1.19.5)\nRequirement already satisfied: pandas>=0.24.0 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from catboost==0.25.1) (1.2.4)\nRequirement already satisfied: plotly in /shared-libs/python3.7/py/lib/python3.7/site-packages (from catboost==0.25.1) (4.14.3)\nRequirement already satisfied: six in /shared-libs/python3.7/py-core/lib/python3.7/site-packages (from catboost==0.25.1) (1.16.0)\nRequirement already satisfied: scipy in /shared-libs/python3.7/py/lib/python3.7/site-packages (from catboost==0.25.1) (1.6.3)\nCollecting graphviz\n  Downloading graphviz-0.16-py2.py3-none-any.whl (19 kB)\nRequirement already satisfied: matplotlib in /shared-libs/python3.7/py/lib/python3.7/site-packages (from catboost==0.25.1) (3.4.1)\nRequirement already satisfied: pytz>=2017.3 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from pandas>=0.24.0->catboost==0.25.1) (2021.1)\nRequirement already satisfied: python-dateutil>=2.7.3 in /shared-libs/python3.7/py-core/lib/python3.7/site-packages (from pandas>=0.24.0->catboost==0.25.1) (2.8.1)\nRequirement already satisfied: kiwisolver>=1.0.1 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from matplotlib->catboost==0.25.1) (1.3.1)\nRequirement already satisfied: pillow>=6.2.0 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from matplotlib->catboost==0.25.1) (8.2.0)\nRequirement already satisfied: cycler>=0.10 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from matplotlib->catboost==0.25.1) (0.10.0)\nRequirement already satisfied: pyparsing>=2.2.1 in /shared-libs/python3.7/py-core/lib/python3.7/site-packages (from matplotlib->catboost==0.25.1) (2.4.7)\nRequirement already satisfied: retrying>=1.3.3 in /shared-libs/python3.7/py/lib/python3.7/site-packages (from plotly->catboost==0.25.1) (1.3.3)\nInstalling collected packages: graphviz, catboost\nSuccessfully installed catboost-0.25.1 graphviz-0.16\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00002-adea1ac8-ca06-4699-910f-5c374a3e0bf4",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "1c427e09",
    "execution_start": 1620383571469,
    "execution_millis": 0,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.linear_model   import LogisticRegression\nfrom sklearn.linear_model   import RidgeClassifier\nfrom sklearn.svm            import SVC\nfrom sklearn.svm            import NuSVC\nfrom sklearn.svm            import LinearSVC\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.neighbors      import KNeighborsClassifier\nfrom sklearn.gaussian_process import GaussianProcessClassifier\nfrom sklearn.gaussian_process.kernels import RBF\nfrom sklearn.naive_bayes    import GaussianNB\nfrom sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\nfrom sklearn.ensemble       import StackingClassifier\n\n#### TREE\nfrom sklearn.tree          import DecisionTreeClassifier\nfrom sklearn.ensemble      import RandomForestClassifier\nfrom sklearn.ensemble      import ExtraTreesClassifier\nfrom sklearn.ensemble      import AdaBoostClassifier\nfrom sklearn.ensemble      import GradientBoostingClassifier\nfrom sklearn.experimental  import enable_hist_gradient_boosting\nfrom sklearn.ensemble      import HistGradientBoostingClassifier\nfrom xgboost               import XGBClassifier\nfrom lightgbm              import LGBMClassifier\nfrom catboost              import CatBoostClassifier\n#from ngboost               import NGBClassifier\n#from rgf.sklearn           import RGFClassifier, FastRGFClassifier\n\n########################################################### REGRESSORS\nfrom sklearn.linear_model  import ElasticNet, Ridge, Lasso, BayesianRidge, ARDRegression, TweedieRegressor\nfrom sklearn.svm           import LinearSVR, NuSVR, SVR\nfrom sklearn.ensemble      import BaggingRegressor\nfrom sklearn.kernel_ridge  import KernelRidge",
   "execution_count": 50,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00002-1a163be0-4239-4df6-af3d-15c5448c05a9",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "fcda6c8e",
    "execution_start": 1620375815875,
    "execution_millis": 0,
    "deepnote_cell_type": "code"
   },
   "source": "df =pd.read_csv('heart.csv')",
   "execution_count": 3,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00003-1d30b783-0e8c-480d-8325-7277b49683b2",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "3f898c48",
    "execution_start": 1620375815876,
    "execution_millis": 37,
    "deepnote_cell_type": "code"
   },
   "source": "df\n",
   "execution_count": 4,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 4,
     "data": {
      "application/vnd.deepnote.dataframe.v2+json": {
       "row_count": 303,
       "column_count": 14,
       "columns": [
        {
         "name": "age",
         "dtype": "int64",
         "stats": {
          "unique_count": 41,
          "nan_count": 0,
          "min": "29",
          "max": "77",
          "histogram": [
           {
            "bin_start": 29,
            "bin_end": 33.8,
            "count": 1
           },
           {
            "bin_start": 33.8,
            "bin_end": 38.6,
            "count": 11
           },
           {
            "bin_start": 38.6,
            "bin_end": 43.4,
            "count": 33
           },
           {
            "bin_start": 43.4,
            "bin_end": 48.2,
            "count": 38
           },
           {
            "bin_start": 48.2,
            "bin_end": 53,
            "count": 37
           },
           {
            "bin_start": 53,
            "bin_end": 57.8,
            "count": 60
           },
           {
            "bin_start": 57.8,
            "bin_end": 62.6,
            "count": 63
           },
           {
            "bin_start": 62.6,
            "bin_end": 67.4,
            "count": 43
           },
           {
            "bin_start": 67.4,
            "bin_end": 72.19999999999999,
            "count": 14
           },
           {
            "bin_start": 72.19999999999999,
            "bin_end": 77,
            "count": 3
           }
          ]
         }
        },
        {
         "name": "sex",
         "dtype": "int64",
         "stats": {
          "unique_count": 2,
          "nan_count": 0,
          "min": "0",
          "max": "1",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.1,
            "count": 96
           },
           {
            "bin_start": 0.1,
            "bin_end": 0.2,
            "count": 0
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.30000000000000004,
            "count": 0
           },
           {
            "bin_start": 0.30000000000000004,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.5,
            "count": 0
           },
           {
            "bin_start": 0.5,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.7000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.7000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 0.9,
            "count": 0
           },
           {
            "bin_start": 0.9,
            "bin_end": 1,
            "count": 207
           }
          ]
         }
        },
        {
         "name": "cp",
         "dtype": "int64",
         "stats": {
          "unique_count": 4,
          "nan_count": 0,
          "min": "0",
          "max": "3",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.3,
            "count": 143
           },
           {
            "bin_start": 0.3,
            "bin_end": 0.6,
            "count": 0
           },
           {
            "bin_start": 0.6,
            "bin_end": 0.8999999999999999,
            "count": 0
           },
           {
            "bin_start": 0.8999999999999999,
            "bin_end": 1.2,
            "count": 50
           },
           {
            "bin_start": 1.2,
            "bin_end": 1.5,
            "count": 0
           },
           {
            "bin_start": 1.5,
            "bin_end": 1.7999999999999998,
            "count": 0
           },
           {
            "bin_start": 1.7999999999999998,
            "bin_end": 2.1,
            "count": 87
           },
           {
            "bin_start": 2.1,
            "bin_end": 2.4,
            "count": 0
           },
           {
            "bin_start": 2.4,
            "bin_end": 2.6999999999999997,
            "count": 0
           },
           {
            "bin_start": 2.6999999999999997,
            "bin_end": 3,
            "count": 23
           }
          ]
         }
        },
        {
         "name": "trestbps",
         "dtype": "int64",
         "stats": {
          "unique_count": 49,
          "nan_count": 0,
          "min": "94",
          "max": "200",
          "histogram": [
           {
            "bin_start": 94,
            "bin_end": 104.6,
            "count": 10
           },
           {
            "bin_start": 104.6,
            "bin_end": 115.2,
            "count": 42
           },
           {
            "bin_start": 115.2,
            "bin_end": 125.8,
            "count": 67
           },
           {
            "bin_start": 125.8,
            "bin_end": 136.4,
            "count": 74
           },
           {
            "bin_start": 136.4,
            "bin_end": 147,
            "count": 57
           },
           {
            "bin_start": 147,
            "bin_end": 157.6,
            "count": 27
           },
           {
            "bin_start": 157.6,
            "bin_end": 168.2,
            "count": 13
           },
           {
            "bin_start": 168.2,
            "bin_end": 178.8,
            "count": 8
           },
           {
            "bin_start": 178.8,
            "bin_end": 189.39999999999998,
            "count": 3
           },
           {
            "bin_start": 189.39999999999998,
            "bin_end": 200,
            "count": 2
           }
          ]
         }
        },
        {
         "name": "chol",
         "dtype": "int64",
         "stats": {
          "unique_count": 152,
          "nan_count": 0,
          "min": "126",
          "max": "564",
          "histogram": [
           {
            "bin_start": 126,
            "bin_end": 169.8,
            "count": 12
           },
           {
            "bin_start": 169.8,
            "bin_end": 213.6,
            "count": 73
           },
           {
            "bin_start": 213.6,
            "bin_end": 257.4,
            "count": 106
           },
           {
            "bin_start": 257.4,
            "bin_end": 301.2,
            "count": 69
           },
           {
            "bin_start": 301.2,
            "bin_end": 345,
            "count": 35
           },
           {
            "bin_start": 345,
            "bin_end": 388.79999999999995,
            "count": 3
           },
           {
            "bin_start": 388.79999999999995,
            "bin_end": 432.59999999999997,
            "count": 4
           },
           {
            "bin_start": 432.59999999999997,
            "bin_end": 476.4,
            "count": 0
           },
           {
            "bin_start": 476.4,
            "bin_end": 520.2,
            "count": 0
           },
           {
            "bin_start": 520.2,
            "bin_end": 564,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "fbs",
         "dtype": "int64",
         "stats": {
          "unique_count": 2,
          "nan_count": 0,
          "min": "0",
          "max": "1",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.1,
            "count": 258
           },
           {
            "bin_start": 0.1,
            "bin_end": 0.2,
            "count": 0
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.30000000000000004,
            "count": 0
           },
           {
            "bin_start": 0.30000000000000004,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.5,
            "count": 0
           },
           {
            "bin_start": 0.5,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.7000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.7000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 0.9,
            "count": 0
           },
           {
            "bin_start": 0.9,
            "bin_end": 1,
            "count": 45
           }
          ]
         }
        },
        {
         "name": "restecg",
         "dtype": "int64",
         "stats": {
          "unique_count": 3,
          "nan_count": 0,
          "min": "0",
          "max": "2",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.2,
            "count": 147
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 1,
            "count": 0
           },
           {
            "bin_start": 1,
            "bin_end": 1.2000000000000002,
            "count": 152
           },
           {
            "bin_start": 1.2000000000000002,
            "bin_end": 1.4000000000000001,
            "count": 0
           },
           {
            "bin_start": 1.4000000000000001,
            "bin_end": 1.6,
            "count": 0
           },
           {
            "bin_start": 1.6,
            "bin_end": 1.8,
            "count": 0
           },
           {
            "bin_start": 1.8,
            "bin_end": 2,
            "count": 4
           }
          ]
         }
        },
        {
         "name": "thalach",
         "dtype": "int64",
         "stats": {
          "unique_count": 91,
          "nan_count": 0,
          "min": "71",
          "max": "202",
          "histogram": [
           {
            "bin_start": 71,
            "bin_end": 84.1,
            "count": 1
           },
           {
            "bin_start": 84.1,
            "bin_end": 97.2,
            "count": 6
           },
           {
            "bin_start": 97.2,
            "bin_end": 110.3,
            "count": 11
           },
           {
            "bin_start": 110.3,
            "bin_end": 123.4,
            "count": 26
           },
           {
            "bin_start": 123.4,
            "bin_end": 136.5,
            "count": 35
           },
           {
            "bin_start": 136.5,
            "bin_end": 149.6,
            "count": 53
           },
           {
            "bin_start": 149.6,
            "bin_end": 162.7,
            "count": 77
           },
           {
            "bin_start": 162.7,
            "bin_end": 175.8,
            "count": 63
           },
           {
            "bin_start": 175.8,
            "bin_end": 188.89999999999998,
            "count": 26
           },
           {
            "bin_start": 188.89999999999998,
            "bin_end": 202,
            "count": 5
           }
          ]
         }
        },
        {
         "name": "exang",
         "dtype": "int64",
         "stats": {
          "unique_count": 2,
          "nan_count": 0,
          "min": "0",
          "max": "1",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.1,
            "count": 204
           },
           {
            "bin_start": 0.1,
            "bin_end": 0.2,
            "count": 0
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.30000000000000004,
            "count": 0
           },
           {
            "bin_start": 0.30000000000000004,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.5,
            "count": 0
           },
           {
            "bin_start": 0.5,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.7000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.7000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 0.9,
            "count": 0
           },
           {
            "bin_start": 0.9,
            "bin_end": 1,
            "count": 99
           }
          ]
         }
        },
        {
         "name": "oldpeak",
         "dtype": "float64",
         "stats": {
          "unique_count": 40,
          "nan_count": 0,
          "min": "0.0",
          "max": "6.2",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.62,
            "count": 149
           },
           {
            "bin_start": 0.62,
            "bin_end": 1.24,
            "count": 50
           },
           {
            "bin_start": 1.24,
            "bin_end": 1.8599999999999999,
            "count": 40
           },
           {
            "bin_start": 1.8599999999999999,
            "bin_end": 2.48,
            "count": 24
           },
           {
            "bin_start": 2.48,
            "bin_end": 3.1,
            "count": 20
           },
           {
            "bin_start": 3.1,
            "bin_end": 3.7199999999999998,
            "count": 11
           },
           {
            "bin_start": 3.7199999999999998,
            "bin_end": 4.34,
            "count": 6
           },
           {
            "bin_start": 4.34,
            "bin_end": 4.96,
            "count": 1
           },
           {
            "bin_start": 4.96,
            "bin_end": 5.58,
            "count": 0
           },
           {
            "bin_start": 5.58,
            "bin_end": 6.2,
            "count": 2
           }
          ]
         }
        },
        {
         "name": "slope",
         "dtype": "int64",
         "stats": {
          "unique_count": 3,
          "nan_count": 0,
          "min": "0",
          "max": "2",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.2,
            "count": 21
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 1,
            "count": 0
           },
           {
            "bin_start": 1,
            "bin_end": 1.2000000000000002,
            "count": 140
           },
           {
            "bin_start": 1.2000000000000002,
            "bin_end": 1.4000000000000001,
            "count": 0
           },
           {
            "bin_start": 1.4000000000000001,
            "bin_end": 1.6,
            "count": 0
           },
           {
            "bin_start": 1.6,
            "bin_end": 1.8,
            "count": 0
           },
           {
            "bin_start": 1.8,
            "bin_end": 2,
            "count": 142
           }
          ]
         }
        },
        {
         "name": "ca",
         "dtype": "int64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0",
          "max": "4",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.4,
            "count": 175
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 1.2000000000000002,
            "count": 65
           },
           {
            "bin_start": 1.2000000000000002,
            "bin_end": 1.6,
            "count": 0
           },
           {
            "bin_start": 1.6,
            "bin_end": 2,
            "count": 0
           },
           {
            "bin_start": 2,
            "bin_end": 2.4000000000000004,
            "count": 38
           },
           {
            "bin_start": 2.4000000000000004,
            "bin_end": 2.8000000000000003,
            "count": 0
           },
           {
            "bin_start": 2.8000000000000003,
            "bin_end": 3.2,
            "count": 20
           },
           {
            "bin_start": 3.2,
            "bin_end": 3.6,
            "count": 0
           },
           {
            "bin_start": 3.6,
            "bin_end": 4,
            "count": 5
           }
          ]
         }
        },
        {
         "name": "thal",
         "dtype": "int64",
         "stats": {
          "unique_count": 4,
          "nan_count": 0,
          "min": "0",
          "max": "3",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.3,
            "count": 2
           },
           {
            "bin_start": 0.3,
            "bin_end": 0.6,
            "count": 0
           },
           {
            "bin_start": 0.6,
            "bin_end": 0.8999999999999999,
            "count": 0
           },
           {
            "bin_start": 0.8999999999999999,
            "bin_end": 1.2,
            "count": 18
           },
           {
            "bin_start": 1.2,
            "bin_end": 1.5,
            "count": 0
           },
           {
            "bin_start": 1.5,
            "bin_end": 1.7999999999999998,
            "count": 0
           },
           {
            "bin_start": 1.7999999999999998,
            "bin_end": 2.1,
            "count": 166
           },
           {
            "bin_start": 2.1,
            "bin_end": 2.4,
            "count": 0
           },
           {
            "bin_start": 2.4,
            "bin_end": 2.6999999999999997,
            "count": 0
           },
           {
            "bin_start": 2.6999999999999997,
            "bin_end": 3,
            "count": 117
           }
          ]
         }
        },
        {
         "name": "target",
         "dtype": "int64",
         "stats": {
          "unique_count": 2,
          "nan_count": 0,
          "min": "0",
          "max": "1",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 0.1,
            "count": 138
           },
           {
            "bin_start": 0.1,
            "bin_end": 0.2,
            "count": 0
           },
           {
            "bin_start": 0.2,
            "bin_end": 0.30000000000000004,
            "count": 0
           },
           {
            "bin_start": 0.30000000000000004,
            "bin_end": 0.4,
            "count": 0
           },
           {
            "bin_start": 0.4,
            "bin_end": 0.5,
            "count": 0
           },
           {
            "bin_start": 0.5,
            "bin_end": 0.6000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.6000000000000001,
            "bin_end": 0.7000000000000001,
            "count": 0
           },
           {
            "bin_start": 0.7000000000000001,
            "bin_end": 0.8,
            "count": 0
           },
           {
            "bin_start": 0.8,
            "bin_end": 0.9,
            "count": 0
           },
           {
            "bin_start": 0.9,
            "bin_end": 1,
            "count": 165
           }
          ]
         }
        },
        {
         "name": "_deepnote_index_column",
         "dtype": "int64"
        }
       ],
       "rows_top": [
        {
         "age": 63,
         "sex": 1,
         "cp": 3,
         "trestbps": 145,
         "chol": 233,
         "fbs": 1,
         "restecg": 0,
         "thalach": 150,
         "exang": 0,
         "oldpeak": 2.3,
         "slope": 0,
         "ca": 0,
         "thal": 1,
         "target": 1,
         "_deepnote_index_column": 0
        },
        {
         "age": 37,
         "sex": 1,
         "cp": 2,
         "trestbps": 130,
         "chol": 250,
         "fbs": 0,
         "restecg": 1,
         "thalach": 187,
         "exang": 0,
         "oldpeak": 3.5,
         "slope": 0,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 1
        },
        {
         "age": 41,
         "sex": 0,
         "cp": 1,
         "trestbps": 130,
         "chol": 204,
         "fbs": 0,
         "restecg": 0,
         "thalach": 172,
         "exang": 0,
         "oldpeak": 1.4,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 2
        },
        {
         "age": 56,
         "sex": 1,
         "cp": 1,
         "trestbps": 120,
         "chol": 236,
         "fbs": 0,
         "restecg": 1,
         "thalach": 178,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 3
        },
        {
         "age": 57,
         "sex": 0,
         "cp": 0,
         "trestbps": 120,
         "chol": 354,
         "fbs": 0,
         "restecg": 1,
         "thalach": 163,
         "exang": 1,
         "oldpeak": 0.6,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 4
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 192,
         "fbs": 0,
         "restecg": 1,
         "thalach": 148,
         "exang": 0,
         "oldpeak": 0.4,
         "slope": 1,
         "ca": 0,
         "thal": 1,
         "target": 1,
         "_deepnote_index_column": 5
        },
        {
         "age": 56,
         "sex": 0,
         "cp": 1,
         "trestbps": 140,
         "chol": 294,
         "fbs": 0,
         "restecg": 0,
         "thalach": 153,
         "exang": 0,
         "oldpeak": 1.3,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 6
        },
        {
         "age": 44,
         "sex": 1,
         "cp": 1,
         "trestbps": 120,
         "chol": 263,
         "fbs": 0,
         "restecg": 1,
         "thalach": 173,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": 7
        },
        {
         "age": 52,
         "sex": 1,
         "cp": 2,
         "trestbps": 172,
         "chol": 199,
         "fbs": 1,
         "restecg": 1,
         "thalach": 162,
         "exang": 0,
         "oldpeak": 0.5,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": 8
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 2,
         "trestbps": 150,
         "chol": 168,
         "fbs": 0,
         "restecg": 1,
         "thalach": 174,
         "exang": 0,
         "oldpeak": 1.6,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 9
        },
        {
         "age": 54,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 239,
         "fbs": 0,
         "restecg": 1,
         "thalach": 160,
         "exang": 0,
         "oldpeak": 1.2,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 10
        },
        {
         "age": 48,
         "sex": 0,
         "cp": 2,
         "trestbps": 130,
         "chol": 275,
         "fbs": 0,
         "restecg": 1,
         "thalach": 139,
         "exang": 0,
         "oldpeak": 0.2,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 11
        },
        {
         "age": 49,
         "sex": 1,
         "cp": 1,
         "trestbps": 130,
         "chol": 266,
         "fbs": 0,
         "restecg": 1,
         "thalach": 171,
         "exang": 0,
         "oldpeak": 0.6,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 12
        },
        {
         "age": 64,
         "sex": 1,
         "cp": 3,
         "trestbps": 110,
         "chol": 211,
         "fbs": 0,
         "restecg": 0,
         "thalach": 144,
         "exang": 1,
         "oldpeak": 1.8,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 13
        },
        {
         "age": 58,
         "sex": 0,
         "cp": 3,
         "trestbps": 150,
         "chol": 283,
         "fbs": 1,
         "restecg": 0,
         "thalach": 162,
         "exang": 0,
         "oldpeak": 1,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 14
        },
        {
         "age": 50,
         "sex": 0,
         "cp": 2,
         "trestbps": 120,
         "chol": 219,
         "fbs": 0,
         "restecg": 1,
         "thalach": 158,
         "exang": 0,
         "oldpeak": 1.6,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 15
        },
        {
         "age": 58,
         "sex": 0,
         "cp": 2,
         "trestbps": 120,
         "chol": 340,
         "fbs": 0,
         "restecg": 1,
         "thalach": 172,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 16
        },
        {
         "age": 66,
         "sex": 0,
         "cp": 3,
         "trestbps": 150,
         "chol": 226,
         "fbs": 0,
         "restecg": 1,
         "thalach": 114,
         "exang": 0,
         "oldpeak": 2.6,
         "slope": 0,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 17
        },
        {
         "age": 43,
         "sex": 1,
         "cp": 0,
         "trestbps": 150,
         "chol": 247,
         "fbs": 0,
         "restecg": 1,
         "thalach": 171,
         "exang": 0,
         "oldpeak": 1.5,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 18
        },
        {
         "age": 69,
         "sex": 0,
         "cp": 3,
         "trestbps": 140,
         "chol": 239,
         "fbs": 0,
         "restecg": 1,
         "thalach": 151,
         "exang": 0,
         "oldpeak": 1.8,
         "slope": 2,
         "ca": 2,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 19
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 0,
         "trestbps": 135,
         "chol": 234,
         "fbs": 0,
         "restecg": 1,
         "thalach": 161,
         "exang": 0,
         "oldpeak": 0.5,
         "slope": 1,
         "ca": 0,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": 20
        },
        {
         "age": 44,
         "sex": 1,
         "cp": 2,
         "trestbps": 130,
         "chol": 233,
         "fbs": 0,
         "restecg": 1,
         "thalach": 179,
         "exang": 1,
         "oldpeak": 0.4,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 21
        },
        {
         "age": 42,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 226,
         "fbs": 0,
         "restecg": 1,
         "thalach": 178,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 22
        },
        {
         "age": 61,
         "sex": 1,
         "cp": 2,
         "trestbps": 150,
         "chol": 243,
         "fbs": 1,
         "restecg": 1,
         "thalach": 137,
         "exang": 1,
         "oldpeak": 1,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 23
        },
        {
         "age": 40,
         "sex": 1,
         "cp": 3,
         "trestbps": 140,
         "chol": 199,
         "fbs": 0,
         "restecg": 1,
         "thalach": 178,
         "exang": 1,
         "oldpeak": 1.4,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": 24
        },
        {
         "age": 71,
         "sex": 0,
         "cp": 1,
         "trestbps": 160,
         "chol": 302,
         "fbs": 0,
         "restecg": 1,
         "thalach": 162,
         "exang": 0,
         "oldpeak": 0.4,
         "slope": 2,
         "ca": 2,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 25
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 2,
         "trestbps": 150,
         "chol": 212,
         "fbs": 1,
         "restecg": 1,
         "thalach": 157,
         "exang": 0,
         "oldpeak": 1.6,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 26
        },
        {
         "age": 51,
         "sex": 1,
         "cp": 2,
         "trestbps": 110,
         "chol": 175,
         "fbs": 0,
         "restecg": 1,
         "thalach": 123,
         "exang": 0,
         "oldpeak": 0.6,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 27
        },
        {
         "age": 65,
         "sex": 0,
         "cp": 2,
         "trestbps": 140,
         "chol": 417,
         "fbs": 1,
         "restecg": 0,
         "thalach": 157,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 2,
         "ca": 1,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 28
        },
        {
         "age": 53,
         "sex": 1,
         "cp": 2,
         "trestbps": 130,
         "chol": 197,
         "fbs": 1,
         "restecg": 0,
         "thalach": 152,
         "exang": 0,
         "oldpeak": 1.2,
         "slope": 0,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 29
        },
        {
         "age": 41,
         "sex": 0,
         "cp": 1,
         "trestbps": 105,
         "chol": 198,
         "fbs": 0,
         "restecg": 1,
         "thalach": 168,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 1,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 30
        },
        {
         "age": 65,
         "sex": 1,
         "cp": 0,
         "trestbps": 120,
         "chol": 177,
         "fbs": 0,
         "restecg": 1,
         "thalach": 140,
         "exang": 0,
         "oldpeak": 0.4,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": 31
        },
        {
         "age": 44,
         "sex": 1,
         "cp": 1,
         "trestbps": 130,
         "chol": 219,
         "fbs": 0,
         "restecg": 0,
         "thalach": 188,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 32
        },
        {
         "age": 54,
         "sex": 1,
         "cp": 2,
         "trestbps": 125,
         "chol": 273,
         "fbs": 0,
         "restecg": 0,
         "thalach": 152,
         "exang": 0,
         "oldpeak": 0.5,
         "slope": 0,
         "ca": 1,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 33
        },
        {
         "age": 51,
         "sex": 1,
         "cp": 3,
         "trestbps": 125,
         "chol": 213,
         "fbs": 0,
         "restecg": 0,
         "thalach": 125,
         "exang": 1,
         "oldpeak": 1.4,
         "slope": 2,
         "ca": 1,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": 34
        }
       ],
       "rows_bottom": [
        {
         "age": 49,
         "sex": 1,
         "cp": 2,
         "trestbps": 118,
         "chol": 149,
         "fbs": 0,
         "restecg": 0,
         "thalach": 126,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 2,
         "ca": 3,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 267
        },
        {
         "age": 54,
         "sex": 1,
         "cp": 0,
         "trestbps": 122,
         "chol": 286,
         "fbs": 0,
         "restecg": 0,
         "thalach": 116,
         "exang": 1,
         "oldpeak": 3.2,
         "slope": 1,
         "ca": 2,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 268
        },
        {
         "age": 56,
         "sex": 1,
         "cp": 0,
         "trestbps": 130,
         "chol": 283,
         "fbs": 1,
         "restecg": 0,
         "thalach": 103,
         "exang": 1,
         "oldpeak": 1.6,
         "slope": 0,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 269
        },
        {
         "age": 46,
         "sex": 1,
         "cp": 0,
         "trestbps": 120,
         "chol": 249,
         "fbs": 0,
         "restecg": 0,
         "thalach": 144,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 270
        },
        {
         "age": 61,
         "sex": 1,
         "cp": 3,
         "trestbps": 134,
         "chol": 234,
         "fbs": 0,
         "restecg": 1,
         "thalach": 145,
         "exang": 0,
         "oldpeak": 2.6,
         "slope": 1,
         "ca": 2,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 271
        },
        {
         "age": 67,
         "sex": 1,
         "cp": 0,
         "trestbps": 120,
         "chol": 237,
         "fbs": 0,
         "restecg": 1,
         "thalach": 71,
         "exang": 0,
         "oldpeak": 1,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 272
        },
        {
         "age": 58,
         "sex": 1,
         "cp": 0,
         "trestbps": 100,
         "chol": 234,
         "fbs": 0,
         "restecg": 1,
         "thalach": 156,
         "exang": 0,
         "oldpeak": 0.1,
         "slope": 2,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 273
        },
        {
         "age": 47,
         "sex": 1,
         "cp": 0,
         "trestbps": 110,
         "chol": 275,
         "fbs": 0,
         "restecg": 0,
         "thalach": 118,
         "exang": 1,
         "oldpeak": 1,
         "slope": 1,
         "ca": 1,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 274
        },
        {
         "age": 52,
         "sex": 1,
         "cp": 0,
         "trestbps": 125,
         "chol": 212,
         "fbs": 0,
         "restecg": 1,
         "thalach": 168,
         "exang": 0,
         "oldpeak": 1,
         "slope": 2,
         "ca": 2,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 275
        },
        {
         "age": 58,
         "sex": 1,
         "cp": 0,
         "trestbps": 146,
         "chol": 218,
         "fbs": 0,
         "restecg": 1,
         "thalach": 105,
         "exang": 0,
         "oldpeak": 2,
         "slope": 1,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 276
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 1,
         "trestbps": 124,
         "chol": 261,
         "fbs": 0,
         "restecg": 1,
         "thalach": 141,
         "exang": 0,
         "oldpeak": 0.3,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 277
        },
        {
         "age": 58,
         "sex": 0,
         "cp": 1,
         "trestbps": 136,
         "chol": 319,
         "fbs": 1,
         "restecg": 0,
         "thalach": 152,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 2,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 278
        },
        {
         "age": 61,
         "sex": 1,
         "cp": 0,
         "trestbps": 138,
         "chol": 166,
         "fbs": 0,
         "restecg": 0,
         "thalach": 125,
         "exang": 1,
         "oldpeak": 3.6,
         "slope": 1,
         "ca": 1,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 279
        },
        {
         "age": 42,
         "sex": 1,
         "cp": 0,
         "trestbps": 136,
         "chol": 315,
         "fbs": 0,
         "restecg": 1,
         "thalach": 125,
         "exang": 1,
         "oldpeak": 1.8,
         "slope": 1,
         "ca": 0,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 280
        },
        {
         "age": 52,
         "sex": 1,
         "cp": 0,
         "trestbps": 128,
         "chol": 204,
         "fbs": 1,
         "restecg": 1,
         "thalach": 156,
         "exang": 1,
         "oldpeak": 1,
         "slope": 1,
         "ca": 0,
         "thal": 0,
         "target": 0,
         "_deepnote_index_column": 281
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 2,
         "trestbps": 126,
         "chol": 218,
         "fbs": 1,
         "restecg": 1,
         "thalach": 134,
         "exang": 0,
         "oldpeak": 2.2,
         "slope": 1,
         "ca": 1,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 282
        },
        {
         "age": 40,
         "sex": 1,
         "cp": 0,
         "trestbps": 152,
         "chol": 223,
         "fbs": 0,
         "restecg": 1,
         "thalach": 181,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 283
        },
        {
         "age": 61,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 207,
         "fbs": 0,
         "restecg": 0,
         "thalach": 138,
         "exang": 1,
         "oldpeak": 1.9,
         "slope": 2,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 284
        },
        {
         "age": 46,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 311,
         "fbs": 0,
         "restecg": 1,
         "thalach": 120,
         "exang": 1,
         "oldpeak": 1.8,
         "slope": 1,
         "ca": 2,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 285
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 3,
         "trestbps": 134,
         "chol": 204,
         "fbs": 0,
         "restecg": 1,
         "thalach": 162,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 2,
         "ca": 2,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 286
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 1,
         "trestbps": 154,
         "chol": 232,
         "fbs": 0,
         "restecg": 0,
         "thalach": 164,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 1,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 287
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 0,
         "trestbps": 110,
         "chol": 335,
         "fbs": 0,
         "restecg": 1,
         "thalach": 143,
         "exang": 1,
         "oldpeak": 3,
         "slope": 1,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 288
        },
        {
         "age": 55,
         "sex": 0,
         "cp": 0,
         "trestbps": 128,
         "chol": 205,
         "fbs": 0,
         "restecg": 2,
         "thalach": 130,
         "exang": 1,
         "oldpeak": 2,
         "slope": 1,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 289
        },
        {
         "age": 61,
         "sex": 1,
         "cp": 0,
         "trestbps": 148,
         "chol": 203,
         "fbs": 0,
         "restecg": 1,
         "thalach": 161,
         "exang": 0,
         "oldpeak": 0,
         "slope": 2,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 290
        },
        {
         "age": 58,
         "sex": 1,
         "cp": 0,
         "trestbps": 114,
         "chol": 318,
         "fbs": 0,
         "restecg": 2,
         "thalach": 140,
         "exang": 0,
         "oldpeak": 4.4,
         "slope": 0,
         "ca": 3,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 291
        },
        {
         "age": 58,
         "sex": 0,
         "cp": 0,
         "trestbps": 170,
         "chol": 225,
         "fbs": 1,
         "restecg": 0,
         "thalach": 146,
         "exang": 1,
         "oldpeak": 2.8,
         "slope": 1,
         "ca": 2,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 292
        },
        {
         "age": 67,
         "sex": 1,
         "cp": 2,
         "trestbps": 152,
         "chol": 212,
         "fbs": 0,
         "restecg": 0,
         "thalach": 150,
         "exang": 0,
         "oldpeak": 0.8,
         "slope": 1,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 293
        },
        {
         "age": 44,
         "sex": 1,
         "cp": 0,
         "trestbps": 120,
         "chol": 169,
         "fbs": 0,
         "restecg": 1,
         "thalach": 144,
         "exang": 1,
         "oldpeak": 2.8,
         "slope": 0,
         "ca": 0,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 294
        },
        {
         "age": 63,
         "sex": 1,
         "cp": 0,
         "trestbps": 140,
         "chol": 187,
         "fbs": 0,
         "restecg": 0,
         "thalach": 144,
         "exang": 1,
         "oldpeak": 4,
         "slope": 2,
         "ca": 2,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 295
        },
        {
         "age": 63,
         "sex": 0,
         "cp": 0,
         "trestbps": 124,
         "chol": 197,
         "fbs": 0,
         "restecg": 1,
         "thalach": 136,
         "exang": 1,
         "oldpeak": 0,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 296
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 0,
         "trestbps": 164,
         "chol": 176,
         "fbs": 1,
         "restecg": 0,
         "thalach": 90,
         "exang": 0,
         "oldpeak": 1,
         "slope": 1,
         "ca": 2,
         "thal": 1,
         "target": 0,
         "_deepnote_index_column": 297
        },
        {
         "age": 57,
         "sex": 0,
         "cp": 0,
         "trestbps": 140,
         "chol": 241,
         "fbs": 0,
         "restecg": 1,
         "thalach": 123,
         "exang": 1,
         "oldpeak": 0.2,
         "slope": 1,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 298
        },
        {
         "age": 45,
         "sex": 1,
         "cp": 3,
         "trestbps": 110,
         "chol": 264,
         "fbs": 0,
         "restecg": 1,
         "thalach": 132,
         "exang": 0,
         "oldpeak": 1.2,
         "slope": 1,
         "ca": 0,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 299
        },
        {
         "age": 68,
         "sex": 1,
         "cp": 0,
         "trestbps": 144,
         "chol": 193,
         "fbs": 1,
         "restecg": 1,
         "thalach": 141,
         "exang": 0,
         "oldpeak": 3.4,
         "slope": 1,
         "ca": 2,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 300
        },
        {
         "age": 57,
         "sex": 1,
         "cp": 0,
         "trestbps": 130,
         "chol": 131,
         "fbs": 0,
         "restecg": 1,
         "thalach": 115,
         "exang": 1,
         "oldpeak": 1.2,
         "slope": 1,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": 301
        },
        {
         "age": 57,
         "sex": 0,
         "cp": 1,
         "trestbps": 130,
         "chol": 236,
         "fbs": 0,
         "restecg": 0,
         "thalach": 174,
         "exang": 0,
         "oldpeak": 0,
         "slope": 1,
         "ca": 1,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": 302
        }
       ]
      },
      "text/plain": "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n0     63    1   3       145   233    1        0      150      0      2.3   \n1     37    1   2       130   250    0        1      187      0      3.5   \n2     41    0   1       130   204    0        0      172      0      1.4   \n3     56    1   1       120   236    0        1      178      0      0.8   \n4     57    0   0       120   354    0        1      163      1      0.6   \n..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   \n298   57    0   0       140   241    0        1      123      1      0.2   \n299   45    1   3       110   264    0        1      132      0      1.2   \n300   68    1   0       144   193    1        1      141      0      3.4   \n301   57    1   0       130   131    0        1      115      1      1.2   \n302   57    0   1       130   236    0        0      174      0      0.0   \n\n     slope  ca  thal  target  \n0        0   0     1       1  \n1        0   0     2       1  \n2        2   0     2       1  \n3        2   0     2       1  \n4        2   0     2       1  \n..     ...  ..   ...     ...  \n298      1   0     3       0  \n299      1   0     3       0  \n300      1   2     3       0  \n301      1   1     3       0  \n302      1   1     2       0  \n\n[303 rows x 14 columns]",
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>age</th>\n      <th>sex</th>\n      <th>cp</th>\n      <th>trestbps</th>\n      <th>chol</th>\n      <th>fbs</th>\n      <th>restecg</th>\n      <th>thalach</th>\n      <th>exang</th>\n      <th>oldpeak</th>\n      <th>slope</th>\n      <th>ca</th>\n      <th>thal</th>\n      <th>target</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>63</td>\n      <td>1</td>\n      <td>3</td>\n      <td>145</td>\n      <td>233</td>\n      <td>1</td>\n      <td>0</td>\n      <td>150</td>\n      <td>0</td>\n      <td>2.3</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>37</td>\n      <td>1</td>\n      <td>2</td>\n      <td>130</td>\n      <td>250</td>\n      <td>0</td>\n      <td>1</td>\n      <td>187</td>\n      <td>0</td>\n      <td>3.5</td>\n      <td>0</td>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>41</td>\n      <td>0</td>\n      <td>1</td>\n      <td>130</td>\n      <td>204</td>\n      <td>0</td>\n      <td>0</td>\n      <td>172</td>\n      <td>0</td>\n      <td>1.4</td>\n      <td>2</td>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>56</td>\n      <td>1</td>\n      <td>1</td>\n      <td>120</td>\n      <td>236</td>\n      <td>0</td>\n      <td>1</td>\n      <td>178</td>\n      <td>0</td>\n      <td>0.8</td>\n      <td>2</td>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>57</td>\n      <td>0</td>\n      <td>0</td>\n      <td>120</td>\n      <td>354</td>\n      <td>0</td>\n      <td>1</td>\n      <td>163</td>\n      <td>1</td>\n      <td>0.6</td>\n      <td>2</td>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>298</th>\n      <td>57</td>\n      <td>0</td>\n      <td>0</td>\n      <td>140</td>\n      <td>241</td>\n      <td>0</td>\n      <td>1</td>\n      <td>123</td>\n      <td>1</td>\n      <td>0.2</td>\n      <td>1</td>\n      <td>0</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>299</th>\n      <td>45</td>\n      <td>1</td>\n      <td>3</td>\n      <td>110</td>\n      <td>264</td>\n      <td>0</td>\n      <td>1</td>\n      <td>132</td>\n      <td>0</td>\n      <td>1.2</td>\n      <td>1</td>\n      <td>0</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>300</th>\n      <td>68</td>\n      <td>1</td>\n      <td>0</td>\n      <td>144</td>\n      <td>193</td>\n      <td>1</td>\n      <td>1</td>\n      <td>141</td>\n      <td>0</td>\n      <td>3.4</td>\n      <td>1</td>\n      <td>2</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>301</th>\n      <td>57</td>\n      <td>1</td>\n      <td>0</td>\n      <td>130</td>\n      <td>131</td>\n      <td>0</td>\n      <td>1</td>\n      <td>115</td>\n      <td>1</td>\n      <td>1.2</td>\n      <td>1</td>\n      <td>1</td>\n      <td>3</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>302</th>\n      <td>57</td>\n      <td>0</td>\n      <td>1</td>\n      <td>130</td>\n      <td>236</td>\n      <td>0</td>\n      <td>0</td>\n      <td>174</td>\n      <td>0</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>2</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>303 rows × 14 columns</p>\n</div>"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00004-c37b06fd-800f-47f0-bed6-b3c2cf59f74f",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "88841b20",
    "execution_start": 1620375815904,
    "execution_millis": 9,
    "deepnote_cell_type": "code"
   },
   "source": "cat_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']\nnum_vars = ['age','trestbps','chol','thalach','oldpeak']\n\nX = df[cat_vars + num_vars]\ny = df.target\nX.shape",
   "execution_count": 5,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 5,
     "data": {
      "text/plain": "(303, 13)"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00005-a9e00669-30fb-4cdd-b8bb-a36f2810aa29",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "f3dd26da",
    "execution_start": 1620375815910,
    "execution_millis": 10,
    "deepnote_cell_type": "code"
   },
   "source": "df.isnull().sum()",
   "execution_count": 6,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 6,
     "data": {
      "text/plain": "age         0\nsex         0\ncp          0\ntrestbps    0\nchol        0\nfbs         0\nrestecg     0\nthalach     0\nexang       0\noldpeak     0\nslope       0\nca          0\nthal        0\ntarget      0\ndtype: int64"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00005-47d59c56-d432-48a7-97ec-2cf4c25c9627",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "aaef0dea",
    "execution_start": 1620375815918,
    "execution_millis": 1029,
    "deepnote_cell_type": "code"
   },
   "source": "sb.set_context(\"paper\", font_scale = 0.8, rc = {\"font.size\": 30,\"axes.titlesize\": 25,\"axes.labelsize\": 10}) \nsb.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())\nplt.title('Variation of Age for each target class')\nplt.show()",
   "execution_count": 7,
   "outputs": [
    {
     "data": {
      "text/plain": "<Figure size 389.844x360 with 1 Axes>",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light",
      "image/png": {
       "width": 478,
       "height": 348
      }
     },
     "output_type": "display_data"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00007-6f90256b-f53a-4d02-9dc5-02498b11db98",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "6c0e851c",
    "execution_start": 1620375816938,
    "execution_millis": 0,
    "deepnote_cell_type": "code"
   },
   "source": "#target = 1 implies that the person is suffering from heart disease and \n#target = 0 implies the person is not suffering.\n#that most people who are suffering are of the age of 58, followed by 57.\n#Majorly, people belonging to the age group 50+ are suffering from the disease",
   "execution_count": 8,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00007-178fa597-5362-47bd-b318-7fe472df8339",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "51429faf",
    "execution_start": 1620377941047,
    "execution_millis": 448,
    "deepnote_cell_type": "code"
   },
   "source": "sb.catplot(kind = 'bar', data = df, y = 'age', x = 'sex', hue = 'target', ci=None)\nplt.title('Distribution of age vs sex with the target class')\nplt.show()\n",
   "execution_count": 22,
   "outputs": [
    {
     "data": {
      "text/plain": "<Figure size 389.844x360 with 1 Axes>",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light",
      "image/png": {
       "width": 598,
       "height": 348
      }
     },
     "output_type": "display_data"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00009-50a82f48-5717-4dc3-9916-1e513484829c",
    "deepnote_cell_type": "code"
   },
   "source": "#0is female and 1 is male\n# females who are suffering from the disease are older than males",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00009-3d78dcbf-e2ea-44d9-93a7-c34dd4f149ac",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "bb89e374",
    "execution_start": 1620377220427,
    "execution_millis": 83,
    "deepnote_cell_type": "code"
   },
   "source": "pos_data = df[df['target']==1]\npos_data.describe()",
   "execution_count": 16,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 16,
     "data": {
      "application/vnd.deepnote.dataframe.v2+json": {
       "row_count": 8,
       "column_count": 14,
       "columns": [
        {
         "name": "age",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "9.550650751946774",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 9.550650751946774,
            "bin_end": 25.095585676752098,
            "count": 1
           },
           {
            "bin_start": 25.095585676752098,
            "bin_end": 40.64052060155742,
            "count": 1
           },
           {
            "bin_start": 40.64052060155742,
            "bin_end": 56.185455526362745,
            "count": 3
           },
           {
            "bin_start": 56.185455526362745,
            "bin_end": 71.73039045116806,
            "count": 1
           },
           {
            "bin_start": 71.73039045116806,
            "bin_end": 87.2753253759734,
            "count": 1
           },
           {
            "bin_start": 87.2753253759734,
            "bin_end": 102.82026030077871,
            "count": 0
           },
           {
            "bin_start": 102.82026030077871,
            "bin_end": 118.36519522558405,
            "count": 0
           },
           {
            "bin_start": 118.36519522558405,
            "bin_end": 133.91013015038936,
            "count": 0
           },
           {
            "bin_start": 133.91013015038936,
            "bin_end": 149.45506507519468,
            "count": 0
           },
           {
            "bin_start": 149.45506507519468,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "sex",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "cp",
         "dtype": "float64",
         "stats": {
          "unique_count": 7,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "trestbps",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "16.169613266874865",
          "max": "180.0",
          "histogram": [
           {
            "bin_start": 16.169613266874865,
            "bin_end": 32.552651940187374,
            "count": 1
           },
           {
            "bin_start": 32.552651940187374,
            "bin_end": 48.93569061349989,
            "count": 0
           },
           {
            "bin_start": 48.93569061349989,
            "bin_end": 65.3187292868124,
            "count": 0
           },
           {
            "bin_start": 65.3187292868124,
            "bin_end": 81.70176796012491,
            "count": 0
           },
           {
            "bin_start": 81.70176796012491,
            "bin_end": 98.08480663343742,
            "count": 1
           },
           {
            "bin_start": 98.08480663343742,
            "bin_end": 114.46784530674994,
            "count": 0
           },
           {
            "bin_start": 114.46784530674994,
            "bin_end": 130.85088398006246,
            "count": 3
           },
           {
            "bin_start": 130.85088398006246,
            "bin_end": 147.23392265337498,
            "count": 1
           },
           {
            "bin_start": 147.23392265337498,
            "bin_end": 163.6169613266875,
            "count": 0
           },
           {
            "bin_start": 163.6169613266875,
            "bin_end": 180,
            "count": 2
           }
          ]
         }
        },
        {
         "name": "chol",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "53.55287155453833",
          "max": "564.0",
          "histogram": [
           {
            "bin_start": 53.55287155453833,
            "bin_end": 104.5975843990845,
            "count": 1
           },
           {
            "bin_start": 104.5975843990845,
            "bin_end": 155.64229724363065,
            "count": 1
           },
           {
            "bin_start": 155.64229724363065,
            "bin_end": 206.68701008817686,
            "count": 1
           },
           {
            "bin_start": 206.68701008817686,
            "bin_end": 257.731722932723,
            "count": 3
           },
           {
            "bin_start": 257.731722932723,
            "bin_end": 308.77643577726917,
            "count": 1
           },
           {
            "bin_start": 308.77643577726917,
            "bin_end": 359.8211486218154,
            "count": 0
           },
           {
            "bin_start": 359.8211486218154,
            "bin_end": 410.86586146636154,
            "count": 0
           },
           {
            "bin_start": 410.86586146636154,
            "bin_end": 461.9105743109077,
            "count": 0
           },
           {
            "bin_start": 461.9105743109077,
            "bin_end": 512.9552871554538,
            "count": 0
           },
           {
            "bin_start": 512.9552871554538,
            "bin_end": 564,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "fbs",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "restecg",
         "dtype": "float64",
         "stats": {
          "unique_count": 6,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "thalach",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "19.174275619393168",
          "max": "202.0",
          "histogram": [
           {
            "bin_start": 19.174275619393168,
            "bin_end": 37.456848057453854,
            "count": 1
           },
           {
            "bin_start": 37.456848057453854,
            "bin_end": 55.73942049551454,
            "count": 0
           },
           {
            "bin_start": 55.73942049551454,
            "bin_end": 74.02199293357522,
            "count": 0
           },
           {
            "bin_start": 74.02199293357522,
            "bin_end": 92.3045653716359,
            "count": 0
           },
           {
            "bin_start": 92.3045653716359,
            "bin_end": 110.58713780969659,
            "count": 1
           },
           {
            "bin_start": 110.58713780969659,
            "bin_end": 128.86971024775727,
            "count": 0
           },
           {
            "bin_start": 128.86971024775727,
            "bin_end": 147.15228268581794,
            "count": 0
           },
           {
            "bin_start": 147.15228268581794,
            "bin_end": 165.43485512387863,
            "count": 4
           },
           {
            "bin_start": 165.43485512387863,
            "bin_end": 183.71742756193933,
            "count": 1
           },
           {
            "bin_start": 183.71742756193933,
            "bin_end": 202,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "exang",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "oldpeak",
         "dtype": "float64",
         "stats": {
          "unique_count": 7,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "slope",
         "dtype": "float64",
         "stats": {
          "unique_count": 6,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "ca",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "thal",
         "dtype": "float64",
         "stats": {
          "unique_count": 6,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "target",
         "dtype": "float64",
         "stats": {
          "unique_count": 3,
          "nan_count": 0,
          "min": "0.0",
          "max": "165.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 16.5,
            "count": 7
           },
           {
            "bin_start": 16.5,
            "bin_end": 33,
            "count": 0
           },
           {
            "bin_start": 33,
            "bin_end": 49.5,
            "count": 0
           },
           {
            "bin_start": 49.5,
            "bin_end": 66,
            "count": 0
           },
           {
            "bin_start": 66,
            "bin_end": 82.5,
            "count": 0
           },
           {
            "bin_start": 82.5,
            "bin_end": 99,
            "count": 0
           },
           {
            "bin_start": 99,
            "bin_end": 115.5,
            "count": 0
           },
           {
            "bin_start": 115.5,
            "bin_end": 132,
            "count": 0
           },
           {
            "bin_start": 132,
            "bin_end": 148.5,
            "count": 0
           },
           {
            "bin_start": 148.5,
            "bin_end": 165,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "_deepnote_index_column",
         "dtype": "object"
        }
       ],
       "rows_top": [
        {
         "age": 165,
         "sex": 165,
         "cp": 165,
         "trestbps": 165,
         "chol": 165,
         "fbs": 165,
         "restecg": 165,
         "thalach": 165,
         "exang": 165,
         "oldpeak": 165,
         "slope": 165,
         "ca": 165,
         "thal": 165,
         "target": 165,
         "_deepnote_index_column": "count"
        },
        {
         "age": 52.4969696969697,
         "sex": 0.5636363636363636,
         "cp": 1.3757575757575757,
         "trestbps": 129.3030303030303,
         "chol": 242.23030303030302,
         "fbs": 0.1393939393939394,
         "restecg": 0.593939393939394,
         "thalach": 158.46666666666667,
         "exang": 0.1393939393939394,
         "oldpeak": 0.583030303030303,
         "slope": 1.593939393939394,
         "ca": 0.36363636363636365,
         "thal": 2.121212121212121,
         "target": 1,
         "_deepnote_index_column": "mean"
        },
        {
         "age": 9.550650751946774,
         "sex": 0.49744357555882157,
         "cp": 0.9522215049717543,
         "trestbps": 16.169613266874865,
         "chol": 53.55287155453833,
         "fbs": 0.34741150297891615,
         "restecg": 0.5048178818796776,
         "thalach": 19.174275619393168,
         "exang": 0.34741150297891615,
         "oldpeak": 0.7806832719018298,
         "slope": 0.5936346262434834,
         "ca": 0.8488938935886287,
         "thal": 0.46575245686060807,
         "target": 0,
         "_deepnote_index_column": "std"
        },
        {
         "age": 29,
         "sex": 0,
         "cp": 0,
         "trestbps": 94,
         "chol": 126,
         "fbs": 0,
         "restecg": 0,
         "thalach": 96,
         "exang": 0,
         "oldpeak": 0,
         "slope": 0,
         "ca": 0,
         "thal": 0,
         "target": 1,
         "_deepnote_index_column": "min"
        },
        {
         "age": 44,
         "sex": 0,
         "cp": 1,
         "trestbps": 120,
         "chol": 208,
         "fbs": 0,
         "restecg": 0,
         "thalach": 149,
         "exang": 0,
         "oldpeak": 0,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": "25%"
        },
        {
         "age": 52,
         "sex": 1,
         "cp": 2,
         "trestbps": 130,
         "chol": 234,
         "fbs": 0,
         "restecg": 1,
         "thalach": 161,
         "exang": 0,
         "oldpeak": 0.2,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": "50%"
        },
        {
         "age": 59,
         "sex": 1,
         "cp": 2,
         "trestbps": 140,
         "chol": 267,
         "fbs": 0,
         "restecg": 1,
         "thalach": 172,
         "exang": 0,
         "oldpeak": 1,
         "slope": 2,
         "ca": 0,
         "thal": 2,
         "target": 1,
         "_deepnote_index_column": "75%"
        },
        {
         "age": 76,
         "sex": 1,
         "cp": 3,
         "trestbps": 180,
         "chol": 564,
         "fbs": 1,
         "restecg": 2,
         "thalach": 202,
         "exang": 1,
         "oldpeak": 4.2,
         "slope": 2,
         "ca": 4,
         "thal": 3,
         "target": 1,
         "_deepnote_index_column": "max"
        }
       ],
       "rows_bottom": null
      },
      "text/plain": "              age         sex          cp    trestbps        chol         fbs  \\\ncount  165.000000  165.000000  165.000000  165.000000  165.000000  165.000000   \nmean    52.496970    0.563636    1.375758  129.303030  242.230303    0.139394   \nstd      9.550651    0.497444    0.952222   16.169613   53.552872    0.347412   \nmin     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000   \n25%     44.000000    0.000000    1.000000  120.000000  208.000000    0.000000   \n50%     52.000000    1.000000    2.000000  130.000000  234.000000    0.000000   \n75%     59.000000    1.000000    2.000000  140.000000  267.000000    0.000000   \nmax     76.000000    1.000000    3.000000  180.000000  564.000000    1.000000   \n\n          restecg     thalach       exang     oldpeak       slope          ca  \\\ncount  165.000000  165.000000  165.000000  165.000000  165.000000  165.000000   \nmean     0.593939  158.466667    0.139394    0.583030    1.593939    0.363636   \nstd      0.504818   19.174276    0.347412    0.780683    0.593635    0.848894   \nmin      0.000000   96.000000    0.000000    0.000000    0.000000    0.000000   \n25%      0.000000  149.000000    0.000000    0.000000    1.000000    0.000000   \n50%      1.000000  161.000000    0.000000    0.200000    2.000000    0.000000   \n75%      1.000000  172.000000    0.000000    1.000000    2.000000    0.000000   \nmax      2.000000  202.000000    1.000000    4.200000    2.000000    4.000000   \n\n             thal  target  \ncount  165.000000   165.0  \nmean     2.121212     1.0  \nstd      0.465752     0.0  \nmin      0.000000     1.0  \n25%      2.000000     1.0  \n50%      2.000000     1.0  \n75%      2.000000     1.0  \nmax      3.000000     1.0  ",
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>age</th>\n      <th>sex</th>\n      <th>cp</th>\n      <th>trestbps</th>\n      <th>chol</th>\n      <th>fbs</th>\n      <th>restecg</th>\n      <th>thalach</th>\n      <th>exang</th>\n      <th>oldpeak</th>\n      <th>slope</th>\n      <th>ca</th>\n      <th>thal</th>\n      <th>target</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.000000</td>\n      <td>165.0</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>52.496970</td>\n      <td>0.563636</td>\n      <td>1.375758</td>\n      <td>129.303030</td>\n      <td>242.230303</td>\n      <td>0.139394</td>\n      <td>0.593939</td>\n      <td>158.466667</td>\n      <td>0.139394</td>\n      <td>0.583030</td>\n      <td>1.593939</td>\n      <td>0.363636</td>\n      <td>2.121212</td>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>9.550651</td>\n      <td>0.497444</td>\n      <td>0.952222</td>\n      <td>16.169613</td>\n      <td>53.552872</td>\n      <td>0.347412</td>\n      <td>0.504818</td>\n      <td>19.174276</td>\n      <td>0.347412</td>\n      <td>0.780683</td>\n      <td>0.593635</td>\n      <td>0.848894</td>\n      <td>0.465752</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>29.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>94.000000</td>\n      <td>126.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>96.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>44.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>120.000000</td>\n      <td>208.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>149.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>2.000000</td>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>52.000000</td>\n      <td>1.000000</td>\n      <td>2.000000</td>\n      <td>130.000000</td>\n      <td>234.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>161.000000</td>\n      <td>0.000000</td>\n      <td>0.200000</td>\n      <td>2.000000</td>\n      <td>0.000000</td>\n      <td>2.000000</td>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>59.000000</td>\n      <td>1.000000</td>\n      <td>2.000000</td>\n      <td>140.000000</td>\n      <td>267.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>172.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>2.000000</td>\n      <td>0.000000</td>\n      <td>2.000000</td>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>76.000000</td>\n      <td>1.000000</td>\n      <td>3.000000</td>\n      <td>180.000000</td>\n      <td>564.000000</td>\n      <td>1.000000</td>\n      <td>2.000000</td>\n      <td>202.000000</td>\n      <td>1.000000</td>\n      <td>4.200000</td>\n      <td>2.000000</td>\n      <td>4.000000</td>\n      <td>3.000000</td>\n      <td>1.0</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00009-68951ac6-c6a3-4538-b87a-7dbeda8cfdc6",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "a7d7189f",
    "execution_start": 1620377230129,
    "execution_millis": 73,
    "deepnote_cell_type": "code"
   },
   "source": "neg_data = df[df['target']==0]\nneg_data.describe()",
   "execution_count": 17,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 17,
     "data": {
      "application/vnd.deepnote.dataframe.v2+json": {
       "row_count": 8,
       "column_count": 14,
       "columns": [
        {
         "name": "age",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "7.96208153750117",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 7.96208153750117,
            "bin_end": 20.965873383751052,
            "count": 1
           },
           {
            "bin_start": 20.965873383751052,
            "bin_end": 33.969665230000935,
            "count": 0
           },
           {
            "bin_start": 33.969665230000935,
            "bin_end": 46.97345707625082,
            "count": 1
           },
           {
            "bin_start": 46.97345707625082,
            "bin_end": 59.9772489225007,
            "count": 3
           },
           {
            "bin_start": 59.9772489225007,
            "bin_end": 72.98104076875057,
            "count": 1
           },
           {
            "bin_start": 72.98104076875057,
            "bin_end": 85.98483261500047,
            "count": 1
           },
           {
            "bin_start": 85.98483261500047,
            "bin_end": 98.98862446125034,
            "count": 0
           },
           {
            "bin_start": 98.98862446125034,
            "bin_end": 111.99241630750024,
            "count": 0
           },
           {
            "bin_start": 111.99241630750024,
            "bin_end": 124.9962081537501,
            "count": 0
           },
           {
            "bin_start": 124.9962081537501,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "sex",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "cp",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "trestbps",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "18.72994396158135",
          "max": "200.0",
          "histogram": [
           {
            "bin_start": 18.72994396158135,
            "bin_end": 36.856949565423214,
            "count": 1
           },
           {
            "bin_start": 36.856949565423214,
            "bin_end": 54.98395516926508,
            "count": 0
           },
           {
            "bin_start": 54.98395516926508,
            "bin_end": 73.11096077310694,
            "count": 0
           },
           {
            "bin_start": 73.11096077310694,
            "bin_end": 91.2379663769488,
            "count": 0
           },
           {
            "bin_start": 91.2379663769488,
            "bin_end": 109.36497198079067,
            "count": 1
           },
           {
            "bin_start": 109.36497198079067,
            "bin_end": 127.49197758463254,
            "count": 1
           },
           {
            "bin_start": 127.49197758463254,
            "bin_end": 145.6189831884744,
            "count": 4
           },
           {
            "bin_start": 145.6189831884744,
            "bin_end": 163.74598879231627,
            "count": 0
           },
           {
            "bin_start": 163.74598879231627,
            "bin_end": 181.87299439615813,
            "count": 0
           },
           {
            "bin_start": 181.87299439615813,
            "bin_end": 200,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "chol",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "49.45461360407159",
          "max": "409.0",
          "histogram": [
           {
            "bin_start": 49.45461360407159,
            "bin_end": 85.40915224366444,
            "count": 1
           },
           {
            "bin_start": 85.40915224366444,
            "bin_end": 121.36369088325728,
            "count": 0
           },
           {
            "bin_start": 121.36369088325728,
            "bin_end": 157.31822952285012,
            "count": 2
           },
           {
            "bin_start": 157.31822952285012,
            "bin_end": 193.27276816244296,
            "count": 0
           },
           {
            "bin_start": 193.27276816244296,
            "bin_end": 229.2273068020358,
            "count": 1
           },
           {
            "bin_start": 229.2273068020358,
            "bin_end": 265.18184544162864,
            "count": 2
           },
           {
            "bin_start": 265.18184544162864,
            "bin_end": 301.1363840812215,
            "count": 1
           },
           {
            "bin_start": 301.1363840812215,
            "bin_end": 337.0909227208143,
            "count": 0
           },
           {
            "bin_start": 337.0909227208143,
            "bin_end": 373.04546136040716,
            "count": 0
           },
           {
            "bin_start": 373.04546136040716,
            "bin_end": 409,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "fbs",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "restecg",
         "dtype": "float64",
         "stats": {
          "unique_count": 6,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "thalach",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "22.598782298785906",
          "max": "195.0",
          "histogram": [
           {
            "bin_start": 22.598782298785906,
            "bin_end": 39.83890406890731,
            "count": 1
           },
           {
            "bin_start": 39.83890406890731,
            "bin_end": 57.07902583902872,
            "count": 0
           },
           {
            "bin_start": 57.07902583902872,
            "bin_end": 74.31914760915014,
            "count": 1
           },
           {
            "bin_start": 74.31914760915014,
            "bin_end": 91.55926937927154,
            "count": 0
           },
           {
            "bin_start": 91.55926937927154,
            "bin_end": 108.79939114939295,
            "count": 0
           },
           {
            "bin_start": 108.79939114939295,
            "bin_end": 126.03951291951437,
            "count": 1
           },
           {
            "bin_start": 126.03951291951437,
            "bin_end": 143.27963468963577,
            "count": 3
           },
           {
            "bin_start": 143.27963468963577,
            "bin_end": 160.51975645975716,
            "count": 1
           },
           {
            "bin_start": 160.51975645975716,
            "bin_end": 177.7598782298786,
            "count": 0
           },
           {
            "bin_start": 177.7598782298786,
            "bin_end": 195,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "exang",
         "dtype": "float64",
         "stats": {
          "unique_count": 5,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "oldpeak",
         "dtype": "float64",
         "stats": {
          "unique_count": 8,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "slope",
         "dtype": "float64",
         "stats": {
          "unique_count": 7,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "ca",
         "dtype": "float64",
         "stats": {
          "unique_count": 7,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "thal",
         "dtype": "float64",
         "stats": {
          "unique_count": 6,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "target",
         "dtype": "float64",
         "stats": {
          "unique_count": 2,
          "nan_count": 0,
          "min": "0.0",
          "max": "138.0",
          "histogram": [
           {
            "bin_start": 0,
            "bin_end": 13.8,
            "count": 7
           },
           {
            "bin_start": 13.8,
            "bin_end": 27.6,
            "count": 0
           },
           {
            "bin_start": 27.6,
            "bin_end": 41.400000000000006,
            "count": 0
           },
           {
            "bin_start": 41.400000000000006,
            "bin_end": 55.2,
            "count": 0
           },
           {
            "bin_start": 55.2,
            "bin_end": 69,
            "count": 0
           },
           {
            "bin_start": 69,
            "bin_end": 82.80000000000001,
            "count": 0
           },
           {
            "bin_start": 82.80000000000001,
            "bin_end": 96.60000000000001,
            "count": 0
           },
           {
            "bin_start": 96.60000000000001,
            "bin_end": 110.4,
            "count": 0
           },
           {
            "bin_start": 110.4,
            "bin_end": 124.2,
            "count": 0
           },
           {
            "bin_start": 124.2,
            "bin_end": 138,
            "count": 1
           }
          ]
         }
        },
        {
         "name": "_deepnote_index_column",
         "dtype": "object"
        }
       ],
       "rows_top": [
        {
         "age": 138,
         "sex": 138,
         "cp": 138,
         "trestbps": 138,
         "chol": 138,
         "fbs": 138,
         "restecg": 138,
         "thalach": 138,
         "exang": 138,
         "oldpeak": 138,
         "slope": 138,
         "ca": 138,
         "thal": 138,
         "target": 138,
         "_deepnote_index_column": "count"
        },
        {
         "age": 56.60144927536232,
         "sex": 0.8260869565217391,
         "cp": 0.4782608695652174,
         "trestbps": 134.3985507246377,
         "chol": 251.08695652173913,
         "fbs": 0.15942028985507245,
         "restecg": 0.4492753623188406,
         "thalach": 139.1014492753623,
         "exang": 0.5507246376811594,
         "oldpeak": 1.5855072463768116,
         "slope": 1.1666666666666667,
         "ca": 1.1666666666666667,
         "thal": 2.5434782608695654,
         "target": 0,
         "_deepnote_index_column": "mean"
        },
        {
         "age": 7.96208153750117,
         "sex": 0.38041551386121214,
         "cp": 0.9059204401375942,
         "trestbps": 18.72994396158135,
         "chol": 49.45461360407159,
         "fbs": 0.3674011473702368,
         "restecg": 0.5413212245494148,
         "thalach": 22.598782298785906,
         "exang": 0.4992324585899055,
         "oldpeak": 1.300339693105365,
         "slope": 0.5613244677999094,
         "ca": 1.0434595276713312,
         "thal": 0.6847618288848198,
         "target": 0,
         "_deepnote_index_column": "std"
        },
        {
         "age": 35,
         "sex": 0,
         "cp": 0,
         "trestbps": 100,
         "chol": 131,
         "fbs": 0,
         "restecg": 0,
         "thalach": 71,
         "exang": 0,
         "oldpeak": 0,
         "slope": 0,
         "ca": 0,
         "thal": 0,
         "target": 0,
         "_deepnote_index_column": "min"
        },
        {
         "age": 52,
         "sex": 1,
         "cp": 0,
         "trestbps": 120,
         "chol": 217.25,
         "fbs": 0,
         "restecg": 0,
         "thalach": 125,
         "exang": 0,
         "oldpeak": 0.6,
         "slope": 1,
         "ca": 0,
         "thal": 2,
         "target": 0,
         "_deepnote_index_column": "25%"
        },
        {
         "age": 58,
         "sex": 1,
         "cp": 0,
         "trestbps": 130,
         "chol": 249,
         "fbs": 0,
         "restecg": 0,
         "thalach": 142,
         "exang": 1,
         "oldpeak": 1.4,
         "slope": 1,
         "ca": 1,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": "50%"
        },
        {
         "age": 62,
         "sex": 1,
         "cp": 0,
         "trestbps": 144.75,
         "chol": 283,
         "fbs": 0,
         "restecg": 1,
         "thalach": 156,
         "exang": 1,
         "oldpeak": 2.5,
         "slope": 1.75,
         "ca": 2,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": "75%"
        },
        {
         "age": 77,
         "sex": 1,
         "cp": 3,
         "trestbps": 200,
         "chol": 409,
         "fbs": 1,
         "restecg": 2,
         "thalach": 195,
         "exang": 1,
         "oldpeak": 6.2,
         "slope": 2,
         "ca": 4,
         "thal": 3,
         "target": 0,
         "_deepnote_index_column": "max"
        }
       ],
       "rows_bottom": null
      },
      "text/plain": "              age         sex          cp    trestbps        chol         fbs  \\\ncount  138.000000  138.000000  138.000000  138.000000  138.000000  138.000000   \nmean    56.601449    0.826087    0.478261  134.398551  251.086957    0.159420   \nstd      7.962082    0.380416    0.905920   18.729944   49.454614    0.367401   \nmin     35.000000    0.000000    0.000000  100.000000  131.000000    0.000000   \n25%     52.000000    1.000000    0.000000  120.000000  217.250000    0.000000   \n50%     58.000000    1.000000    0.000000  130.000000  249.000000    0.000000   \n75%     62.000000    1.000000    0.000000  144.750000  283.000000    0.000000   \nmax     77.000000    1.000000    3.000000  200.000000  409.000000    1.000000   \n\n          restecg     thalach       exang     oldpeak       slope          ca  \\\ncount  138.000000  138.000000  138.000000  138.000000  138.000000  138.000000   \nmean     0.449275  139.101449    0.550725    1.585507    1.166667    1.166667   \nstd      0.541321   22.598782    0.499232    1.300340    0.561324    1.043460   \nmin      0.000000   71.000000    0.000000    0.000000    0.000000    0.000000   \n25%      0.000000  125.000000    0.000000    0.600000    1.000000    0.000000   \n50%      0.000000  142.000000    1.000000    1.400000    1.000000    1.000000   \n75%      1.000000  156.000000    1.000000    2.500000    1.750000    2.000000   \nmax      2.000000  195.000000    1.000000    6.200000    2.000000    4.000000   \n\n             thal  target  \ncount  138.000000   138.0  \nmean     2.543478     0.0  \nstd      0.684762     0.0  \nmin      0.000000     0.0  \n25%      2.000000     0.0  \n50%      3.000000     0.0  \n75%      3.000000     0.0  \nmax      3.000000     0.0  ",
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>age</th>\n      <th>sex</th>\n      <th>cp</th>\n      <th>trestbps</th>\n      <th>chol</th>\n      <th>fbs</th>\n      <th>restecg</th>\n      <th>thalach</th>\n      <th>exang</th>\n      <th>oldpeak</th>\n      <th>slope</th>\n      <th>ca</th>\n      <th>thal</th>\n      <th>target</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.000000</td>\n      <td>138.0</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>56.601449</td>\n      <td>0.826087</td>\n      <td>0.478261</td>\n      <td>134.398551</td>\n      <td>251.086957</td>\n      <td>0.159420</td>\n      <td>0.449275</td>\n      <td>139.101449</td>\n      <td>0.550725</td>\n      <td>1.585507</td>\n      <td>1.166667</td>\n      <td>1.166667</td>\n      <td>2.543478</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>7.962082</td>\n      <td>0.380416</td>\n      <td>0.905920</td>\n      <td>18.729944</td>\n      <td>49.454614</td>\n      <td>0.367401</td>\n      <td>0.541321</td>\n      <td>22.598782</td>\n      <td>0.499232</td>\n      <td>1.300340</td>\n      <td>0.561324</td>\n      <td>1.043460</td>\n      <td>0.684762</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>35.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>100.000000</td>\n      <td>131.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>71.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>52.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>120.000000</td>\n      <td>217.250000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>125.000000</td>\n      <td>0.000000</td>\n      <td>0.600000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>2.000000</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>58.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>130.000000</td>\n      <td>249.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>142.000000</td>\n      <td>1.000000</td>\n      <td>1.400000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>3.000000</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>62.000000</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>144.750000</td>\n      <td>283.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>156.000000</td>\n      <td>1.000000</td>\n      <td>2.500000</td>\n      <td>1.750000</td>\n      <td>2.000000</td>\n      <td>3.000000</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>77.000000</td>\n      <td>1.000000</td>\n      <td>3.000000</td>\n      <td>200.000000</td>\n      <td>409.000000</td>\n      <td>1.000000</td>\n      <td>2.000000</td>\n      <td>195.000000</td>\n      <td>1.000000</td>\n      <td>6.200000</td>\n      <td>2.000000</td>\n      <td>4.000000</td>\n      <td>3.000000</td>\n      <td>0.0</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00011-2b048df8-03de-4def-a870-0dc2030dca38",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "3b5d8617",
    "execution_start": 1620377240337,
    "execution_millis": 6,
    "deepnote_cell_type": "code"
   },
   "source": "print(\"(Positive Patients ST depression): \" + str(pos_data['oldpeak'].mean()))\nprint(\"(Negative Patients ST depression): \" + str(neg_data['oldpeak'].mean()))",
   "execution_count": 18,
   "outputs": [
    {
     "name": "stdout",
     "text": "(Positive Patients ST depression): 0.583030303030303\n(Negative Patients ST depression): 1.5855072463768116\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00011-73517cb8-3136-4bfc-a06c-694cf6dec05f",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "1fa0120f",
    "execution_start": 1620377244892,
    "execution_millis": 9,
    "deepnote_cell_type": "code"
   },
   "source": "print(\"(Positive Patients thalach): \" + str(pos_data['thalach'].mean()))\nprint(\"(Negative Patients thalach): \" + str(neg_data['thalach'].mean()))",
   "execution_count": 19,
   "outputs": [
    {
     "name": "stdout",
     "text": "(Positive Patients thalach): 158.46666666666667\n(Negative Patients thalach): 139.1014492753623\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00008-c57308ab-a6a7-4b47-b637-377b4892d734",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "d78e2a9",
    "execution_start": 1620379791482,
    "execution_millis": 10,
    "deepnote_cell_type": "code"
   },
   "source": "num_4_Models = pipeline.Pipeline(steps=[\n\n    ('scalar',preprocessing.QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=10))\n    \n])\n\ncat_4_Models = pipeline.Pipeline(steps=[\n  ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))\n])\n\nmult_prepro = compose.ColumnTransformer(transformers=[\n    ('num', num_4_Models, num_vars),\n    ('cat', cat_4_Models, cat_vars),\n], remainder='drop') \nmult_prepro",
   "execution_count": 25,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 25,
     "data": {
      "text/plain": "ColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('scalar',\n                                                  QuantileTransformer(n_quantiles=200,\n                                                                      output_distribution='normal',\n                                                                      random_state=10))]),\n                                 ['age', 'trestbps', 'chol', 'thalach',\n                                  'oldpeak']),\n                                ('cat',\n                                 Pipeline(steps=[('onehot',\n                                                  OneHotEncoder(handle_unknown='use_encoded_value'))]),\n                                 ['sex', 'cp', 'fbs', 'restecg', 'exang',\n                                  'slope', 'ca', 'thal'])])",
      "text/html": "<style>#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 {color: black;background-color: white;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 pre{padding: 0;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-toggleable {background-color: white;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-item {z-index: 1;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-parallel-item:only-child::after {width: 0;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-27569bfa-752f-47d2-babc-6a3870bfc7b6 div.sk-container {display: inline-block;position: relative;}</style><div id=\"sk-27569bfa-752f-47d2-babc-6a3870bfc7b6\" class\"sk-top-container\"><div class=\"sk-container\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"209fb882-6b0a-4592-8453-6666d9c47c97\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"209fb882-6b0a-4592-8453-6666d9c47c97\">ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('scalar',\n                                                  QuantileTransformer(n_quantiles=200,\n                                                                      output_distribution='normal',\n                                                                      random_state=10))]),\n                                 ['age', 'trestbps', 'chol', 'thalach',\n                                  'oldpeak']),\n                                ('cat',\n                                 Pipeline(steps=[('onehot',\n                                                  OneHotEncoder(handle_unknown='use_encoded_value'))]),\n                                 ['sex', 'cp', 'fbs', 'restecg', 'exang',\n                                  'slope', 'ca', 'thal'])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"8582fbf2-6563-44c1-804e-3cdb493820fb\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"8582fbf2-6563-44c1-804e-3cdb493820fb\">num</label><div class=\"sk-toggleable__content\"><pre>['age', 'trestbps', 'chol', 'thalach', 'oldpeak']</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"6ef06517-0da4-40ed-b35e-86f777a95772\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"6ef06517-0da4-40ed-b35e-86f777a95772\">QuantileTransformer</label><div class=\"sk-toggleable__content\"><pre>QuantileTransformer(n_quantiles=200, output_distribution='normal',\n                    random_state=10)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"06afa705-e346-47ba-9db1-a6561f4f5afc\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"06afa705-e346-47ba-9db1-a6561f4f5afc\">cat</label><div class=\"sk-toggleable__content\"><pre>['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"66a53e70-2ea9-4226-8d7d-1d479384ff57\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"66a53e70-2ea9-4226-8d7d-1d479384ff57\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown='use_encoded_value')</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00019-8c2aff84-9b7f-4559-b826-754c17b9d44e",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "1c33e1f1",
    "execution_start": 1620383745042,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.svm import SVC\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nimport lightgbm as lgb\nfrom xgboost import XGBClassifier",
   "execution_count": 52,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00015-ccaec440-e838-4dec-9e09-dd27e2d8048e",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "79afbbb4",
    "execution_start": 1620384328203,
    "execution_millis": 31,
    "deepnote_cell_type": "code"
   },
   "source": "classifiers = {\n'SVM' :SVC(kernel = 'rbf'),\n'Gaussian': GaussianNB(),\n\"Logistic\":LogisticRegression(),\n\"Decision Tree\": DecisionTreeClassifier(),\n'Random_forest_classifier':RandomForestClassifier(n_estimators = 10),\n'xgboost':XGBClassifier(),\n}\nall_pipelines = {name: pipeline.make_pipeline(mult_prepro, model) for name, model in classifiers.items()}\nall_pipelines['Logistic']",
   "execution_count": 63,
   "outputs": [
    {
     "output_type": "execute_result",
     "execution_count": 63,
     "data": {
      "text/plain": "Pipeline(steps=[('columntransformer',\n                 ColumnTransformer(transformers=[('num',\n                                                  Pipeline(steps=[('scalar',\n                                                                   QuantileTransformer(n_quantiles=200,\n                                                                                       output_distribution='normal',\n                                                                                       random_state=10))]),\n                                                  ['age', 'trestbps', 'chol',\n                                                   'thalach', 'oldpeak']),\n                                                 ('cat',\n                                                  Pipeline(steps=[('onehot',\n                                                                   OneHotEncoder(handle_unknown='use_encoded_value'))]),\n                                                  ['sex', 'cp', 'fbs',\n                                                   'restecg', 'exang', 'slope',\n                                                   'ca', 'thal'])])),\n                ('logisticregression', LogisticRegression())])",
      "text/html": "<style>#sk-0c1583ad-22ba-467e-b998-0d25437183c2 {color: black;background-color: white;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 pre{padding: 0;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-toggleable {background-color: white;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-item {z-index: 1;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-parallel-item:only-child::after {width: 0;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-0c1583ad-22ba-467e-b998-0d25437183c2 div.sk-container {display: inline-block;position: relative;}</style><div id=\"sk-0c1583ad-22ba-467e-b998-0d25437183c2\" class\"sk-top-container\"><div class=\"sk-container\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"e747a56e-47c6-4d85-b402-b410e66b566d\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"e747a56e-47c6-4d85-b402-b410e66b566d\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[('columntransformer',\n                 ColumnTransformer(transformers=[('num',\n                                                  Pipeline(steps=[('scalar',\n                                                                   QuantileTransformer(n_quantiles=200,\n                                                                                       output_distribution='normal',\n                                                                                       random_state=10))]),\n                                                  ['age', 'trestbps', 'chol',\n                                                   'thalach', 'oldpeak']),\n                                                 ('cat',\n                                                  Pipeline(steps=[('onehot',\n                                                                   OneHotEncoder(handle_unknown='use_encoded_value'))]),\n                                                  ['sex', 'cp', 'fbs',\n                                                   'restecg', 'exang', 'slope',\n                                                   'ca', 'thal'])])),\n                ('logisticregression', LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"046229d9-93b8-47d4-b9b1-67bf6597c146\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"046229d9-93b8-47d4-b9b1-67bf6597c146\">columntransformer: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('scalar',\n                                                  QuantileTransformer(n_quantiles=200,\n                                                                      output_distribution='normal',\n                                                                      random_state=10))]),\n                                 ['age', 'trestbps', 'chol', 'thalach',\n                                  'oldpeak']),\n                                ('cat',\n                                 Pipeline(steps=[('onehot',\n                                                  OneHotEncoder(handle_unknown='use_encoded_value'))]),\n                                 ['sex', 'cp', 'fbs', 'restecg', 'exang',\n                                  'slope', 'ca', 'thal'])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"07821885-982c-4317-affd-b1fd56bc9f14\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"07821885-982c-4317-affd-b1fd56bc9f14\">num</label><div class=\"sk-toggleable__content\"><pre>['age', 'trestbps', 'chol', 'thalach', 'oldpeak']</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"dd4cbcf5-4d8d-4028-aded-c3418fa3572d\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"dd4cbcf5-4d8d-4028-aded-c3418fa3572d\">QuantileTransformer</label><div class=\"sk-toggleable__content\"><pre>QuantileTransformer(n_quantiles=200, output_distribution='normal',\n                    random_state=10)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"a77450fb-48c3-4157-8472-195f4a785479\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"a77450fb-48c3-4157-8472-195f4a785479\">cat</label><div class=\"sk-toggleable__content\"><pre>['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"c982fddc-6efe-41e2-b6f3-a1ebfd13d343\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"c982fddc-6efe-41e2-b6f3-a1ebfd13d343\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown='use_encoded_value')</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"39c6f625-02e3-46fd-be1d-91afd315bdb7\" type=\"checkbox\" ><label class=\"sk-toggleable__label\" for=\"39c6f625-02e3-46fd-be1d-91afd315bdb7\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>"
     },
     "metadata": {}
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00021-e2f62098-14d3-435c-ba59-cfbb8b15a0c6",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "6ef570e5",
    "execution_start": 1620384560569,
    "execution_millis": 21,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.metrics import mean_squared_error\npipelines = {name: pipeline.make_pipeline(m_prepro, model) for name, model in classifiers.items()}\nx_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)\n\n\n\nfor name, pipe in pipelines.items():\n  \n\n    start_time = time.time()\n    pipe.fit(x_train,y_train)\n    preds = pipe.predict(x_valid)\n    total_time = time.time() - start_time\n    results = results.append({\"Model\": name,\n                              \"MSE\":  mean_squared_error(y_valid,preds),\n                              \"RMSE\":  mean_squared_error(y_valid,preds,squared=False),\n                              \"Time\":  total_time},\n                              ignore_index=True)\n    \n    \n    results_ord = results.sort_values(by=['RMSE'], ascending=True, ignore_index=True)\n    results_ord.index += 1 \n    clear_output()\n    display(results_ord.style.bar(subset=['MSE', 'RMSE'], vmin=0, color='#5fba7d'))",
   "execution_count": 65,
   "outputs": [
    {
     "output_type": "error",
     "ename": "NameError",
     "evalue": "name 'm_prepro' is not defined",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-65-5076b14161d7>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmetrics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmean_squared_error\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mpipelines\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mpipeline\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_pipeline\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mm_prepro\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mclassifiers\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_valid\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_valid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel_selection\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_test_split\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.4\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'Model'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'MSE'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'RMSE'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'Time'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-65-5076b14161d7>\u001b[0m in \u001b[0;36m<dictcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmetrics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmean_squared_error\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mpipelines\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mpipeline\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_pipeline\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mm_prepro\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mclassifiers\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_valid\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_valid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel_selection\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_test_split\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.4\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'Model'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'MSE'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'RMSE'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'Time'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'm_prepro' is not defined"
     ]
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-f6ef31e8-dcea-48bb-b97d-79f590dd9fc3",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "b4a13aba",
    "execution_start": 1620381422967,
    "execution_millis": 0,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.model_selection import train_test_split\nX_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state =0)",
   "execution_count": 39,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-3cc48e39-cb9d-4df5-b978-8e9ee5f65974",
    "deepnote_cell_type": "code"
   },
   "source": "logreg=logisticRegression()\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-b0ec7724-afac-4456-bad1-2ecd34790775",
    "deepnote_cell_type": "code"
   },
   "source": "df['slope*oldpeak'] = df['slope'] * df['oldpeak']\ndf['slope*thalach'] = df['slope'] * df['thalach']\ndf['oldpeak*thalach'] = df['oldpeak'] * df['thalach']\ndf['exang*oldpeak'] = df['exang'] * df['oldpeak']\ndf['exang*cp'] = df['exang'] * df['cp']\ndf['fbs*chol'] = df['fbs'] * df['chol']",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-f175d7e3-f47c-4e1f-851f-744f8bf07ba2",
    "deepnote_cell_type": "code"
   },
   "source": "cat_vars  = ['sex', 'cp', 'fbs','restecg','exang','ca','slope','thal','exang*cp']        \nnum_vars  = ['age', \n             'trestbps',\n             'chol',\n             'thalach',\n             'oldpeak',\n             'slope*oldpeak',\n             'slope*thalach',\n            'oldpeak*thalach',\n             'exang*oldpeak',\n             'fbs*chol'\n            ] \n\nX = df[cat_vars + num_vars]\ny = df.target",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-95787242-b232-4591-9c78-43c0e2982154",
    "deepnote_cell_type": "code"
   },
   "source": "",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00016-85cea616-888d-451a-acde-103090475795",
    "deepnote_cell_type": "code"
   },
   "source": "",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00010-133d2803-912c-4bda-8d8f-0269f33011d0",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "d8c5a559",
    "execution_start": 1620376261105,
    "execution_millis": 1,
    "deepnote_cell_type": "code"
   },
   "source": "\n\n",
   "execution_count": 11,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00011-df478e77-3265-4614-ad36-853104cf153f",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "be725a3e",
    "execution_start": 1620376374690,
    "execution_millis": 33,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.linear_model import LogisticRegression\n\nclassifier =LogisticRegression()\nclassifier.fit(X_train,y_train)\ny_pred =classifier.predict(X_test)",
   "execution_count": 14,
   "outputs": [
    {
     "name": "stderr",
     "text": "/shared-libs/python3.7/py/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):\nSTOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n\nIncrease the number of iterations (max_iter) or scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00012-485d4d6e-fbce-44e4-8a01-c09bd144d1e3",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "5c08030f",
    "execution_start": 1620377264678,
    "execution_millis": 1,
    "deepnote_cell_type": "code"
   },
   "source": "from sklearn.metrics import confusion_matrix\ncm_test = confusion_matrix(y_pred, y_test)\n\ny_pred_train = classifier.predict(X_train)\ncm_train = confusion_matrix(y_pred_train, y_train)\n",
   "execution_count": 20,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00017-135d0ffc-8ed7-48c1-9242-02acbe76770f",
    "deepnote_to_be_reexecuted": false,
    "source_hash": "840e1045",
    "execution_start": 1620377294109,
    "execution_millis": 1,
    "deepnote_cell_type": "code"
   },
   "source": "print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))\nprint('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))\n",
   "execution_count": 21,
   "outputs": [
    {
     "name": "stdout",
     "text": "Accuracy for training set for Logistic Regression = 0.8553719008264463\nAccuracy for test set for Logistic Regression = 0.8524590163934426\n",
     "output_type": "stream"
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {
    "tags": [],
    "cell_id": "00018-04b78beb-fa5b-4fe9-8c35-d8cc7e2d3aa1",
    "deepnote_cell_type": "code"
   },
   "source": "",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "source": "<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ea7ce3b3-a072-4534-8285-878511d24c52' target=\"_blank\">\n<img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>\nCreated in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>",
   "metadata": {
    "tags": [],
    "created_in_deepnote_cell": true,
    "deepnote_cell_type": "markdown"
   }
  }
 ],
 "nbformat": 4,
 "nbformat_minor": 2,
 "metadata": {
  "orig_nbformat": 2,
  "deepnote": {
   "is_reactive": false
  },
  "deepnote_notebook_id": "962358e8-7317-465c-97b2-32d8506b4fa1",
  "deepnote_execution_queue": []
 }
}