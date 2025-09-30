#import librerie

import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, train_test_split, learning_curve
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_error, r2_score, log_loss, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, RocCurveDisplay, roc_auc_score, auc, PrecisionRecallDisplay
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from scipy.stats import pointbiserialr, skewtest, ttest_ind, chi2_contingency, pearsonr, stats, shapiro, kstest
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer, make_column_selector
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
import warnings
RANDOM_SEED=0