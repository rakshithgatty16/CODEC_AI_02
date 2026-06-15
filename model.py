import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("churn.csv")

print("Before Encoding")
print(df.head())

le = LabelEncoder()

categorical_cols = [
    "Gender",
    "Partner",
    "Dependents",
    "Contract",
    "PaymentMethod",
    "Churn"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("After Encoding")
print(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model Saved Successfully")