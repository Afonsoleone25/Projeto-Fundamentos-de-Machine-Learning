
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv"

df_raw = pd.read_csv(url)

df = df_raw[[
    "mandante",
    "visitante",
    "mandante_placar",
    "visitante_placar"
]].copy()

df = df.rename(columns={
    "mandante": "HomeTeam",
    "visitante": "AwayTeam",
    "mandante_placar": "FTHG",
    "visitante_placar": "FTAG"
})

def get_result(row):
    if row["FTHG"] > row["FTAG"]:
        return "H"
    elif row["FTHG"] < row["FTAG"]:
        return "A"
    else:
        return "D"

df["FTR"] = df.apply(get_result, axis=1)

df = df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_result = LabelEncoder()

df["HomeTeam"] = le_home.fit_transform(df["HomeTeam"])
df["AwayTeam"] = le_away.fit_transform(df["AwayTeam"])
df["FTR"] = le_result.fit_transform(df["FTR"])

X = df[["HomeTeam","AwayTeam","FTHG","FTAG"]]
y = df["FTR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, pred))
