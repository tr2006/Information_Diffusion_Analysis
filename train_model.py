import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv(
    "dataset/higgs-activity_time.txt",
    sep=" ",
    header=None
)

print("Dataset loaded:", df.shape)

# 2. Create features
total_interactions = df.groupby(0).size()

retweets = df[df[3] == "RT"].groupby(0).size()
mentions = df[df[3] == "MT"].groupby(0).size()
replies = df[df[3] == "RE"].groupby(0).size()

unique_reached = df.groupby(0)[1].nunique()

features = pd.DataFrame({
    "Total_Interactions": total_interactions,
    "Retweets": retweets,
    "Mentions": mentions,
    "Replies": replies,
    "Unique_Users_Reached": unique_reached
}).fillna(0)

# 3. Create diffusion score
features["Diffusion_Score"] = (
    features["Unique_Users_Reached"] * 0.6 +
    features["Retweets"] * 0.3 +
    features["Mentions"] * 0.1
)

# 4. Create diffusion levels
features["Diffusion_Level"] = pd.cut(
    features["Unique_Users_Reached"],
    bins=[0, 1, 3, float("inf")],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# 5. Select input features
X = features[
    [
        "Total_Interactions",
        "Retweets",
        "Mentions",
        "Replies"
    ]
]

# 6. Select target
y = features["Diffusion_Level"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Save model
joblib.dump(model, "model/information_diffusion_model.pkl")

print("Model training completed!")
print("Model saved successfully!")