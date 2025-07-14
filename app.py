import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from itertools import cycle
<<<<<<< HEAD

=======
>>>>>>> 6db9bdd (Intial Commit)
# Load pre-trained KMeans model and scaler
with open("kmeans_model.pkl", "rb") as f_model:
    kmeans = pickle.load(f_model)

with open("scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)

# Load customer data
df = pd.read_csv("Mall_Customers.csv")

# Features used by the model
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# App Title
st.title("🛍️ Customer Grouping App")
st.markdown("""
This tool helps you understand your customers by grouping them based on their **age**, **income**, and **spending behavior**.
""")

# Step 1: Choose what to show on the graph
st.subheader("1. Choose Features to Display")
selected_features = []
default = ['Annual Income (k$)', 'Spending Score (1-100)']
for col in features:
    if st.checkbox(f"Show {col}", value=(col in default)):
        selected_features.append(col)

if len(selected_features) >= 2:
    # Step 2: Choose X and Y axis
    st.subheader("2. Choose What to Show on the Graph")
    x_axis = st.selectbox("X-Axis (bottom):", selected_features)
    y_axis = st.selectbox("Y-Axis (left):", [f for f in selected_features if f != x_axis])

    # Step 3: Customize age, income, and spending ranges
    st.subheader("3. Define Age, Income, and Spending Ranges")

    st.markdown("**Age Groups**")
    young_age = st.slider("Maximum age for 'Young' customers:", 18, 60, value=35)
    old_age = st.slider("Minimum age for 'Older' customers:", young_age + 1, 80, value=55)

    st.markdown("**Income Groups (in $1,000s)**")
    low_income = st.slider("Maximum income for 'Low Income':", 0, 100, value=40)
    high_income = st.slider("Minimum income for 'High Income':", low_income + 1, 150, value=70)

    st.markdown("**Spending Score (1 to 100)**")
    low_spend = st.slider("Maximum for 'Low Spender':", 0, 100, value=40)
    high_spend = st.slider("Minimum for 'High Spender':", low_spend + 1, 100, value=60)

    # Step 4: Predict customer groups
    scaled_data = scaler.transform(df[features])
    df['Group'] = kmeans.predict(scaled_data)

    # Define custom labels
    def get_age_group(age):
        if age <= young_age:
            return "Young"
        elif age >= old_age:
            return "Older"
        else:
            return "Middle-aged"

    def get_income_group(income):
        if income <= low_income:
            return "Low Income"
        elif income >= high_income:
            return "High Income"
        else:
            return "Average Income"

    def get_spending_group(score):
        if score <= low_spend:
            return "Low Spender"
        elif score >= high_spend:
            return "High Spender"
        else:
            return "Average Spender"

    # Create readable labels
    df['Age Group'] = df['Age'].apply(get_age_group)
    df['Income Group'] = df['Annual Income (k$)'].apply(get_income_group)
    df['Spending Group'] = df['Spending Score (1-100)'].apply(get_spending_group)
    df['Customer Type'] = df['Age Group'] + ", " + df['Income Group'] + ", " + df['Spending Group']

    # Step 5: 2D Visualization
    st.subheader("4. 2D Customer Group Plot")

    color_list = [
        'red', 'blue', 'green', 'orange', 'purple',
        'brown', 'cyan', 'magenta', 'olive', 'pink'
    ]
    color_cycle = cycle(color_list)

    fig, ax = plt.subplots()
    for group_id, color in zip(sorted(df['Group'].unique()), color_cycle):
        group_data = df[df['Group'] == group_id]
        ax.scatter(
            group_data[x_axis],
            group_data[y_axis],
            color=color,
            label=f"Group {group_id}",
            edgecolors='black',
            alpha=0.7,
            s=80
        )

    # Show centers of each group
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=features)
    ax.scatter(
        centers_df[x_axis],
        centers_df[y_axis],
        color='black',
        s=200,
        marker='X',
        label='Group Center'
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Customer Groups: {x_axis} vs {y_axis}")
    ax.legend()
    st.pyplot(fig)

    # Step 6: Show summary
    st.subheader("5. Group Overview Table")
    summary = df.groupby('Group')[features].mean().round(1).reset_index()
    summary = summary.rename(columns={
        'Age': 'Avg Age',
        'Annual Income (k$)': 'Avg Income ($k)',
        'Spending Score (1-100)': 'Avg Spending Score'
    })

    labels = df.groupby('Group')[['Age Group', 'Income Group', 'Spending Group', 'Customer Type']] \
        .agg(lambda x: x.value_counts().idxmax()).reset_index()

    final_summary = pd.merge(summary, labels, on='Group')
    st.dataframe(final_summary)

    # Step 7: 3D Interactive View
    st.subheader("6. 3D Interactive Customer View")

    fig3d = px.scatter_3d(
        df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color=df['Group'].astype(str),
        symbol=df['Group'].astype(str),
        size_max=10,
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['Gender', 'Customer Type']
    )

    fig3d.update_traces(marker=dict(size=6, line=dict(width=1, color='black')))
    fig3d.update_layout(
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Income ($k)',
            zaxis_title='Spending Score'
        ),
        legend_title="Customer Group"
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # Step 8: Download
    st.download_button("⬇️ Download Grouped Data", data=df.to_csv(index=False),
                       file_name="customer_groups.csv", mime="text/csv")

    st.markdown("🔎 This tool uses a trained model to find patterns in your customer data.")

else:
    st.warning("Please select at least two features to display.")
