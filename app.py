import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])
    data = pd.merge(ratings, movies, on='movie_id')
    return data


data = load_data()

# Sidebar
st.sidebar.title("MovieLens Recommender")
page = st.sidebar.radio("Go to", ["Overview", "Top Movies", "Find Similar Movies"])

if page == "Overview":
    st.title("🎬 MovieLens 100k Analysis")

    st.write("## Distribution of Ratings")
    fig, ax = plt.subplots()
    sns.histplot(data['rating'], bins=5, kde=False, ax=ax)
    st.pyplot(fig)

    st.write("## Top 10 Most Rated Movies")
    top_rated = data['title'].value_counts().head(10)
    st.bar_chart(top_rated)

elif page == "Top Movies":
    st.title("⭐ Top Rated Movies (with 50+ ratings)")
    movie_stats = data.groupby('title').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['average_rating', 'rating_count']
    filtered = movie_stats[movie_stats['rating_count'] >= 50]
    top_movies = filtered.sort_values('average_rating', ascending=False).head(10)

    st.dataframe(top_movies)

elif page == "Find Similar Movies":
    st.title("🔍 Movie Recommender")
    movie_user_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

    movie_list = data['title'].unique().tolist()
    movie_input = st.selectbox("Choose a movie you like", sorted(movie_list))

    if movie_input:
        movie_corr = movie_user_matrix.corrwith(movie_user_matrix[movie_input])
        corr_df = pd.DataFrame(movie_corr, columns=['correlation'])
        corr_df.dropna(inplace=True)

        rating_stats = data.groupby('title').agg({'rating': ['mean', 'count']})
        rating_stats.columns = ['mean_rating', 'rating_count']
        corr_df = corr_df.join(rating_stats)

        recommendations = corr_df[corr_df['rating_count'] > 50].sort_values("correlation", ascending=False)

        st.write(f"### Movies similar to: `{movie_input}`")
        st.dataframe(recommendations.head(10))
