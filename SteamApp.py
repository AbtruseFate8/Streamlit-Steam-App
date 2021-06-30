import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time
import altair as alt
import matplotlib as mpl
import math, nltk, warnings
from nltk.corpus import wordnet
from wordcloud import WordCloud, STOPWORDS

# Author: Zach Droog, Juan Toro

warnings.filterwarnings('ignore')
PS = nltk.stem.PorterStemmer()

sns.set_context("poster")
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 14}
mpl.rc('font', **font)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Clean and Prepare Data


df_url = 'https://github.com/AbtruseFate8/Streamlit-Steam-App/blob/92dd4f76697d73a279ac93fbb30887b53d2df244/steam.csv'
df = pd.read_csv(df_url, index_col= 0)

df_desc_url = "https://github.com/AbtruseFate8/Streamlit-Steam-App/blob/57960bea5a34b707b59b37ddd99257b6d7495dbc/steam_description_data.csv"
df_desc = pd.read_csv(df_desc_url, index_col = 0)

df_images_url = "https://github.com/AbtruseFate8/Streamlit-Steam-App/blob/57960bea5a34b707b59b37ddd99257b6d7495dbc/steam_media_data.csv"
df_images = pd.read_csv(df_images_url, index_col = 0)

#df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Python Scripts\Operations_Analytics\steam.csv")
#df_desc = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Python Scripts\Operations_Analytics\steam_description_data.csv")
#df_images = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Python Scripts\Operations_Analytics\steam_media_data.csv")


# Link description to each game
df_desc.rename(columns={'steam_appid':'appid'}, inplace=True)
df_merged = pd.merge(df, df_desc, on = "appid")
df_merged.drop(columns = ['detailed_description','about_the_game'],
              inplace=True, axis=1)

# link image headers to each game
df_images.rename(columns={'steam_appid':'appid'}, inplace=True)
df_merged = pd.merge(df_merged, df_images, on = "appid")
df_merged.drop(columns = ['screenshots','background', 'movies'],
              inplace=True, axis=1)


# Calculate ratings 
def calc_rating(row):
    pos = row['positive_ratings']
    neg = row['negative_ratings']
    total_reviews = pos + neg
    average = pos / total_reviews

    # pulls score towards 50, pulls more strongly for games with few reviews
    score = average - (average*0.5) * 2**(-math.log10(total_reviews + 1))
    return score * 100

df_merged['total_ratings'] = df_merged['positive_ratings'] + df_merged['negative_ratings']
df_merged['rating_ratio'] = df_merged['positive_ratings'] / df_merged['total_ratings']
df_merged['rating'] = df_merged.apply(calc_rating, axis=1)

# Create Free vs Paid Flag
df_merged['price_type'] = "Free"
df_merged.loc[df_merged['price'] > 0, 'price_type'] = 'Paid'

# convert release_date to datetime type and create separate column for release_year
df_merged['release_date'] = df_merged['release_date'].astype('datetime64[ns]')
df_merged['release_year'] = df_merged['release_date'].apply(lambda x: x.year)

# keep lower bound of owners column, as integer then split into categoricals
df_merged['owners'] = df_merged['owners'].str.split('-').apply(lambda x: x[0]).astype(int)
df_merged['owners_flag'] = 'Under 20,000 owners'
df_merged.loc[df_merged['owners'] >= 20000, 'owners_flag'] = '20,000+ owners'

# create combined content for content based filtering rec engine
df_merged['combined_content'] = df_merged['name'] + " " + df_merged['steamspy_tags'] + " " + df_merged['genres']
df_merged['combined_content'] = df_merged['combined_content'].str.replace(";", " ")

# create working dataframe
df_full = df_merged.copy()

# Dropping all games for rec engine that do not have at least 1000 positive reviews
df_pos = df_full[df_full['positive_ratings'] > 1000].reset_index(drop = True)

# Descriptive plots

alt.data_transformers.disable_max_rows()

selector = alt.selection(type="single", empty='none', on='mouseover')
steamblue = '#00adee'

PosBar = alt.Chart(df_full.sort_values(by = 'positive_ratings', ascending = False).iloc[:10]
                  ).mark_bar().encode(
         alt.Y('name', title = 'Game Title', sort = alt.EncodingSortField(field = 'positive_ratings', 
                                                                          order = "descending")),
         alt.X('positive_ratings', title = 'Positive Ratings'),
         tooltip = 'positive_ratings',
         color = alt.condition(selector,
                              alt.value('darkblue'),
                              alt.value(steamblue))).add_selection(selector)

PosBar_insight = ("""
In terms of the games with the most positive ratings,
Counter-Strike:Global Offensive is by far the most positively
rated game with 2644404 positive ratings, followed by Dota 2 (863507)
and Team Fortress 2 (515879).
""")

x = df_full['developer'].value_counts().iloc[:20]
topdev = pd.DataFrame({'Developer': x.index, 'game_count': x.values})

DevBar = alt.Chart(topdev).mark_bar().encode(
         alt.Y('Developer', title = 'Developer', sort = alt.EncodingSortField(field = 'game_count', 
                                                                              order = "descending")),
         alt.X('game_count', title = 'Games Published'),
         tooltip = 'game_count',
         color = alt.condition(selector,
                              alt.value('darkblue'),
                              alt.value(steamblue))).add_selection(selector)

DevBar_insight = ("""
Across all developers that have published games on Stream,
Choice of Games has the the most games published with 94 
distinct games followed by KOEI TECMO GAMES (72) and
Ripknot Systems (62).
""")

x = df_full['release_year'].value_counts()
newgames = pd.DataFrame({'release_year': x.index, 'game_count': x.values})
newgames = newgames[newgames['release_year'] > 2006] #very few releases pre 2006 so excluded
newgames = newgames[newgames['release_year'] < 2019] #incomplete 2019 data

Hist = alt.Chart(newgames).mark_bar().encode(
    alt.X('release_year:O', title = "Year"),
    alt.Y('game_count', title = "New Game Releases"),
    tooltip = 'game_count',
    color = alt.condition(selector,
                          alt.value('darkblue'),
                          alt.value(steamblue))).add_selection(selector)

Hist_insight = ("""
Since Steam was first launced, the number of games released each year has
continued to trend upward. Since 2012/2013, there has been a noticable
increase in the amount of games released each year. This was likely caused
in response to major performance updates released on the platfrom around
that time including family game sharing, streaming functionality and redesigned
storefront. These upgrades likely attracted developers to the platform
as a viable medium to distribute their products. Note that years 1997-2007
been removed due to low game counts and drop in 2019  due to the data being
extracted prior to year end.
""")

Freegames = pd.DataFrame({"Free": df_full[df_full['price_type'] == 'Free'].groupby(['release_year'])['price_type'].size()}).reset_index()
Paidgames = pd.DataFrame({"Paid": df_full[df_full['price_type'] == 'Paid'].groupby(['release_year'])['price_type'].size()}).reset_index()
FreevsPaid = Paidgames.merge(Freegames, how = 'left').fillna(0)

FreevsPaid = FreevsPaid[(FreevsPaid['release_year'] > 2006) ]
FreevsPaid = FreevsPaid[(FreevsPaid['release_year'] < 2019) ] 

Free = alt.Chart(FreevsPaid).mark_line(point = True).encode(
        alt.X('release_year:N', title = "Year"),
        alt.Y('Free:Q', title = "Game Releases"),
        color = alt.value('orange'),
        tooltip = [alt.Tooltip('Free', title='Free Games')])

Paid = alt.Chart(FreevsPaid).mark_line(point = True).encode(
        alt.X('release_year:N', title = "Year"),
        alt.Y('Paid:Q', title = "Game Releases"),
        color = alt.value(steamblue),
        tooltip = [alt.Tooltip('Paid', title='Paid Games')])

FreePaid_insight = ("""
Breaking down the distribution of games published on Steam by pricing,
the majority of games are paid but the amount of free games on the platform
is still growing but at a much slower rate. Note that years 1997-2007
been removed due to low game counts and drop in 2019  due to the data being
extracted prior to year end.
""")

Rating = alt.Chart(df_pos.sort_values(by = 'positive_ratings', ascending = False).iloc[1:len(df_pos)]).mark_circle(size=50).encode(
         alt.Y('total_ratings:Q', title = ' Total Ratings'),
         alt.X('rating:Q', title = 'Overall Rating'),
         tooltip = ['name', 'price_type', 'rating', 'total_ratings'],
         color = alt.Color('price_type:N',
                           scale=alt.Scale(
                               domain=['Free', 'Paid'],
                               range=['orange', steamblue])))

Rating_insight = ("""
Based on games with at least 1000 positive ratings, overall ratings trend above a
score of 50 or higher. This is due to the rating formula pulls closer to 50 the less
ratings a game has. Note that Counter-Stike: Global Offensive was removed from the above
graphic as an outlier with an extremely large number of total ratings (127873)
with a rating of 95.97.
""")

# Text Clustering 

df_full['combined_content_cluster'] = df_full['categories'].str.cat(df_full['steamspy_tags'],sep=";")

def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
            tone = 100
            h = int(360.0 * tone / 255.0)
            s = int(100.0 * 255.0 / 255.0)
            l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
            return "hsl({}, {}%, {}%)".format(h, s, l)

def remove_non_english(df_full):
    # keep only rows marked as supporting english
    df_full = df_full[df_full['english'] == 1].copy()
    
    # keep rows which don't contain 3 or more non-ascii characters in succession
    df_full = df_full[~df_full['name'].str.contains('[^\u0001-\u007F]{3,}')]
    
    # remove english column, now redundant
    df_full = df_full.drop('english', axis=1)
    
    return df_full

set_keywords = set()
for list_keywords in df_full['combined_content_cluster'].str.split(';').values:
    if type(list_keywords) == float: continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(list_keywords)

def count_word(df_full, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df_full[ref_col].str.split(';'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords: 
            if pd.notnull(s): keyword_count[s] += 1

    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences_comb = []
    for k,v in keyword_count.items():
        keyword_occurences_comb.append([k,v])
    keyword_occurences_comb.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences_comb, keyword_count

keyword_occurences_comb, dum = count_word(df_full, 'combined_content_cluster', set_keywords)

occ_total_insight = ("""
Having combined the information from the 'Category' and 'Tags' columns into a consolidated 
'Combined Content' column, a list of keywords was created to find out the number of times 
each of them appeared in the dataset. This graph represents the content of the most 
popular categories on steam - according to the dataset. The following are the top 10 most 
frequently occuring game categories/tags:

1. Single-Player (25678)
2. Indie (16232)
3. Steam Achievements (14130)
4. Action (10322)
5. Casual (8205)
6. Steam Trading Cards (7918)
7. Adventure (7770)
8. Steam Cloud (7219)
9. Full Controller Support (5695)
10. Partial Controller Support (4234)
""")

occ_genre_insight = ("""
Having conducted a text clustering analysis on the most frequently occuring genres in  
the dataset, we can see the top game genres on steam. This graph represents the content 
of the most popular game genres on steam. The following are the top 10 most 
frequently occuring game genres:

1. Indie (19421)
2. Action (11903)
3. Casual (10210)
4. Adventure (10032)
5. Strategy (5247)
6. Simulation (5194)
7. RPG (4311)
8. Early Access (2954)
9. Free to Play (1704)
10. Sports (1322)
""")

instructions = ("""
Like a game and want to play something similar? Simply enter the the title of a game you
love and the number of recommendations you would like to see to generate some potential
next purchases on Steam!

Note: Due to technical constraints, this engine only works on games with at least 1000 
positive ratings.
""")

# Import "About this project" markdown

from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

About = read_markdown_file(r"https://github.com/AbtruseFate8/Streamlit-Steam-App/blob/69bad8d7bedb07cdf55ad91a155ccc54a8a385f8/About.md")

# Recommendation Engine

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Declare Algorithm

cv = CountVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
cv_matrix = cv.fit_transform(df_pos['combined_content'])
cosine_similarities = cosine_similarity(cv_matrix)

results = {}
for idx, row in df_pos.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], df_pos['name'][i]) for i in similar_indices]
    results[row['name']] = similar_items[1:]
print('Calculations Complete')

# See results

@st.cache(suppress_st_warning = True, show_spinner = False)
def cv_recommend(name, num_rec):
    if name == "":
        st.write("Please input a game title to receive recommendations")
    else:    
        st.write("Recommending " + str(num_rec) + " games similar to " + str(name) + "...")
        st.write("-------")
        time.sleep(5)
        recs = results[name][:num_rec]
        for rec in recs:
            st.write("Recommended: " + str(rec[1]) + " (Similarity score: " + str(round(rec[0]*100,3)) + "%)")
            x = df_pos.loc[df_pos['name'] == str(rec[1]), 'header_image']
            link = x.iloc[0]
            image = st.image(link)

# App Functionality            

def main():
    menu = ["Descriptive Insights", "Text Clustering", "Game Recommendations", "About this Project"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Descriptive Insights":
        st.title("Steam Report - Descriptive Insights")
        st.subheader("Top 10 Most Positively Rated Games")
        st.altair_chart(PosBar, use_container_width=True)
        st.markdown(PosBar_insight) 
        st.subheader("Top 20 Developers by Title Releases")
        st.altair_chart(DevBar, use_container_width=True)
        st.markdown(DevBar_insight)
        st.subheader("Games Releases by Year")
        st.altair_chart(Hist, use_container_width=True)
        st.markdown(Hist_insight)
        st.subheader("Games Releases by Year + Price Type")
        st.altair_chart(Free + Paid, use_container_width=True)
        st.markdown(FreePaid_insight)
        st.subheader("Overall Rating by Total Ratings")
        st.altair_chart(Rating, use_container_width=True)
        st.markdown(Rating_insight)
        
    elif choice == "Text Clustering":
        st.title("Steam Report - Text Clustering Insights")
        st.subheader("Most Frequently Occuring Categories & Tags")
        fig = plt.figure(figsize=(18,13))
        ax1 = fig.add_subplot(2,1,1)
        words = dict()
        trunc_occurences_comb = keyword_occurences_comb[0:50]
        for s in trunc_occurences_comb:
            words[s[0]] = s[1]
        tone = 100
        wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)
        ax1.imshow(wordcloud, interpolation="bilinear")
        ax1.axis('off')
        st.pyplot()
        st.markdown(occ_total_insight)
        #
        st.subheader("Most Frequently Occuring Genres")
        genre_labels = set()
        for s in df_full['genres'].str.split(';').values:
            genre_labels = genre_labels.union(set(s))

        keyword_occurences, dum = count_word(df_full, 'genres', genre_labels)
        words = dict()
        trunc_occurences_comb = keyword_occurences[0:100]
        for s in trunc_occurences_comb:
            words[s[0]] = s[1]
        wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()
        st.pyplot()
        st.markdown(occ_genre_insight)

    elif choice == "Game Recommendations":
        st.title("Steam Report - Game Recommendations")
        st.markdown(instructions)
        form = st.form(key = 'my_form')
        name = form.text_input("Game Title")
        num_rec = form.slider("Number of Recommendations",1,10)
        submit = form.form_submit_button('Recommend')
        if submit:
            try:
                result = cv_recommend(name, num_rec)
            except:
                result = "Not Found"
                st.write(result)
    else:
        st.title("Steam Report - About this Project")
        st.markdown(About, unsafe_allow_html=True)  

if __name__ == '__main__':
    main()
