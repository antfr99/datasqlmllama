import streamlit as st
import pandas as pd
import pandasql as ps
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors



# --- Page Config ---
st.set_page_config(
    layout="wide",
    page_title="IMDb/SQL/PYTHON Data Project 🎬"
)

st.title("IMDb/SQL/PYTHON Data Project 🎬")
st.write("""
This is a film/data project that integrates several Python libraries, including Pandas, PandasQL, NumPy, Streamlit, Scikit-learn, SciPy, TextBlob, Matplotlib, Seaborn, NetworkX and Sentence-Transformers. It also incorporates SQL, OMDb API, AI, GitHub, and IMDb.
""")

# --- Load Excel files ---
try:
    IMDB_Ratings = pd.read_excel("imdbratings.xlsx")
    IMDB_Ratings_2019 = pd.read_excel("imdbratings2019onwards.xlsx")  # New workbook
    My_Ratings = pd.read_excel("myratings.xlsx")
    Votes = pd.read_excel("votes.xlsx")  # Optional votes source
except Exception as e:
    st.error(f"Error loading Excel files: {e}")
    IMDB_Ratings = pd.DataFrame()
    IMDB_Ratings_2019 = pd.DataFrame()
    My_Ratings = pd.DataFrame()
    Votes = pd.DataFrame()

# --- Clean unnamed columns ---
def clean_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

IMDB_Ratings = clean_unnamed_columns(IMDB_Ratings)
IMDB_Ratings_2019 = clean_unnamed_columns(IMDB_Ratings_2019)
My_Ratings = clean_unnamed_columns(My_Ratings)
Votes = clean_unnamed_columns(Votes)

# --- Append and remove duplicates ---
if not IMDB_Ratings_2019.empty:
    IMDB_Ratings = pd.concat([IMDB_Ratings, IMDB_Ratings_2019], ignore_index=True)
    IMDB_Ratings = IMDB_Ratings.drop_duplicates(subset=["Movie ID"], keep="last")

# --- Merge votes ---
if not Votes.empty:
    IMDB_Ratings = IMDB_Ratings.merge(Votes, on="Movie ID", how="left")

# --- Show Tables ---
st.write("---")
st.write("### IMDb Ratings Table")
if not IMDB_Ratings.empty:
    st.dataframe(IMDB_Ratings, width="stretch", height=400)
else:
    st.warning("IMDb Ratings table is empty or failed to load.")

st.write("### My Ratings Table")
if not My_Ratings.empty:
    My_Ratings['Year_Sort'] = pd.to_numeric(My_Ratings['Year'], errors='coerce')
    My_Ratings_sorted = My_Ratings.sort_values(by="Year_Sort", ascending=False)
        # Rename column only for display
    display_ratings = My_Ratings_sorted.rename(columns={"Your Rating": "My Ratings"})
    display_ratings = display_ratings.drop(columns=['Year_Sort'])
    st.dataframe(display_ratings, width="stretch", height=400)
else:
    st.warning("My Ratings table is empty or failed to load.")

# --- Scenarios ---

scenario = st.radio(
    "Choose a scenario:",
    [
        "1 – Highlight Disagreements (SQL)",
        "2 – Hybrid Recommendations (SQL)",
        "3 – Top Unseen Films by Decade (SQL)",
        "4 – Statistical Insights by Genre (Agreement)",
        "5 – Statistical Insights by Director (t-test)",
        "6 – Review Analysis (Sentiment, Subjectivity)",
        "7 – Poster Image Analysis (OMDb API)",
        "8 – Graph Based Movie Relationships",
        "9 – Natural-Language Film Q&A Assistant",
        "10 – Predict My Ratings (ML)", 
        "11 – Model Evaluation (Feature Importance)",
        "12 – Feature Hypothesis Testing",
        "13 – Semantic Genre & Recommendations (Deep Learning / NLP)",
        "14 – Live Ratings Monitor (MLOps + CI/CD + Monitoring)",
        "15 – Psycho 1960 Film (Trained AI Model)"
                
                
    ]
)




# --- Scenario 1: SQL Playground ---
if scenario == "1 – Highlight Disagreements (SQL)":
    st.header("1 – Highlight Disagreements (SQL)")
    st.write("Movies where my rating differs from IMDb by more than 2 points.")

    default_query_1 = """SELECT 
       pr.Title,
       pr.[Your Rating] AS [My Rating],
       ir.[IMDb Rating],
       ABS(CAST(pr.[Your Rating] AS FLOAT) - CAST(ir.[IMDb Rating] AS FLOAT)) AS Rating_Diff,
       CASE 
            WHEN pr.[Your Rating] > ir.[IMDb Rating] THEN 'I Liked More'
            ELSE 'I Liked Less'
       END AS Disagreement_Type
FROM My_Ratings pr
JOIN IMDB_Ratings ir
    ON pr.[Movie ID] = ir.[Movie ID]
WHERE ABS(CAST(pr.[Your Rating] AS FLOAT) - CAST(ir.[IMDb Rating] AS FLOAT)) > 2
ORDER BY Rating_Diff DESC, ir.[Num Votes] DESC
LIMIT 1000;"""

    user_query = st.text_area("Enter SQL query:", default_query_1, height=500, key="sql1")
    if st.button("Run SQL Query – Find my disagreements", key="run_sql1"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=800)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")

# --- Scenario 2: SQL Playground ---
if scenario == "2 – Hybrid Recommendations (SQL)":
    st.header("2 – Hybrid Recommendations (SQL)")
    st.write("""
    Recommend movies I haven't seen yet with a bonus point system:  
    - Director I liked before → +1 point  
    - Genre is Comedy or Drama → +0.5  
    - Other genres → +0.2
    """)

    default_query_2 = """SELECT ir.Title,
       ir.[IMDb Rating],
       ir.Director,
       ir.Genre,
       ir.Year,
       CASE WHEN ir.Director IN (SELECT DISTINCT Director FROM My_Ratings WHERE [Your Rating] >= 7) THEN 1 ELSE 0 END AS Director_Bonus,
       CASE WHEN ir.Genre IN ('Comedy','Drama') THEN 0.5 ELSE 0.2 END AS Genre_Bonus,
       ir.[IMDb Rating] 
       + CASE WHEN ir.Director IN (SELECT DISTINCT Director FROM My_Ratings WHERE [Your Rating] >= 7) THEN 1 ELSE 0 END
       + CASE WHEN ir.Genre IN ('Comedy','Drama') THEN 0.5 ELSE 0.2 END AS Recommendation_Score
FROM IMDB_Ratings ir
LEFT JOIN My_Ratings pr
    ON ir.[Movie ID] = pr.[Movie ID]
WHERE pr.[Your Rating] IS NULL
  AND ir.[Num Votes] > 40000
ORDER BY Recommendation_Score DESC
LIMIT 10000;"""

    user_query = st.text_area("Enter SQL query:", default_query_2, height=500, key="sql2")
    if st.button("Run SQL Query – Recommend movies", key="run_sql2"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=800)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")



# --- Scenario 3: SQL Playground ---
if scenario == "3 – Top Unseen Films by Decade (SQL)":
    st.header("3 – Top Unseen Films by Decade (SQL)")
    st.write("""
    Shows the highest-rated unseen films grouped by decade.  
    Uses Python deduplication and limits results to the top 20 per decade.
    """)

    # Cleaner SQL – no redundant CTE
    default_query_3 = """
SELECT *
FROM (
    SELECT ir.[Movie ID], 
           ir.Title,
           ir.[IMDb Rating],
           ir.[Num Votes],
           ir.Genre,
           ir.Director,
           ir.Year,
           (ir.Year / 10) * 10 AS Decade,
           ROW_NUMBER() OVER (
               PARTITION BY (ir.Year / 10) * 10 
               ORDER BY ir.[IMDb Rating] DESC, ir.[Num Votes] DESC
           ) AS RankInDecade
    FROM IMDB_Ratings ir
    LEFT JOIN My_Ratings pr
        ON ir.[Movie ID] = pr.[Movie ID]
    WHERE pr.[Your Rating] IS NULL
      AND ir.[Num Votes] > 50000
) ranked
WHERE RankInDecade <= 20
ORDER BY Decade, [IMDb Rating] DESC, [Num Votes] DESC;
"""

    # Text area to allow user edits
    user_query = st.text_area("Enter SQL query:", default_query_3, height=600, key="sql3")

    # Run button
    if st.button("Run SQL Query – Top unseen films", key="run_sql3"):
        try:
            result = ps.sqldf(user_query, {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings})
            st.dataframe(result, width="stretch", height=800)
        except Exception as e:
            st.error(f"Error in SQL query: {e}")



# --- Scenario 9: Python ML ---
if scenario == "10 – Predict My Ratings (ML)":
    st.header("10 – Predict My Ratings (ML)")
    st.write("""
    Predict my ratings for unseen movies using a machine learning model.

    **How it works:**
    1. The model uses my existing ratings (`My_Ratings`) as training data.
    2. Features used include:  
       - IMDb Rating  
       - Genre  
       - Director  
       - Year of release  
       - Number of votes
    3. A Random Forest Regressor learns patterns from the movies I've already rated.
    4. The model predicts how I might rate movies I haven't seen yet (`Predicted Rating`).

    """)

    ml_code = '''
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
train_df = df_ml[df_ml['Your Rating'].notna()]
predict_df = df_ml[df_ml['Your Rating'].isna()]


categorical_features = ['Genre', 'Director']
numerical_features = ['IMDb Rating', 'Num Votes', 'Year']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

model = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train = train_df[categorical_features + numerical_features]
y_train = train_df['Your Rating']
model.fit(X_train, y_train)
X_pred = predict_df[categorical_features + numerical_features]
predict_df['Predicted Rating'] = model.predict(X_pred)
predict_df
'''

    user_ml_code = st.text_area("Python ML Code (editable)", ml_code, height=1000)

    st.sidebar.header("ML Options")
    min_votes = st.sidebar.slider("Minimum IMDb Votes", 0, 500000, 50000, step=5000)
    top_n = st.sidebar.slider("Number of Top Predictions", 5, 50, 30, step=5)

    if st.button("Run Python ML Code", key="run_ml"):
        try:
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_ml_code, {}, local_vars)
            predict_df = local_vars['predict_df']
            predict_df = predict_df[predict_df['Num Votes'] >= min_votes]
            st.dataframe(
                predict_df[['Title','IMDb Rating','Genre','Director','Predicted Rating']]
                .sort_values(by='Predicted Rating', ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
        except Exception as e:
            st.error(f"Error running ML code: {e}")




# --- Scenario 4: Statistical Insights ---
if scenario == "4 – Statistical Insights by Genre (Agreement)":
    st.header("4 – Statistical Insights by Genre (Agreement)")
    st.write("""
    This analysis measures how often my ratings align with IMDb ratings **within a tolerance band of ±1 point**.  
    Results are grouped by genre, showing agreements, disagreements, and overall percentages.
    """)

    stats_code = '''
df_compare = IMDB_Ratings.merge(
    My_Ratings[['Movie ID','Your Rating']],
    on='Movie ID', how='inner'
)

df_compare['Agreement'] = (
    (df_compare['Your Rating'] - df_compare['IMDb Rating']).abs() <= 1
)

genre_agreement = (
    df_compare.groupby('Genre')
    .agg(
        Total_Movies=('Movie ID','count'),
        Agreements=('Agreement','sum')
    )
    .reset_index()
)

genre_agreement['Disagreements'] = (
    genre_agreement['Total_Movies'] - genre_agreement['Agreements']
)
genre_agreement['Agreement_%'] = (
    genre_agreement['Agreements'] / genre_agreement['Total_Movies'] * 100
).round(2)

genre_agreement.sort_values(by='Agreement_%', ascending=False)
'''

    # Editable code box
    user_stats_code = st.text_area("Python Statistical Code (editable)", stats_code, height=600)

    if st.button("Run Statistical Analysis", key="run_stats5"):
        try:
            # Run the code entered in the text area
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_stats_code, {}, local_vars)

            # Retrieve dataframe if created
            if "genre_agreement" in local_vars:
                st.dataframe(local_vars["genre_agreement"], width="stretch", height=500)
            else:
                st.warning("No output dataframe named 'genre_agreement' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running Statistical Analysis code: {e}")






# --- Scenario 5: Statistical Insights (t-test per Director) ---
if scenario == "5 – Statistical Insights by Director (t-test)":
    st.header("5 – Statistical Insights by Director (t-test)")
    st.write("""
This analysis compares my ratings with IMDb ratings on a director-by-director basis using a **paired t-test**.  
The test checks whether the differences between my ratings and IMDb’s are statistically significant for each director.  

- **t-statistic**: shows the size and direction of the difference (positive = I rate higher than IMDb, negative = I rate lower).  
- **p-value**: shows whether the difference is statistically significant or could be due to chance. p < 0.05 (significant) → Unlikely the difference is due to chance. I consistently rate this director higher or lower than IMDb. 
""")

    # Sidebar slider for minimum movies per director
    min_movies = st.sidebar.slider("Minimum movies per director for t-test", 2, 10, 5)

    # Editable t-test code
    ttest_code_director = f'''
from scipy.stats import ttest_rel
import numpy as np
import pandas as pd

df_ttest = IMDB_Ratings.merge(
    My_Ratings[['Movie ID','Your Rating']],
    on='Movie ID', how='inner'
)

results = []

for director, group in df_ttest.groupby('Director'):
    n = len(group)
    if n >= {min_movies}:
        differences = group['Your Rating'] - group['IMDb Rating']

        
        if differences.std() == 0:
            stat, pval = np.nan, np.nan
            interpretation = "All differences identical — t-test undefined"
        else:
            stat, pval = ttest_rel(group['Your Rating'], group['IMDb Rating'])
            if pval < 0.05:
                if n <= 2*{min_movies}:
                    interpretation = "Significant (p < 0.05) — small sample, interpret cautiously"
                else:
                    interpretation = "Significant (p < 0.05)"
            else:
                interpretation = "Not Significant"

        results.append({{
            "Director": director,
            "Num_Movies": n,
            "Mean_IMDb": group['IMDb Rating'].mean().round(2),
            "Mean_Mine": group['Your Rating'].mean().round(2),
            "t_statistic": round(stat, 3) if not np.isnan(stat) else np.nan,
            "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "Interpretation": interpretation
        }})


df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="p_value")
'''

    user_ttest_code_director = st.text_area("Python t-test per Director Code (editable)", ttest_code_director, height=650)

    if st.button("Run t-test Analysis", key="run_ttest_director6"):
        try:
            local_vars = {"IMDB_Ratings": IMDB_Ratings, "My_Ratings": My_Ratings}
            exec(user_ttest_code_director, {}, local_vars)

            if "df_results" in local_vars:
                st.dataframe(local_vars["df_results"], width="stretch", height=500)
            else:
                st.warning("No dataframe named 'df_results' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running t-test analysis: {e}")




# --- SCENARIO 6 ---

from textblob import TextBlob
import pandas as pd
import streamlit as st


if scenario == "6 – Review Analysis (Sentiment, Subjectivity)":
    st.header("6 – Review Analysis (Sentiment, Subjectivity)")

    # --- Short explanation ---
    st.markdown("""
    This scenario analyzes **audience reviews** of *Mother! (2017)*.  
    Each review is processed with natural language techniques to calculate:
    - **Sentiment** (negative to positive tone, -1 → +1)  
    - **Subjectivity** (objective → opinionated, 0 → 1)  
    The results include a summary table, aggregate metrics, and sample snippets.  
    """)

    # --- All reviews stored in a multi-line string ---
    reviews_text = """
Religious allegories abound but really it's just pretentious nonsense
Now I'm not one to disparage the director, I liked Requiem for a Dream and loved Black Swan, but this is a stinker and just simply boring. It's all just packed full of cod biblical allegories spread thickly throughout which tries to twist between different types of horror genres, but leaved me unintrigued. 
Granted the settings, claustrophobic direction and acting are top notch but it shouldn't mask for what otherwise is a poor uninteresting movie. It unsettles and bores, way too much to care, and as the ending dragged on I was left increasingly frustrated as it refused to just shut up shop. 
It's totally split opinion from what I've seen so far, and you'll struggle to find anyone in the middle on this one. In fairness, some credit to the film studios for risking this effort in launching it into mainstream cinemas but without the director it would have rightfully languished on cable late night showings.
There's no point going anymore into this. I simply hated it, and that despite being a major admirer of offbeat horror and psychological movies, but this isn't in the same league as for example Raw or Get Out, which is a shame. I'd recommend you pass on this there are far better films out there to go watch.
Aronofsky's mother! will be hated by many, but loved by a precious few

Horrifying. Just.. horrifying. Aronofsky really got me with this one. Not only did he manage to grab me on an intellectual level, but also on an emotional one...
# (Include all remaining reviews here, each separated by an empty line)

Usually this is where I put my plot description but it's best that you go into Darren Aronofsky's latest knowing as little as possible. Lets just say that Jennifer Lawrence and Javier Bardem are living in a large house all alone when a surprise visit sets them off into madness.
It really shocks me that Paramount would try to push MOTHER! onto the masses. For starters, the majority of moviegoers today do not want to think and they certainly don't want to see a movie where everything isn't explained. In fact, most people need everything explained in the trailer before they'll even go see a movie. A movie like MOTHER! is something that never explains itself and it constantly keeps you guessing from one scene to the next. What's it about? It's really hard to say as every viewer is going to come away with something different. With all of that said, it's easy to see why the film bombed at the box office and why those who did see it gave it a F rating.
what I loved most about this movie is that the setting is just so perfect. You've got a large beautiful house out in the middle of nowhere and it's surrounded by beautiful grass and trees. From the very first scene we can just tell that something isn't quite right and Aronofsky puts us in this beautiful place with confusing surroundings. What makes the film so special is the fact that nothing is ever explained and with each new plot twist your brain just becomes more confused as to what's going on. We know something is happening and we know something bad is going to happen but you're constantly trying to guess what.
Of course, a movie like this wouldn't work without a terrific cast to pull it off. Lawrence turns in another terrific performance and I thought she as fabulous at showing how fractured this character was. We're often questioning her mental state and I thought Lawrence managed to make you feel for the character and go along with her confusion to everything that is happening. Bardem actually steals the show with his fiery performance and I really loved the rage and anger he brought to the film as well as another side that I won't spoil to prevent giving away aspects of the plot. Both Ed Harris and Michelle Pfeiffer were also terrific but, again, I'll hold off commenting more to prevent plot points.
The cinematography is terrific and on a technical level the film is quite flawless. The story is a very interesting one and one that keeps you guessing throughout. The performances just seal the deal. With that said, the film certainly goes downright insane at times and the ending is just one that will have you staggering out of the theater. I must say that I thought the finale went on a bit too long and that it would have worked better had it been edited down a bit. Still, MOTHER! is a film that I really loved and one that I really respected but at the same time I'm not sure who I'd recommend it to.

Went to the first matinée available locally and I am still thinking the picture over. Will definitely see this one again, if it hasn't left the theatre abruptly. I was certainly horrified by the film, which is a good thing, as I had assumed it was a horror picture. It is, of course, much more than that. Nonetheless, it is NOT The Conjuring or Get Out (both good films, for sure), so just be warned.
By now you are aware that the film has been controversial, also a good thing. Jennifer Lawrence does a fine job and her career is certainly not going to suffer for her performance. I am not exactly a JLaw "fan" (could live without the Hunger Games), although I will pay closer attention to her future performances, especially if she pulls off more roles like this one (really liked Winter's Bone, by the way). As I understand the Hollywood scene, it is a respectable personal decision to take on a challenging role in an avant garde picture, especially if you have already banked serious money from popular roles in blockbusters. Javier Bardem, Michelle Pfeiffer, and Ed Harris also do their respective parts justice--a well-acted film by A-listers, overall. Camera work and special effects are also impressive.
The story is genuinely disturbing in a Requiem for a Dream way, so don't go if you can't handle that sort of thing. Some of the violence is, indeed, OVER THE TOP. Seriously, not for the faint of heart. Aside from the biblical allegory stuff, I found the character portrayals creepy as hell in a (sur?)realistic David Lynch-esque way. Hell is other people!
I applaud Mr. Aronofsky for keeping his vision intact all the way to the big screen. For reference, I just don't need any more movies based on superheros, comic books (except The Tenth or Gen 13), children's cartoons, vampires fighting werewolves, or horror stick about unfriending weirdos on facebook. 
You will have to make up your own mind on this one, so please do just that. Even if you end up despising the film, try to remember that, to quote Rob Zombie, "Art's Not Safe."

A married couple live in an isolated country house. He is a celebrated poet, suffering from writer's block, and she is working on renovating the house. Then a guest, a stranger, suddenly drops in and nothing will ever be the same again. 
Written and directed by Darren Aronofsky who gave us masterpieces like 'The Wrestler' and 'Requiem for a Dream', as well as the excellent 'Black Swan'. The fact that he wrote and directed this was the only reason I watched it, hoping that he was back to the form of those movies as his previous movie was the craptacular-beyond-belief 'Noah'.
Unfortunately, no, he isn't, though initially there was a glimmer of hope. The movie started interestingly enough, with some decent character development and some interesting themes. However, from the outset it was slow, plus there were signs this wasn't going to be a character-based drama but something symbolic, and pretentious.
Plus it was annoying. The only likeable character was Jennifer Lawrence's. Javier Bardem's was selfish and egotistical and every single other character was incredibly irritating. 
Still, I was hoping this would all develop into something interesting and profound. Wrong again. It develops into anarchy and some sort of badly-thought-out horror movie, and the annoyance factor gets pushed to the max. Of course, it's all meant to be symbolic, but figuring out everything would require you to think about the movie, and do so you would have had to have concentrated all through the tidal wave of excrement that was the movie.
Pretentious and annoying, and evidence that, sadly, Darren Aronofsky has run out of ideas.

I have been going to the movies for 45 years. This is, hands down, the worst movie I have ever seen. I mean, I hated this movie. Plan 9 From Outer Space and The Room were at least entertaining. This is like being locked in a cell with a stoned college student who can't shut up and thinks that every opinion they have, is the final word on a subject for 2 hours. Jennifer Lawrence should stick to roles that require her to paint herself blue or shoot arrows. Darren Aronofsky wants to be Luis Buñuel but he's closer to Uwe Boll. He cites The Exterminating Angel as the inspiration for Mother! I agree, in the sense that I did feel like one of the dinner guests who can't leave in Buñuel's classic during the course of watching Mother after paying 13 bucks to see this pretentious, heavy handed waste of time. Do yourself a favor, don't go see this movie, you won't get the 2 hours of your life back if you do. When it shows up on The Movie Channel playing at 3 in the morning in a couple of months, don't even set your DVR to record it. There are infomercials about gardening tools on at the same time, that are much more entertaining to watch that this.

I thought this was worth its salt even though it did tend towards cliché as it wore on. The disappointing aspect of this film is that Jennifer Lawrence somehow portrays an ego that is beyond the character. It's a kind of "you know that I know I'm only acting this and the real movie is me" that seems to have perpetuated in every film she had made since Silver linings Playbook, bar X-Men (when she was covered in paint and having to "live in" the previous "humble" shoes of Rebecca Romijn) and American Hustle (where she was greedy White Trash). She needs a director who can "humble her down", in the same way Eastwood did for Jolie in Changeling, so that her ego is less of a distraction for her acting.

Where to start - I've literally just finished watching this and spent the last hour questioning if I had been transported to another universe.
This movie had all the potential to be something great, from the cast to the secluded creepy setting - but no, we got almost 2 hours of the what could only be described as one of those brainwashing experimental videos where you have no idea what's going on.
If you like movies which make you feel uneasy, and make you think you're going mad then this might be the movies for you. Otherwise, I wouldn't bother.
Edit:So after having a day or so to ponder over the meaning of this movie - I've changed my rating and edited my review based on what I have come to know.
I can now say that once you understand the characters and why they represent, you'll understand the meaning and it could change your entire view of this movie.

If you just watch the pictures, the are confusing, disturbing, chaotic but their actual meaning is the representation of a person giving everything for someone else and still is not enough and everything she gives is topped by something else, every precious moment is taken to be shared.
I don't want to go into details of the meaning of every scene and why the movie develops the way it does, it would probably take a book to describe.
Every scene is symbolic, really well done.

This is a phenomenal film, full of details, full of symbolism and references to the Bible, to man's relationship to the mother Earth, to the state of consciousness in which we find ourselves as humanity. The atmosphere is superb and the actors are exceptional. I think this is my favorite Darren Aronofsky's movie. And I'm a bit sad because people in their reviews give it worst grade just because the movie does not have "enough action", just because is slow or not fun enough, just because the super heroes in it doesn't shoot and fly in the air and perform all kinds of spectacular things. I think, there is internet, there is IMDB and before you go to see a movie (especially if it is a an intellectually demanding like this one) read about it and see, decide if you are interested in such a thing. If it is not for you, there will always be something else to your liking out there An extraordinary movie in every aspect.

I have never watch a movie about it :).Dont try to learn something about the film before watching. Actually, it tells very good the whole life, and theatral aspect was wonderful in the movie. I strongly suggest that movie but, first, you have to leave your superstitions and prejudice . Just watch as an art and movie. But this movie, is not for superhero lovers and childs.
    """

    # --- Full editable code block for this scenario ---
    review_code = f'''
from textblob import TextBlob
import pandas as pd

# --- Reviews input ---
reviews_text = """{reviews_text.strip()}"""

# Convert multi-line text to list of reviews
reviews = [r.strip() for r in reviews_text.split("\\n\\n") if r.strip()]

review_records = []
review_counter = 1
for review in reviews:
    words = review.split()
    if len(words) < 5:
        continue  

    tb = TextBlob(review)
    sentiment = tb.sentiment.polarity
    subjectivity = tb.sentiment.subjectivity

    snippet = review[:500].strip()
    if not snippet:
        continue

    review_records.append({{
        "ReviewID": review_counter,
        "Words": len(words),
        "Sentiment": round(sentiment, 3),
        "Subjectivity": round(subjectivity, 3),
        "Snippet": snippet + ("..." if len(review) > 500 else "")
    }})
    review_counter += 1

df_reviews = pd.DataFrame(review_records)
df_reviews.reset_index(drop=True, inplace=True)
df_reviews['ReviewID'] = df_reviews.index + 1
'''

    # --- Editable code input (like Scenario 5) ---
    user_review_code = st.text_area(
        "Python Review Sentiment Code (editable)",
        review_code,
        height=700
    )

    # --- Run button ---
    if st.button("Run Sentiment Analysis", key="run_sentiment6"):
        try:
            local_vars = {}
            exec(user_review_code, {}, local_vars)

            if "df_reviews" in local_vars:
                df_reviews = local_vars["df_reviews"]

                st.subheader("Reviews Overview")
                st.dataframe(df_reviews, width="stretch", height=400)

                st.subheader("Aggregate Insights")
                st.write(f"**Average sentiment:** {df_reviews['Sentiment'].mean():.3f}")
                st.write(f"**Average subjectivity:** {df_reviews['Subjectivity'].mean():.3f}")

                st.markdown("""
                **What these metrics mean:**
                - **Sentiment**: ranges from -1 (negative) to +1 (positive).  
                - **Subjectivity**: ranges from 0 (objective) to 1 (subjective/opinionated).  
                - **Snippet**: first 500 characters of the review.  
                """)

                st.markdown("""
                ---
                **How TextBlob works (in simple terms):**  
                - TextBlob uses a built-in **lexicon** (a dictionary of words) where each word has a sentiment score  
                  (e.g., *"great"* → +0.8, *"boring"* → -0.6).  
                - When it processes a review, it breaks the text into words and phrases, looks them up in the lexicon,  
                  and then averages the scores to estimate overall **sentiment**.  
                - For **subjectivity**, it checks how opinion-based the words are. Words like *"amazing"* or *"terrible"*  
                  are subjective, while factual words like *"movie length"* are objective.  
                - The result is a quick, automated way of measuring tone and bias without needing manual labeling.  

                ⚠️ **Note:** TextBlob is rule-based and doesn’t “understand” context deeply.  
                For example, sarcasm or irony might confuse it (e.g., *"What a masterpiece..."* said negatively will still be read as **positive**). 
                """)

                # --- Full reviews ---
                st.markdown("---")
                with st.expander("Full Reviews (click to expand)"):
                    for r in local_vars["reviews"]:
                        if len(r.split()) >= 5:
                            st.markdown(f"<div style='color:gray; padding:5px;'>{r}</div>", unsafe_allow_html=True)
            else:
                st.warning("No dataframe named 'df_reviews' was produced. Please check your code.")

        except Exception as e:
            st.error(f"Error running sentiment analysis: {e}")




# --- Scenario 11---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scenario 11 ---
if scenario == "11 – Model Evaluation (Feature Importance)":
    st.header("11 – Model Evaluation: Feature Importance")

    st.write("""
    We analyze which features matter most for predicting **my movie ratings** using a Random Forest model.  

    **Feature Importance:**  
    - Higher score → stronger influence on predictions.  
    - Lower score → weaker influence.  

    *(Requires a trained model from Scenario 9.)*
    """)

    # --- Retrain model if not in session ---
    if 'model' not in st.session_state:
        st.warning("Model not found. Retrain here.")
        if st.button("Run Scenario 9 ( Predit My Ratings ) Training Now"):
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline

            df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
            train_df = df_ml[df_ml['Your Rating'].notna()]

            # Treat Year as categorical
            categorical_features = ['Genre', 'Director', 'Year']
            numerical_features = ['IMDb Rating', 'Num Votes']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                    ('num', 'passthrough', numerical_features)
                ]
            )

            model = Pipeline([
                ('prep', preprocessor),
                ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            X_train = train_df[categorical_features + numerical_features]
            y_train = train_df['Your Rating']
            model.fit(X_train, y_train)

            st.session_state['model'] = model
            st.success("Model trained successfully! You can now view feature importance.")

    # --- Show feature importance if model exists ---
    if 'model' in st.session_state:
        trained_model = st.session_state['model']
        rf = trained_model.named_steps['reg']
        preproc = trained_model.named_steps['prep']

        # Feature names
        cat_features = preproc.named_transformers_['cat'].get_feature_names_out(['Genre','Director','Year'])
        numerical_features = ['IMDb Rating', 'Num Votes']
        all_features = np.concatenate([cat_features, numerical_features])
        importances = rf.feature_importances_

        fi_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # --- Top N individual features ---
        top_n = 20
        fi_top = fi_df.head(top_n)

        st.subheader(f"Top {top_n} Feature Importances")
        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=fi_top, palette='viridis')
        plt.title("Top Feature Importances")
        plt.tight_layout()
        st.pyplot(plt)

        # --- Automatic explanation for top Director ---
        director_features = fi_df[fi_df['Feature'].str.startswith('Director')]
        if not director_features.empty:
            top_director = director_features.sort_values(by='Importance', ascending=False).iloc[0]
            feature = top_director['Feature']
            importance = top_director['Importance']
            director_name = feature.replace('Director_','')

            st.write("**Specific Insight:**")
            st.write(f"""
            **{feature}** (importance {importance:.3f}):

            **What the feature represents:**  
            For `{feature}`, the model uses a one-hot encoded feature to distinguish {director_name} movies from all other movies.  
            In other words, whether a movie is directed by {director_name} significantly affects the model's predictions.  
            My rating behavior for {director_name} movies is distinct from my average ratings, and therefore the model relies on this pattern to make predictions.
            """)

        # --- Aggregated by category ---
        fi_df['Category'] = fi_df['Feature'].str.split('_').str[0]
        agg_df = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)

        st.subheader("Feature Importance by Category")
        plt.figure(figsize=(8, 4))
        sns.barplot(x=agg_df.values, y=agg_df.index, palette='magma')
        plt.title("Aggregated Importances")
        plt.tight_layout()
        st.pyplot(plt)

        # --- Summary explanation (only shows when model exists) ---
        st.write("""
        **Interpretation:**  
        Aggregating features by category shows the bigger picture of what drives my ratings. If `Director` is high, it means certain directors consistently shape how I score movies.  

        **Why this matters for me:**  
        I bring my own personal insight into how I feel about directors — their style, storytelling, or reputation.  
        The model simply quantifies what I already sense: that my ratings often rise or fall depending on who directed the film.  

        **Why movies are my choice for all scenarios:**  
        Movies are personal. Unlike abstract datasets, I have close experience with films and directors.  
        This makes the insights richer — I can interpret the model’s patterns through my own perspective as a movie fan.  
        That connection is why I chose film as the subject matter to explore these scenarios.
        """)





# --- Scenario 12: Feature Hypothesis Testing ---
if scenario == "12 – Feature Hypothesis Testing":
    st.header("12 – Feature Hypothesis Testing & Predictions")

    st.markdown("""
    Select features to test if they **improve model predictions** for your ratings.
    After running, you'll see:
    1. Statistical test results
    2. Detailed explanation of feature impact
    3. Example predicted ratings for unseen movies with reasoning
    4. Annotated RMSE comparison with interpretation
    """)

    # --- Feature selection ---
    candidate_features = ['Director', 'Genre', 'Year', 'Num Votes', 'IMDb Rating']
    selected_features = st.multiselect(
        "Select feature(s) to test", 
        candidate_features, 
        default=['Director'] 

    )
    if 'scenario10_result' not in st.session_state:
        st.session_state['scenario10_result'] = None

    if st.button("Run Test & Show Predictions"):
        import numpy as np
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor
        from scipy.stats import ttest_rel
        import matplotlib.pyplot as plt

        # --- Prepare training data ---
        df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
        train_df = df_ml[df_ml['Your Rating'].notna()]
        y = train_df['Your Rating']  # Target variable: your ratings

        # --- Baseline model (numeric only) ---
        baseline_features = ['Num Votes','IMDb Rating']
        X_base = train_df[baseline_features]
        model_base = RandomForestRegressor(n_estimators=100, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_base = -cross_val_score(model_base, X_base, y, cv=cv, scoring='neg_root_mean_squared_error')

        # --- Feature-added model ---
        categorical_features = [f for f in selected_features if f in ['Director','Genre','Year']]
        numerical_features = [f for f in selected_features if f in ['Num Votes','IMDb Rating']]
        features_to_use = categorical_features + numerical_features

        if features_to_use:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                    ('num', 'passthrough', numerical_features)
                ]
            )
            X_test = train_df[features_to_use]
            model_test = Pipeline([
                ('prep', preprocessor),
                ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            scores_test = -cross_val_score(model_test, X_test, y, cv=cv, scoring='neg_root_mean_squared_error')

            # --- Paired t-test ---
            t_stat, p_val = ttest_rel(scores_base, scores_test)

            # --- Retrain for predictions ---
            model_test.fit(X_test, y)

            # --- Predict all unseen movies ---
            unseen_df = df_ml[df_ml['Your Rating'].isna()]
            if not unseen_df.empty:
                X_unseen = unseen_df[features_to_use]
                preds = model_test.predict(X_unseen)
                pred_df = unseen_df[['Movie ID','Title','Year','IMDb Rating']].copy()
                pred_df['Predicted Rating'] = np.round(preds,1)

                # --- Features considered per movie ---
                features_list = []
                for idx, row in unseen_df.iterrows():
                    feature_values = {f: row.get(f,'?') for f in selected_features}
                    features_list.append(", ".join([f"{k}={v}" for k,v in feature_values.items()]))
                pred_df['Features Considered'] = features_list

                # --- Sort by Year descending ---
                pred_df = pred_df.sort_values(by='Year', ascending=False)
            else:
                pred_df = pd.DataFrame()

            # --- RMSE summary & automatic interpretation ---
            rmse_base_mean = np.mean(scores_base)
            rmse_test_mean = np.mean(scores_test)
            rmse_diff = rmse_base_mean - rmse_test_mean

            if p_val < 0.05:
                if rmse_diff > 0:
                    stat_explanation = (
                        f"✅ Adding {', '.join(selected_features)} improved the model.\n"
                        f"- Average RMSE decreased from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                        f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → statistically significant improvement."
                    )
                else:
                    stat_explanation = (
                        f"❌ Adding {', '.join(selected_features)} worsened the model.\n"
                        f"- Average RMSE increased from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                        f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → statistically significant deterioration."
                    )
            else:
                stat_explanation = (
                    f"ℹ️ Adding {', '.join(selected_features)} did NOT meaningfully change the model.\n"
                    f"- Average RMSE changed from {rmse_base_mean:.2f} → {rmse_test_mean:.2f}.\n"
                    f"- t-value = {t_stat:.3f}, p-value = {p_val:.4f} → no statistically significant difference."
                )

            st.session_state['scenario10_result'] = {
                't_stat': t_stat,
                'p_val': p_val,
                'stat_explanation': stat_explanation,
                'predictions': pred_df,
                'scores_base': scores_base,
                'scores_test': scores_test,
                'selected_features': selected_features
            }

    # --- Display results ---
    if st.session_state['scenario10_result']:
        result = st.session_state['scenario10_result']

        # --- Predictions table ---
        st.write("### Predictions Table (All Unrated Movies)")
        if not result['predictions'].empty:
            st.dataframe(result['predictions'])

            # --- Statistical significance explanation ---
            st.write("### Statistical Significance of Improvement")
            st.info(result['stat_explanation'])

            # --- Explanation of predicted rating changes ---
            st.write("### Why Predicted Ratings Change")
            st.markdown(f"""
            The predicted ratings change when you modify the selected features because the model learns patterns from your past ratings.  

            **Current features used:** {', '.join(result['selected_features'])}  

            - **Director:** captures your preferences for specific directors.  
            - **Genre:** captures your preferences for specific types of films.  
            - **Year:** considers how your ratings vary over time.  
            - **IMDb Rating & Num Votes:** reflect general popularity and consensus quality.  

            When features are added or removed, the model adjusts the predictions based on the patterns it learned from your historical ratings.
            """)
        else:
            st.warning("No unseen movies available for prediction.")

        # --- Annotated RMSE boxplot ---
        plt.figure(figsize=(7,4))
        rmse_base_mean = np.mean(result['scores_base'])
        rmse_test_mean = np.mean(result['scores_test'])
        plt.boxplot([result['scores_base'], result['scores_test']], labels=['Baseline', 'With Feature(s)'])
        plt.ylabel("RMSE")
        plt.title("Cross-Validated RMSE Comparison")
        plt.text(1, rmse_base_mean + 0.02, f"{rmse_base_mean:.2f}", ha='center', color='blue')
        plt.text(2, rmse_test_mean + 0.02, f"{rmse_test_mean:.2f}", ha='center', color='green')
        st.pyplot(plt)

        # --- RMSE interpretation ---
        st.write("""
        **Interpretation of RMSE Boxplot and Model Comparison**

        **1: Baseline Model (Numeric Features Only)**
        - Uses only `IMDb Rating` and `Num Votes`.
        - Captures general popularity and average rating information.
        - Higher RMSE → predictions deviate more from your actual ratings.
        - Wide spread → inconsistent performance across movies.

        **2: Feature-Added Model (Selected Features Included)**
        - Includes additional features such as `Director`, `Genre`, `Year`.
        - Provides context about your personal preferences.
        - Lower RMSE → predictions closer to your actual ratings.
        - Tighter spread → more consistent performance.

        **Takeaway**
        - RMSE decrease + p-value < 0.05 → features improve model accuracy.
        - RMSE increase + p-value < 0.05 → features worsen predictions.
        - p-value ≥ 0.05 → no significant change.
        """)



# --- Scenario 8: Graph-Based Movie Relationships ---
if scenario == "8 – Graph Based Movie Relationships":
    st.header("8 – Graph-Based Movie Relationships")
    st.write("""
    This scenario models the dataset as a **graph**:
    - **Nodes**: Movies, Directors, Genres  
    - **Edges**: Relationships between them.  
    
    Use the filters below to narrow down by **Year**, **Director(s)**, and **Genre**, then run the graph builder.
    """)

    # --- Filters ---
    directors = sorted(IMDB_Ratings["Director"].dropna().unique()) if not IMDB_Ratings.empty else []
    genres = []
    if "Genre" in IMDB_Ratings.columns:
        genres = sorted({g.strip() for sublist in IMDB_Ratings["Genre"].dropna().str.split(",") for g in sublist})
    years = sorted(IMDB_Ratings["Year"].dropna().unique().astype(int).tolist()) if "Year" in IMDB_Ratings.columns else []

    # --- Default Selections ---
    default_year = "All"
    default_genre = "Drama"
    default_director = [d for d in ["Alfred Hitchcock", "Stanley Kubrick", "Francis Ford Coppola"] if d in directors]

    selected_year = st.selectbox(
        "Filter by Year",
        ["All"] + [str(y) for y in years],
        index=(["All"] + [str(y) for y in years]).index(default_year)
    )
    selected_directors = st.multiselect("Filter by Director(s)", directors, default=default_director)
    selected_genre = st.selectbox(
        "Filter by Genre",
        ["All"] + genres,
        index=(["All"] + genres).index(default_genre) if default_genre in genres else 0
    )

    # --- Editable code template ---
    graph_code = '''
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

df_graph = IMDB_Ratings.copy()

if selected_year != "All":
    df_graph = df_graph[df_graph["Year"] == int(selected_year)]
if selected_directors:
    df_graph = df_graph[df_graph["Director"].isin(selected_directors)]
if selected_genre != "All":
    df_graph = df_graph[df_graph["Genre"].str.contains(selected_genre, na=False)]

G = nx.Graph()
for _, row in df_graph.iterrows():
    movie = row.get("Title")
    director = row.get("Director")
    genre = row.get("Genre")

    if pd.notna(movie):
        G.add_node(movie, type="movie")
    if pd.notna(director):
        G.add_node(director, type="director")
        G.add_edge(director, movie)
    if pd.notna(genre):
        for g in str(genre).split(", "):
            G.add_node(g, type="genre")
            G.add_edge(movie, g)

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3, iterations=25)
color_map = []
for node, data in G.nodes(data=True):
    if data["type"] == "movie":
        color_map.append("skyblue")
    elif data["type"] == "director":
        color_map.append("lightgreen")
    else:
        color_map.append("salmon")

nx.draw(G, pos, with_labels=True, node_size=800, node_color=color_map, font_size=8, edge_color="gray", ax=ax)
st.pyplot(fig)
st.write(f"Graph built with **{len(G.nodes)} nodes** and **{len(G.edges)} edges**.")
'''

    user_graph_code = st.text_area("Python Graph Code (editable)", graph_code, height=600)

    if st.button("Run Graph Analysis", key="run_graph11"):
        try:
            local_vars = {
                "IMDB_Ratings": IMDB_Ratings,
                "selected_year": selected_year,
                "selected_directors": selected_directors,
                "selected_genre": selected_genre,
                "st": st,
                "pd": pd
            }
            exec(user_graph_code, {}, local_vars)

            # --- Clean Explanation ---
            st.markdown("""
### Understanding the Graph

**Nodes:**  
- Movies  
- Directors  
- Genres  

**Edges:**  
- Director → Movie  
- Movie → Genre  

**Why this matters:**  
- Identify which directors specialize in which genres  
- Discover genre clusters with many movies  
- Explore connections between directors through shared genres or collaborations  

This visualization helps you explore the movie dataset’s structure and uncover patterns and relationships clearly.
""")
        except Exception as e:
            st.error(f"Error running Graph Analysis code: {e}")




# --- Scenario 7 Poster Analysis ---
if scenario == "7 – Poster Image Analysis (OMDb API)":
    st.header("7 – Poster Image & Mood Analysis")
    st.markdown("""
    Select a movie, then click **Fetch Poster & Analyze** to display the poster, 
    dominant colors, and an easy-to-understand mood analysis.
    """)

    import requests
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans

    # --- Editable code block ---
    poster_code = '''

imdb_id = IMDB_Ratings.loc[IMDB_Ratings['Title'] == selected_film, 'Movie ID'].values[0]


url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
response = requests.get(url).json()
poster_url = response.get('Poster')

if poster_url and poster_url != "N/A":
    img = Image.open(requests.get(poster_url, stream=True).raw).convert("RGB")
    img_small = img.resize((150, 150))
    img_array = np.array(img_small).reshape(-1, 3)

    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(img_array)
    dominant_colors = kmeans.cluster_centers_

    
    st.image(poster_url, width=300)

    
    st.write("🎨 Dominant Colors:")
    cols = st.columns(len(dominant_colors))
    for idx, color in enumerate(dominant_colors.astype(int)):
        hex_color = '#%02x%02x%02x' % tuple(color)
        cols[idx].markdown(
            "<div style='width:60px; height:60px; background:{}; border-radius:8px; border:1px solid #000'></div>".format(hex_color),
            unsafe_allow_html=True
        )

    
    brightness = np.mean(img_array)
    if brightness < 100:
        mood = "dark and moody"
        cluster_name = "Cluster 0 – Thriller / Horror style"
        mood_tag = "🌑 Dark Thriller vibes"
    elif brightness < 170:
        mood = "balanced"
        cluster_name = "Cluster 1 – Drama / Realistic style"
        mood_tag = "🎭 Dramatic tone"
    else:
        mood = "bright and vivid"
        cluster_name = "Cluster 2 – Comedy / Family style"
        mood_tag = "😂 Lighthearted & Fun"

    
    st.success("🎬 Poster assigned to: **{}**".format(cluster_name))
    st.info("The poster looks **{}**, suggesting **{}**.\\n\\n👉 Mood tag: **{}**".format(
        mood, cluster_name.split('–')[1].strip(), mood_tag
    ))
else:
    st.warning("Poster not found.")
'''

    # --- Editable text area ---
    user_poster_code = st.text_area("Python Poster Analysis Code (editable)", poster_code, height=650)

    # --- Hidden API key ---
    OMDB_API_KEY = "cbbdb8f8"  # Keep this hidden in production

    # --- Movie selection ---
    film_list = IMDB_Ratings['Title'].dropna().unique().tolist()
    selected_film = st.selectbox("Select a movie to analyze poster:", film_list)

    # --- Run button ---
    if st.button("Fetch Poster & Analyze"):
        try:
            local_vars = {
                "IMDB_Ratings": IMDB_Ratings,
                "selected_film": selected_film,
                "OMDB_API_KEY": OMDB_API_KEY,
                "st": st,
                "np": np,
                "KMeans": KMeans,
                "requests": requests,
                "Image": Image
            }
            exec(user_poster_code, {}, local_vars)
        except Exception as e:
            st.error(f"Error running poster analysis: {e}")




# --- Scenario 13: Deep Learning Semantic Genre Analysis (Dynamic) ---
if scenario == "13 – Semantic Genre & Recommendations (Deep Learning / NLP)":
    st.header("13 – Semantic Genre & Recommendations (Deep Learning / NLP)")
    st.markdown("""
    This scenario uses **sentence embeddings** to determine the main genre of films by analyzing the plot.  
    The table shows:
    - Film title
    - Plot snippet
    - OMDb listed genres
    - Embedding similarity with each genre
    - Predicted main genre
    """)

    # --- Dynamic list of directors from the dataset ---
    directors_list = IMDB_Ratings["Director"].dropna().unique().tolist()
    directors_list.sort()
    selected_director = st.selectbox("Choose a director:", directors_list)

    # --- Hidden OMDb API key ---
    OMDB_API_KEY = "72466310"  # keep this private

    # --- Cached function to fetch OMDb data ---
    @st.cache_data(show_spinner=False)
    def fetch_movie_data(title):
        import requests
        response = requests.get(f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}&plot=full").json()
        plot = response.get("Plot") or "Plot missing"
        genres = response.get("Genre").split(", ") if response.get("Genre") else ["Unknown"]
        return {"Title": response.get("Title") or title, "Plot": plot, "Genre": genres}

    # --- Run button ---
    if st.button("Run Deep Learning Genre Analysis"):
        # Get all movies for the selected director dynamically
        movies = IMDB_Ratings[IMDB_Ratings["Director"] == selected_director]["Title"].dropna().tolist()

        if not movies:
            st.warning(f"No movies found for {selected_director}")
        else:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")

            results = []

            for title in movies:
                movie_data = fetch_movie_data(title)
                plot = movie_data["Plot"]
                genres = movie_data["Genre"]

                # Compute plot embedding
                plot_embedding = model.encode(plot, convert_to_tensor=True)

                # Compute similarity for each genre
                similarities = {}
                for g in genres:
                    g_embedding = model.encode(g, convert_to_tensor=True)
                    sim = util.cos_sim(plot_embedding, g_embedding).item()
                    similarities[g] = round(sim, 3)

                # Main genre = highest similarity
                main_genre = max(similarities, key=similarities.get) if similarities else "Unknown"

                results.append({
                    "Film": movie_data["Title"],
                    "OMDb Genres": ", ".join(genres),
                    "Embedding Similarity": similarities,
                    "Main Genre (Predicted)": main_genre,
                    "Plot": plot[:200] + "..." if len(plot) > 200 else plot
                })

            df_results = pd.DataFrame(results)
            st.success(f"Analysis complete for {selected_director} ✅")
            st.dataframe(df_results, use_container_width=True)

            st.markdown("""
            **Explanation:**  
            - Each **plot** is converted into a vector (embedding).  
            - Each **genre** is also converted into a vector.  
            - **Cosine similarity** measures semantic closeness (0 to 1).  
            - The genre with the highest similarity is predicted as the **main genre**.  
            - This helps when OMDb lists multiple genres, showing the most semantically relevant one.
            """)


# --- Scenario 14: Live Ratings Monitor + Supervised ML Predictions (English only) ---
if scenario == "14 – Live Ratings Monitor (MLOps + CI/CD + Monitoring)":
    st.header("14 – Live Ratings Monitor (MLOps + CI/CD + Monitoring)")

    st.markdown("""
**MLOps + CI/CD + Monitoring (Brief)**  

- **MLOps:** Automates data collection (live IMDb ratings), logs historical differences, and retrains ML models to predict future rating changes.  
- **CI/CD:** Modular code can be version-controlled; in a full setup, changes would trigger automated testing and deployment.  
- **Monitoring:** Tracks rating differences over time with timestamps, enabling detection of trends, anomalies, or shifts in popularity.

**Supervised Machine Learning:**  
The model uses my existing ratings (`My_Ratings`) as training data to learn patterns in how I rate movies.  
Given movie features (IMDb rating, genre, director, year, votes), the model predicts my rating for unseen films - Horror Films only.  
""")

    # --- OMDb API key ---
    OMDB_API_KEY = "e9476c0a"

    # --- Select top 250 films ---
    top250_films = IMDB_Ratings[
    IMDB_Ratings['Genre'].str.contains("Horror", case=False, na=False)
    ].sort_values(by="IMDb Rating", ascending=False).head(250)


    # --- Run Button ---
    if st.button("Run Live Ratings Check"):
        import requests
        from datetime import datetime
        import os
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        history_file = "live_ratings_history.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load previous history
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
        else:
            history_df = pd.DataFrame()

        results = []

        # --- Fetch live ratings from OMDb using Movie ID (IMDb ID) ---
        for _, row in top250_films.iterrows():
            movie_id = row["Movie ID"]
            static_rating = row["IMDb Rating"]

            try:
                url = f"http://www.omdbapi.com/?i={movie_id}&apikey={OMDB_API_KEY}"
                resp = requests.get(url).json()

                if resp.get("Response") == "True":
                    # Normalize languages: split, strip, lowercase
                    languages = [lang.strip().lower() for lang in resp.get("Language", "").split(",")]
                    live_rating = float(resp.get("imdbRating", 0)) if resp.get("imdbRating") else None

                   
                    if "english" not in languages:
                        continue
                else:
                    live_rating = None
                    languages = []
            except Exception:
                live_rating = None
                languages = []

            rating_diff = live_rating - static_rating if live_rating is not None else None

            results.append({
                "Title": row["Title"],
                "IMDb Rating (Static)": static_rating,
                "IMDb Rating (Live)": live_rating,
                "Rating Difference": rating_diff,
                "CheckedAt": timestamp,
                "Movie ID": movie_id,
                "Genre": row.get("Genre"),
                "Director": row.get("Director"),
                "Year": row.get("Year"),
                "Num Votes": row.get("Num Votes"),
                "Language": ", ".join([lang.capitalize() for lang in languages])
            })

        new_df = pd.DataFrame(results)

        # Only keep rows with non-zero rating differences if the column exists
        if not new_df.empty and "Rating Difference" in new_df.columns:
            new_df = new_df[new_df["Rating Difference"] != 0]
        else:
            new_df = pd.DataFrame()  # Ensure it’s still a DataFrame even if empty

        st.success("Live ratings check complete ✅")

        # --- Show sorted results by Rating Difference ---
        if not new_df.empty:
            st.subheader("📊 Current Run - Live Ratings Comparison")
            st.dataframe(
                new_df.sort_values(by="Rating Difference", ascending=False).reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.warning("No English-language films with rating changes found in this run.")

        # --- Supervised ML: Predict My Ratings for Movies with Changed Live Ratings ---
        df_ml = IMDB_Ratings.merge(My_Ratings[['Movie ID','Your Rating']], on='Movie ID', how='left')
        df_ml = df_ml.merge(new_df[['Movie ID','Rating Difference']], on='Movie ID', how='left')

        # Only predict for unseen movies from the current Horror subset with rating changes
        predict_df = df_ml[
        (df_ml['Movie ID'].isin(top250_films['Movie ID'])) &
        (df_ml['Rating Difference'].notna()) &
        (df_ml['Your Rating'].isna())
        ].copy()
        train_df = df_ml[df_ml['Your Rating'].notna()]

        categorical_features = ['Genre', 'Director']
        numerical_features = ['IMDb Rating', 'Num Votes', 'Year']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ]
        )

        model = Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        X_train = train_df[categorical_features + numerical_features]
        y_train = train_df['Your Rating']
        model.fit(X_train, y_train)

        X_pred = predict_df[categorical_features + numerical_features]
        predict_df['Predicted Rating'] = model.predict(X_pred)

        if not predict_df.empty:
            st.subheader("🤖 Predicted Ratings for Unseen Movies with Changed Ratings")
            st.dataframe(
                predict_df[['Title','IMDb Rating','Genre','Director','Rating Difference','Predicted Rating']]
                .sort_values(by='Predicted Rating', ascending=False)
                .reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("No new movies available for prediction this run.")

        # --- Explain how Python and packages make predictions ---
        st.markdown("""
**How the Predictions Work (Technical Explanation):**  

1. **Data Preparation**  
   - Features used: `Genre`, `Director` (categorical), `IMDb Rating`, `Num Votes`, `Year` (numerical).  
   - `My Rating` is the target variable for supervised learning.

2. **Feature Encoding with `ColumnTransformer` and `OneHotEncoder`**  
   - Categorical features are converted to **one-hot encoded vectors**.  
   - Numerical features are passed through unchanged.  

3. **Pipeline with `RandomForestRegressor`**  
   - Combines preprocessing and model training.  
   - Random forest is an **ensemble of decision trees**:  
     - Each tree predicts independently.  
     - The final prediction is the average across all trees.  
     - This reduces overfitting and improves accuracy.

4. **Training**  
   - Model learns patterns from movies I have rated (`Your Rating`).  

5. **Prediction**  
   - Model predicts ratings for movies I haven’t rated based on learned patterns.  

6. **Why this works**  
   - Handles non-linear relationships and feature interactions naturally.  
   - One-hot encoding allows categorical variables like directors and genres to be used.  
   - Random forests are robust to overfitting and can generalize well to unseen movies.
""")


# --- Scenario 9: Natural-Language Film Q&A Assistant (final version) ---

# --- Scenario 9: Natural-Language Film Q&A Assistant (final version, cleaned) ---

if scenario.startswith("9"):
    import streamlit as st
    import pandas as pd
    import textwrap
    import re

    st.subheader("🎬 9 – Natural-Language Film Q&A Assistant")

    st.markdown("""
This scenario allows you to ask **natural-language questions** about my personal film ratings.

- When asking about directors, include only the **director’s surname** (last name).  
- You can also filter by genre (e.g., comedy, horror, drama).  
- Words like **top/highest/best** or **lowest/worst/bottom** control sorting.
""")

    st.markdown("**Example questions you can ask:**")
    for q in [
        "Which Hitchcock films did I rate the highest?",
        "Top films by Spielberg?",
        "Which drama films did I rate the lowest?",
        "Show me films by Cameron"
    ]:
        st.write(f"- {q}")

    try:
        My_Ratings = pd.read_excel("myratings.xlsx")
        IMDB_Ratings = pd.read_excel("imdbratings.xlsx")
    except Exception as e:
        st.error(f"Error loading Excel files: {e}")
        My_Ratings = pd.DataFrame()
        IMDB_Ratings = pd.DataFrame()

    # --- Editable logic code (cleaned: no unused comments or stopwords) ---
    logic_code = textwrap.dedent(r"""
        question_lower = user_question.lower()
        filtered = My_Ratings.copy()
        question_tokens = set(re.findall(r"\b[\w']+\b", question_lower))

        genres = ["comedy", "horror", "action", "drama", "sci-fi", "thriller", "romance"]
        filtered_genre = False
        for g in genres:
            if g in question_tokens or g in question_lower:
                filtered = filtered[filtered['Genre'].str.lower().str.contains(g, na=False)]
                filtered_genre = True
                break

        all_directors = My_Ratings['Director'].dropna().unique()
        matches = []

        for d in all_directors:
            last_name = d.split()[-1].lower()
            if re.search(r'\b' + re.escape(last_name) + r'\b', question_lower):
                matches.append(d)

        if matches:
            filtered = filtered[filtered['Director'].str.lower().isin([m.lower() for m in matches])]
        elif not filtered_genre:
            filtered = filtered.iloc[0:0]

        sort_col = "IMDb Rating" if "imdb" in question_lower else "Your Rating"
        if any(w in question_tokens for w in ["highest", "top", "best"]):
            ascending = False
        elif any(w in question_tokens for w in ["lowest", "worst", "bottom"]):
            ascending = True
        else:
            ascending = False
    """)

    st.markdown("#### 🔧 Filtering and Sorting Logic (editable)")
    editable_code = st.text_area("Modify logic if needed:", logic_code, height=400)

    user_question = st.text_input(
        "🎥 Ask a question:",
        placeholder="Which comedy films did I rate the highest?"
    )

    if user_question and not My_Ratings.empty:
        exec_ns = {"My_Ratings": My_Ratings, "user_question": user_question, "re": re}
        try:
            exec(editable_code, exec_ns)
        except Exception as e:
            st.error(f"Error running logic: {e}")
            exec_ns.setdefault("filtered", My_Ratings.copy())
            exec_ns.setdefault("sort_col", "Your Rating")
            exec_ns.setdefault("ascending", False)

        filtered = exec_ns.get("filtered", My_Ratings.copy())
        sort_col = exec_ns.get("sort_col", "Your Rating")
        ascending = exec_ns.get("ascending", False)

        if not filtered.empty:
            filtered_sorted = filtered.sort_values(by=sort_col, ascending=ascending)
            st.dataframe(filtered_sorted)
        else:
            st.info("No matching films found. Try a different director surname or genre keyword.")

if scenario != "15 – Psycho 1960 Film (Trained AI Model)":

    st.write("---")
    st.write("### IMDb Ratings Table")
    if not IMDB_Ratings.empty:
        st.dataframe(IMDB_Ratings, width="stretch", height=400)
    else:
        st.warning("IMDb Ratings table is empty or failed to load.")

    st.write("### My Ratings Table")
    if not My_Ratings.empty:
        My_Ratings['Year_Sort'] = pd.to_numeric(My_Ratings['Year'], errors='coerce')
        My_Ratings_sorted = My_Ratings.sort_values(by="Year_Sort", ascending=False)
        display_ratings = My_Ratings_sorted.rename(columns={"Your Rating": "My Ratings"})
        display_ratings = display_ratings.drop(columns=['Year_Sort'])
        st.dataframe(display_ratings, width="stretch", height=400)
    else:
        st.warning("My Ratings table is empty or failed to load.")

# --- Scenario 15: Psycho 1960 AI Quiz only ---
if scenario == "15 – Psycho 1960 Film (Trained AI Model)":
    
    from movie_quiz import ask_psycho_question  # only function, no widgets in movie_quiz.py

    st.header("🎬 Psycho 1960 - Trained AI Model")

    st.markdown("""
    This AI model was **trained locally** using TinyLlama 1.1B Chat + LoRA.
    Ask any question about the 1960 film *Psycho* and get AI-generated answers.
    """)

    question = st.text_input("Ask a question about the 1960 film Psycho:")
    if question:
        try:
            answer = ask_psycho_question(question)
            st.success(f"🎬 Answer: {answer}")
        except Exception as e:
            st.error(f"⚠️ Failed to get answer: {e}")
