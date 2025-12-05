import json
import os
import logging
import sys
import re
import glob
import pandas as pd
import dotenv
import concurrent.futures
import threading
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Any

# LangChain imports (matching dinsql.py)
from langchain_community.utilities import SQLDatabase
# Removed LangChain LLM imports
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )

from src.llm.client import get_competition

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Locks for thread safety
file_lock = threading.Lock()
print_lock = threading.Lock()
eval_lock = threading.Lock()

# ==========================================
# DIN-SQL Prompts & Helpers
# ==========================================

SYSTEM_SCHEMA_LINKING_TEMPLATE = """
You are an agent designed to find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.
Hint helps you to fine the correct schema_links.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie
#
Q: Which year has the least number of movies that was released and what is the title of the movie in that year that has the highest number of rating score of 1?
Hint: least number of movies refers to MIN(movie_release_year); highest rating score refers to MAX(SUM(movie_id) where rating_score = '1')
A: Let’s think step by step. In the question , we are asked:
"Which year" so we need column = [movies.movie_release_year]
"number of movies" so we need column = [movies.movie_id]
"title of the movie" so we need column = [movies.movie_title]
"rating score" so we need column = [ratings.rating_score]
Hint also refers to the columns = [movies.movie_release_year, movies.movie_id, ratings.rating_score]
Based on the columns and tables, we need these Foreign_keys = [movies.movie_id = ratings.movie_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1]. So the Schema_links are:
Schema_links: [movies.movie_release_year, movies.movie_title, ratings.rating_score, movies.movie_id=ratings.movie_id, 1]

Schema of the database with sample rows:
#
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)

/*
3 rows from lists table:
user_id list_id list_title      list_movie_number       list_update_timestamp_utc       list_creation_timestamp_utc     list_followers  list_url        list_commentslist_description list_cover_image_url    list_first_image_url    list_second_image_url   list_third_image_url
88260493        1       Films that made your kid sister cry     5       2019-01-24 19:16:18     2009-11-11 00:02:21     5       http://mubi.com/lists/films-that-made-your-kid-sister-cry     3       <p>Don’t be such a baby!!</p>
<p><strong>bold</strong></p>    https://assets.mubicdn.net/images/film/3822/image-w1280.jpg?1445914994  https://assets.mubicdn.net/images/film/3822/image-w320.jpg?1445914994 https://assets.mubicdn.net/images/film/506/image-w320.jpg?1543838422    https://assets.mubicdn.net/images/film/485/image-w320.jpg?1575331204
45204418        2       Headscratchers  3       2018-12-03 15:12:20     2009-11-11 00:05:11     1       http://mubi.com/lists/headscratchers    2       <p>Films that need at least two viewings to really make sense.</p>
<p>Or at least… they did for <em>       https://assets.mubicdn.net/images/film/4343/image-w1280.jpg?1583331932  https://assets.mubicdn.net/images/film/4343/image-w320.jpg?1583331932 https://assets.mubicdn.net/images/film/159/image-w320.jpg?1548864573    https://assets.mubicdn.net/images/film/142/image-w320.jpg?1544094102
48905025        3       Romantic Movies        7       2019-05-30 03:00:07     2009-11-11 00:20:00     6       http://mubi.com/lists/romantic-movies  5       <p>Films about romance. In development.</p>
<p>Remarks</p>
<p><strong>Enter the    https://assets.mubicdn.net/images/film/3491/image-w1280.jpg?1564112978  https://assets.mubicdn.net/images/film/3491/image-w320.jpg?1564112978https://assets.mubicdn.net/images/film/2377/image-w320.jpg?1564675204    https://assets.mubicdn.net/images/film/2874/image-w320.jpg?1546574412
*/

CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)

/*
3 rows from lists_users table:
user_id list_id list_update_date_utc    list_creation_date_utc  user_trialist   user_subscriber user_avatar_image_url   user_cover_image_url    user_eligible_for_trial       user_has_payment_method
85981819        1969    2019-11-26      2009-12-18      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        3946    2020-05-01      2010-01-30      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        6683    2020-04-12      2010-03-31      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
*/

Table: lists
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_title: column description -> Name of the list
Column list_movie_number: column description -> Number of movies added to the list
Column list_update_timestamp_utc: column description -> Last update timestamp for the list
Column list_creation_timestamp_utc: column description -> Creation timestamp for the list
Column list_followers: column description -> Number of followers on the list
Column list_url: column description -> URL to the list page on Mubi
Column list_comments: column description -> Number of comments on the list
Column list_description: column description -> List description made by the user

Table: lists_users
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_update_date_utc: column description -> Last update date for the list, value description -> YYYY-MM-DD
Column list_creation_date_utc: column description -> Creation date for the list, value description -> YYYY-MM-DD
Column user_trialist: column description -> whether the user was a tralist when he created the list , value description -> 1 = the user was a trialist when he created the list 0 = the user was not a trialist when he created the list
Column user_subscriber: column description -> whether the user was a subscriber when he created the list , value description -> 1 = the user was a subscriber when he created the list 0 = the user was not a subscriber when he created the list
Column user_avatar_image_url: column description -> User profile image URL on Mubi
Column user_cover_image_url: column description -> User profile cover image URL on Mubi
Column user_eligible_for_trial: column description -> whether the user was eligible for trial when he created the list , value description -> 1 = the user was eligible for trial when he created the list 0 = the user was not eligible for trial when he created the list
Column user_has_payment_method : column description -> whether the user was a paying subscriber when he created the list , value description -> 1 = the user was a paying subscriber when he created the list 0 = the user was not a paying subscriber when he created the list
#
Q: Among the lists created by user 4208563, which one has the highest number of followers? Indicate how many followers it has and whether the user was a subscriber or not when he created the list.
Hint: User 4208563 refers to user_id;highest number of followers refers to MAX(list_followers); user_subscriber = 1 means that the user was a subscriber when he created the list; user_subscriber = 0 means the user was not a subscriber when he created the list (to replace)
A: Let’s think step by step. In the question , we are asked:
"user" so we need column = [lists_users.user_id]
"number of followers" so we need column = [lists.list_followers]
"user was a subscriber or not" so we need column = [lists_users.user_subscriber]
Hint also refers to the columns = [lists_users.user_id,lists.list_followers,lists_users.user_subscriber]
Based on the columns and tables, we need these Foreign_keys = [lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1, 4208563]. So the Schema_links are:
Schema_links: [lists.list_followers,lists_users.user_subscriber,lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id, lists_users.user_id, 4208563, 1]

"""

HUMAN_SCHEMA_LINKING_TEMPLATE = """
For the given question, find the schema links between the question and the table.
Hint helps you to fine the correct schema_links.
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
A: Let's think step by step. In the question , we are asked:
"""

def get_database_schema(DB_URI: str) -> str:
    db = SQLDatabase.from_uri("sqlite:///"+DB_URI)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw()

def table_descriptions_parser(database_dir):
    csv_files = glob.glob(f"{database_dir}/*.csv")
    db_descriptions = ""
    for file_path in csv_files:
        table_name: str = os.path.basename(file_path).replace(".csv", "")
        db_descriptions += f"Table: {table_name}\n"
        try:
            table_df = pd.read_csv(file_path, encoding='latin-1')
            for _,row in table_df.iterrows():
                if pd.notna(row.iloc[2]):
                    col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))
                    val_description = re.sub(r'\s+', ' ', str(row.iloc[4])) if pd.notna(row.iloc[4]) else ""
                    if val_description:
                        db_descriptions += f"Column {row.iloc[0]}: column description -> {col_description}, value description -> {val_description}\n"
                    else:
                        db_descriptions += f"Column {row.iloc[0]}: column description -> {col_description}\n"
        except Exception as e:
            # logger.warning(f"Error reading CSV {file_path}: {e}")
            db_descriptions += "No column description\n"
        db_descriptions += "\n"
    return db_descriptions

def extract_schema_links(input_text: str) -> List[str]:
    pattern = r'Schema_links:\s*\[(.*?)\]'
    match = re.search(pattern, input_text)
    if match:
        schema_links_str = match.group(1)
        schema_links = [link.strip() for link in schema_links_str.split(',')]
        return schema_links
    else:
        return []

# ==========================================
# DinSQL Retriever Class
# ==========================================

class DinSQLRetriever:
    def __init__(self, db_root_path):
        self.db_root_path = db_root_path
        # Removed LangChain LLM initialization
        # self.llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=2000)
        
        # system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_SCHEMA_LINKING_TEMPLATE)
        # human_prompt = HumanMessagePromptTemplate.from_template(HUMAN_SCHEMA_LINKING_TEMPLATE)
        # self.prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        # self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def retrieve_schema_links(self, db_id: str, question: str, evidence: str) -> Dict[str, List[str]]:
        """
        Executes DIN-SQL schema linking and converts result to {table: [columns]} format.
        """
        db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
        desc_path = os.path.join(self.db_root_path, db_id, "database_description")
        
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return {}

        try:
            schema_str = get_database_schema(db_path)
            columns_desc = table_descriptions_parser(desc_path)
            
            # Construct messages manually
            system_message = {"role": "system", "content": SYSTEM_SCHEMA_LINKING_TEMPLATE}
            user_content = HUMAN_SCHEMA_LINKING_TEMPLATE.format(
                schema=schema_str,
                columns_descriptions=columns_desc,
                question=question,
                hint=evidence
            )
            user_message = {"role": "user", "content": user_content}
            
            messages = [system_message, user_message]
            
            # Call project LLM client
            response = get_competition(messages)
            
            raw_links = extract_schema_links(response)
            return self._parse_links_to_dict(raw_links)
            
        except Exception as e:
            logger.error(f"Error in DIN-SQL retrieval for {db_id}: {e}")
            return {}

    def _parse_links_to_dict(self, raw_links: List[str]) -> Dict[str, List[str]]:
        """
        Converts list like ['table.col', 'table', 'value'] to {'table': ['col']}
        """
        result = defaultdict(list)
        for link in raw_links:
            link = link.strip()
            if '.' in link:
                parts = link.split('.')
                if len(parts) == 2:
                    tbl, col = parts
                    # Simple heuristic: assume it's a valid table/col if it looks like one
                    # Ideally we should validate against actual schema, but for baseline this is okay
                    # as we want to measure what the LLM *thinks* is relevant.
                    result[tbl].append(col)
                elif '=' in link:
                    # Handle foreign key notation like T1.C1 = T2.C2
                    # We extract both sides
                    sides = link.split('=')
                    for side in sides:
                        side = side.strip()
                        if '.' in side:
                            t, c = side.split('.')
                            result[t].append(c)
            else:
                # Might be just a table name or a value
                # We can't easily distinguish without schema lookup, 
                # but usually values don't look like table names in this context.
                # For now, if it's a single word, we treat it as a table if it's not a number.
                if not link.replace('.', '', 1).isdigit():
                     if link not in result:
                         result[link] = []
        
        # Remove duplicates
        final_result = {}
        for tbl, cols in result.items():
            final_result[tbl] = sorted(list(set(cols)))
            
        return final_result

# ==========================================
# Evaluation Logic (Adapted from evaluate_retrieval.py)
# ==========================================

def calculate_metrics(retrieved, golden):
    golden_tables = set(golden.keys())
    golden_columns = set()
    for tbl, cols in golden.items():
        for col in cols:
            golden_columns.add(f"{tbl}.{col}")

    retrieved_tables = set(retrieved.keys())
    retrieved_columns = set()
    for tbl, cols in retrieved.items():
        for col in cols:
            retrieved_columns.add(f"{tbl}.{col}")

    tbl_tp = len(retrieved_tables & golden_tables)
    tbl_fp = len(retrieved_tables - golden_tables)
    tbl_fn = len(golden_tables - retrieved_tables)

    col_tp = len(retrieved_columns & golden_columns)
    col_fp = len(retrieved_columns - golden_columns)
    col_fn = len(golden_columns - retrieved_columns)

    return {
        "tbl_tp": tbl_tp, "tbl_fp": tbl_fp, "tbl_fn": tbl_fn,
        "col_tp": col_tp, "col_fp": col_fp, "col_fn": col_fn
    }

def process_single_db(db_id, cases, saved_results, output_file, eval_results, eval_output_file, db_root_path):
    local_metrics = {
        "tbl_tp": 0, "tbl_fp": 0, "tbl_fn": 0,
        "col_tp": 0, "col_fp": 0, "col_fn": 0
    }
    
    # Initialize DIN-SQL Retriever
    retriever = DinSQLRetriever(db_root_path)
    
    for case in cases:
        question = case['question']
        evidence = case.get('evidence', '')
        golden_link = case['golden_schema_link']
        question_id = str(case.get('question_id'))
        
        retrieved_link = {}

        # Check cache
        if question_id in saved_results:
            retrieved_link = saved_results[question_id]
        else:
            logger.info(f"Retrieving Q: {question[:60]}...")
            try:
                retrieved_link = retriever.retrieve_schema_links(db_id, question, evidence)
                
                with file_lock:
                    saved_results[question_id] = retrieved_link
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(saved_results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue

        try:
            metrics = calculate_metrics(retrieved_link, golden_link)

            # Log mismatches (optional, similar to original script)
            # ... (omitted for brevity, logic is same as original) ...

            for k in local_metrics:
                local_metrics[k] += metrics[k]
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            
    return local_metrics

def evaluate(db_name, test_file_path, output_dir, db_root_path,report_save_dir, max_workers=16):
    logger.info(f"Loading test cases from {test_file_path}")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    output_file = os.path.join(output_dir, f"dinsql_retrieval_results_{db_name}.json")
    if os.path.exists(output_file):
        logger.info(f"Loading cached results from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
    else:
        saved_results = {}
        os.makedirs(output_dir, exist_ok=True)

    reprot_save_dir = os.path.join(report_save_dir, f"dinsql_retrieval_report_{db_name}.txt")
    os.makedirs(os.path.dirname(reprot_save_dir), exist_ok=True)
    eval_results = {}

    cases_by_db = defaultdict(list)
    for case in test_cases:
        cases_by_db[case['db_id']].append(case)

    total_metrics = {
        "tbl_tp": 0, "tbl_fp": 0, "tbl_fn": 0,
        "col_tp": 0, "col_fp": 0, "col_fn": 0
    }
    
    total_cases = len(test_cases)
    logger.info(f"Starting evaluation with {max_workers} workers for {len(cases_by_db)} databases.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cases = {}
        for db_id, cases in cases_by_db.items():
            future = executor.submit(
                process_single_db,
                db_id,
                cases,
                saved_results,
                output_file,
                eval_results,
                reprot_save_dir,
                db_root_path,
            )
            future_to_cases[future] = len(cases)

        progress = tqdm(total=total_cases, desc="Evaluating", unit="q", mininterval=1.0, monitor_interval=0)
        try:
            for future in concurrent.futures.as_completed(future_to_cases):
                num_cases = future_to_cases[future]
                try:
                    local_metrics = future.result()
                    for k in total_metrics:
                        total_metrics[k] += local_metrics[k]
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                finally:
                    # filepath: /home/zjw/project/Text2SQL/schema_kg/scripts/evaluate/evaluate_dinsql.py
                    progress.update(num_cases)
        finally:
            progress.close()

    def calc_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    tbl_p, tbl_r, tbl_f1 = calc_f1(total_metrics['tbl_tp'], total_metrics['tbl_fp'], total_metrics['tbl_fn'])
    col_p, col_r, col_f1 = calc_f1(total_metrics['col_tp'], total_metrics['col_fp'], total_metrics['col_fn'])

    print("\n" + "="*40)
    print("DIN-SQL RETRIEVAL EVALUATION REPORT")
    print("="*40)
    print(f"Total Questions Evaluated: {total_cases}")
    print("-" * 20)
    print(f"TABLE LEVEL:")
    print(f"  Precision : {tbl_p:.4f}")
    print(f"  Recall    : {tbl_r:.4f}")
    print(f"  F1 Score  : {tbl_f1:.4f}")
    print("-" * 20)
    print(f"COLUMN LEVEL:")
    print(f"  Precision : {col_p:.4f}")
    print(f"  Recall    : {col_r:.4f}")
    print(f"  F1 Score  : {col_f1:.4f}")
    print("="*40)
    
if __name__ == "__main__":
    # Configuration
    db_name = "california_schools" # Or pass as arg
    test_file = f"bird_data/golden_link/golden_schema_link_{db_name}.json"
    output_dir = "scripts/evaluate/cache"
    db_root_path = "bird_data/bird/llm/data/dev_databases" # Path to where BIRD databases are stored (unzipped)
    report_save_dir = "scripts/evaluate/result"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    if os.path.exists(test_file):
        evaluate(db_name, test_file, output_dir, db_root_path, report_save_dir)
    else:
        logger.error(f"Test file not found: {test_file}")