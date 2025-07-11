import re


def replace_cur_year(query: str) -> str:
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )

def query_preprocessing(sql:str):
    """
    Preprocess the generated SQL query
    """
    sql = " ".join(sql.replace("\n", " ").split())
    sql = sql.strip().split("/*")[0]
    if sql.startswith("SELECT"):
        return sql + "\n"
    elif sql.startswith(" "):
        return "SELECT" + sql + "\n"
    else:
        return "SELECT " + sql + "\n"

def clean_sql_output(gpt_output):
    """ Special post-processing for gpt-4o SQL outputs """
    # Use regex to remove ```sql tags and any extraneous SELECT pre-text
    cleaned_output = re.sub(r"^SELECT ```sql\s*|```$", "", gpt_output.strip(), flags=re.MULTILINE)
    return cleaned_output.strip()

def query_postprocessing(sql:str):
    """
    Postprocess the generated SQL query
    """
    sql = " ".join(sql.replace("\n", " ").split())
    sql = sql.strip().split("/*")[0]
    sql = clean_sql_output(sql) # for gpt-4o
    if sql.lower().startswith("select"):
        return sql + "\n"
    elif sql.startswith(" "):
        return "SELECT" + sql + "\n"
    else:
        return "SELECT " + sql + "\n"
    