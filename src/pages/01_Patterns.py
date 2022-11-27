import streamlit as st  
import pandas as pd
from DataFrame import DF
import itertools

st.set_page_config(layout="wide")

def get_data(df, clean=False, encode=True):
    df.missing_values_imputation()
    if encode:
        df.encode_str()
    if clean: 
        for att in list(df.data.keys()):
            df.outliers_median_imputation(att)
        for att in list(df.data.keys()):
            df.normalize_minmax(att)
    if encode:
        return [[df.encode_dict[att][df.data[att][i]] for att in list(df.data.keys())] for i in range(len(df.data.get('Watcher')))]
    return df 

def get_transactions(df):
    transactions = {}
    for k, v in get_data(df):
        if k in transactions.keys():
            if v not in transactions[k]:
                transactions[k] += tuple([v])
        else:
            transactions[k] = tuple([v]) 
    return list(transactions.values())

def Apriori(transactions, min_support = 0.2):
    out = {}
    c = {}
    min_support = min_support * len(transactions)

    for transaction in transactions:
        for item in transaction: 
            c[(item,)] = c[(item,)] + 1 if (item,) in c.keys() else 1
    l = {k: v for k, v in c.items() if v >= min_support}
    out.update(l)
    
    while c:
        c = {} 
        for item_set1 in list(l.keys()):
            for item_set2 in list(l.keys()):
                if item_set1 != item_set2 and item_set1 != item_set2[::-1]: 
                    c[tuple(set(item_set1) | set(item_set2))] = 0   
        for transaction in transactions:
            for item_set in c.keys():
                if (set(item_set).issubset(transaction)):
                    c[item_set] = c[item_set] + 1
        l = {k: v for k, v in c.items() if v >= min_support}
        out.update(l) 
    out = dict(sorted(out.items(), key=lambda item: item[1], reverse=True))
    return l, c, out

def possibilities(L):
    if len(L) == 1:
        return
    perm = tuple(itertools.permutations(L)) 
    out = {}
    for p in perm:
        for i in range(1, len(p)):
            out[p[:i]] = p[i:]
    return out

def get_rules(transactions, out, min_support=False, min_conf=False):
    rules = []
    for itemset, support in out.items():
        if len(itemset) > 1:
            for antecedents, consequents in possibilities(itemset).items():
                try:
                    if consequents not in out.keys():
                        consequents = consequents[::-1]
                    support = out[tuple(set(antecedents) | set(consequents))]
                    confidence = out[tuple(set(antecedents) | set(consequents))] / out[antecedents]
                    lift = out[tuple(set(antecedents) | set(consequents))] / (out[antecedents] * out[consequents])
                    rules.append([antecedents, consequents, support, confidence, lift])
                except:
                    pass    
    if min_support:
        min_support = min_support * len(transactions)
        rules = [rule for rule in rules if rule[2] >= min_support]
    if min_conf:
        rules = [rule for rule in rules if rule[3] >= min_conf]
    return rules  

def get_pred(watcher, transactions, rules):
    interests = transactions[watcher]
    return get_pred_by_cats(interests, rules)

def get_pred_by_cats(categories, rules):
    change = True
    interests = categories
    while(change):
        change = False
        for rule in rules:
            if set(rule[0]).issubset(set(categories)):
                if not set(rule[1]).issubset(set(categories)):
                    categories = set(categories) | set(rule[1])
                    change = True
    return [i for i in categories if i not in interests]

st.sidebar.markdown('Frequent items')
min_suport = st.sidebar.slider('Min support', 0.05, 1.0, 0.4)
st.sidebar.markdown("""---""")
st.sidebar.markdown('Association rules')
min_suport_rules, min_conf_rules = st.sidebar.slider('Min support', 0.05, 1.0, 0.0), st.sidebar.slider('Min confidence', 0.05, 1.0, 0.0)

df = DF(remove=['definition', 'videoCategoryId'])
transactions = get_transactions(df)

l, c, out = Apriori(transactions, min_suport)  
output = pd.DataFrame([[k, v] for k, v in zip(out.keys(), out.values())]) 
output.columns = ['itemsets', 'support'] 



st.title('Frequent items')
st.dataframe(output)

col1, col2 = st.columns([4, 1])

with col1:
    st.title('Association rules and Recommendations')

    try:
        rules = get_rules(transactions, out, min_suport_rules, min_conf_rules)
        rules_to_display = pd.DataFrame(rules)  
        rules_to_display.columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift'] 
        rules_to_display = rules_to_display.sort_values(by=['support'], ascending=False)
        rules_to_display.reset_index(drop=True, inplace=True)

        st.dataframe(rules_to_display)
    except:
        st.write('No association rules')

with col2:
    st.title('')
    st.title('')
    df = DF(remove=['definition', 'videoCategoryId'])
    data = get_data(df, encode=False)
    watchers = data.unique('Watcher')
    watcher = st.selectbox('Select a watcher', watchers)
    df.encode_str()
    index = {v: k for k, v in df.encode_dict['Watcher'].items()}
    watcher = index[watcher]
    st.markdown('Interests')
    st.write(transactions[watcher])
    st.markdown('Predictions')
    st.write(get_pred(watcher, transactions, rules))

df = DF(remove=['definition', 'videoCategoryId'])
data = get_data(df, encode=False)
categories = st.multiselect(
    'What are your favorite colors',
    data.unique('videoCategoryLabel'),
    [])

st.markdown('Predictions')
st.write(get_pred_by_cats(categories, rules))
# pull add commit push