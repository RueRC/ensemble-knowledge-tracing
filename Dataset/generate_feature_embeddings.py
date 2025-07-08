"""
This is a template for generating feature embeddings from the original public dataset,
using Assist2017 as an example.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import src.configs as C

def main():

    df = pd.read_csv('assistment2017.csv',
                     usecols=['studentId', 'AveKnow', 'AveCarelessness', 'AveCorrect',
                              'AveResBored', 'AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResOfftask',
                              'AveResGaming', 'correct','original','hint','scaffold',
                              'timeTaken', 'bottomHint', 'attemptCount', 'frIsHelpRequest',
                              'totalFrTimeOnSkill', 'timeSinceSkill', 'totalFrAttempted', 'totalFrSkillOpportunities',
                              'confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
                              'confidence(OFF TASK)', 'confidence(FRUSTRATED)', 'confidence(GAMING)', 'RES_BORED',
                              'RES_CONCENTRATING', 'RES_CONFUSED', 'RES_FRUSTRATED', 'RES_OFFTASK', 'RES_GAMING', 'MCAS',
                              'Ln-1', 'Ln'])

    # time taken - standardize
    scaler = StandardScaler()
    # scaler.fit(df.loc[:,'timeTaken'])
    df.loc[:, 'timeTaken'] = scaler.fit_transform(df.loc[:, 'timeTaken'].values.reshape(-1, 1))

    # attemptCount - standardize
    scaler = StandardScaler()
    df.loc[:, 'attemptCount'] = scaler.fit_transform(df.loc[:, 'attemptCount'].values.reshape(-1, 1))

    #others - standardize
    df.loc[:,'AveKnow'] = scaler.fit_transform(df.loc[:,'AveKnow'].values.reshape(-1,1))
    df.loc[:,'AveCarelessness'] = scaler.fit_transform(df.loc[:,'AveCarelessness'].values.reshape(-1,1))
    df.loc[:,'AveCorrect'] = scaler.fit_transform(df.loc[:,'AveCorrect'].values.reshape(-1,1))
    df.loc[:,'confidence(CONCENTRATING)'] = scaler.fit_transform(df.loc[:,'confidence(CONCENTRATING)'].values.reshape(-1,1))
    df.loc[:,'confidence(BORED)'] = scaler.fit_transform(df.loc[:,'confidence(BORED)'].values.reshape(-1,1))
    df.loc[:,'confidence(CONFUSED)'] = scaler.fit_transform(df.loc[:,'confidence(CONFUSED)'].values.reshape(-1,1))
    df.loc[:,'confidence(OFF TASK)'] = scaler.fit_transform(df.loc[:,'confidence(OFF TASK)'].values.reshape(-1,1))
    df.loc[:,'Ln-1'] = scaler.fit_transform(df.loc[:,'Ln-1'].values.reshape(-1,1))
    df.loc[:,'Ln'] = scaler.fit_transform(df.loc[:,'Ln'].values.reshape(-1,1))

    # bottomHint - onehot
    bottomHint1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'bottomHint'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(bottomHint1hot)], axis=1)
    df.rename(columns={0: 'bottomHintfalse', 1: 'bottomHinttrue'}, inplace=True)

    # frIsHelpRequest - onehot
    frIsHelpRequest1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'frIsHelpRequest'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(frIsHelpRequest1hot)], axis=1)

    df.rename(columns={0: 'frIsHelpRequestfalse', 1: 'frIsHelpRequesttrue'}, inplace=True)
    df.drop(["bottomHint", "frIsHelpRequest"], axis=1, inplace=True)

    # 'AveResBored','AveResEngcon','AveResConf','AveResFrust','AveResOfftask','AveResGaming'
    df.loc[:, 'AveResBored'] = scaler.fit_transform(df.loc[:, 'AveResBored'].values.reshape(-1, 1))
    df.loc[:, 'AveResEngcon'] = scaler.fit_transform(df.loc[:, 'AveResEngcon'].values.reshape(-1, 1))
    df.loc[:, 'AveResConf'] = scaler.fit_transform(df.loc[:, 'AveResConf'].values.reshape(-1, 1))
    df.loc[:, 'AveResFrust'] = scaler.fit_transform(df.loc[:, 'AveResFrust'].values.reshape(-1, 1))
    df.loc[:, 'AveResOfftask'] = scaler.fit_transform(df.loc[:, 'AveResOfftask'].values.reshape(-1, 1))
    df.loc[:, 'AveResGaming'] = scaler.fit_transform(df.loc[:, 'AveResGaming'].values.reshape(-1, 1))
    # 'totalFrTimeOnSkill','timeSinceSkill','totalFrAttempted','totalFrSkillOpportunities'
    df.loc[:, 'totalFrTimeOnSkill'] = scaler.fit_transform(df.loc[:, 'totalFrTimeOnSkill'].values.reshape(-1, 1))
    df.loc[:, 'timeSinceSkill'] = scaler.fit_transform(df.loc[:, 'timeSinceSkill'].values.reshape(-1, 1))
    df.loc[:, 'totalFrAttempted'] = scaler.fit_transform(df.loc[:, 'totalFrAttempted'].values.reshape(-1, 1))
    df.loc[:, 'totalFrSkillOpportunities'] = scaler.fit_transform(
        df.loc[:, 'totalFrSkillOpportunities'].values.reshape(-1, 1))
    # 'confidence(FRUSTRATED)','confidence(GAMING)','RES_BORED','RES_CONCENTRATING','RES_CONFUSED'
    df.loc[:, 'confidence(FRUSTRATED)'] = scaler.fit_transform(
        df.loc[:, 'confidence(FRUSTRATED)'].values.reshape(-1, 1))
    df.loc[:, 'confidence(GAMING)'] = scaler.fit_transform(df.loc[:, 'confidence(GAMING)'].values.reshape(-1, 1))
    df.loc[:, 'RES_BORED'] = scaler.fit_transform(df.loc[:, 'RES_BORED'].values.reshape(-1, 1))
    df.loc[:, 'RES_CONCENTRATING'] = scaler.fit_transform(df.loc[:, 'RES_CONCENTRATING'].values.reshape(-1, 1))
    df.loc[:, 'RES_CONFUSED'] = scaler.fit_transform(df.loc[:, 'RES_CONFUSED'].values.reshape(-1, 1))
    # 'RES_FRUSTRATED','RES_OFFTASK','RES_GAMING','MCAS'
    df.loc[:, 'RES_FRUSTRATED'] = scaler.fit_transform(df.loc[:, 'RES_FRUSTRATED'].values.reshape(-1, 1))
    df.loc[:, 'RES_OFFTASK'] = scaler.fit_transform(df.loc[:, 'RES_OFFTASK'].values.reshape(-1, 1))
    df.loc[:, 'RES_GAMING'] = scaler.fit_transform(df.loc[:, 'RES_GAMING'].values.reshape(-1, 1))
    df.loc[:, 'MCAS'] = scaler.fit_transform(df.loc[:, 'MCAS'].values.reshape(-1, 1))

    # 'correct','original','hint','scaffold'
    # correct - onehot
    correct1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'correct'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(correct1hot)], axis=1)
    df.rename(columns={0: 'correctfalse', 1: 'correcttrue'}, inplace=True)
    # original - onehot
    original1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'original'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(original1hot)], axis=1)
    df.rename(columns={0: 'originalfalse', 1: 'originaltrue'}, inplace=True)
    # hint - onehot
    hint1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'hint'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(hint1hot)], axis=1)
    df.rename(columns={0: 'hintfalse', 1: 'hinttrue'}, inplace=True)
    # scaffold - onehot
    scaffold1hot = OneHotEncoder(categories='auto').fit_transform(df.loc[:, 'scaffold'].values.reshape(-1, 1)).toarray()
    df = pd.concat([df, pd.DataFrame(scaffold1hot)], axis=1)
    df.rename(columns={0: 'scaffoldfalse', 1: 'scaffoldtrue'}, inplace=True)
    df.drop(["correct", 'original', 'hint', 'scaffold'], axis=1, inplace=True)


    df.to_csv(path_or_buf=C.ASSIST2017_FEA_PATH, index=False, header=False)

if __name__ == '__main__':
    main()