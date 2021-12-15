# from __future__ import division
import logging
import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
import shap
import sys
# from shap.plots import colors
# from shap.plots import labels
from scipy.stats import gaussian_kde
import multiprocessing as mp
import time
from scipy import stats
import requests
# regex = re.compile(r"\[|\]|<", re.IGNORECASE)
try:
    import matplotlib.pyplot as pl
except ImportError as e:
    warnings.warn("matplotlib could not be loaded!", e)
    pass
import warnings
warnings.filterwarnings("ignore")

class InterpretableModel:
    
    def __init__(self, data_obj, metric, account_id,gan_flag ,lang_flag,num_feats=10):

        self._metric = metric
        self._data_obj = data_obj
        self._account_id = account_id
        self._ads_used = None
        self._b_feats_used = None
        self._s_feats_used = None
        self.coeffs = None
        self.testlist = None
        self.savedollar = None
        self.confidence = None
        self._gan_flag = gan_flag
        self._lang_flag = lang_flag

    def interpreting_model(self):
        return self._random_forest_pipline(self._metric, self._account_id, num_feats=10)

    def prediction_test(self):
        return self.predict_ads(self._metric, self._account_id, num_feats=10)

    def summary_plot(self,shap_values,features=None, feature_names=None, max_display=None, sort=True, importance = None):
        min_percent_impact = []
        max_percent_impact = []
        # print("I am in analysis.py 1----------------------------------->")
        ad_id = []
        def get_max(row):
            if abs(row['min_percent_impact']) > abs(row['max_percent_impact']):
                return row['min_percent_impact']
            else:
                return row['max_percent_impact']
        # print("I am in analysis.py 2------------------------------------------------------->")
        multi_class = False
        if isinstance(shap_values, list):
            multi_class = True
            plot_type = "bar" # only type supported for now
        else:
            assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."
        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            # print("I am in analysis.py 3 ===============================------------------------------------------------------->")
            if feature_names is None:
                # feature_names = features.columns
                features.columns=[regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in features.columns.values]
                feature_names = features.columns
                # print("In the analysis-------------------------------------------------------------------------------------",feature_names)
            dataframe = features
            # print("In the analysis 1.1    -------------------------------------------------------------------------------------",dataframe)
            features = features[feature_names].values
            # print("In the analysis 1.2    -------------------------------------------------------------------------------------",features)
        elif isinstance(features, list):
            if feature_names is None:
                # print("In the analysis 1.3    -------------------------------------------------------------------------------------",features)
                feature_names = features
            features = None
        elif (features is not None) and len(features.shape) == 1 and feature_names is None:
            # print("In the analysis 1.41       -------------------------------------------------------------------------------------",feature_names)
            feature_names = features
            features = None

        num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

        if feature_names is None:
            feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

        if max_display is None:
            max_display = features.shape[1]
        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=0), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
                feature_confidence = np.sum(np.abs(shap_values), axis=0)
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
        row_height = 0.4
        for pos, i in enumerate(feature_order):
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            #print("shaps:",shaps)
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))
            if features is not None:
                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                #nan_mask = np.isnan(values)
                nan_mask  = (values != 1) | np.isnan(values)
                min_percent_impact.append(np.mean(shaps[np.invert(nan_mask)][shaps[np.invert(nan_mask)]<0]))
                max_percent_impact.append(np.mean(shaps[np.invert(nan_mask)][shaps[np.invert(nan_mask)]>=0]))
                #print("min_per:",min_percent_impact)
                #print("max_per:",max_percent_impact)
       # print("^"*100,"INSIDE SUMMARY_PLOT")

       # print(set(feature_names))
       # print(feature_order)
        #print([feature_names[i] if feature_  for i in feature_order])
        df = pd.DataFrame({"feature_name":[feature_names[i] for i in feature_order],'confidence':[importance[feature_names[i]] for i in feature_order],
                            "min_percent_impact":min_percent_impact,"max_percent_impact":max_percent_impact,"ad_id":[self.get_ad_id(dataframe, feature_names[i]) for i in feature_order]})
        #print(df)
        #print(df[''])
        df = df.sort_values(by = 'confidence')
        #print(df)
        d_f = df.iloc[::-1]
        d_f = d_f.fillna(0)
        #print(d_f)
        d_f['max_impact'] = d_f.apply(get_max, axis = 1)
        d_f = d_f[d_f['max_impact'] != 0]
        #print(d_f)
        #print(d_f['max_impact'])
        #print(d_f[['feature_name','max_impact']])
        #print(d_f['feature_name'])
        
        print("Exited Summary Plot")
        return d_f[['feature_name','max_impact','confidence','ad_id']]

    def predict_ads(self,metric, account_id, num_feats=10):
        dftemp, dummies, xvars = self._filter_df()
        assert metric in ['cpp','ctr', 'cpm', 'cpc', 'relevance_score', 'roas','impressions','cpr','cpi',"cpl","cpreg","cpatc"]       
        feats = list((dummies|xvars))
        df_pred_all = dftemp
        body_text, headline, link_description = list(dftemp['body'])[-1], list(dftemp['title'])[-1], list(dftemp['link_description'])[-1]
        df_prd = list(dftemp[feats].values[-1])
        for n, i in enumerate(df_prd):
            if i == "":
                df_prd[n] = 0
        dftemp = dftemp.iloc[:-1,:]
        dataframe = dftemp
        dftemp = dftemp[[metric,'impressions'] + feats]
        dftemp = dftemp.apply(pd.to_numeric, errors = 'coerce')
        dftemp.dropna(inplace=True, axis=0)
        self._ads_used, self._b_feats_used, self._s_feats_used = dftemp.index.tolist(), dummies,xvars
        logging.info("# of matching ads: {}".format(len(dftemp)))
        logging.info("# of dataframe columns: {}".format(len(list(dftemp))))
        #best_feats = rfe_feat_selection(dftemp[feats],dftemp[metric])
        best_feats = feats
        rf = RandomForestRegressor(random_state = 0,n_jobs = -1)
        # rf = lgb.LGBMRegressor(random_state=0)
        rf.fit(dftemp[best_feats].values, dftemp[metric].values)
        feature_importance = dict(zip(best_feats,rf.feature_importances_))
        Score = rf.score(dftemp[feats].values, dftemp[metric].values)
        calculate_percentage = np.average(rf.predict(dftemp[best_feats].values))
        #print("BEST FEATS", set(best_feats))
        shap_values = shap.TreeExplainer(rf).shap_values(dftemp[best_feats])
        dffinal = self.summary_plot(shap_values,features = dataframe,feature_names = best_feats, importance = feature_importance)
        dffinal['max_impact'] = (dffinal['max_impact']/calculate_percentage)* 100
        pred = rf.predict([df_prd])[0]
        row = df_pred_all[feats].tail(1)
        avg_metric = float(np.mean(dftemp[metric].values))
        std_metric = float(np.std(dftemp[metric].values))
        z_metric = float((pred - avg_metric)/std_metric)      
        row_feats = []
        row = row.reset_index()
        for col in list(best_feats):
            if row.loc[0, col] != 0: 
                row_feats.append(col)
        input_feats = list(set(row_feats))
        dffinal_input = dffinal[dffinal['feature_name'].isin(input_feats)] #with input feats
        #print(dffinal_input)
        dffinal = dffinal[~dffinal['feature_name'].isin(input_feats)]
        dffinal['feature_name'] = dffinal['feature_name'].apply(lambda x:self.make_readable(x))
        dffinal_input['feature_name'] = dffinal_input['feature_name'].apply(lambda x:self.make_readable(x)) 
        if metric.lower() != 'cpr':
            tmp_metric = metric
        else:
            tmp_metric = 'Cost per Result'
        def get_con(x):
            if x['max_impact'] > 0:
                x['feature_name'] = x['feature_name']+" will increase "+tmp_metric.upper()+" by"
                x['max_impact'] = "{0:.2f}".format(abs(x['max_impact']))
                return x
            else:
                x['feature_name'] = x['feature_name']+" will  decrease "+tmp_metric.upper()+" by"
                x['max_impact'] = "{0:.2f}".format(abs(x['max_impact']))
                return x 
        # if metric=='relevance_score' or metric=='ctr' or metric=='roas' or metric=='impressions':
        if metric=='relevance_score' or metric=='roas' or metric=='impressions' or metric=='ctr':
            dffinal = dffinal[dffinal['max_impact']>0]
            dffinal = dffinal.append(dffinal_input[dffinal_input['max_impact'] < 0])
            dffinal = dffinal.sort_values(by='confidence', ascending=False)
            dffinal['max_impact'] = dffinal['max_impact'].apply(lambda x:"{0:.2f}".format(x))
        else:
            dffinal = dffinal[dffinal['max_impact']<0]
            dffinal = dffinal.append(dffinal_input[dffinal_input['max_impact'] > 0])
            dffinal = dffinal.sort_values(by='confidence', ascending=False)
            dffinal['max_impact'] = dffinal['max_impact'].apply(lambda x:"{0:.2f}".format(x)) 
        try:
            dffinal = self.order_format_dataframe(dffinal.head(10))
            add_below_feat = [i for i in list(dffinal['feature_name'] + "@@"+dffinal['max_impact'])]
        except:
            dffinal = self.order_format_dataframe(dffinal)
            add_below_feat = [i for i in list(dffinal['feature_name'] + "@@"+dffinal['max_impact'])]
        if metric.lower() == 'ctr':
            dict_result = {"link_description":link_description,"headline":headline,"body_text":body_text,"add_below_features" :  add_below_feat, "performance":"Predicted "+metric.upper()+ ": " + "{0:.2f}".format(pred)+" %","Average": metric.upper()+ ": " + "{0:.2f}".format(np.average(dftemp[metric].values))+" %"}
        else:
            if metric.lower() == 'cpr':
                dict_result = {"link_description":link_description,"headline":headline,"body_text":body_text,"add_below_features" :  add_below_feat, "performance":"Predicted Cost Per Result"+ ": " +"$"+ "{0:.2f}".format(pred),"Average":"Cost Per Result"+ ": " +"$"+ "{0:.2f}".format(np.average(dftemp[metric].values))}
            else:
                dict_result = {"link_description":link_description,"headline":headline,"body_text":body_text,"add_below_features" :  add_below_feat, "performance":"Predicted "+metric.upper()+ ": " +"$"+ "{0:.2f}".format(pred),"Average": metric.upper()+ ": " + " $ "+"{0:.2f}".format(np.average(dftemp[metric].values))}
        dict_result['r2_Score'] = Score
        return dict_result

    def _random_forest_pipline(self,metric, account_id, num_feats=10):
        tm = time.time()
        dftemp, dummies, xvars = self._filter_df()
        print("find filter time","*"*10,time.time()-tm)
        dataframe = dftemp
        # print("In the random forest pipeline-------------------------------------------------------------------------------------------",dftemp.columns)
        assert metric in ['cpp','ctr', 'cpm', 'cpc', 'relevance_score', 'roas','impressions','cpr',"cpi","cpl","cpreg","cpatc"]
        # print("Below therandom forest pipeline 1 -------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp[feats])
        feats = list((dummies|xvars)) 
        # print("Below the random forest pipeline 1.1-----------------------------------------------------------------------------------------------------------------------------------------------------------------",feats)
        spend = np.mean(dftemp['spend'])
        clicks = np.mean(dftemp['clicks'])
        # print("Below the random forest pipeline 1.2-------------------------------------------------------------------------------------------------------------------------------------------------------------------",spend)
        impression = np.mean(dftemp['impressions'])
        if metric=='impressions':
            dftemp = dftemp[[metric] + feats]
            # print("Below the random forest pipeline 1.3-------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp)
        else:ars = ['nlp_link_description_sentiment_mag_mean',
         'nlp_link_description_sentiment_score_mean',
         'nlp_link_description_documentSentiment_mag',
         'nlp_link_description_documentSentiment_score',
         'nlp_title_sentiment_mag_mean',
         'nlp_title_sentiment_score_mean',
         'nlp_title_documentSentiment_mag',
         'nlp_title_documentSentiment_score',
         'nlp_body_sentiment_mag_mean',
         'nlp_body_sentiment_score_mean',
         'nlp_body_documentSentiment_mag',
         'nlp_body_documentSentiment_score',
         'nlp_video_transcription_sentiment_mag_mean',
         'nlp_video_transcription_sentiment_score_mean',
         'nlp_video_transcription_documentSentiment_mag',
         'nlp_video_transcription_documentSentiment_score','VID_segment_time']

        # Exclude binary features not found in the dataframe')
        # print("Below the ml model 1 -------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp.columns)
        dftemp.dropna(inplace=True, axis=0)
        self._ads_used, self._b_feats_used, self._s_feats_used = dftemp.index.tolist(), dummies,xvars
        logging.info("# of matching ads: {}".format(len(dftemp)))
        logging.info("# of dataframe columns: {}".format(len(list(feats))))
        rf = RandomForestRegressor(random_state = 0)
        # rf = lgb.LGBMRegressor(random_state=0)
        # rf = XGBRegressor(random_state=0)
        # print("Below the ml model-----------------------------------------------------------------------------------------------------",dftemp[feats].values)
        # print("Below the ml model 1.1-----------------------------------------------------------------------------------------------------",dftemp[metric].values)
        rf.fit(dftemp[feats].values, dftemp[metric].values)
        dftttmp = dftemp[feats]
        dftttmp['ctr'] = dftemp[metric].values
        Score = rf.score(dftemp[feats].values, dftemp[metric].values)
        calculate_percentage = float(np.average(rf.predict(dftemp[feats].values)))
        feature_importance = dict(zip(feats,rf.feature_importances_))
        shap_values = shap.TreeExplainer(rf).shap_values(dftemp[feats])
        #shap.summary_plot(shap_values,dftemp[feats])
        dffinal = self.summary_plot(shap_values,features = dataframe,feature_names = feats, importance = feature_importance)
        if len(dffinal)==0:
            raise Exception("Not enough data to analyze!")
        #print("$"*100,"dffinal['max_impact']")
        #print(dffinal['max_impact'])
        dffinal['percent_impact'] = (dffinal['max_impact']/calculate_percentage)* 100
        if metric.lower() == 'ctr':
            dffinal['save_dollar'] = dffinal['max_impact'] * spend
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpc':
            dffinal['save_dollar'] = dffinal['max_impact'] * clicks
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpm':
            dffinal['save_dollar'] = dffinal['max_impact'] * (impression/1000)
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpp':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpr':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpl':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpreg':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpatc':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpi':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'relevance_score':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'roas':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'impressions':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        self.coeffs = list(dffinal['percent_impact'])#coeffs
        self.testlist = list(dffinal['feature_name'])#df['feats'][:n_feats]
        self.confidence = list(dffinal['confidence'])
        #print(dffinal['feature_name']) 
        # print("LET's SEE NOW!")
        dffinal['feature_name'] = dffinal['feature_name'].apply(self.make_readable)
        return len(dftemp),Score, dffinal



    def _filter_df(self):
        df = self._data_obj.df
        dummies = self._data_obj.binary_feats
        xvars = self._data_obj.scalar_feats
        dummies_low_var = set()
        dummies_many_null = set()
        xvars_many_null = set()
        df_tmp  =pd.DataFrame(1 - df[list(dummies)].sum(axis = 0)/len(df), columns = ['percent'])
        # print("Inside the filter_df---------------------------------------------------------------------------------------------",df_temp.columns)
        dummies_low_var = set(list(df_tmp[df_tmp['percent']>.96].index))
        temp_series = 1 - df[list(dummies)].isna().sum()/len(df)
        dummies_many_null = set(list(temp_series[temp_series<.99].index))
        sparse_vars = dummies_low_var | dummies_many_null | xvars_many_null
        logging.info('Remaining sparse features: {}'.format(sparse_vars))
        df = df.drop(sparse_vars, axis=1)

        # Filter down dummies and xvars
        binary_feats = dummies  - sparse_vars
        # print("+"*100,"Binary feats:")
        # print(binary_feats)
        scalar_feats = xvars - sparse_vars
        return df, binary_feats, scalar_feats

    def get_salients(self):
        omit_list = ['vision_entities_font', 'vid_all_entities_font']
        df_used = self._data_obj.df.loc[self._ads_used]
        analysis = []
        NUM_ADS_PER_FEAT = 20
        #print("!"*100)
        #print(self.coeffs)
        TRUNCATE_VALUE = max(self.coeffs)
        unique_keyword=set()
        unique_keyword.add("Ad type is video")
        anlysis_lstid = []
        flag1=False
        #print("READABLE NAMES:")
        for feat, coeff, dollar, conf in zip(self.testlist, self.coeffs, self.save_dollar,self.confidence):
            if feat not in omit_list:
                # print(" condition 2 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",feat)
                coeff = max(-TRUNCATE_VALUE, min(coeff, TRUNCATE_VALUE))
                if coeff>0 and coeff<0.05:
                    continue
                if coeff<0 and coeff>-0.05:
                    continue
                flag1=False
                l1=feat.split()
                for j in range(len(l1)):
                    if len(l1[j])<4:
                        flag1=True
                    continue
                # print(" condition 3----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",coeff)
                if flag1==False:
                    readable_name = self.make_readable(feat)
                else:
                    continue
                if readable_name.find("is")!=-1:
                    continue
                # print(" condition 4 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",readable_name)
                # print(" condition 5 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",type(readable_name))

                readable_name = readable_name.rstrip()
                readable_name = readable_name.lstrip()
                readable_name = re.sub(r'^https?:\/\/.*[\r\n]*','', readable_name)
                readable_name = re.sub(r'http\S+','',readable_name)
                readable_name = re.sub(r'\.+', ".", readable_name)
                readable_name=readable_name.replace("\\","")
                # text_tokens = word_tokenize(readable_name)
                # tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
                # filtered_sentence = (" ").join(tokens_without_sw)
                # readable_name=filtered_sentence
                if readable_name in unique_keyword:
                    continue
                else:
                    unique_keyword.add(readable_name)
                # print(" condition 7----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",readable_name)
                #print(ascii(readable_name))    
                feat_dict = {'name': feat, 'coeff': coeff, 'ads': [], 'human_readable': readable_name, "dollar_info":dollar, "confidence":conf}  
                # Get used rows that have the feature:
                # TODO: Weight with impressions?
                if feat in self._b_feats_used:
                    df_feat = df_used.loc[df_used[feat] == 1]
                    df_feat = df_feat.sort_values(self._metric, ascending=False)
                    feat_dict['ads'] = df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist()
                    anlysis_lstid.append(df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist())         
                elif feat in self._s_feats_used:
                    # TODO: Deal with scalars properly (still want good performers)
                    df_feat = df_used.sort_values(feat, ascending=False)
                    feat_dict['ads'] = df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist()
                    anlysis_lstid.append(df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist())        
                else:
                    raise ValueError('Feature {} is neither binary nor scalar'.format(feat))         
                analysis.append(feat_dict)
            else:
                print("Omitted feat ", feat)
        # lst = ["id","ad_name" ,"source" ,"type" ,"campaign_name","adset_name","ad_name","thumbnail","ad_preview_link","body","cta_button","title",
        # "link_description","status","nlp_title_entities","nlp_body_entities","nlp_link_description_entities","vid_all_entities","vision_entities",
        # "impressions","primary_color","vision_api_full_text","vid_all_catentities","vid_text_entities","spend",self._metric] 
        lst=["id","ad_name" ,"source" ,"type" ,"campaign_name","adset_name","ad_name","thumbnail","ad_preview_link","body","cta_button","title",
        "link_description","status","spend",self._metric,"ctr","cpc","impressions","clicks","cpr","roas","reach","cpm","similiar_ads","cpp","cpl","cpi","cpreg","cpatc"] 
        return analysis,self._data_obj.df[lst].rename(columns={self._metric:"metric"}).set_index('id').to_dict('index')#list(set(sum(anlysis_lstid, [])))
            
    #To make human readable forms
    def make_readable(self, orig_name): 
        #print("INSIDE make_readable")
        language_str = self._lang_flag
        # print(" condition 4 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",language_str)
        # print("orig_name:")
        # print(orig_name)
        name =  orig_name.strip().replace("-", "_").split('_')
        # print(" condition 5 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",name)
        while name[-1] == "":
            name = name[:-1]
        # print(" condition 6 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",name)
        #handling these three type of variables for now, for more variable types, these rules will be amended.
        if (name[0]).lower() == 'vid':
            if (name[1]).lower() == 'text':
                final_name = ('"' + name[-1].lower() + '"' + ' Text in Video'.lower())
                # print("In the analysis.py vid---------- ---------------------=================================>",final_name)
            elif language_str:
                # print(" condition 7. in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",language_str)
                if language_str!='en':
                    translated_name = self.translate_text(text=name[-1].lower(),target_language_string=language_str)
                    final_name = '"'+ translated_name  +'"'+ ' in Video'.lower()
                else:
                    final_name ='"'+  name[-1].lower() +'"'+ ' in Video'.lower()
            else:
                final_name =  '"'+ name[-1].lower() +'"'+ ' in Video'.lower()
        elif (name[0]).lower() == 'nlp' :
            if (name[1]).lower() == 'link':
                final_name = ('"' + name[-1].lower() + '"' + ' in link description'.lower())
            elif ((name[1]).lower() == 'body'):
                # print(name[-1].lower())
                final_name = ('"' + name[-1].lower() + '"' + ' in Body text'.lower())
            else:
                final_name = ('"' + name[-1].lower() + '"' + ' in ' + name[1].lower().replace('title','headline'))
        elif (name[0]).lower()  == 'age':
            final_name = ('Targeting for ' + name[-2].lower() + '-' + '"' + name[-1].lower() + '"' + ' years old'.lower() )
        elif name[0].lower() == 'type':
            final_name = ('Ad type is ' + '"' + name[-1].lower() + '"')
        elif (name[0]).lower()  == 'vision':
            if 'vision_api_full_text' in orig_name:
                final_name =  '"' + name[-1].lower() + '"' + ' in Image'.lower()
            else:
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text=name[-1].lower(),target_language_string=language_str)
                        final_name = translated_name  + ' in Image'.lower()
                    else:
                        final_name = '"'+name[-1].lower() + '"'+ ' in Image'.lower()
                else:
                    final_name = '"'+name[-1].lower() + '"'+ ' in Image'.lower()
        elif (name[1] == 'color'):
            if ('primary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Primary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name =  translated_name + ' in Image'.lower()
                    else:                
                        final_name =  'Primary color is ' + name[-1].lower() + ' in Image'.lower()
                else:                
                    final_name =  'Primary color is ' + name[-1].lower() + ' in Image'.lower()
            elif ('secondary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Secondary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name = '"' + translated_name + '"' + 'in Image'.lower()
                    else:
                        final_name =  'Secondary color is ' + name[-1].lower() + ' in Image'.lower()
                else:
                    final_name =  'Secondary color is ' + name[-1].lower() + ' in Image'.lower()
            elif ('tertiary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Tertiary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name = '"'+translated_name + '"'+ 'in Image'.lower()
                    else:
                        final_name =  'Tertiary color is ' + name[-1].lower() + ' in Image'.lower()
                else:
                    final_name =  'Tertiary color is ' + name[-1].lower() + ' in Image'.lower()
        elif (int(name[0])%2==0):
            if language_str:
                if language_str!='en':
                    translated_name = self.translate_text(text = orig_name+" words", target_language_string = language_str)
                    final_name = '"' + translated_name + '"' + " in body text"
                else:    
                    final_name = '"'+orig_name+" words"+'"' +" in body text"
            else:    
                final_name = '"'+orig_name+" words"+'"' +" in body text"
            
            #print(final_name)
        else:
            final_name = orig_name
        # print(final_name)
        return final_name

    def order_format_dataframe(self,dct):
        def get_order(x):
            if 'video' in x.lower():
                return 0
            elif 'image' in x.lower():
                return 1
            elif 'headline' in x.lower():
                return 2
            elif 'body text' in x.lower():
                return 3
            elif 'link description' in x.lower():
                return 4
            else:
                return 5
        dct['feature_sort'] = dct['feature_name'].apply(lambda x: get_order(x))
        dct = dct.sort_values(by=['feature_sort'])
        dct = dct.drop(['feature_sort'], axis=1)
        return dct

    def get_ad_id(self,df_used, feat):
        df_feat = df_used.loc[df_used[feat] == 1]
        #df_feat = df_feat.sort_values(self._metric, ascending=False)
        return df_feat['id'].tolist()
    
    
    def translate_text(self, text, target_language_string):
        """Translating Text."""
        # print("Translating Text")
        
        json_data = {
                        'q': text,
                        'source': 'en',
                        'target': target_language_string,
                        'format': 'text'
                    }
        try:
            req = requests.post('https://translation.googleapis.com/language/translate/v2?key=AIzaSyBSSP5BNxjVUOFtP54jrGcPnk2WOkLEiSo', json=json_data)
            response = req.json()
            #print(dict(response)['data']['translations'][0]['translatedText'])
            translated_text = dict(response)['data']['translations'][0]['translatedText']
        except:
            translated_text = text
        return translated_text



