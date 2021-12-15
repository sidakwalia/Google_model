# IDEA:
# Shahid gets a query object, which is then converted to a dataframe, which is further deserialized here
# TODO: Deserialize JSON data at moment of download. For now we work on the fly.

import logging
import json
import re
import webcolors
import numpy as np
import pandas as pd
import sys
import ast
import time
import math
class AdData:

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, val):
        self._df = val

    @property
    def binary_feats(self):
        return self._binary_feats

    @binary_feats.setter
    def binary_feats(self, val):
        self._binary_feats = val

    @property
    def scalar_feats(self):
        return self._scalar_feats

    @scalar_feats.setter
    def scalar_feats(self, val):
        self._scalar_feats = val


class FBVideoData(AdData):
    def __init__(self, data,mode = "binary"):
        self._LISTVARS = ['nlp_title_entities']
        self._df = data
        entities = ['nlp_title_entities']
        self._df.reset_index(drop = True)
        t2 = time.time()
        for entity in entities:
            if mode == "binary":
                self._df = self.get_binaries_for_entities(entity)
                print("Entities are in binary----------------------------",self._df['nlp_title_entities'])    
            else:
                self._df = self.get_scalars_for_entities(entity)
                print("Entities are in scalers----------------------------",self._df['nlp_title_entities'])     
        print("*binaries*"*5,time.time()-t2)
        tm1 = time.time()
        # self._df = self.to_df(self._df)
        tm2 = time.time()
        print("*to_df*"*5,tm2-tm1)
        self._binary_feats = self._get_binaries(mode)
        tm3 = time.time()
        print("*_binary_feats*"*5,tm3-tm2)
        self._scalar_feats = self._get_scalars(mode)
    

    def _get_binaries(self,mode):
        df_cols = set(self._df.columns)
        print("**************COLS IN GET BINARIES**")
        #print(df_cols)
        dummies = []
        if mode == "binary":
                for i in self._LISTVARS:
                      for j in df_cols:
                          if j.startswith(i+'_'):
                              dummies.append(j)
        get_dummies_col = ['primary_color_','secondary_color_','tertiary_color_']
        for i in get_dummies_col:
            dummies = dummies + [x for x in df_cols if x.startswith(i)]
            # dummies = dummies + [x for x in df_cols if x.startswith(i)] + ['type_video','type_image','type_carousel']
        #df = self._df
        #self._df, length_feats = self.get_df_with_length_feats()
        #dummies.extend(length_feats)
        #print("########After getDF#########")
        print(set(dummies))
        return set(dummies)


    def _get_scalars(self,mode):

        xvars = [
         'nlp_title_sentiment_mag_mean',
         'nlp_title_sentiment_score_mean',
         'nlp_title_documentSentiment_mag',
         'nlp_title_documentSentiment_score']

        # Exclude binary features not found in the dataframe
        df_cols = set(self._df.columns)
        if mode == "score":
             scalar = []
             for i in self._LISTVARS:
                  for j in df_cols:
                       if j.startswith(i+'_'):
                            scalar.append(j)
             xvars = set(xvars + scalar)
        else:
             xvars = set(xvars)
        scalar_feats = (xvars & df_cols)
        # Log the features not found
        logging.warn('Did not find these scalar features: {}'.format(xvars - df_cols))

        return scalar_feats


    def to_df(self, data):
        def get_type_feature(x,key):
            if str(x).lower() == key:
                return 1
            else:
                return 0
        df = data    
        # df = df.loc[df['type'].isin(['video', 'carousel','image'])] 
        # for i in ['type_video','type_image','type_carousel']:
        #     df[i] = df['type'].apply(lambda x: get_type_feature(x,i.split("_")[-1]))
        df = self.get_transcription_entities(df)
        df = self.text_in_image(df)
        #print('-'*150)
        #print("Before get_binaries_for_color")
        # #print(set(df.columns))
        # df = self.get_binaries_for_color(df)
        #print("After get_binaries_for_color")
        #print(set(df.columns))
        #df = self.get_df_with_length_feats(df)
        #print("**************DF after adding length_feats**********************")
        #print(set(df.columns))
        return df
    
    def get_scalars_for_entities(self, entity):
        """This function will use for scalar extraction from entities on future"""
        df = self._df.copy()
        df.reset_index(drop = True)
        all_entities = set()
        def set_valid_keys(x):
            try:
                all_entities.update(json.loads(x).keys())
            except:
                pass
        def key_mapping_to_value(x, key):
            try:
                return json.loads(x)[key]
            except:
                return 0

        temp = df[entity].apply(lambda x: set_valid_keys(x))
        df1=pd.DataFrame([[0]*len(all_entities)] * df.shape[0], columns=([entity + "_" + x for x in list(all_entities)]))
        df = pd.concat([df,pd.DataFrame([[0]*len(all_entities)] * df.shape[0], columns=([entity + "_" + x for x in list(all_entities)]))], axis =1)
        df = df[pd.notna(df['account_id'])]
        for key in all_entities:
            df[entity+'_'+key] = df[entity].apply(lambda x: key_mapping_to_value(x, key))
        return df
    
    def get_binaries_for_entities(self, entity):
        """Extracting binaries features from *_entities columns"""
        df = self._df
        def get_clean_data(x):
            temp = [re.sub('[0-9]', '', i).replace("_","").replace("''","").replace('""',"") for i in x] 
            temp = [i for i in temp if len(i) > 2]
            return temp
        def set_valid_keys(x):
            try:
                return list(set([i.lower().replace("segment_label_","").replace("shot_label_","") for i in list(ast.literal_eval(str(x)).keys())]))
            except Exception as e:
                return []
        def get_vid_vision_keys(x):
            #print("INSIDE GET VISION KEYS")
            try:
                if isinstance(ast.literal_eval(str(x)),list):
                    return list(set([list(i.keys())[0] for i in ast.literal_eval(str(x)) if list(i.values())[0]>0.8]))
                return list(set([i.lower() for i in list(dict((k, v) for k, v in ast.literal_eval(str(x)).items() if v >= .8))]))
                #print("="*100)
                #print("VISION ENTITIES")
                #print(set([i.lower() for i in list(dict((k, v) for k, v in ast.literal_eval(str(x)).items() if v >= .8))]))
            except Exception as e:
                #print(e)
                return []
        def key_mapping_to_value(x, key):
            try:
                if key in x:
                    return 1
                else:
                    return 0
            except Exception as e:
                #print(e)
                return 0
        
        if entity in ['vision_entities']:
            df[entity] = df[entity].apply(lambda x: get_vid_vision_keys(x))
        else:
            df[entity] = df[entity].apply(lambda x: set_valid_keys(x))
        if entity in ['nlp_link_description_entities','nlp_title_entities','nlp_body_entities','vid_text_entities']:
            df[entity] = df[entity].apply(lambda x: get_clean_data(ast.literal_eval(str(x))))
        all_entities = [i for i in list(set(sum(list(df[entity]),[]))) if len(i) > 2]
        #print("%"*100,"set(all_entities)")
        #print(set(all_entities))
        for key in all_entities:
            df[entity+'_'+key] = df[entity].apply(lambda x: key_mapping_to_value(x, key))
        df = df[pd.notna(df['account_id'])]
        return df

    def get_transcription_entities(self,df):
        def find(key, dictionary):
            if type(dictionary) is type({}):
                for k, v in dictionary.items():
                    if k == key:
                        yield v
                    elif isinstance(v, dict):
                        for result in find(key, v):
                            yield result
                    elif isinstance(v, list):
                        for d in v:
                            for result in find(key, d):
                                yield result
                                
        def get_temp_intermediate(x):
            try:
                return list(find('name',ast.literal_eval(str(x))))
            except:
                return []
                                
        def get_ent(x):
            lst = []
            try:
                for i in ast.literal_eval(str(x))['entities']:
                    if i['type'] != 'OTHER':
                        lst.append(i['name'])
                return list(set(lst))
            except Exception as e:
                return []
            
        def keymapping(x,key):
            try:
                if key in x:
                    return 1
                else:
                    return 0
            except Exception as e:
                return 0
        # temp_series = df['nlp_video_transcription'].apply(lambda x: get_temp_intermediate(x))
        # all_ent = df['nlp_video_transcription'].apply(lambda x: get_ent(x))
        # all_entities = list(set(sum(list(all_ent),[])))
        # for k in all_entities:
        #     df['nlp_video_transcription' +'_'+k] = temp_series.apply(lambda x: keymapping(x,k))
        return df


    # def get_binaries_for_color(self,df):
        def find(key, dictionary):
            if type(dictionary) is type({}):
                for k, v in dictionary.items():
                    if k == key:
                        yield v
                    elif isinstance(v, dict):
                        for result in find(key, v):
                            yield result
                    elif isinstance(v, list):
                        for d in v:
                            for result in find(key, d):
                                yield result
        #all_entities = ['primary_color','secondary_color']
        all_entities = ['primary_color']
        def closest_colour(requested_colour):
            min_colours = {}
            for key, name in webcolors.css3_hex_to_names.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - requested_colour[0]) ** 2
                gd = (g_c - requested_colour[1]) ** 2
                bd = (b_c - requested_colour[2]) ** 2
                min_colours[(rd + gd + bd)] = name
            return min_colours[min(min_colours.keys())]

        def get_colour_name(requested_colour):
            try:
                closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
            except ValueError:
                closest_name = closest_colour(requested_colour)
                actual_name = None
            return actual_name, closest_name
        """
        def Actual_colour(closest_name_val):
            colour_list = ['yellow','orange','red','green','blue','brown','white','voilet','gray','pink','purple']
            flag = False
            for i in colour_list:
                if i in closest_name_val.lower():
                    flag = True
                    break
            if flag:
                return i
            else:
                return closest_name_val.lower()
        """
        def Actual_colour(closest_name_val):
            prefix_suffix_clr_lst = ['dim','dark','deep','blanched','olive','floral','forest','dodger','hot','lawn','light','le,mon','medium','midnight','mint','pale','misty','rosy','royal','saddle','sandy','sea','slate','lemon','yellow','orange','red','dark','green','blue','brown','white','voilet','gray','grey','pink','purple']
            clr_name = closest_name_val.lower()
            for clr in prefix_suffix_clr_lst:
                if closest_name_val.lower().endswith(clr): 
                    clr_name = closest_name_val.lower()[:-len(clr)] + " " +clr
                    break
                elif closest_name_val.lower().startswith(clr):
                    clr_name = clr + " " + closest_name_val.lower()[len(clr):]
                    break
            clr_name  = clr_name.strip()
            return clr_name

        def get_top_three_color(x,key):
            try:
                if x['vision_api_image_properties'] == '' or type(x['vision_api_image_properties']) == type(np.nan):
                    return None
                if x['type'] == 'carousel':
                    lst = [ast.literal_eval(i) for i in x['vision_api_image_properties'].split("|separator|")[:-1]]
                    df = pd.DataFrame({"r":list(find("red",lst[0])),"g":list(find("green",lst[0])),"b":list(find("blue",lst[0])),"score":list(find("score",lst[0]))})
                    for k in lst[1:]:
                        try:
                            df = df.append(pd.DataFrame({"r":list(find("red",k)),"g":list(find("green",k)),"b":list(find("blue",k)),"score":list(find("score",k))}))
                        except:
                            pass
                    df = df.sort_values(by = 'score', ascending=False)
                    clr = list(zip(list(df['r']),list(df['g']),list(df['b'])))
                if x['type'] == 'image':
                    lst = ast.literal_eval(x['vision_api_image_properties'])
                    df = pd.DataFrame({"r":list(find("red",lst)),"g":list(find("green",lst)),"b":list(find("blue",lst)),"score":list(find("score",lst))})
                    df = df.sort_values(by = 'score', ascending=False)
                    clr = list(zip(list(df['r']),list(df['g']),list(df['b'])))
                if key == "primary_color":
                    return Actual_colour(get_colour_name(clr[0])[1])
                elif key == "secondary_color":
                    return Actual_colour(get_colour_name(clr[1])[1])
                elif key == "tertiary_color":
                    return Actual_colour(get_colour_name(clr[2])[1])
            except Exception as e:
                print(e)
                return None
        for key in all_entities:
            df[key] = df.apply(lambda x: get_top_three_color(x, key),axis=1)
            #print("@"*100,"COLOURS")
            #print(key)
            #print(df[key])
        backup_color_column = df[all_entities]
        #df = pd.get_dummies(df, prefix=['primary_color_is','secondary_color_is'], columns=['primary_color','secondary_color'])
        df = pd.get_dummies(df, prefix=['primary_color_is'], columns=['primary_color'])
        df = pd.concat([df, backup_color_column], axis=1)
        print("@"*100,"colors:")
        print(df)
        return df

    def text_in_image(self, df):
        def get_text_in_image_or_video(x):
            try:
                fnl = []
                if '|separator|' in str(x):
                    all_content = str(x).split('|separator|')
                    all_content = [a for a in all_content if a.strip() not in ["", '""', "''"]]
                    for k in all_content:
                        k = ast.literal_eval(str(k))
                        fnl.extend(list((set(k['text'].split("\n"))- set(["", '""', "''"]))))
                    if fnl == []:
                        return []
                    else:
                        return fnl
                else:
                    x = ast.literal_eval(str(x))
                    return (list(set(x['text'].split("\n")) - set(["", '""', "''"])))
            except Exception as e:
                return []
            
        def keymapping(x,i):
            try:
                if i in ast.literal_eval(str(x)):
                    return 1
                else:
                    return 0
            except:
                return 0
        df['vision_api_full_text'] = df['vision_api_full_text'].apply(get_text_in_image_or_video)
        df['vision_api_full_text'] = df['vision_api_full_text'].apply(lambda x: list(set([" ".join(re.findall("[一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[a-zA-Z]+|[ａ-ｚＡ-Ｚ]+|[々〆〤]+", word)) if len(" ".join(re.findall("[一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[a-zA-Z]+|[ａ-ｚＡ-Ｚ]+|[々〆〤]+", word))) > 3 else "" for word in ast.literal_eval(str(x))])-set(["", '""', "''"])))
        all_entities = list(set(sum(list(df['vision_api_full_text']),[])))
        for i in all_entities:
            df['vision_api_full_text'+'_'+i] = df['vision_api_full_text'].apply(lambda x: keymapping(x,i))
        return df

    def get_df_with_length_feats(self):
        #print("########Inside get_Df###########")
        #df = self._df
        #print("-"*100,"BEFORE:")
        #print(len(self._df.columns))
        def fun(text):
            #print("TEXT")
            #print(text)
            length=0
            try:
                length = len(text.split())
                range_str =  str(2**int(math.log2(length)))+'-'+str(2**(int(math.log2(length)+1)))
            except Exception as e:
                print(e)
                range_str = 'NA'
            return range_str
        dummy_df = pd.get_dummies(self._df['body'].apply(fun))
        #print("^"*100)
        #print(set(dummy_df.columns))
        if 'NA' in dummy_df.columns:
            dummy_df = dummy_df.drop(labels=['NA'],axis=1)
        self._df = pd.concat([self._df,dummy_df],axis=1)
        #print("-"*100,"AFTER:")
        #print(len(self._df.columns))
        #print(set(self._df.columns))
        return self._df, dummy_df.columns
    


