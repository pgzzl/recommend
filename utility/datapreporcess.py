import pandas as pd
import os


current_path=os.getcwd()
dataset_path=os.path.join(current_path,'original dataset','category_del_repet','appInfoCOMICS.json')
csv_applib_path=os.path.join(current_path,'original dataset','app_lib.csv')
csv_app_path=os.path.join(current_path,'original dataset','app.csv')
csv_lib_path=os.path.join(current_path,'original dataset','lib.csv')

print('dataset:'+dataset_path)

df=pd.read_json(dataset_path)
selected_df=df.loc[:,['package','libraries']]
print(selected_df.head())

def find_key_by_value(dictionary, value):
    # 遍历字典中的键值对
    for key, val in dictionary.items():
        # 如果找到与目标值匹配的值，则返回对应的键
        if val == value:
            return key
    # 如果未找到匹配的值，则返回 None
    return None

def save_app_lib(apppath,libpath,applib_path,df):
    app_dict={}
    lib_dict={}
    app_lib_df=pd.DataFrame()
    app_lib_df['app']=None
    app_lib_df['lib']=None
    new_rows=[]
    lib_index=0
    for index,row in df.iterrows():
        app_dict[index]=row[0]
        if row[1]:
            for item in row[1]:
                if item not in lib_dict.values():
                    lib_dict[lib_index]=item
                    new_rows.append({'app':index,'lib':lib_index})
                    lib_index+=1
                else:
                    i=find_key_by_value(lib_dict,item)
                    new_rows.append({'app':index,'lib':i})
    app_lib_df = app_lib_df.append(new_rows, ignore_index=True)
    app_lib_df=app_lib_df.sort_values(by=['app','lib'])                
    app_df = pd.DataFrame.from_dict(app_dict, orient='index')
    lib_df=pd.DataFrame.from_dict(lib_dict, orient='index')
    print(app_df.head())
    app_df.to_csv(apppath,index=True,header=False)
    lib_df.to_csv(libpath,index=True,header=False)
    app_lib_df.to_csv(applib_path,index=False,header=False)



save_app_lib(csv_app_path,csv_lib_path,csv_applib_path,selected_df)








