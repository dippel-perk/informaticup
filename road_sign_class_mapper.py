import pandas as pd


class RoadSignClassMapper:
    def __init__(self, file_name='road_sign_names.csv'):
        self.name_class_df = pd.read_csv(file_name)

    def get_class_by_name(self, name: str) -> int:
        result_df = self.name_class_df[self.name_class_df['SignName'] == name]
        if result_df.shape[0] == 1:
            return result_df.iloc[0]["ClassId"]
        else:
            raise ValueError('Name: {} not found'.format(name))

    def get_name_by_class(self, classId: int) -> str:
        result_df = self.name_class_df[self.name_class_df['ClassId'] == classId]
        if result_df.shape[0] == 1:
            return result_df.iloc[0]["SignName"]
        else:
            raise ValueError('ClassId: {} not found'.format(classId))
