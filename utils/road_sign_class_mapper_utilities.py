import os
import pandas as pd

from config.road_sign_mapper_configuration import RoadSignMapperConfiguration


class RoadSignClassMapperUtilities:
    """
    Provides basic methods for dealing with road sign class ids and names.
    """

    _NAME_CLASS_DATAFRAME = pd.read_csv(os.path.join(RoadSignMapperConfiguration.RESOURCES_PATH,
                                                     RoadSignMapperConfiguration.ROAD_SIGN_CSV_FILE_NAME))

    @staticmethod
    def get_class_by_name(name: str) -> int:
        """
        Given a road sign name, this method returns the corresponding class id.
        :param name: The road sign name.
        :return: The class id.
        """
        result_df = RoadSignClassMapperUtilities._NAME_CLASS_DATAFRAME[
            RoadSignClassMapperUtilities._NAME_CLASS_DATAFRAME[RoadSignMapperConfiguration.LABEL_SIGN_NAME] == name]

        if result_df.shape[0] == 1:
            return result_df.iloc[0][RoadSignMapperConfiguration.LABEL_CLASS_ID]
        else:
            raise ValueError('Name: {} not found'.format(name))

    @staticmethod
    def get_name_by_class(class_id: int) -> str:
        """
        Given a class id this methods returns the corresponding road sign name.
        :param class_id: The class id.
        :return: The class/road sign name.
        """
        result_df = RoadSignClassMapperUtilities._NAME_CLASS_DATAFRAME[
            RoadSignClassMapperUtilities._NAME_CLASS_DATAFRAME[RoadSignMapperConfiguration.LABEL_CLASS_ID] == class_id]
        if result_df.shape[0] == 1:
            name = result_df.iloc[0][RoadSignMapperConfiguration.LABEL_SIGN_NAME]
            if name.startswith(RoadSignMapperConfiguration.LABEL_NULL):
                return None
            return name
        else:
            raise ValueError('ClassId: {} not found'.format(class_id))
